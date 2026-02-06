#!/usr/bin/env python3
"""
GPU Worker Entry Point

Main entry point for the GPU worker that processes TTS and ASR jobs from GitLab.
Scans for available jobs, claims them, executes handlers, and uploads results.

Usage:
    # Normal mode: scan and claim next available job
    python worker_entry.py --token YOUR_TOKEN --worker-id gpu-worker-1
    python worker_entry.py --loop-interval 300  # Poll every 5 minutes
    python worker_entry.py --work-dir /tmp/gpu_work --keep-jobs 10

    # Force mode: skip claim, run a specific already-claimed job
    python worker_entry.py --token YOUR_TOKEN --worker-id any --force-job 123:my_job_1
"""

import os
import sys
import time
import shutil
import importlib
import argparse
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import yaml

from gitlab.exceptions import GitlabCreateError
from gitlab_jobs import GitLabJobScanner, NoJobsAvailableError, DEFAULT_GITLAB_URL, RESPONSE_FILE_PATH_TEMPLATE


class JobCanceledException(Exception):
    """Raised when a job is canceled. Handler can catch to cleanup before re-raising."""


class TeeLogger:
    """Write to both file and stdout/stderr."""

    def __init__(self, log_path: Path, original_stream):
        self.log_file = open(log_path, 'a', encoding='utf-8')
        self.original_stream = original_stream

    def write(self, message):
        self.log_file.write(message)
        self.log_file.flush()
        self.original_stream.write(message)
        self.original_stream.flush()

    def flush(self):
        self.log_file.flush()
        self.original_stream.flush()

    def close(self):
        self.log_file.close()


class JobCallbacks:
    """
    Callbacks object passed to job handlers.
    Provides shared functionality for heartbeat, file operations, etc.
    """

    HEARTBEAT_INTERVAL = 300  # 5 minutes

    def __init__(
        self,
        job_context: Dict[str, Any],
        scanner: GitLabJobScanner,
        work_dir: Path,
        worker_id: str,
        log_path: Path
    ):
        self.job_context = job_context
        self.scanner = scanner
        self.work_dir = work_dir
        self.worker_id = worker_id
        self.log_path = log_path
        self.project_id = job_context['project_id']
        self.job_id = job_context['job_id']
        self._last_heartbeat = time.time()
        self._epochs_completed = 0

    def heartbeat(self, epochs_completed: Optional[int] = None, message: Optional[str] = None):
        """
        Update response.yaml with timestamp and optional progress.
        Checks for cancellation - raises JobCanceledException if canceled.

        Args:
            epochs_completed: Optional progress indicator
            message: Optional status message

        Raises:
            JobCanceledException: If job is canceled or claimed by another worker
        """
        if epochs_completed is not None:
            self._epochs_completed = epochs_completed

        # Check if job still exists and we still own it
        job_status = self.scanner.get_job_status(self.project_id, self.job_id)

        if not job_status:
            raise JobCanceledException("Job removed from manifest")

        if job_status.get('canceled'):
            raise JobCanceledException("Job canceled by user")

        response = job_status.get('response', {})
        if response and response.get('worker_id') != self.worker_id:
            raise JobCanceledException("Job claimed by another worker")

        # Update response.yaml with heartbeat
        self._update_response(
            state='running',
            epochs_completed=self._epochs_completed,
            message=message
        )

        self._last_heartbeat = time.time()

    def check_heartbeat_needed(self):
        """Check if heartbeat is needed based on time elapsed."""
        if time.time() - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.heartbeat()

    def _update_response(
        self,
        state: str,
        epochs_completed: Optional[int] = None,
        error_message: Optional[str] = None,
        termination_reason: Optional[str] = None,
        message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ):
        """Update the response.yaml file in GitLab."""

        response_data = {
            'worker_id': self.worker_id,
            'state': state,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if epochs_completed is not None:
            response_data['epochs_completed'] = epochs_completed

        if error_message:
            response_data['error_message'] = error_message

        if termination_reason:
            response_data['termination_reason'] = termination_reason

        if message:
            response_data['status_message'] = message

        # Include additional result data from handler (metrics, paths, etc.)
        if result_data:
            response_data['result'] = result_data

        yaml_content = yaml.dump(response_data, default_flow_style=False, sort_keys=False)

        response_path = f"gpu_jobs/job_{self.job_id}/response.yaml"

        try:
            self.scanner.update_file(
                project_id=self.project_id,
                file_path=response_path,
                content=yaml_content,
                commit_message=f"Worker {self.worker_id} updates job {self.job_id} - {state}"
            )
        except Exception as e:
            print(f"Warning: Failed to update response.yaml: {e}", file=sys.stderr)

    def get_work_dir(self) -> Path:
        """Get the job-specific work directory."""
        return self.work_dir

    def get_job_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from the job context with default."""
        return self.job_context.get(key, default)

    def mark_complete(self, result_data: Optional[Dict[str, Any]] = None):
        """Mark the job as completed with optional result data."""
        self._update_response(
            state='completed',
            epochs_completed=self._epochs_completed,
            result_data=result_data
        )

    def mark_failed(self, error_message: str):
        """Mark the job as failed with an error message."""
        self._update_response(
            state='failed',
            epochs_completed=self._epochs_completed,
            error_message=error_message
        )

    def mark_canceled(self, reason: str):
        """Mark the job as canceled with a reason."""
        self._update_response(
            state='canceled',
            epochs_completed=self._epochs_completed,
            termination_reason=reason
        )


def validate_checkpoint_path(path: str) -> bool:
    """
    Validate that checkpoint path is safe.
    Must start with gpu_jobs/ and not contain path traversal.
    """
    if not path:
        return True  # Empty path is valid (no checkpoint)

    # Must start with gpu_jobs/
    if not path.startswith('gpu_jobs/'):
        return False

    # Must not contain path traversal
    if '..' in path:
        return False

    return True


def load_handler(job_type: str, model_type: str, mode: str):
    """
    Dynamically load the handler module based on job configuration.

    Args:
        job_type: 'tts' or 'asr'
        model_type: e.g., 'StableTTS', 'W2V2BERT'
        mode: 'training', 'inference', or 'training_and_inference'

    Returns:
        The handler module with a run() function
    """
    # Normalize model type to lowercase for module path
    model_type_lower = model_type.lower().replace('-', '').replace('_', '')

    # Map common variations
    model_type_map = {
        'stabletts': 'stabletts',
        'w2v2bert': 'w2v2bert',
        'wav2vec2bert': 'w2v2bert',
    }

    model_module = model_type_map.get(model_type_lower, model_type_lower)

    # Build module path: handlers.{job_type}.{model}.{mode}
    module_path = f"handlers.{job_type}.{model_module}.{mode}"

    try:
        handler_module = importlib.import_module(module_path)

        if not hasattr(handler_module, 'run'):
            raise ImportError(f"Handler module {module_path} does not have a run() function")

        return handler_module

    except ImportError as e:
        raise ImportError(
            f"Could not load handler for job_type={job_type}, model={model_type}, mode={mode}. "
            f"Expected module at {module_path}. Error: {e}"
        ) from e


def setup_job_logging(work_dir: Path) -> Path:
    """
    Setup logging for a job - redirect stdout/stderr to both file and console.

    Returns:
        Path to the log file
    """
    log_path = work_dir / "logs.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create tee loggers
    sys.stdout = TeeLogger(log_path, sys.__stdout__)
    sys.stderr = TeeLogger(log_path, sys.__stderr__)

    return log_path


def restore_logging():
    """Restore original stdout/stderr."""
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    if isinstance(sys.stderr, TeeLogger):
        sys.stderr.close()
        sys.stderr = sys.__stderr__


def upload_logs(scanner: GitLabJobScanner, project_id: int, job_id: str, log_path: Path):
    """Upload log file to GitLab."""
    if not log_path.exists():
        return

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        remote_path = f"gpu_jobs/job_{job_id}/logs.txt"

        # Try to create or update the file
        try:
            scanner.create_file(
                project_id=project_id,
                file_path=remote_path,
                content=log_content,
                commit_message=f"Upload logs for job {job_id}"
            )
        except GitlabCreateError:
            # File exists, update it
            scanner.update_file(
                project_id=project_id,
                file_path=remote_path,
                content=log_content,
                commit_message=f"Update logs for job {job_id}"
            )

        print(f"Uploaded logs to {remote_path}")

    except Exception as e:
        print(f"Warning: Failed to upload logs: {e}", file=sys.stderr)


def cleanup_old_jobs(work_dir: Path, keep_count: int):
    """Keep only the N most recent job directories."""
    if keep_count < 0:
        return  # Keep all

    # Find all job directories
    job_dirs = []
    if work_dir.exists():
        for item in work_dir.iterdir():
            if item.is_dir() and item.name.startswith('job_'):
                job_dirs.append(item)

    if len(job_dirs) <= keep_count:
        return

    # Sort by modification time (newest first)
    job_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Remove old directories
    for old_dir in job_dirs[keep_count:]:
        try:
            shutil.rmtree(old_dir)
            print(f"Cleaned up old job directory: {old_dir}")
        except Exception as e:
            print(f"Warning: Failed to cleanup {old_dir}: {e}", file=sys.stderr)


def process_job(
    job: Dict[str, Any],
    scanner: GitLabJobScanner,
    worker_id: str,
    base_work_dir: Path,
    force_mode: bool = False
) -> bool:
    """
    Process a single job.

    Returns:
        True if job completed successfully, False otherwise
    """
    job_id = job['job_id']
    project_id = job['project_id']

    # Create job-specific work directory.
    # In force mode (debugging), omit the timestamp so the same directory is
    # reused across runs — this allows previously downloaded data to be found.
    if force_mode:
        work_dir = base_work_dir / f"job_{job_id}"
    else:
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        work_dir = base_work_dir / f"job_{job_id}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = setup_job_logging(work_dir)

    print("=" * 60)
    print(f"Processing Job: {job_id}")
    print(f"Project ID: {project_id}")
    print(f"Work Directory: {work_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Create callbacks
    callbacks = JobCallbacks(
        job_context=job,
        scanner=scanner,
        work_dir=work_dir,
        worker_id=worker_id,
        log_path=log_path
    )

    success = False
    should_upload_logs = True

    try:
        # Extract job configuration
        job_type = job.get('job_type', 'tts').lower()
        model_config = job.get('model', {})
        model_type = model_config.get('type', 'StableTTS')
        mode = job.get('mode', 'training')

        # Validate checkpoint path if specified
        base_checkpoint = model_config.get('base_checkpoint', '')
        if not validate_checkpoint_path(base_checkpoint):
            raise ValueError(
                f"Invalid checkpoint path: {base_checkpoint}. "
                "Must start with 'gpu_jobs/' and not contain '..'"
            )

        print(f"Job Type: {job_type}")
        print(f"Model Type: {model_type}")
        print(f"Mode: {mode}")
        print()

        # Load handler
        print(f"Loading handler for {job_type}/{model_type}/{mode}...")
        handler = load_handler(job_type, model_type, mode)
        print(f"Handler loaded: {handler.__name__}")
        print()

        # Execute handler
        print("Executing handler...")
        result = handler.run(job, callbacks)

        if result.get('success', False):
            print("\nJob completed successfully!")
            # Extract result data to include in response.yaml
            # Remove internal fields, keep metrics and paths
            result_data = {k: v for k, v in result.items()
                          if k not in ('success', 'error_message') and v is not None}
            callbacks.mark_complete(result_data=result_data if result_data else None)
            success = True
        else:
            error_msg = result.get('error_message', 'Unknown error')
            print(f"\nJob failed: {error_msg}")
            callbacks.mark_failed(error_msg)

    except JobCanceledException as e:
        reason = str(e)
        print(f"\nJob canceled: {reason}")

        # Only update response if we still own the job
        if "claimed by another worker" not in reason.lower():
            callbacks.mark_canceled(reason)
        else:
            should_upload_logs = False

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nJob failed with exception: {error_msg}")
        print(traceback.format_exc())
        callbacks.mark_failed(error_msg)

    finally:
        # Restore logging before uploading
        restore_logging()

        # Upload logs
        if should_upload_logs:
            upload_logs(scanner, project_id, job_id, log_path)

        print()
        print(f"Finished: {datetime.now().isoformat()}")
        print("=" * 60)

    return success


def main():
    parser = argparse.ArgumentParser(
        description='GPU Worker - Process TTS and ASR jobs from GitLab',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Authentication
    parser.add_argument(
        '--token',
        type=str,
        default=os.environ.get('GITLAB_TOKEN'),
        help='GitLab access token (or set GITLAB_TOKEN env var)'
    )
    parser.add_argument(
        '--worker-id',
        type=str,
        default=os.environ.get('WORKER_ID'),
        help='Unique worker identifier (or set WORKER_ID env var)'
    )

    # GitLab configuration
    parser.add_argument(
        '--gitlab-url',
        type=str,
        default=DEFAULT_GITLAB_URL,
        help='GitLab server URL'
    )

    # Work directory
    parser.add_argument(
        '--work-dir',
        type=str,
        default=os.environ.get('WORK_DIR', './work'),
        help='Base directory for job work files (or set WORK_DIR env var)'
    )

    # Loop configuration
    parser.add_argument(
        '--loop-interval',
        type=int,
        default=-1,
        help='Seconds between job checks (-1 = exit when no jobs available)'
    )

    # Cleanup configuration
    parser.add_argument(
        '--keep-jobs',
        type=int,
        default=5,
        help='Number of completed job directories to keep (-1 = keep all)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    # Force-run a specific job (skip claim)
    parser.add_argument(
        '--force-job',
        type=str,
        default=None,
        help='Force-run a specific job, skipping claim. Format: PROJECT_ID:JOB_ID'
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.token:
        print("ERROR: GitLab token required. Use --token or set GITLAB_TOKEN env var.")
        sys.exit(1)

    if not args.worker_id:
        print("ERROR: Worker ID required. Use --worker-id or set WORKER_ID env var.")
        sys.exit(1)

    # Setup
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GPU Worker Starting")
    print("=" * 60)
    print(f"Worker ID: {args.worker_id}")
    print(f"GitLab URL: {args.gitlab_url}")
    print(f"Work Directory: {work_dir.absolute()}")
    if args.force_job:
        print(f"FORCE MODE: {args.force_job}")
    else:
        print(f"Loop Interval: {args.loop_interval}s" if args.loop_interval >= 0 else "Loop Interval: Exit when done")
    print(f"Keep Jobs: {args.keep_jobs}")
    print("=" * 60)
    print()

    # Create scanner
    try:
        scanner = GitLabJobScanner(
            token=args.token,
            gitlab_url=args.gitlab_url,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"ERROR: Failed to connect to GitLab: {e}")
        sys.exit(1)

    # Main loop
    jobs_processed = 0
    jobs_succeeded = 0

    while True:
        try:
            if args.force_job:
                # === FORCE MODE: skip claim, run a specific job ===
                try:
                    project_id_str, job_id = args.force_job.split(':', 1)
                    project_id = int(project_id_str)
                except ValueError:
                    print(f"ERROR: Invalid --force-job format: '{args.force_job}'")
                    print("Expected format: PROJECT_ID:JOB_ID (e.g., 123:my_job_1)")
                    sys.exit(1)

                print(f"FORCE MODE: Running job {job_id} from project {project_id} (skipping claim)")

                # Fetch job metadata without claiming
                job = scanner.get_job_status(project_id, job_id)
                if not job:
                    print(f"ERROR: Job {job_id} not found in project {project_id}")
                    sys.exit(1)

                if job.get('is_claimed') and job.get('response'):
                    # Job already claimed — adopt the existing worker_id
                    # so heartbeat checks pass without modifying response.yaml
                    existing_worker_id = job['response'].get('worker_id')
                    if existing_worker_id:
                        print(f"FORCE MODE: Adopting existing worker_id: {existing_worker_id}")
                        args.worker_id = existing_worker_id
                else:
                    # Job not yet claimed — create claim with our worker_id
                    print(f"FORCE MODE: Job not claimed yet, creating claim with worker_id: {args.worker_id}")
                    response_path = RESPONSE_FILE_PATH_TEMPLATE.format(job_id=job_id)
                    response_data = {
                        'worker_id': args.worker_id,
                        'state': 'running',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    scanner.create_file(
                        project_id=project_id,
                        file_path=response_path,
                        content=yaml.dump(response_data, default_flow_style=False, sort_keys=False),
                        commit_message=f"Worker {args.worker_id} force-claims job {job_id}"
                    )
                    # Re-fetch job status now that it's claimed
                    job = scanner.get_job_status(project_id, job_id)

                # Run the job
                success = process_job(job, scanner, args.worker_id, work_dir, force_mode=True)
                jobs_processed += 1
                if success:
                    jobs_succeeded += 1

                # Cleanup old job directories
                cleanup_old_jobs(work_dir, args.keep_jobs)

                # Force mode: run once and exit
                break

            else:
                # === NORMAL MODE: scan and claim next available job ===
                print("Scanning for available jobs...")
                job = scanner.claim_next_job(args.worker_id)

                print(f"Claimed job: {job['job_id']}")

                success = process_job(job, scanner, args.worker_id, work_dir)

                jobs_processed += 1
                if success:
                    jobs_succeeded += 1

                # Cleanup old job directories
                cleanup_old_jobs(work_dir, args.keep_jobs)

        except NoJobsAvailableError:
            print("No jobs available.")

            if args.loop_interval < 0:
                print("Exiting (no loop configured).")
                break

            print(f"Waiting {args.loop_interval} seconds before next scan...")
            time.sleep(args.loop_interval)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            print(traceback.format_exc())

            # In force mode, don't loop on error — exit immediately
            if args.force_job:
                sys.exit(1)

            if args.loop_interval < 0:
                sys.exit(1)

            print(f"Waiting {args.loop_interval} seconds before retry...")
            time.sleep(args.loop_interval)

    # Summary
    print()
    print("=" * 60)
    print("GPU Worker Summary")
    print("=" * 60)
    print(f"Jobs Processed: {jobs_processed}")
    print(f"Jobs Succeeded: {jobs_succeeded}")
    print(f"Jobs Failed: {jobs_processed - jobs_succeeded}")
    print("=" * 60)


if __name__ == '__main__':
    main()
