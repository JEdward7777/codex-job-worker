#!/usr/bin/env python3
"""
Monitor Process — Auto-scaling GPU Worker Manager

This script runs a single monitoring cycle:
  1. Scans GitLab for active jobs (pending + running)
  2. Calculates how many workers are needed: ceil(active_jobs / JOBS_PER_WORKER)
  3. Checks how many SkyPilot workers are currently running
  4. Launches new workers if more are needed (up to MAX_WORKERS)
  5. Cleans up workers in invalid states (STOPPED, stale INIT)

Designed to be run via cron (e.g., every 10 minutes). If the process crashes,
cron simply runs it again next cycle — no long-running daemon to babysit.

Usage:
    # Single monitoring cycle (for cron)
    uv run python monitor.py run

    # Dry run — verify GitLab scanning and SkyPilot connectivity
    uv run python monitor.py run --dry-run

    # Override settings
    uv run python monitor.py run --max-workers=3 --jobs-per-worker=2

    # Show current status without taking action
    uv run python monitor.py status

    # Test SkyPilot with a lightweight dryrun VM
    uv run python monitor.py test_launch --dry-run

    # Test full worker pipeline (launches real worker, scans for jobs, exits)
    uv run python monitor.py test_launch

Cron example (every 10 minutes):
    */10 * * * * flock -n /tmp/monitor.lock -c 'cd /path/to/launcher_project && bash run_monitor_cron.sh' >> /var/log/monitor_cron.log 2>&1
"""

import gzip
import io
import logging
import logging.handlers
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import fire
from dotenv import load_dotenv

# Add parent directory to path so we can import gitlab_jobs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gitlab_jobs import GitLabJobScanner  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKER_PREFIX = "codex-worker-"
EXIT_CODE_UPDATE_RESTART = 22

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _gzip_rotator(source: str, dest: str) -> None:
    """Compress rotated log files with gzip."""
    with open(source, 'rb') as f_in:
        with gzip.open(f'{dest}.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(source)


def _gzip_namer(name: str) -> str:
    """Append .gz to rotated log file names."""
    return name + '.gz'


def setup_logging(log_file: str = 'monitor.log') -> logging.Logger:
    """
    Configure logging to both stdout and a rotating log file with gzip compression.

    Args:
        log_file: Path to the log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger('monitor')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplicate messages from root logger

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Rotating file handler with gzip compression
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.rotator = _gzip_rotator
    file_handler.namer = _gzip_namer
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# SkyPilot helpers
# ---------------------------------------------------------------------------

def _get_sky_clusters(logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Get all SkyPilot clusters matching our worker prefix.

    Returns a list of dicts with keys: name, status, launched_at, status_updated_at
    """
    try:
        import sky
        request_id = sky.status(refresh=sky.StatusRefreshMode.AUTO)
        # follow=True is required — follow=False returns None immediately
        # without waiting for the API server to produce the result.
        clusters = sky.stream_and_get(request_id, follow=True)
    except Exception as e:
        logger.error(f"Failed to get sky status: {e}")
        return []

    if not clusters:
        return []

    our_clusters = []
    for cluster in clusters:
        name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', '')
        if name.startswith(WORKER_PREFIX):
            status = cluster.status if hasattr(cluster, 'status') else cluster.get('status')
            launched_at = (cluster.launched_at if hasattr(cluster, 'launched_at')
                          else cluster.get('launched_at'))
            status_updated_at = (cluster.status_updated_at
                                 if hasattr(cluster, 'status_updated_at')
                                 else cluster.get('status_updated_at'))
            our_clusters.append({
                'name': name,
                'status': status,
                'launched_at': launched_at,
                'status_updated_at': status_updated_at,
            })

    return our_clusters


def _sky_down(cluster_name: str, logger: logging.Logger) -> bool:
    """Tear down a SkyPilot cluster. Returns True on success."""
    try:
        import sky
        logger.info(f"Tearing down cluster: {cluster_name}")
        request_id = sky.down(cluster_name)
        # follow=True is required — follow=False returns immediately
        # without waiting for the teardown to complete.
        sky.stream_and_get(request_id, follow=True)
        logger.info(f"Successfully tore down: {cluster_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to tear down {cluster_name}: {e}")
        return False


def _sky_launch(
    yaml_path: str,
    cluster_name: str,
    envs: Dict[str, str],
    logger: logging.Logger,
    dry_run: bool = False,
    down: bool = True,
    idle_minutes_to_autostop: Optional[int] = None,
    stream: bool = False,
) -> bool:
    """
    Launch a SkyPilot cluster using the Python API.

    Args:
        yaml_path: Path to the SkyPilot YAML task file.
        cluster_name: Name for the SkyPilot cluster.
        envs: Environment variables to set on the task.
        logger: Logger instance.
        dry_run: If True, log what would happen but don't launch.
        down: If True, auto-terminate VM when the run script exits.
        idle_minutes_to_autostop: Minutes of idleness before auto-stopping.
            Combined with down=True, the VM tears down after this many idle
            minutes. If None, SkyPilot uses its default (1 minute).
        stream: If True, stream logs and wait for the run to complete.
                If False, return after provisioning + setup (don't wait for run).

    Returns True on success.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would launch cluster '{cluster_name}' "
                     f"from {yaml_path} (down={down}, "
                     f"idle_minutes_to_autostop={idle_minutes_to_autostop}, "
                     f"envs={list(envs.keys())})")
        return True

    logger.info(f"Launching cluster: {cluster_name} from {yaml_path}")

    try:
        import sky

        # Load the task from YAML
        task = sky.Task.from_yaml(yaml_path)

        # Set environment variables on the task
        task.update_envs(envs)

        # Launch the task
        launch_kwargs = dict(
            task=task,
            cluster_name=cluster_name,
            down=down,
        )
        if idle_minutes_to_autostop is not None:
            launch_kwargs['idle_minutes_to_autostop'] = idle_minutes_to_autostop

        request_id = sky.launch(**launch_kwargs)

        # stream_and_get processes the request and returns the result.
        # With follow=True, it streams logs to stdout (for interactive use).
        # With follow=False, it returns immediately — use only when you
        # don't need the result (fire-and-forget launch).
        sky.stream_and_get(request_id, follow=stream)

        logger.info(f"Successfully launched: {cluster_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to launch {cluster_name}: {e}")
        return False


def _check_skypilot_configured(logger: logging.Logger) -> bool:
    """
    Verify that SkyPilot is installed and cloud credentials are configured.

    Returns True if at least one cloud is enabled.
    """
    try:
        result = subprocess.run(
            ['sky', 'check'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # sky check returns 0 even if some clouds fail, as long as it runs
        if result.returncode != 0:
            logger.error(
                "SkyPilot check failed. Please run 'sky check' and configure "
                "your cloud credentials.\n"
                "See: https://skypilot.readthedocs.io/en/latest/getting-started/installation.html"
            )
            return False

        # Check if at least one cloud is enabled
        output = result.stdout + result.stderr
        if 'enabled' not in output.lower():
            logger.error(
                "No cloud providers enabled in SkyPilot. Please run 'sky check' "
                "and configure at least one cloud provider.\n"
                "For Vast.ai: https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot"
            )
            return False

        return True
    except FileNotFoundError:
        logger.error(
            "SkyPilot CLI not found. Install it with: pip install 'skypilot[vast]'\n"
            "Or: uv add 'skypilot[vast]'"
        )
        return False
    except Exception as e:
        logger.error(f"Error checking SkyPilot: {e}")
        return False


# ---------------------------------------------------------------------------
# Job counting
# ---------------------------------------------------------------------------

def count_active_jobs(scanner: GitLabJobScanner, logger: logging.Logger) -> int:
    """
    Count jobs that are in pending or running state.

    Active = not completed, not failed, not canceled.

    Uses list_available_jobs(include_claimed=True) to get both pending and
    running jobs (excluding canceled), then filters out completed and failed.
    """
    try:
        # include_claimed=True returns both unclaimed (pending) and claimed jobs
        # It already excludes canceled jobs
        all_jobs = scanner.list_available_jobs(include_claimed=True)
    except Exception as e:
        logger.error(f"Failed to scan GitLab for jobs: {e}")
        return 0

    active_count = 0
    for job in all_jobs:
        # Check response state — if claimed, look at the response
        response = job.get('response')
        if response and isinstance(response, dict):
            state = response.get('state', '')
            if state in ('completed', 'failed', 'canceled'):
                continue

        active_count += 1

    logger.info(f"Found {active_count} active jobs ({len(all_jobs)} total non-canceled)")
    return active_count


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class Monitor:
    """
    GPU Worker Monitor — auto-scales SkyPilot workers based on job demand.

    Designed to be run as a single cycle via cron. Each invocation:
    1. Scans GitLab for active jobs
    2. Calculates desired worker count
    3. Cleans up invalid clusters
    4. Launches new workers if needed
    """

    def run(
        self,
        dry_run: bool = False,
        gitlab_token: Optional[str] = None,
        gitlab_url: Optional[str] = None,
        max_workers: Optional[int] = None,
        min_workers: Optional[int] = None,
        jobs_per_worker: Optional[int] = None,
        init_timeout: Optional[int] = None,
        worker_yaml: Optional[str] = None,
        log_file: Optional[str] = None,
        idle_minutes: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Run a single monitoring cycle.

        Args:
            dry_run: If True, scan jobs and check status but don't launch or
                     tear down any clusters. Logs what would happen.
            gitlab_token: GitLab access token (default: from .env or GITLAB_TOKEN env var).
            gitlab_url: GitLab server URL (default: from .env or https://git.genesisrnd.com).
            max_workers: Maximum concurrent workers (default: from .env or 5).
            min_workers: Minimum workers to keep running even with 0 jobs (default: 0).
                         Useful for testing worker detection. Workers launched to meet
                         the minimum use idle_minutes auto-teardown instead of immediate
                         exit, so they stay alive long enough for the next cycle to see them.
            jobs_per_worker: Jobs per worker ratio (default: from .env or 4).
            init_timeout: Seconds before tearing down stale INIT clusters (default: from .env or 1800).
            worker_yaml: Path to SkyPilot worker YAML (default: from .env or skypilot_worker.yaml).
            log_file: Path to log file (default: from .env or monitor.log).
            idle_minutes: Minutes of idle time before auto-teardown (default: from
                          .env or 7). Passed to SkyPilot as idle_minutes_to_autostop.
                          Combined with --down, the VM tears down after being idle
                          for this many minutes. If not set, SkyPilot uses its default.
            verbose: Enable verbose GitLab scanning output.
        """
        # Load .env file (does not override existing env vars)
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)

        # Resolve configuration: CLI arg > env var > default
        gitlab_token = gitlab_token or os.environ.get('GITLAB_TOKEN')
        gitlab_url = gitlab_url or os.environ.get('GITLAB_URL', 'https://git.genesisrnd.com')
        max_workers = max_workers if max_workers is not None else int(os.environ.get('MAX_WORKERS', '5'))
        min_workers = min_workers if min_workers is not None else int(os.environ.get('MIN_WORKERS', '0'))
        jobs_per_worker = jobs_per_worker if jobs_per_worker is not None else int(os.environ.get('JOBS_PER_WORKER', '4'))
        init_timeout = init_timeout if init_timeout is not None else int(os.environ.get('INIT_TIMEOUT', '1800'))
        worker_yaml = worker_yaml or os.environ.get('WORKER_YAML', 'skypilot_worker.yaml')
        log_file = log_file or os.environ.get('LOG_FILE', 'monitor.log')
        idle_minutes = idle_minutes if idle_minutes is not None else int(os.environ.get('IDLE_MINUTES', '1'))

        # Setup logging
        logger = setup_logging(log_file)

        mode_label = "[DRY RUN] " if dry_run else ""
        logger.info(f"{mode_label}=== Monitor cycle starting ===")
        logger.info(f"{mode_label}Config: max_workers={max_workers}, min_workers={min_workers}, "
                     f"jobs_per_worker={jobs_per_worker}, idle_minutes={idle_minutes}, "
                     f"init_timeout={init_timeout}s, worker_yaml={worker_yaml}")

        # --- Preflight checks ---

        if not gitlab_token:
            logger.error(
                "GITLAB_TOKEN not set. Provide it via:\n"
                "  1. launcher_project/.env file (copy from .env.template)\n"
                "  2. GITLAB_TOKEN environment variable\n"
                "  3. --gitlab-token CLI argument"
            )
            sys.exit(1)

        # Resolve worker YAML path relative to this script's directory
        script_dir = Path(__file__).parent
        worker_yaml_path = script_dir / worker_yaml
        if not worker_yaml_path.exists():
            logger.error(f"Worker YAML not found: {worker_yaml_path}")
            sys.exit(1)

        if not dry_run:
            if not _check_skypilot_configured(logger):
                sys.exit(1)

        # --- Step 1: Scan GitLab for active jobs ---

        logger.info("Scanning GitLab for active jobs...")
        try:
            scanner = GitLabJobScanner(
                token=gitlab_token,
                gitlab_url=gitlab_url,
                verbose=verbose,
                quiet=not verbose,
            )
        except Exception as e:
            logger.error(f"Failed to connect to GitLab: {e}")
            sys.exit(1)

        active_jobs = count_active_jobs(scanner, logger)

        # --- Step 2: Calculate desired workers ---

        if active_jobs == 0:
            desired_workers = 0
        else:
            desired_workers = min(math.ceil(active_jobs / jobs_per_worker), max_workers)

        # Apply minimum workers floor
        if min_workers > 0 and desired_workers < min_workers:
            logger.info(f"Applying min_workers floor: {desired_workers} -> {min_workers}")
            desired_workers = min_workers

        logger.info(f"Active jobs: {active_jobs}, desired workers: {desired_workers}")

        # --- Step 3: Get current cluster status ---

        logger.info("Checking SkyPilot cluster status...")
        clusters = _get_sky_clusters(logger)
        logger.info(f"Found {len(clusters)} codex-worker-* clusters")

        for c in clusters:
            logger.debug(f"  {c['name']}: status={c['status']}, "
                         f"launched_at={c.get('launched_at')}, "
                         f"status_updated_at={c.get('status_updated_at')}")

        # --- Step 4: Clean up invalid clusters ---

        import sky
        now = datetime.now(timezone.utc)

        # Tear down STOPPED clusters
        stopped = [c for c in clusters if c['status'] == sky.ClusterStatus.STOPPED]
        for c in stopped:
            logger.info(f"Cluster {c['name']} is STOPPED — tearing down")
            if not dry_run:
                _sky_down(c['name'], logger)
            else:
                logger.info(f"[DRY RUN] Would tear down STOPPED cluster: {c['name']}")

        # Tear down stale INIT clusters
        init_clusters = [c for c in clusters if c['status'] == sky.ClusterStatus.INIT]
        for c in init_clusters:
            updated_at = c.get('status_updated_at')
            if updated_at is not None:
                # status_updated_at might be a timestamp (int/float) or datetime
                if isinstance(updated_at, (int, float)):
                    age_seconds = (now - datetime.fromtimestamp(updated_at, tz=timezone.utc)).total_seconds()
                elif isinstance(updated_at, datetime):
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    age_seconds = (now - updated_at).total_seconds()
                else:
                    age_seconds = 0  # Can't determine age, skip

                if age_seconds > init_timeout:
                    logger.info(
                        f"Cluster {c['name']} stuck in INIT for {age_seconds:.0f}s "
                        f"(>{init_timeout}s) — tearing down"
                    )
                    if not dry_run:
                        _sky_down(c['name'], logger)
                    else:
                        logger.info(f"[DRY RUN] Would tear down stale INIT cluster: {c['name']}")

        # --- Step 5: Count healthy running workers ---

        # Re-fetch after cleanup (or just filter the original list)
        healthy_statuses = {sky.ClusterStatus.INIT, sky.ClusterStatus.UP}
        # Exclude clusters we just tore down
        torn_down_names = {c['name'] for c in stopped}
        # Also exclude stale INIT clusters we tore down
        for c in init_clusters:
            updated_at = c.get('status_updated_at')
            if updated_at is not None:
                if isinstance(updated_at, (int, float)):
                    age_seconds = (now - datetime.fromtimestamp(updated_at, tz=timezone.utc)).total_seconds()
                elif isinstance(updated_at, datetime):
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    age_seconds = (now - updated_at).total_seconds()
                else:
                    age_seconds = 0
                if age_seconds > init_timeout:
                    torn_down_names.add(c['name'])

        current_workers = sum(
            1 for c in clusters
            if c['status'] in healthy_statuses and c['name'] not in torn_down_names
        )

        logger.info(f"Current healthy workers: {current_workers}, desired: {desired_workers}")

        # --- Step 6: Launch new workers if needed ---

        workers_to_launch = max(0, desired_workers - current_workers)

        if workers_to_launch > 0:
            logger.info(f"Need to launch {workers_to_launch} new worker(s)")
            for i in range(workers_to_launch):
                short_uuid = uuid.uuid4().hex[:8]
                cluster_name = f"{WORKER_PREFIX}{short_uuid}"
                success = _sky_launch(
                    yaml_path=str(worker_yaml_path),
                    cluster_name=cluster_name,
                    envs={
                        'GITLAB_TOKEN': gitlab_token,
                        'WORKER_ID': cluster_name,
                    },
                    logger=logger,
                    dry_run=dry_run,
                    down=True,
                    idle_minutes_to_autostop=idle_minutes,
                    stream=False,  # Don't block — launch and move on
                )
                if not success:
                    logger.warning(f"Failed to launch worker {i+1}/{workers_to_launch}, stopping launches")
                    break
        else:
            logger.info("No new workers needed")

        # --- Summary ---

        logger.info(
            f"{mode_label}=== Monitor cycle complete === "
            f"active_jobs={active_jobs} desired={desired_workers} "
            f"current={current_workers} launched={workers_to_launch} "
            f"cleaned_up={len(torn_down_names)}"
        )

    def status(
        self,
        gitlab_token: Optional[str] = None,
        gitlab_url: Optional[str] = None,
        log_file: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Show current status: active jobs and running workers.

        Does not launch or tear down anything.
        """
        # Load .env
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)

        gitlab_token = gitlab_token or os.environ.get('GITLAB_TOKEN')
        gitlab_url = gitlab_url or os.environ.get('GITLAB_URL', 'https://git.genesisrnd.com')
        log_file = log_file or os.environ.get('LOG_FILE', 'monitor.log')

        logger = setup_logging(log_file)

        if not gitlab_token:
            logger.error("GITLAB_TOKEN not set.")
            sys.exit(1)

        # Scan jobs
        logger.info("Scanning GitLab for jobs...")
        scanner = GitLabJobScanner(
            token=gitlab_token,
            gitlab_url=gitlab_url,
            verbose=verbose,
            quiet=not verbose,
        )
        active_jobs = count_active_jobs(scanner, logger)

        # Get clusters
        logger.info("Checking SkyPilot clusters...")
        clusters = _get_sky_clusters(logger)

        print(f"\n{'='*60}")
        print(f"Monitor Status Report")
        print(f"{'='*60}")
        print(f"Active jobs (pending + running): {active_jobs}")
        print(f"Worker clusters (codex-worker-*): {len(clusters)}")
        for c in clusters:
            print(f"  {c['name']}: {c['status']}")
        print(f"{'='*60}\n")


    def test_launch(
        self,
        dry_run: bool = False,
        gitlab_token: Optional[str] = None,
        gitlab_url: Optional[str] = None,
        worker_yaml: Optional[str] = None,
        log_file: Optional[str] = None,
        keep_alive: bool = False,
        verbose: bool = False,
    ):
        """
        Test SkyPilot by launching a VM.

        Without --dry-run, this launches the actual worker YAML
        (skypilot_worker.yaml) with --down. The worker will:
        1. Provision a GPU VM on Vast.ai
        2. Clone the repo, install dependencies
        3. Scan GitLab for jobs, process any it finds
        4. Auto-terminate when no more jobs are available

        This is useful for verifying the full worker pipeline end-to-end,
        or for running a one-off job before the monitor cron is set up.

        With --dry-run, this launches skypilot_dryrun.yaml instead — a
        lightweight YAML that just echoes a message and exits. This only
        tests that SkyPilot can provision a VM on Vast.ai.

        Use --keep-alive to keep the VM alive after it finishes (useful
        for SSH exploration; you must manually tear it down).

        Args:
            dry_run: If True, use skypilot_dryrun.yaml (lightweight test).
                     If False (default), use the actual worker YAML.
            gitlab_token: GitLab access token.
            gitlab_url: GitLab server URL.
            worker_yaml: Path to worker YAML (default: from .env or skypilot_worker.yaml).
            log_file: Path to log file.
            keep_alive: If True, launch without --down (VM stays alive).
                        If False (default), auto-terminate when done.
            verbose: Enable verbose output.
        """
        # Load .env
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)

        gitlab_token = gitlab_token or os.environ.get('GITLAB_TOKEN')
        gitlab_url = gitlab_url or os.environ.get('GITLAB_URL', 'https://git.genesisrnd.com')
        log_file = log_file or os.environ.get('LOG_FILE', 'monitor.log')

        logger = setup_logging(log_file)

        if not gitlab_token:
            logger.error("GITLAB_TOKEN not set.")
            sys.exit(1)

        # Check SkyPilot is configured
        if not _check_skypilot_configured(logger):
            sys.exit(1)

        script_dir = Path(__file__).parent
        cluster_name = "codex-test-launch"

        if dry_run:
            # Lightweight dryrun — just test VM provisioning
            yaml_file = script_dir / 'skypilot_dryrun.yaml'
            if not yaml_file.exists():
                logger.error(f"Dryrun YAML not found: {yaml_file}")
                sys.exit(1)
            mode_desc = "dryrun (lightweight VM test)"
        else:
            # Full worker launch — scan for jobs, process them, exit
            worker_yaml = worker_yaml or os.environ.get('WORKER_YAML', 'skypilot_worker.yaml')
            yaml_file = script_dir / worker_yaml
            if not yaml_file.exists():
                logger.error(f"Worker YAML not found: {yaml_file}")
                sys.exit(1)
            mode_desc = "full worker (will scan for jobs and process them)"

        use_down = not keep_alive

        if keep_alive:
            logger.info(f"Launching test VM: {mode_desc} (keep-alive — no --down)")
            logger.info("NOTE: VM will stay alive! Tear down with: sky down codex-test-launch")
        else:
            logger.info(f"Launching test VM: {mode_desc} (with --down)")
            logger.info("VM will auto-terminate when done")

        logger.info(f"Using YAML: {yaml_file}")

        envs = {'GITLAB_TOKEN': gitlab_token}
        if not dry_run:
            envs['WORKER_ID'] = cluster_name

        success = _sky_launch(
            yaml_path=str(yaml_file),
            cluster_name=cluster_name,
            envs=envs,
            logger=logger,
            dry_run=False,
            down=use_down,
            stream=True,  # Stream logs so user can see the output
        )

        if success:
            logger.info("Test launch completed successfully!")
            if keep_alive:
                logger.info(f"SSH in with: sky ssh {cluster_name}")
                logger.info(f"Tear down with: sky down {cluster_name}")
            else:
                logger.info("VM should auto-terminate shortly (--down mode)")
        else:
            logger.error("Test launch failed!")
            sys.exit(1)


def main():
    """Entry point for Fire CLI."""
    fire.Fire(Monitor)


if __name__ == '__main__':
    main()
