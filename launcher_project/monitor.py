#!/usr/bin/env python3
"""
Monitor Process — Auto-scaling GPU Worker Manager

This script runs a single monitoring cycle:
  1. Scans GitLab for active jobs (pending + running)
  2. Calculates how many workers are needed: ceil(active_jobs / JOBS_PER_WORKER)
  3. Checks how many workers are currently running (via cloud provider)
  4. Launches new workers if more are needed (up to MAX_WORKERS)
  5. Cleans up workers in invalid states (stopped, stale init)

Supports multiple cloud providers via the CloudProvider interface:
  - vast: Direct Vast.ai REST API for listing, SkyPilot for launch/destroy
  - skypilot: Pure SkyPilot (original implementation)

Designed to be run via cron (e.g., every 10 minutes). If the process crashes,
cron simply runs it again next cycle — no long-running daemon to babysit.

Usage:
    # Single monitoring cycle (for cron)
    uv run python monitor.py run

    # Dry run — verify GitLab scanning and cloud provider connectivity
    uv run python monitor.py run --dry-run

    # Override settings
    uv run python monitor.py run --max-workers=3 --jobs-per-worker=2

    # Show current status without taking action
    uv run python monitor.py status

    # Test launch with a lightweight dryrun VM
    uv run python monitor.py test_launch --dry-run

    # Test full worker pipeline (launches real worker, scans for jobs, exits)
    uv run python monitor.py test_launch

Cron example (every 10 minutes):
    */10 * * * * flock -n /tmp/monitor.lock -c 'cd /path/to/launcher_project && bash run_monitor_cron.sh' >> /var/log/monitor_cron.log 2>&1
"""

import gzip
import json
import logging
import logging.handlers
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import fire  # pylint: disable=import-error
from dotenv import load_dotenv  # pylint: disable=import-error

# Add parent directory to path so we can import gitlab_jobs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gitlab_jobs import GitLabJobScanner  # noqa: E402  # pylint: disable=import-error

from cloud_provider import CloudProvider, InstanceInfo  # noqa: E402  # pylint: disable=import-error

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
# Provider factory
# ---------------------------------------------------------------------------

def _create_provider(
    cloud_provider: str,
    logger: logging.Logger,
) -> CloudProvider:
    """
    Create a cloud provider instance based on the CLOUD_PROVIDER setting.

    Args:
        cloud_provider: Provider name — 'vast' or 'skypilot'.
        logger: Logger instance.

    Returns:
        A CloudProvider implementation.
    """
    if cloud_provider == 'vast':
        from vast_provider import VastCloudProvider

        api_key = os.environ.get('VAST_API_KEY')
        if not api_key:
            logger.error(
                "VAST_API_KEY not set. Required when CLOUD_PROVIDER=vast.\n"
                "Set it in launcher_project/.env or as an environment variable."
            )
            sys.exit(1)

        return VastCloudProvider(api_key=api_key, logger=logger)

    elif cloud_provider == 'skypilot':
        from skypilot_provider import SkyPilotProvider
        return SkyPilotProvider(logger=logger)

    else:
        logger.error(
            f"Unknown CLOUD_PROVIDER: '{cloud_provider}'. "
            f"Supported values: 'vast', 'skypilot'"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Job counting
# ---------------------------------------------------------------------------

def count_active_jobs(
    scanner: GitLabJobScanner,
    logger: logging.Logger,
    all_projects: Optional[List[Dict]] = None,
) -> int:
    """
    Count jobs that are in pending or running state.

    Active = not completed, not failed, not canceled.

    Uses list_available_jobs(include_claimed=True) to get both pending and
    running jobs (excluding canceled), then filters out completed and failed.

    Args:
        scanner: GitLabJobScanner instance.
        logger: Logger instance.
        all_projects: Optional pre-scanned project data from scanner.scan_all_projects().
                      If provided, jobs are extracted from this data instead of
                      re-scanning GitLab. If None, scanner.list_available_jobs() is called.

    Returns:
        Number of active (pending or running) jobs.
    """
    try:
        if all_projects is not None:
            # Extract jobs from pre-scanned project data
            all_jobs = []
            for project in all_projects:
                if not project.get('jobs'):
                    continue
                for job in project['jobs']:
                    # Exclude canceled jobs (same as list_available_jobs)
                    if job.get('canceled') is True:
                        continue
                    all_jobs.append(job)
        else:
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
# Stale project tracking
# ---------------------------------------------------------------------------

DEFAULT_STALE_PROJECTS_FILE = 'stale_projects.json'
DEFAULT_PROJECT_UNSHARE_HOURS = 24


def _load_stale_projects(file_path: str) -> Dict[str, str]:
    """
    Load the stale projects tracking file.

    The file maps project_id (as string) -> ISO timestamp of when the project
    was first observed as unactionable.

    Args:
        file_path: Path to the JSON tracking file.

    Returns:
        Dictionary mapping project_id strings to ISO timestamp strings.
        Returns empty dict if file doesn't exist or is invalid.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def _save_stale_projects(file_path: str, data: Dict[str, str]) -> None:
    """
    Save the stale projects tracking file.

    Args:
        file_path: Path to the JSON tracking file.
        data: Dictionary mapping project_id strings to ISO timestamp strings.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _has_actionable_jobs(project_data: Dict) -> bool:
    """
    Check if a project has any actionable jobs (unclaimed or running).

    A job is actionable if:
    - It is not canceled AND
    - It is either unclaimed (pending) OR claimed with state 'running'

    Args:
        project_data: Project dict from scanner.scan_all_projects().

    Returns:
        True if the project has at least one actionable job.
    """
    jobs = project_data.get('jobs')
    if not jobs:
        return False

    for job in jobs:
        # Skip canceled jobs
        if job.get('canceled') is True:
            continue

        # Unclaimed job = actionable (pending)
        if not job.get('is_claimed'):
            return True

        # Claimed job — check response state
        response = job.get('response')
        if response and isinstance(response, dict):
            state = response.get('state', '')
            if state not in ('completed', 'failed', 'canceled'):
                # Still running or in some active state
                return True
        else:
            # Claimed but no parseable response — treat as actionable
            return True

    return False


def cleanup_stale_projects(
    scanner: GitLabJobScanner,
    all_projects: List[Dict],
    stale_file: str,
    unshare_hours: float,
    logger: logging.Logger,
    dry_run: bool = False,
) -> int:
    """
    Track and clean up stale shared projects.

    This function:
    1. Loads the stale projects tracking file
    2. For each visible project, checks if it has actionable jobs
    3. If actionable: removes it from the stale tracker (resets timer)
    4. If unactionable: adds it to the tracker with current timestamp
       (or keeps existing timestamp if already tracked)
    5. If a project has been unactionable for longer than unshare_hours:
       unshares it (removes the worker from the project)
    6. Removes entries for projects that are no longer visible
       (handles manual unshares and prevents JSON accumulation)
    7. Saves the updated tracking file

    Args:
        scanner: GitLabJobScanner instance (for calling unshare_project).
        all_projects: Pre-scanned project data from scanner.scan_all_projects().
        stale_file: Path to the stale projects JSON tracking file.
        unshare_hours: Hours a project must be unactionable before unsharing.
        logger: Logger instance.
        dry_run: If True, log what would happen but don't unshare.

    Returns:
        Number of projects unshared in this cycle.
    """
    now = datetime.now(timezone.utc)
    stale_data = _load_stale_projects(stale_file)

    # Build set of currently visible project IDs
    visible_project_ids = {str(p['project_id']) for p in all_projects}

    # Remove entries for projects no longer visible (manual unshare, etc.)
    stale_keys_to_remove = [
        pid for pid in stale_data if pid not in visible_project_ids
    ]
    for pid in stale_keys_to_remove:
        logger.debug(f"Removing stale tracker entry for project {pid} (no longer visible)")
        del stale_data[pid]

    # Process each visible project
    unshared_count = 0
    projects_to_unshare = []

    for project_data in all_projects:
        project_id = str(project_data['project_id'])
        project_path = project_data.get('project_path', project_id)

        if _has_actionable_jobs(project_data):
            # Project is actionable — remove from stale tracker
            if project_id in stale_data:
                logger.debug(f"Project {project_path} has actionable jobs — removing from stale tracker")
                del stale_data[project_id]
        else:
            # Project is unactionable
            if project_id not in stale_data:
                # First time seeing this project as unactionable — record timestamp
                stale_data[project_id] = now.isoformat()
                logger.info(f"Project {project_path} has no actionable jobs — starting stale timer")
            else:
                # Already tracked — check if it's been long enough
                first_seen_str = stale_data[project_id]
                try:
                    first_seen = datetime.fromisoformat(first_seen_str)
                    if first_seen.tzinfo is None:
                        first_seen = first_seen.replace(tzinfo=timezone.utc)
                    hours_stale = (now - first_seen).total_seconds() / 3600
                except (ValueError, TypeError):
                    # Invalid timestamp — reset it
                    stale_data[project_id] = now.isoformat()
                    hours_stale = 0

                if hours_stale >= unshare_hours:
                    projects_to_unshare.append((project_data, hours_stale))

    # Unshare stale projects
    for project_data, hours_stale in projects_to_unshare:
        project_id = project_data['project_id']
        project_path = project_data.get('project_path', str(project_id))

        if dry_run:
            logger.info(
                f"[DRY RUN] Would unshare project {project_path} (ID: {project_id}) "
                f"— stale for {hours_stale:.1f} hours (threshold: {unshare_hours}h)"
            )
        else:
            try:
                scanner.unshare_project(project_id=project_id)
                logger.info(
                    f"Unshared project {project_path} (ID: {project_id}) "
                    f"— stale for {hours_stale:.1f} hours"
                )
                unshared_count += 1
            except Exception as e:
                logger.error(f"Failed to unshare project {project_path} (ID: {project_id}): {e}")

        # Remove from stale tracker regardless (either unshared or will retry next cycle)
        stale_data.pop(str(project_id), None)

    # Save updated tracking data
    _save_stale_projects(stale_file, stale_data)

    return unshared_count


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class Monitor:
    """
    GPU Worker Monitor — auto-scales workers based on job demand.

    Supports multiple cloud providers via the CloudProvider interface.
    Designed to be run as a single cycle via cron. Each invocation:
    1. Scans GitLab for active jobs
    2. Calculates desired worker count
    3. Cleans up invalid instances
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
        cloud_provider: Optional[str] = None,
        project_unshare_hours: Optional[float] = None,
        stale_projects_file: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Run a single monitoring cycle.

        Args:
            dry_run: If True, scan jobs and check status but don't launch or
                     tear down any instances. Logs what would happen.
            gitlab_token: GitLab access token (default: from .env or GITLAB_TOKEN env var).
            gitlab_url: GitLab server URL (default: from .env or https://git.genesisrnd.com).
            max_workers: Maximum concurrent workers (default: from .env or 5).
            min_workers: Minimum workers to keep running even with 0 jobs (default: 0).
                         Useful for testing worker detection. Workers launched to meet
                         the minimum use idle_minutes auto-teardown instead of immediate
                         exit, so they stay alive long enough for the next cycle to see them.
            jobs_per_worker: Jobs per worker ratio (default: from .env or 4).
            init_timeout: Seconds before tearing down stale INIT instances (default: from .env or 1800).
            worker_yaml: Path to SkyPilot worker YAML (default: from .env or skypilot_worker.yaml).
            log_file: Path to log file (default: from .env or monitor.log).
            idle_minutes: Minutes of idle time before auto-teardown (default: from
                          .env or 7). Passed to SkyPilot as idle_minutes_to_autostop.
                          Combined with --down, the VM tears down after being idle
                          for this many minutes. If not set, SkyPilot uses its default.
            cloud_provider: Cloud provider to use — 'vast' or 'skypilot'
                            (default: from .env or 'vast').
            project_unshare_hours: Hours a project must have no actionable jobs before
                                   the worker unshares itself from it (default: from
                                   .env or 24). Set to 0 to disable stale project cleanup.
            stale_projects_file: Path to the JSON file tracking stale projects
                                 (default: from .env or stale_projects.json).
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
        cloud_provider = cloud_provider or os.environ.get('CLOUD_PROVIDER', 'vast')
        project_unshare_hours = (
            project_unshare_hours if project_unshare_hours is not None
            else float(os.environ.get('PROJECT_UNSHARE_HOURS', str(DEFAULT_PROJECT_UNSHARE_HOURS)))
        )
        stale_projects_file = stale_projects_file or os.environ.get(
            'STALE_PROJECTS_FILE', DEFAULT_STALE_PROJECTS_FILE
        )

        # Setup logging
        logger = setup_logging(log_file)

        mode_label = "[DRY RUN] " if dry_run else ""
        logger.info(f"{mode_label}=== Monitor cycle starting ===")
        logger.info(f"{mode_label}Config: cloud_provider={cloud_provider}, "
                     f"max_workers={max_workers}, min_workers={min_workers}, "
                     f"jobs_per_worker={jobs_per_worker}, idle_minutes={idle_minutes}, "
                     f"init_timeout={init_timeout}s, worker_yaml={worker_yaml}, "
                     f"project_unshare_hours={project_unshare_hours}h")

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

        # Create cloud provider
        provider = _create_provider(cloud_provider, logger)

        if not dry_run:
            if not provider.check_configured():
                sys.exit(1)

        # --- Step 1: Scan GitLab for all projects and count active jobs ---

        logger.info("Scanning GitLab for projects and jobs...")
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

        # Scan all projects once — reuse for both job counting and stale project tracking
        all_projects = scanner.scan_all_projects()
        active_jobs = count_active_jobs(scanner, logger, all_projects=all_projects)

        # --- Step 1b: Clean up stale shared projects ---

        if project_unshare_hours > 0:
            # Resolve stale_projects_file path relative to this script's directory
            stale_file_path = str(Path(__file__).parent / stale_projects_file)
            unshared = cleanup_stale_projects(
                scanner=scanner,
                all_projects=all_projects,
                stale_file=stale_file_path,
                unshare_hours=project_unshare_hours,
                logger=logger,
                dry_run=dry_run,
            )
            if unshared > 0:
                logger.info(f"Unshared {unshared} stale project(s)")
        else:
            logger.debug("Stale project cleanup disabled (project_unshare_hours=0)")

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

        # --- Step 3: Get current instance status ---

        logger.info(f"Checking {cloud_provider} instance status...")
        instances = provider.list_instances(prefix=WORKER_PREFIX)
        logger.info(f"Found {len(instances)} {WORKER_PREFIX}* instances")

        for inst in instances:
            logger.debug(f"  {inst.name}: status={inst.status}, "
                         f"gpu={inst.gpu_name}, created_at={inst.created_at}, "
                         f"cost=${inst.cost_per_hour:.3f}/hr")

        # --- Step 4: Clean up invalid instances ---

        now = datetime.now(timezone.utc)
        torn_down_names: set = set()

        # Tear down stopped instances
        stopped = [inst for inst in instances if inst.status == 'stopped']
        for inst in stopped:
            logger.info(f"Instance {inst.name} ({inst.id}) is stopped — tearing down")
            if not dry_run:
                provider.destroy_instance(inst.id)
            else:
                logger.info(f"[DRY RUN] Would tear down stopped instance: {inst.name}")
            torn_down_names.add(inst.name)

        # Tear down error instances
        errored = [inst for inst in instances if inst.status == 'error']
        for inst in errored:
            logger.info(f"Instance {inst.name} ({inst.id}) is in error state — tearing down")
            if not dry_run:
                provider.destroy_instance(inst.id)
            else:
                logger.info(f"[DRY RUN] Would tear down errored instance: {inst.name}")
            torn_down_names.add(inst.name)

        # Tear down stale init instances
        init_instances = [inst for inst in instances if inst.status == 'init']
        for inst in init_instances:
            if inst.created_at is not None:
                created_at = inst.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                age_seconds = (now - created_at).total_seconds()

                if age_seconds > init_timeout:
                    logger.info(
                        f"Instance {inst.name} ({inst.id}) stuck in init for "
                        f"{age_seconds:.0f}s (>{init_timeout}s) — tearing down"
                    )
                    if not dry_run:
                        provider.destroy_instance(inst.id)
                    else:
                        logger.info(f"[DRY RUN] Would tear down stale init instance: {inst.name}")
                    torn_down_names.add(inst.name)

        # --- Step 5: Count healthy running workers ---

        healthy_statuses = {'running', 'init'}
        current_workers = sum(
            1 for inst in instances
            if inst.status in healthy_statuses and inst.name not in torn_down_names
        )

        logger.info(f"Current healthy workers: {current_workers}, desired: {desired_workers}")

        # --- Step 6: Launch new workers if needed ---

        workers_to_launch = max(0, desired_workers - current_workers)

        if workers_to_launch > 0:
            logger.info(f"Need to launch {workers_to_launch} new worker(s)")
            for i in range(workers_to_launch):
                short_uuid = uuid.uuid4().hex[:8]
                cluster_name = f"{WORKER_PREFIX}{short_uuid}"

                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would launch worker '{cluster_name}' "
                        f"from {worker_yaml_path} (idle_minutes={idle_minutes})"
                    )
                    continue

                result = provider.launch_instance(
                    name=cluster_name,
                    yaml_path=str(worker_yaml_path),
                    envs={
                        'GITLAB_TOKEN': gitlab_token,
                        'WORKER_ID': cluster_name,
                    },
                    down=True,
                    idle_minutes_to_autostop=idle_minutes,
                    stream=False,  # Don't block — launch and move on
                )
                if result is None:
                    logger.warning(f"Failed to launch worker {i+1}/{workers_to_launch}, stopping launches")
                    break
        else:
            logger.info("No new workers needed")

        # --- Summary ---

        logger.info(
            f"{mode_label}=== Monitor cycle complete === "
            f"active_jobs={active_jobs} desired={desired_workers} "
            f"current={current_workers} launched={workers_to_launch} "
            f"cleaned_up={len(torn_down_names)} "
            f"projects_scanned={len(all_projects)}"
        )

    def status(
        self,
        gitlab_token: Optional[str] = None,
        gitlab_url: Optional[str] = None,
        log_file: Optional[str] = None,
        cloud_provider: Optional[str] = None,
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
        cloud_provider = cloud_provider or os.environ.get('CLOUD_PROVIDER', 'vast')

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

        # Get instances
        provider = _create_provider(cloud_provider, logger)
        logger.info(f"Checking {cloud_provider} instances...")
        instances = provider.list_instances(prefix=WORKER_PREFIX)

        print(f"\n{'='*60}")
        print(f"Monitor Status Report (provider: {cloud_provider})")
        print(f"{'='*60}")
        print(f"Active jobs (pending + running): {active_jobs}")
        print(f"Worker instances ({WORKER_PREFIX}*): {len(instances)}")
        for inst in instances:
            gpu_info = f" [{inst.gpu_name}]" if inst.gpu_name else ""
            cost_info = f" ${inst.cost_per_hour:.3f}/hr" if inst.cost_per_hour else ""
            print(f"  {inst.name}: {inst.status}{gpu_info}{cost_info}")
        print(f"{'='*60}\n")


    def test_launch(
        self,
        dry_run: bool = False,
        gitlab_token: Optional[str] = None,
        gitlab_url: Optional[str] = None,
        worker_yaml: Optional[str] = None,
        log_file: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        keep_alive: bool = False,
        verbose: bool = False,
    ):
        """
        Test the cloud provider by launching a VM.

        Without --dry-run, this launches the actual worker YAML
        (skypilot_worker.yaml) with --down. The worker will:
        1. Provision a GPU VM on the configured cloud provider
        2. Clone the repo, install dependencies
        3. Scan GitLab for jobs, process any it finds
        4. Auto-terminate when no more jobs are available

        This is useful for verifying the full worker pipeline end-to-end,
        or for running a one-off job before the monitor cron is set up.

        With --dry-run, this launches skypilot_dryrun.yaml instead — a
        lightweight YAML that just echoes a message and exits. This only
        tests that the cloud provider can provision a VM.

        Use --keep-alive to keep the VM alive after it finishes (useful
        for SSH exploration; you must manually tear it down).

        Args:
            dry_run: If True, use skypilot_dryrun.yaml (lightweight test).
                     If False (default), use the actual worker YAML.
            gitlab_token: GitLab access token.
            gitlab_url: GitLab server URL.
            worker_yaml: Path to worker YAML (default: from .env or skypilot_worker.yaml).
            log_file: Path to log file.
            cloud_provider: Cloud provider to use — 'vast' or 'skypilot'
                            (default: from .env or 'vast').
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
        cloud_provider = cloud_provider or os.environ.get('CLOUD_PROVIDER', 'vast')

        logger = setup_logging(log_file)

        if not gitlab_token:
            logger.error("GITLAB_TOKEN not set.")
            sys.exit(1)

        # Create provider and check configuration
        provider = _create_provider(cloud_provider, logger)
        if not provider.check_configured():
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
            logger.info("NOTE: VM will stay alive! Tear down manually.")
        else:
            logger.info(f"Launching test VM: {mode_desc} (with --down)")
            logger.info("VM will auto-terminate when done")

        logger.info(f"Using YAML: {yaml_file}")
        logger.info(f"Cloud provider: {cloud_provider}")

        envs = {'GITLAB_TOKEN': gitlab_token}
        if not dry_run:
            envs['WORKER_ID'] = cluster_name

        result = provider.launch_instance(
            name=cluster_name,
            yaml_path=str(yaml_file),
            envs=envs,
            down=use_down,
            stream=True,  # Stream logs so user can see the output
        )

        if result is not None:
            logger.info("Test launch completed successfully!")
            if keep_alive:
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
