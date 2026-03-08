#!/usr/bin/env python3
"""
GitLab GPU Job Scanner

This script scans GitLab projects for GPU jobs defined in manifest files.
It identifies available jobs that haven't been claimed yet.

Usage:
    # List available jobs
    python gitlab_jobs.py list_available_jobs --token=YOUR_TOKEN
    python gitlab_jobs.py list_available_jobs --token=YOUR_TOKEN --verbose
    python gitlab_jobs.py list_available_jobs --gitlab-url=https://custom.gitlab.com

    # Create a file in a repository
    python gitlab_jobs.py create_file --token=YOUR_TOKEN --project-id=123 --file-path=test.txt --content="Hello World"
    python gitlab_jobs.py create_file --token=YOUR_TOKEN --project-id=123 --file-path=config.yaml --file-source=./local-config.yaml

    # Unshare a project (remove yourself as a member)
    python gitlab_jobs.py unshare_project --token=YOUR_TOKEN --project-id=123
    python gitlab_jobs.py unshare_project --token=YOUR_TOKEN --project-id=123 --user-id=551
"""

import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Iterator

import fire
import yaml
import gitlab
from gitlab.exceptions import GitlabError, GitlabAuthenticationError, GitlabGetError, GitlabCreateError, GitlabUpdateError, GitlabDeleteError


# Constants for retry logic
INITIAL_BACKOFF_DELAY = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_DELAY = 60.0  # seconds
DEFAULT_MAX_RETRIES = 10
MAX_JOB_CLAIM_RETRY = 3

# Default GitLab server URL
DEFAULT_GITLAB_URL = "https://git.genesisrnd.com"

# Manifest file location
MANIFEST_PATH = "gpu_jobs/manifest.yaml"
RESPONSE_FILE_PATH_TEMPLATE = "gpu_jobs/job_{job_id}/response.yaml"

# Default claim timeout: hours after which a claimed job with no heartbeat
# update is considered stale and eligible for re-claiming.
DEFAULT_CLAIM_TIMEOUT_HOURS = 24.0

# Default maximum number of times a stale job can be re-claimed before
# being marked as permanently failed.
DEFAULT_MAX_JOB_RETRIES = 3


class GitLabConnectionError(Exception):
    """Raised when unable to connect to GitLab server."""

class GitLabAuthenticationError(Exception):
    """Raised when authentication fails."""

class NoTokenProvidedError(Exception):
    """Raised when no GitLab token is provided."""

class NoJobsAvailableError(Exception):
    """Raised when no jobs are available to claim."""

class MaxRetriesExceededError(Exception):
    """Raised when a job has exceeded its maximum retry count."""

def retry_with_backoff(max_retries: int = DEFAULT_MAX_RETRIES):
    """
    Decorator to retry functions with exponential backoff.

    Retries on network errors, rate limiting, and server errors.
    Does not retry on authentication errors or client errors.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = INITIAL_BACKOFF_DELAY

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)

                except GitlabCreateError:
                    # Don't retry create errors
                    raise
                except GitlabAuthenticationError:
                    # Don't retry authentication errors
                    raise
                except GitlabGetError as e:
                    # Don't retry 404s or other client errors
                    if hasattr(e, 'response_code') and 400 <= e.response_code < 500:
                        raise
                    # Retry server errors and network issues
                    if retries >= max_retries:
                        raise
                except (GitlabError, ConnectionError, TimeoutError):
                    # Retry network and server errors
                    if retries >= max_retries:
                        raise

                retries += 1
                if retries <= max_retries:
                    time.sleep(delay)
                    delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_DELAY)

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _is_stale_claim(job: Dict[str, Any], claim_timeout_hours: float) -> bool:
    """
    Check if a claimed job's heartbeat has gone stale.

    A claim is considered stale when:
    - The job has a response with a parseable ``timestamp`` field, AND
    - That timestamp is older than ``claim_timeout_hours``, AND
    - The response state is not a terminal state (completed/failed/canceled).

    If the timestamp field is missing or unparseable, the claim is NOT
    considered stale (conservative: avoid stealing from a legitimately
    running worker that may have older code without the timestamp field).

    Args:
        job: Job dictionary with ``response`` sub-dict from get_job_status().
        claim_timeout_hours: Hours after which a claim is considered stale.

    Returns:
        True if the claim is stale and the job should be treated as unclaimed.
    """
    response = job.get('response')
    if not response or not isinstance(response, dict):
        return False

    # Terminal states are never stale — they're done
    state = response.get('state', '')
    if state in ('completed', 'failed', 'canceled'):
        return False

    # If no timestamp, don't consider stale (conservative)
    timestamp_str = response.get('timestamp')
    if not timestamp_str:
        return False

    try:
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        # Unparseable timestamp — don't consider stale
        return False

    now = datetime.now(timezone.utc)
    age_hours = (now - ts).total_seconds() / 3600
    return age_hours >= claim_timeout_hours


class GitLabJobScanner:
    """Scanner for GitLab GPU jobs."""

    def __init__(
        self,
        token: Optional[str] = None,
        gitlab_url: str = DEFAULT_GITLAB_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        verbose: bool = False,
        quiet: bool = False
    ):
        """
        Initialize the GitLab job scanner.

        Args:
            token: GitLab access token (or use GITLAB_TOKEN env var)
            gitlab_url: GitLab server URL
            max_retries: Maximum number of retry attempts for API calls
            verbose: Enable verbose output to stderr
            quiet: Suppress all output except final JSON
        """
        self.gitlab_url = gitlab_url
        self.max_retries = max_retries
        self.verbose = verbose and not quiet
        self.quiet = quiet

        # Get token from parameter or environment
        self.token = token or os.environ.get('GITLAB_TOKEN')
        if not self.token:
            raise NoTokenProvidedError(
                "No GitLab token provided. Use --token parameter or set GITLAB_TOKEN environment variable."
            )

        # Add aliases for compatibility with GitLabDatasetDownloader interface
        # Handlers expect these attribute names when creating GitLabDatasetDownloader instances
        self.server_url = self.gitlab_url
        self.access_token = self.token

        # Initialize GitLab connection
        self.gl = self._connect_to_gitlab()

    def _log(self, message: str, force: bool = False):
        """Log message to stderr if verbose mode is enabled."""
        if (self.verbose or force) and not self.quiet:
            print(message, file=sys.stderr)

    def _connect_to_gitlab(self) -> gitlab.Gitlab:
        """
        Connect to GitLab server and verify authentication.

        Returns:
            Authenticated GitLab instance

        Raises:
            GitLabConnectionError: If unable to connect to server
            GitLabAuthenticationError: If authentication fails
        """
        try:
            self._log(f"Connecting to GitLab at {self.gitlab_url}...")
            gl = gitlab.Gitlab(self.gitlab_url, private_token=self.token, timeout=30)

            # Verify authentication by getting current user
            gl.auth()
            user = gl.user
            self._log(f"Successfully authenticated as: {user.username}")

            return gl

        except GitlabAuthenticationError as e:
            raise GitLabAuthenticationError(
                f"Authentication failed. Please check your GitLab token. Error: {e}"
            ) from e
        except (ConnectionError, TimeoutError) as e:
            raise GitLabConnectionError(
                f"Unable to connect to GitLab server at {self.gitlab_url}. Error: {e}"
            ) from e
        except Exception as e:
            raise GitLabConnectionError(
                f"Unexpected error connecting to GitLab: {e}"
            ) from e

    @retry_with_backoff()
    def _get_file_content(self, project, file_path: str, ref: str) -> Optional[str]:
        """
        Get file content from GitLab repository.

        Args:
            project: GitLab project object
            file_path: Path to file in repository
            ref: Branch/tag/commit reference

        Returns:
            File content as string, or None if file doesn't exist
        """
        try:
            file_obj = project.files.get(file_path=file_path, ref=ref)
            return file_obj.decode().decode('utf-8')
        except GitlabGetError as e:
            if hasattr(e, 'response_code') and e.response_code == 404:
                return None
            raise

    def _check_job_claimed(self, project, job_id: str, default_branch: str) -> bool:
        """
        Check if a job has been claimed by looking for response.yaml file.

        Args:
            project: GitLab project object
            job_id: Job ID to check
            default_branch: Default branch of the project

        Returns:
            True if job is claimed (response.yaml exists), False otherwise
        """
        response_path = RESPONSE_FILE_PATH_TEMPLATE.format(job_id=job_id)
        content = self._get_file_content(project, response_path, default_branch)
        return content is not None

    def _build_manifest_url(
        self,
        project_path: str,
        default_branch: str
    ) -> str:
        """
        Build URL to view manifest file in GitLab web UI.

        Args:
            project_path: Project path (namespace/project)
            default_branch: Default branch name

        Returns:
            URL to manifest file in GitLab web UI
        """
        return f"{self.gitlab_url}/{project_path}/-/blob/{default_branch}/{MANIFEST_PATH}"

    def _scan_project(self, project) -> Dict[str, Any]:
        """
        Scan a single project for GPU jobs.

        Args:
            project: GitLab project object

        Returns:
            Dictionary with scan results containing:
            - project_id, project_name, project_path, default_branch
            - manifest_url, manifest_data
            - jobs: List of job dicts with status
            - error: Error message if any
        """
        project_data = {
            'project_id': project.id,
            'project_name': project.name,
            'project_path': project.path_with_namespace,
            'default_branch': project.default_branch or 'main',
            'manifest_url': None,
            'manifest_data': None,
            'jobs': None,
            'error': None
        }

        try:
            # Try to get manifest file
            manifest_content = self._get_file_content(
                project,
                MANIFEST_PATH,
                project_data['default_branch']
            )

            if manifest_content is None:
                # No manifest file - not an error, just no GPU jobs
                return project_data

            # Build manifest URL
            project_data['manifest_url'] = self._build_manifest_url(
                project_data['project_path'],
                project_data['default_branch']
            )

            # Parse manifest YAML
            try:
                manifest_data = yaml.safe_load(manifest_content)
                project_data['manifest_data'] = manifest_data

                # Validate version
                version = manifest_data.get('version')
                if version is None:
                    project_data['error'] = "Manifest missing 'version' field"
                    return project_data

                if version > 1:
                    project_data['error'] = f"Manifest version {version} is not supported (max version: 1)"
                    return project_data

                # Get jobs
                jobs = manifest_data.get('jobs', [])
                if not isinstance(jobs, list):
                    project_data['error'] = "Manifest 'jobs' field is not a list"
                    return project_data

                # Process each job using get_job_status
                processed_jobs = []
                for job in jobs:
                    if not isinstance(job, dict):
                        continue

                    job_id = job.get('job_id')
                    if job_id is None:
                        continue

                    # Get full job status including claim info
                    job_status = self.get_job_status(project_data['project_id'], str(job_id))

                    if job_status:
                        processed_jobs.append(job_status)

                project_data['jobs'] = processed_jobs

            except yaml.YAMLError as e:
                project_data['error'] = f"Failed to parse manifest YAML: {e}"
            except Exception as e:
                project_data['error'] = f"Error processing manifest: {e}"

        except Exception as e:
            project_data['error'] = f"Error scanning project: {e}"

        return project_data

    def _iter_projects(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through all accessible projects and scan them.

        Yields:
            Dictionary for each scanned project
        """
        try:
            # Get all projects with pagination
            projects = self.gl.projects.list(iterator=True)

            project_count = 0
            for project in projects:
                project_count += 1
                self._log(f"Scanning project {project_count}: {project.path_with_namespace}")

                yield self._scan_project(project)

        except Exception as e:
            self._log(f"Error iterating projects: {e}", force=True)
            raise

    def scan_all_projects(self) -> List[Dict[str, Any]]:
        """
        Scan all accessible projects for GPU jobs.

        Returns:
            List of project data dictionaries
        """
        self._log("Starting project scan...")

        all_projects = []
        for project_data in self._iter_projects():
            all_projects.append(project_data)

        self._log(f"Scan complete. Scanned {len(all_projects)} projects.")
        return all_projects

    def list_available_jobs(
        self,
        include_claimed: bool = False,
        claim_timeout_hours: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        List GPU jobs.

        By default, lists only available (unclaimed, non-canceled) jobs.
        With include_claimed=True, also includes claimed jobs.

        When ``claim_timeout_hours`` is set, claimed jobs whose last heartbeat
        timestamp is older than the timeout are treated as unclaimed (stale
        claims).  These jobs will have ``is_stale_claim=True`` set on them and
        the existing ``retry_count`` carried forward from the old response.
        If the response has no parseable timestamp, the claim is NOT considered
        stale (conservative — avoids stealing from a legitimately running
        worker).

        Args:
            include_claimed: If True, include claimed jobs in the results.
                           Canceled jobs are always excluded.
            claim_timeout_hours: Hours after which a claimed job with no
                                 heartbeat update is considered stale and
                                 treated as unclaimed.  None = no timeout.

        Returns:
            List of job dictionaries (same format as from get_job_status)
        """
        all_projects = self.scan_all_projects()

        available_jobs = []

        for project in all_projects:
            # Skip projects without jobs
            if not project.get('jobs'):
                continue

            for job in project['jobs']:
                # Always skip canceled jobs
                if job.get('canceled') is True:
                    continue

                is_claimed = job.get('is_claimed') is True

                # Check for stale claims
                if is_claimed and claim_timeout_hours is not None:
                    if _is_stale_claim(job, claim_timeout_hours):
                        # Treat as unclaimed — carry forward retry_count
                        response = job.get('response') or {}
                        job['is_stale_claim'] = True
                        job['retry_count'] = response.get('retry_count', 0)
                        self._log(
                            f"Job {job.get('job_id')} in project "
                            f"{job.get('project_id')} has stale claim "
                            f"(retry_count={job['retry_count']})"
                        )
                        # Mark as effectively unclaimed for filtering below
                        is_claimed = False

                # Skip claimed jobs unless include_claimed is set
                if is_claimed and not include_claimed:
                    continue

                # Job passes filters - add it
                available_jobs.append(job)

        label = "total" if include_claimed else "available"
        self._log(f"Found {len(available_jobs)} {label} jobs.")
        return available_jobs

    @retry_with_backoff()
    def _create_repository_file(
        self,
        project_id: int,
        file_path: str,
        content: str,
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new file in a GitLab repository.

        This is the internal method that performs the actual file creation.
        It relies on the GitLab API to detect if the file already exists,
        which prevents race conditions.

        Args:
            project_id: GitLab project ID
            file_path: Path where the file should be created in the repository
            content: Content to write to the file
            commit_message: Commit message (defaults to "Add {filename}")

        Returns:
            Dictionary containing:
            - file_path: Path of the created file
            - branch: Branch where file was created
            - project_id: Project ID
            - file_object: The GitLab file object

        Raises:
            GitLabFileExistsError: If the file already exists
            GitlabGetError: If project not found or other API errors
        """
        # Get the project
        project = self.gl.projects.get(project_id)
        default_branch = project.default_branch or 'main'

        self._log(f"Creating file '{file_path}' in project {project_id} on branch '{default_branch}'")

        # Generate default commit message if not provided
        if commit_message is None:
            filename = file_path.split('/')[-1]
            commit_message = f"Add {filename}"

        # Create the file - let the API tell us if it already exists
        file_data = {
            'file_path': file_path,
            'branch': default_branch,
            'content': content,
            'commit_message': commit_message
        }

        created_file = project.files.create(file_data)

        self._log(f"Successfully created file '{file_path}'")

        # Return metadata
        return {
            'file_path': file_path,
            'branch': default_branch,
            'project_id': project_id,
            'file_object': created_file
        }



    def create_file(
        self,
        project_id: int,
        file_path: str,
        content: Optional[str] = None,
        file_source: Optional[str] = None,
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new file in a GitLab repository (CLI wrapper).

        This method can be called from the command line using Fire.
        It accepts either direct content or a path to a local file.

        Args:
            project_id: GitLab project ID
            file_path: Path where the file should be created in the repository
            content: Direct content to write (mutually exclusive with file_source)
            file_source: Path to local file to read content from (mutually exclusive with content)
            commit_message: Commit message (defaults to "Add {filename}")

        Returns:
            Dictionary containing file metadata

        Raises:
            ValueError: If both or neither content and file_source are provided
            GitLabFileExistsError: If the file already exists in the repository
            FileNotFoundError: If file_source doesn't exist

        Examples:
            # Create file with direct content
            python gitlab_jobs.py create_file --project-id=123 --file-path=test.txt --content="Hello World"

            # Create file from local file
            python gitlab_jobs.py create_file --project-id=123 --file-path=config.yaml --file-source=./local-config.yaml

            # With custom commit message
            python gitlab_jobs.py create_file --project-id=123 --file-path=README.md --content="# Project" --commit-message="Initial README"
        """
        # Validate input
        if content is not None and file_source is not None:
            raise ValueError("Cannot specify both 'content' and 'file_source'. Please provide only one.")

        if content is None and file_source is None:
            raise ValueError("Must specify either 'content' or 'file_source'.")

        # Read content from file if file_source is provided
        if file_source is not None:
            self._log(f"Reading content from local file: {file_source}")
            try:
                with open(file_source, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Local file not found: {file_source}") from e
            except Exception as e:
                raise Exception(f"Error reading file '{file_source}': {e}") from e

        # Create the file
        result = self._create_repository_file(
            project_id=project_id,
            file_path=file_path,
            content=content,
            commit_message=commit_message
        )

        # Log success
        self._log(
            f"File created successfully:\n"
            f"  Path: {result['file_path']}\n"
            f"  Branch: {result['branch']}\n"
            f"  Project: {result['project_id']}",
            force=True
        )

        return result

    @retry_with_backoff()
    def _update_repository_file(
        self,
        project_id: int,
        file_path: str,
        content: str,
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing file in a GitLab repository.

        This is the internal method that performs the actual file update.

        Args:
            project_id: GitLab project ID
            file_path: Path to the file in the repository
            content: New content for the file
            commit_message: Commit message (defaults to "Update {filename}")

        Returns:
            Dictionary containing:
            - file_path: Path of the updated file
            - branch: Branch where file was updated
            - project_id: Project ID

        Raises:
            GitlabUpdateError: If the file doesn't exist or update fails
            GitlabGetError: If project not found or other API errors
        """
        # Get the project
        project = self.gl.projects.get(project_id)
        default_branch = project.default_branch or 'main'

        self._log(f"Updating file '{file_path}' in project {project_id} on branch '{default_branch}'")

        # Generate default commit message if not provided
        if commit_message is None:
            filename = file_path.split('/')[-1]
            commit_message = f"Update {filename}"

        # Get the existing file to update it
        try:
            file_obj = project.files.get(file_path=file_path, ref=default_branch)
        except GitlabGetError as e:
            if hasattr(e, 'response_code') and e.response_code == 404:
                raise GitlabUpdateError(f"File not found: {file_path}") from e
            raise

        # Update the file
        file_obj.content = content
        file_obj.save(branch=default_branch, commit_message=commit_message)

        self._log(f"Successfully updated file '{file_path}'")

        # Return metadata
        return {
            'file_path': file_path,
            'branch': default_branch,
            'project_id': project_id
        }

    def update_file(
        self,
        project_id: int,
        file_path: str,
        content: Optional[str] = None,
        file_source: Optional[str] = None,
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing file in a GitLab repository (CLI wrapper).

        This method can be called from the command line using Fire.
        It accepts either direct content or a path to a local file.

        Args:
            project_id: GitLab project ID
            file_path: Path to the file in the repository
            content: Direct content to write (mutually exclusive with file_source)
            file_source: Path to local file to read content from (mutually exclusive with content)
            commit_message: Commit message (defaults to "Update {filename}")

        Returns:
            Dictionary containing file metadata

        Raises:
            ValueError: If both or neither content and file_source are provided
            GitlabUpdateError: If the file doesn't exist in the repository
            FileNotFoundError: If file_source doesn't exist

        Examples:
            # Update file with direct content
            python gitlab_jobs.py update_file --project-id=123 --file-path=test.txt --content="New content"

            # Update file from local file
            python gitlab_jobs.py update_file --project-id=123 --file-path=config.yaml --file-source=./local-config.yaml

            # With custom commit message
            python gitlab_jobs.py update_file --project-id=123 --file-path=README.md --content="# Updated" --commit-message="Update README"
        """
        # Validate input
        if content is not None and file_source is not None:
            raise ValueError("Cannot specify both 'content' and 'file_source'. Please provide only one.")

        if content is None and file_source is None:
            raise ValueError("Must specify either 'content' or 'file_source'.")

        # Read content from file if file_source is provided
        if file_source is not None:
            self._log(f"Reading content from local file: {file_source}")
            try:
                with open(file_source, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Local file not found: {file_source}") from e
            except Exception as e:
                raise Exception(f"Error reading file '{file_source}': {e}") from e

        # Update the file
        result = self._update_repository_file(
            project_id=project_id,
            file_path=file_path,
            content=content,
            commit_message=commit_message
        )

        # Log success
        self._log(
            f"File updated successfully:\n"
            f"  Path: {result['file_path']}\n"
            f"  Branch: {result['branch']}\n"
            f"  Project: {result['project_id']}",
            force=True
        )

        return result

    def get_job_status(
        self,
        project_id: int,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get the current status of a specific job.

        This method fetches fresh status information for a job by:
        1. Getting the project and its manifest
        2. Finding the job in the manifest
        3. Checking if it's claimed (response.yaml exists)
        4. Parsing response.yaml if it exists

        Args:
            project_id: GitLab project ID
            job_id: Job ID to check

        Returns:
            Dictionary containing:
            - All fields from the job in the manifest
            - project_id: The project ID
            - is_claimed: Boolean indicating if job is claimed
            - response: Dict with response.yaml content (if claimed), otherwise None

            Returns empty dict {} if job not found.
        """
        # Get the project
        project = self.gl.projects.get(project_id)
        default_branch = project.default_branch or 'main'

        # Get manifest content
        manifest_content = self._get_file_content(
            project,
            MANIFEST_PATH,
            default_branch
        )

        if manifest_content is None:
            return {}

        # Parse manifest
        try:
            manifest_data = yaml.safe_load(manifest_content)
        except yaml.YAMLError:
            return {}

        # Find the job
        jobs = manifest_data.get('jobs', [])
        job_dict = None
        for job in jobs:
            if isinstance(job, dict) and str(job.get('job_id')) == str(job_id):
                job_dict = job.copy()
                break

        if job_dict is None:
            return {}

        # Check if claimed and get response data
        response_path = RESPONSE_FILE_PATH_TEMPLATE.format(job_id=job_id)
        response_content = self._get_file_content(project, response_path, default_branch)

        is_claimed = response_content is not None
        response_data = None

        if is_claimed:
            try:
                response_data = yaml.safe_load(response_content)
            except yaml.YAMLError:
                response_data = None

        # Build result
        job_dict['project_id'] = project_id
        job_dict['is_claimed'] = is_claimed
        job_dict['response'] = response_data

        return job_dict

    def claim_job(
        self,
        project_id: int,
        job_id: str,
        worker_id: str,
        is_stale_claim: bool = False,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Claim a job by creating (or overwriting) a response.yaml file.

        For fresh claims, this creates a new ``response.yaml`` at
        ``gpu_jobs/job_{job_id}/response.yaml``.  If the file already exists,
        ``GitlabCreateError`` is raised (race-condition safe).

        For stale re-claims (``is_stale_claim=True``), the existing
        ``response.yaml`` is overwritten via ``update_file()`` with the new
        worker's ID and an incremented ``retry_count``.

        After writing the file, the claim is verified by reading back
        ``response.yaml`` and checking that the ``worker_id`` matches.

        Args:
            project_id: GitLab project ID
            job_id: Job ID to claim
            worker_id: Unique identifier for this worker
            is_stale_claim: If True, overwrite the existing response.yaml
                            instead of creating a new one.
            retry_count: Current retry count from the stale response.
                         Will be incremented and written to the new response.

        Returns:
            Dictionary containing the claimed job status (from get_job_status):
            - All fields from the job manifest
            - project_id: The project ID
            - is_claimed: True
            - response: Dict with response.yaml content including worker_id
              and retry_count

        Raises:
            GitlabCreateError: If the claim file already exists (fresh claim)
                               or another worker claimed it
            GitlabGetError: If project not found or other API errors
        """
        for attempt in range(MAX_JOB_CLAIM_RETRY):
            # Build the claim file path
            claim_file_path = RESPONSE_FILE_PATH_TEMPLATE.format(job_id=job_id)

            # Build response.yaml content with retry_count
            new_retry_count = retry_count + 1 if is_stale_claim else 1
            response_data = {
                'worker_id': worker_id,
                'state': 'running',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'retry_count': new_retry_count,
            }

            # Convert to YAML
            yaml_content = yaml.dump(response_data, default_flow_style=False, sort_keys=False)

            # Create commit message
            if is_stale_claim:
                commit_message = (
                    f"Worker {worker_id} re-claims stale job {job_id} "
                    f"(retry {new_retry_count})"
                )
            else:
                commit_message = f"Worker {worker_id} claims job {job_id}"

            # Write the claim file
            if is_stale_claim:
                # Overwrite existing stale response.yaml
                self._update_repository_file(
                    project_id=project_id,
                    file_path=claim_file_path,
                    content=yaml_content,
                    commit_message=commit_message,
                )
            else:
                # Fresh claim — create new file (fails if already exists)
                self._create_repository_file(
                    project_id=project_id,
                    file_path=claim_file_path,
                    content=yaml_content,
                    commit_message=commit_message,
                )

            # Verify the claim by reading back the status
            job_status = self.get_job_status(project_id, job_id)

            if job_status.get('is_claimed'):
                response = job_status.get('response', {})
                claimed_worker_id = response.get('worker_id') if response else None

                if claimed_worker_id == worker_id:
                    # Successfully claimed and verified
                    return job_status
                else:
                    # Another worker claimed it - don't retry
                    raise GitlabCreateError("Job claimed by another worker")

            # Job still not claimed - retry if attempts remain
            if attempt < MAX_JOB_CLAIM_RETRY - 1:
                time.sleep(0.5)  # Brief delay before retry

        # All retries exhausted and job still not claimed
        raise GitlabCreateError("Claim verification failed - job not claimed after retries")

    def claim_next_job(
        self,
        worker_id: str,
        claim_timeout_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Attempt to claim the next available job, prioritized by submission time.

        This method:
        1. Gets list of available jobs (including stale-claimed jobs when
           ``claim_timeout_hours`` is set)
        2. Sorts them by submitted_at (oldest first) for fair FIFO ordering.
           Jobs without submitted_at are shuffled randomly and placed at the back.
        3. Iterates through them attempting to claim each
        4. Returns the first successfully claimed job (verified)

        Args:
            worker_id: Unique identifier for this worker
            claim_timeout_hours: Hours after which a claimed job with no
                                 heartbeat update is considered stale and
                                 eligible for re-claiming.  None = no timeout
                                 (only truly unclaimed jobs are returned).

        Returns:
            Dictionary containing the claimed job with:
            - All fields from the job manifest
            - project_id: The project ID
            - is_claimed: True
            - response: Dict with response.yaml content including worker_id
              and retry_count

        Raises:
            NoJobsAvailableError: If no jobs are available to claim
            GitlabGetError: If project not found or other API errors
        """
        self._log("Fetching available jobs...")
        available_jobs = self.list_available_jobs(
            claim_timeout_hours=claim_timeout_hours,
        )

        # Fair job ordering: FIFO by submitted_at (oldest first).
        # Jobs without submitted_at get a high default so they sort to the back.
        # Shuffle first so jobs with the same (or missing) timestamp get random order
        # (Python's sort is stable, so equal keys preserve the shuffled order).
        random.shuffle(available_jobs)
        available_jobs.sort(key=lambda j: j.get('submitted_at', '9999-12-31T23:59:59Z'))

        if not available_jobs:
            raise NoJobsAvailableError("No jobs available to claim")

        self._log(f"Found {len(available_jobs)} available jobs. Attempting to claim...")

        for job_entry in available_jobs:
            project_id = job_entry['project_id']
            job_id = job_entry['job_id']
            stale = job_entry.get('is_stale_claim', False)
            prev_retry_count = job_entry.get('retry_count', 0)

            if stale:
                self._log(
                    f"Attempting to re-claim stale job {job_id} in project "
                    f"{project_id} (retry_count={prev_retry_count})..."
                )
            else:
                self._log(f"Attempting to claim job {job_id} in project {project_id}...")

            try:
                # Attempt to claim the job (includes verification)
                # Returns the job status directly
                job_status = self.claim_job(
                    project_id=project_id,
                    job_id=job_id,
                    worker_id=worker_id,
                    is_stale_claim=stale,
                    retry_count=prev_retry_count,
                )

                self._log(f"Successfully claimed job {job_id}")
                return job_status

            except GitlabCreateError:
                # Job already claimed or verification failed, continue to next
                self._log(f"Job {job_id} already claimed, continuing...")
                continue
            except Exception as e:
                # Log error but continue trying other jobs
                self._log(f"Error claiming job {job_id}: {e}, continuing...")
                continue

        # If we get here, no jobs were successfully claimed
        raise NoJobsAvailableError("Failed to claim any available jobs")

    @retry_with_backoff()
    def _remove_project_member(
        self,
        project_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Remove a user from a project's member list.

        This is the internal method that performs the actual member removal
        via the GitLab Members API (DELETE /projects/:id/members/:user_id).

        Args:
            project_id: GitLab project ID
            user_id: User ID to remove from the project

        Returns:
            Dictionary containing:
            - project_id: The project ID
            - user_id: The removed user's ID
            - project_path: The project's path with namespace

        Raises:
            GitlabDeleteError: If the removal fails (e.g., user is not a member)
            GitlabGetError: If project not found or other API errors
            ValueError: If attempting to remove the project owner
        """
        # Get the project
        project = self.gl.projects.get(project_id)

        # Safety check: prevent removing the project owner
        if hasattr(project, 'owner') and project.owner:
            owner_id = project.owner.get('id') if isinstance(project.owner, dict) else getattr(project.owner, 'id', None)
            if owner_id == user_id:
                raise ValueError(
                    f"Cannot remove user {user_id} from project {project_id}: "
                    f"user is the project owner"
                )

        # Also check creator_id as a fallback ownership indicator
        if hasattr(project, 'creator_id') and project.creator_id == user_id:
            raise ValueError(
                f"Cannot remove user {user_id} from project {project_id}: "
                f"user is the project creator"
            )

        self._log(
            f"Removing user {user_id} from project {project_id} "
            f"({project.path_with_namespace})"
        )

        # Remove the member via the Members API
        project.members.delete(user_id)

        self._log(f"Successfully removed user {user_id} from project {project_id}")

        return {
            'project_id': project_id,
            'user_id': user_id,
            'project_path': project.path_with_namespace
        }

    def unshare_project(
        self,
        project_id: int,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Remove a user from a project's member list (unshare).

        By default, removes the currently authenticated user from the project.
        This is used by the GPU worker to remove itself from projects after
        completing jobs, reducing security exposure and speeding up job scanning.

        Includes a safety check to prevent accidentally removing the project
        owner/creator.

        Args:
            project_id: GitLab project ID
            user_id: User ID to remove. Defaults to the currently authenticated user.

        Returns:
            Dictionary containing:
            - project_id: The project ID
            - user_id: The removed user's ID
            - project_path: The project's path with namespace

        Raises:
            ValueError: If attempting to remove the project owner/creator
            GitlabDeleteError: If the removal fails
            GitlabGetError: If project not found or other API errors

        Examples:
            # Remove yourself from a project
            python gitlab_jobs.py unshare_project --token=YOUR_TOKEN --project-id=123

            # Remove a specific user from a project
            python gitlab_jobs.py unshare_project --token=YOUR_TOKEN --project-id=123 --user-id=551
        """
        # Default to the currently authenticated user
        if user_id is None:
            user_id = self.gl.user.id
            self._log(f"No user_id specified, using current user: {user_id} ({self.gl.user.username})")

        # Perform the removal
        result = self._remove_project_member(
            project_id=project_id,
            user_id=user_id
        )

        # Log success
        self._log(
            f"Project unshared successfully:\n"
            f"  Project: {result['project_id']} ({result['project_path']})\n"
            f"  Removed user: {result['user_id']}",
            force=True
        )

        return result


def main():
    """Main entry point for Fire CLI."""
    fire.Fire(GitLabJobScanner)


if __name__ == '__main__':
    main()
