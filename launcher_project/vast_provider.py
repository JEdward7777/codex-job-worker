"""
Vast.ai Cloud Provider — Direct integration with Vast.ai REST API.

Bypasses SkyPilot for instance listing (and eventually launching/destroying)
to avoid SkyPilot's fileno bugs and provide better control over GPU selection
(e.g., filtering out low-VRAM RTX 3060 Ti variants).

For operations not yet reimplemented (launch, destroy), delegates to an
internal SkyPilotProvider instance.

Usage:
    from vast_provider import VastCloudProvider

    provider = VastCloudProvider(api_key='...', logger=logger)
    instances = provider.list_instances(prefix='codex-worker-')

    # Launch still goes through SkyPilot (for now)
    provider.launch_instance(name='codex-worker-abc', yaml_path='worker.yaml', ...)
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests  # pylint: disable=import-error

from cloud_provider import CloudProvider, InstanceInfo  # pylint: disable=import-error
from skypilot_provider import SkyPilotProvider  # pylint: disable=import-error

# Vast.ai API base URL
VAST_API_BASE = 'https://console.vast.ai/api/v0'

# Retry settings for Vast.ai API calls (exponential backoff)
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 2.0

# SkyPilot names Vast.ai instances as "{cluster_name}-{host_id}-head".
# This regex extracts the cluster name portion from the full Vast.ai label.
# Example: "codex-worker-006479f5-413d3bc4-head" → "codex-worker-006479f5"
_SKYPILOT_LABEL_RE = re.compile(r'^(.+)-[0-9a-f]{8}-head$')

# Map Vast.ai actual_status values to our normalized statuses
_VAST_STATUS_MAP = {
    'running': 'running',
    'loading': 'init',
    'exited': 'stopped',
    'stopped': 'stopped',
    'created': 'init',
    'starting': 'init',
    'error': 'error',
}


class VastCloudProvider(CloudProvider):
    """
    Cloud provider implementation using the Vast.ai REST API directly.

    Uses the Vast.ai REST API for listing instances (reliable, no fileno
    issues). Delegates to SkyPilotProvider for launch/destroy until those
    are reimplemented with direct Vast.ai API calls.
    """

    def __init__(
        self,
        api_key: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Vast.ai provider.

        Args:
            api_key: Vast.ai API key.
            logger: Logger instance.
        """
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self._headers = {'Authorization': f'Bearer {api_key}'}

        # Internal SkyPilot provider for operations not yet reimplemented
        self._skypilot = SkyPilotProvider(logger=self.logger)

        # Cache mapping Vast.ai instance IDs to their labels (populated by
        # list_instances).  Used by destroy_instance to resolve the SkyPilot
        # cluster name from a numeric Vast.ai ID.
        self._id_to_label: Dict[str, str] = {}

    def list_instances(self, prefix: str = '') -> Optional[List[InstanceInfo]]:
        """
        List Vast.ai instances using the REST API.

        Args:
            prefix: If provided, only return instances whose label starts
                    with this prefix (e.g., 'codex-worker-').

        Returns:
            List of InstanceInfo objects for matching instances, or *None*
            if the API request failed (e.g., rate limit, network error).
            Returning None (rather than []) lets the caller distinguish
            "no instances exist" from "we couldn't check".
        """
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(
                    f'{VAST_API_BASE}/instances',
                    params={'owner': 'me'},
                    headers=self._headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break  # Success — exit retry loop
            except requests.RequestException as e:
                last_error = e
                is_rate_limit = (
                    isinstance(e, requests.HTTPError)
                    and e.response is not None
                    and e.response.status_code == 429
                )
                if is_rate_limit:
                    # Log full 429 response details so we can tune retry
                    # behaviour based on what Vast.ai actually sends back.
                    resp_obj = e.response
                    self.logger.debug(
                        f"429 response headers: {dict(resp_obj.headers)}"
                    )
                    try:
                        body_text = resp_obj.text[:500]  # cap at 500 chars
                    except Exception:
                        body_text = '<unreadable>'
                    self.logger.debug(
                        f"429 response body (first 500 chars): {body_text}"
                    )
                if attempt < _MAX_RETRIES - 1 and is_rate_limit:
                    # Respect the Retry-After header if the server provides
                    # one; otherwise fall back to exponential backoff.
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after is not None:
                        self.logger.debug(
                            f"Retry-After header present: {retry_after!r}"
                        )
                        try:
                            backoff = float(retry_after)
                        except (ValueError, TypeError):
                            self.logger.debug(
                                f"Could not parse Retry-After as float, "
                                f"falling back to exponential backoff"
                            )
                            backoff = _INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    else:
                        self.logger.debug(
                            "No Retry-After header in 429 response, "
                            "using exponential backoff"
                        )
                        backoff = _INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    self.logger.warning(
                        f"Vast.ai API rate limited (429), "
                        f"retrying in {backoff:.1f}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                    )
                    time.sleep(backoff)
                else:
                    self.logger.error(
                        f"Failed to list Vast.ai instances: {e}"
                    )
                    return None
        else:
            # All retries exhausted
            self.logger.error(
                f"Failed to list Vast.ai instances after {_MAX_RETRIES} "
                f"attempts: {last_error}"
            )
            return None

        instances_data = data.get('instances', [])
        if not instances_data:
            return []

        # Refresh the ID-to-label cache so destroy_instance can resolve
        # Vast.ai numeric IDs to SkyPilot cluster names.
        self._id_to_label = {}
        for inst in instances_data:
            inst_id = str(inst.get('id', ''))
            inst_label = inst.get('label') or ''
            if inst_id and inst_label:
                self._id_to_label[inst_id] = inst_label

        results = []
        for inst in instances_data:
            # Get the instance label (name)
            label = inst.get('label') or ''

            # Filter by prefix if specified
            if prefix and not label.startswith(prefix):
                continue

            # Map status
            raw_status = inst.get('actual_status', inst.get('status_msg', ''))
            status = _VAST_STATUS_MAP.get(raw_status, 'unknown')

            # Parse created_at timestamp
            created_at = None
            start_date = inst.get('start_date')
            if start_date is not None:
                try:
                    if isinstance(start_date, (int, float)):
                        created_at = datetime.fromtimestamp(
                            start_date, tz=timezone.utc
                        )
                    elif isinstance(start_date, str):
                        created_at = datetime.fromisoformat(
                            start_date.replace('Z', '+00:00')
                        )
                except (ValueError, OSError):
                    pass

            # Extract GPU info
            gpu_name = inst.get('gpu_name', '')
            gpu_count = inst.get('num_gpus', 0)
            # gpu_ram is in MB in the API
            gpu_ram_mb = int(inst.get('gpu_ram', 0))

            # Network info
            ip = inst.get('public_ipaddr', '')
            ssh_port = inst.get('ssh_port', 0)

            # Cost
            cost_per_hour = float(inst.get('dph_total', 0))

            results.append(InstanceInfo(
                id=str(inst.get('id', '')),
                name=label,
                status=status,
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                gpu_ram_mb=gpu_ram_mb,
                ip=ip,
                ssh_port=ssh_port,
                created_at=created_at,
                cost_per_hour=cost_per_hour,
                raw=inst,
            ))

        return results

    def launch_instance(
        self,
        name: str,
        yaml_path: str,
        envs: Dict[str, str],
        down: bool = True,
        idle_minutes_to_autostop: Optional[int] = None,
        stream: bool = False,
    ) -> Optional[str]:
        """
        Launch a new instance.

        Currently delegates to SkyPilot. Will be reimplemented with
        direct Vast.ai API calls in the future.
        """
        self.logger.debug(
            "VastCloudProvider.launch_instance: delegating to SkyPilot"
        )
        return self._skypilot.launch_instance(
            name=name,
            yaml_path=yaml_path,
            envs=envs,
            down=down,
            idle_minutes_to_autostop=idle_minutes_to_autostop,
            stream=stream,
        )

    @staticmethod
    def _extract_cluster_name(label: str) -> Optional[str]:
        """
        Extract the SkyPilot cluster name from a Vast.ai instance label.

        SkyPilot names Vast.ai instances as ``{cluster}-{host_id}-head``.
        For example, ``codex-worker-006479f5-413d3bc4-head`` belongs to
        cluster ``codex-worker-006479f5``.

        Args:
            label: The Vast.ai instance label.

        Returns:
            The SkyPilot cluster name, or *None* if the label doesn't
            match the expected pattern.
        """
        m = _SKYPILOT_LABEL_RE.match(label)
        return m.group(1) if m else None

    def destroy_instance(self, instance_id: str) -> bool:
        """
        Destroy a Vast.ai instance by tearing down its SkyPilot cluster.

        The monitor passes the Vast.ai numeric instance ID here, but
        SkyPilot's ``sky.down()`` expects a *cluster name*.  We resolve
        the cluster name from the instance label (cached by the most
        recent ``list_instances()`` call) and pass that to SkyPilot.

        If the label cannot be resolved or doesn't match the expected
        SkyPilot naming pattern, we fall back to passing the raw
        ``instance_id`` (which may or may not work, but at least won't
        silently destroy the wrong cluster).
        """
        # Look up the Vast.ai label for this instance ID
        label = self._id_to_label.get(instance_id, '')
        cluster_name = self._extract_cluster_name(label) if label else None

        if cluster_name:
            self.logger.info(
                f"Resolved Vast.ai instance {instance_id} "
                f"(label={label}) to SkyPilot cluster: {cluster_name}"
            )
        else:
            # Fallback — log a warning so we know something is off
            self.logger.warning(
                f"Could not resolve SkyPilot cluster name for Vast.ai "
                f"instance {instance_id} (label={label!r}). "
                f"Passing raw instance_id to sky.down() — this may not work."
            )
            cluster_name = instance_id

        self.logger.debug(
            "VastCloudProvider.destroy_instance: delegating to SkyPilot "
            f"with cluster_name={cluster_name}"
        )
        return self._skypilot.destroy_instance(cluster_name)

    def check_configured(self) -> bool:
        """
        Verify that the Vast.ai API key is valid by making a test request.

        Also checks SkyPilot configuration since launch/destroy still
        delegate to it.
        """
        # Check Vast.ai API key
        try:
            resp = requests.get(
                f'{VAST_API_BASE}/instances',
                params={'owner': 'me'},
                headers=self._headers,
                timeout=15,
            )
            if resp.status_code == 401:
                self.logger.error(
                    "Vast.ai API key is invalid. Check VAST_API_KEY in .env"
                )
                return False
            resp.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to connect to Vast.ai API: {e}")
            return False

        # Also check SkyPilot since launch/destroy still use it
        if not self._skypilot.check_configured():
            return False

        return True
