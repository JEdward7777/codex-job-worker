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
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests  # pylint: disable=import-error

from cloud_provider import CloudProvider, InstanceInfo  # pylint: disable=import-error
from skypilot_provider import SkyPilotProvider  # pylint: disable=import-error

# Vast.ai API base URL
VAST_API_BASE = 'https://console.vast.ai/api/v0'

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

    def list_instances(self, prefix: str = '') -> List[InstanceInfo]:
        """
        List Vast.ai instances using the REST API.

        Args:
            prefix: If provided, only return instances whose label starts
                    with this prefix (e.g., 'codex-worker-').

        Returns:
            List of InstanceInfo objects for matching instances.
        """
        try:
            resp = requests.get(
                f'{VAST_API_BASE}/instances',
                params={'owner': 'me'},
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to list Vast.ai instances: {e}")
            return []

        instances_data = data.get('instances', [])
        if not instances_data:
            return []

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

    def destroy_instance(self, instance_id: str) -> bool:
        """
        Destroy a Vast.ai instance.

        Currently delegates to SkyPilot. Will be reimplemented with
        direct Vast.ai API calls in the future.
        """
        self.logger.debug(
            "VastCloudProvider.destroy_instance: delegating to SkyPilot"
        )
        return self._skypilot.destroy_instance(instance_id)

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
