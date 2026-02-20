"""
SkyPilot Cloud Provider — Manages GPU VMs via SkyPilot CLI/API.

This provider wraps SkyPilot's CLI and Python API to implement the
CloudProvider interface. It handles the fileno bugs in SkyPilot's
API server by using CLI subprocess calls where needed.

Usage:
    from skypilot_provider import SkyPilotProvider

    provider = SkyPilotProvider(logger=logger)
    instances = provider.list_instances(prefix='codex-worker-')
"""

import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional

from cloud_provider import CloudProvider, InstanceInfo  # pylint: disable=import-error


class SkyPilotProvider(CloudProvider):
    """
    Cloud provider implementation using SkyPilot.

    Uses the SkyPilot Python API for status queries and the CLI
    (subprocess) for launching instances to avoid fileno errors.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def list_instances(self, prefix: str = '') -> List[InstanceInfo]:
        """
        List SkyPilot clusters, optionally filtered by name prefix.

        Uses AUTO refresh mode to query the cloud provider for accurate
        status. Falls back to NONE (cached) if AUTO fails with the
        Vast.ai fileno bug.
        """
        try:
            import sky  # pylint: disable=import-error

            # Try AUTO first — queries the cloud for accurate status.
            try:
                request_id = sky.status(refresh=sky.StatusRefreshMode.AUTO)
                clusters = sky.get(request_id)
            except Exception as auto_err:
                # AUTO fails with "fileno" error on Vast.ai in the API
                # server context. Fall back to NONE (cached status).
                self.logger.warning(
                    f"sky.status(AUTO) failed ({auto_err}), "
                    f"falling back to cached status"
                )
                request_id = sky.status(refresh=sky.StatusRefreshMode.NONE)
                clusters = sky.get(request_id)
        except Exception as e:
            self.logger.error(f"Failed to get sky status: {e}")
            return []

        if not clusters:
            return []

        results = []
        for cluster in clusters:
            name = (cluster.name if hasattr(cluster, 'name')
                    else cluster.get('name', ''))

            if prefix and not name.startswith(prefix):
                continue

            # Extract status
            raw_status = (cluster.status if hasattr(cluster, 'status')
                          else cluster.get('status'))

            import sky as sky_module  # pylint: disable=import-error
            if raw_status == sky_module.ClusterStatus.UP:
                status = 'running'
            elif raw_status == sky_module.ClusterStatus.INIT:
                status = 'init'
            elif raw_status == sky_module.ClusterStatus.STOPPED:
                status = 'stopped'
            else:
                status = 'unknown'

            # Extract timestamps
            launched_at = (cluster.launched_at if hasattr(cluster, 'launched_at')
                           else cluster.get('launched_at'))
            created_at = None
            if launched_at is not None:
                if isinstance(launched_at, (int, float)):
                    created_at = datetime.fromtimestamp(launched_at, tz=timezone.utc)
                elif isinstance(launched_at, datetime):
                    created_at = launched_at

            # Extract status_updated_at for the raw dict
            status_updated_at = (
                cluster.status_updated_at
                if hasattr(cluster, 'status_updated_at')
                else cluster.get('status_updated_at')
            )

            results.append(InstanceInfo(
                id=name,  # SkyPilot uses cluster name as ID
                name=name,
                status=status,
                created_at=created_at,
                raw={
                    'status_updated_at': status_updated_at,
                    'sky_status': raw_status,
                },
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
        Launch a SkyPilot cluster using the CLI (subprocess).

        The Python API hits fileno errors in SSH/cron contexts, so we
        use the CLI which works reliably.

        When stream=False (the default for monitor cycles), uses --detach-run
        so that sky launch returns immediately after submitting the job
        to the SkyPilot API server. The provisioning continues in the
        background. This prevents the monitor from blocking for 5-10
        minutes per worker launch.

        When stream=True (for interactive test_launch), follows the logs
        so the user can see provisioning progress.

        Returns the cluster name on success, None on failure.
        """
        self.logger.info(f"Launching cluster: {name} from {yaml_path}")

        try:
            # Build the sky launch command
            cmd = ['sky', 'launch', '-c', name, yaml_path]

            if down:
                cmd.append('--down')

            if idle_minutes_to_autostop is not None:
                cmd.extend(['-i', str(idle_minutes_to_autostop)])

            # Add environment variables
            for key, value in envs.items():
                cmd.extend(['--env', f'{key}={value}'])

            # Always use -y to skip confirmation prompts
            cmd.append('-y')

            # When not streaming, use --detach-run so sky launch blocks
            # during VM provisioning (which takes ~4 min) but returns
            # immediately once the run phase starts. This is intentional:
            # the flock in run_monitor_cron.sh prevents overlapping
            # monitor cycles, so blocking during provisioning acts as a
            # natural guard against launching duplicate workers.
            if not stream:
                cmd.append('--detach-run')

            # Run the command
            if stream:
                result = subprocess.run(cmd, check=False)
            else:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

            if result.returncode != 0:
                error_msg = (result.stderr if not stream and result.stderr
                             else "see logs above")
                self.logger.error(
                    f"sky launch failed with exit code {result.returncode}: "
                    f"{error_msg}"
                )
                return None

            self.logger.info(f"Successfully launched: {name}")
            return name

        except Exception as e:
            self.logger.error(f"Failed to launch {name}: {e}")
            return None

    def destroy_instance(self, instance_id: str) -> bool:
        """
        Tear down a SkyPilot cluster.

        Uses the Python API with follow=False (fire-and-forget) to avoid
        fileno errors.
        """
        try:
            import sky  # pylint: disable=import-error
            self.logger.info(f"Tearing down cluster: {instance_id}")
            request_id = sky.down(instance_id)
            sky.stream_and_get(request_id, follow=False)
            self.logger.info(f"Tear down initiated: {instance_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to tear down {instance_id}: {e}")
            return False

    def check_configured(self) -> bool:
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
            if result.returncode != 0:
                self.logger.error(
                    "SkyPilot check failed. Please run 'sky check' and "
                    "configure your cloud credentials."
                )
                return False

            output = result.stdout + result.stderr
            if 'enabled' not in output.lower():
                self.logger.error(
                    "No cloud providers enabled in SkyPilot. "
                    "Please run 'sky check'."
                )
                return False

            return True
        except FileNotFoundError:
            self.logger.error(
                "SkyPilot CLI not found. Install with: "
                "uv add 'skypilot[vast]'"
            )
            return False
        except Exception as e:
            self.logger.error(f"Error checking SkyPilot: {e}")
            return False
