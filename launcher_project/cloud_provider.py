"""
Cloud Provider Interface — Abstract base class for cloud VM management.

Defines the contract that all cloud providers must implement, enabling
the monitor to work with different clouds (Vast.ai, SkyPilot, etc.)
without cloud-specific code.

Usage:
    from cloud_provider import CloudProvider, InstanceInfo
    from vast_provider import VastCloudProvider

    provider = VastCloudProvider(api_key='...')
    instances = provider.list_instances(prefix='codex-worker-')
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class InstanceInfo:
    """
    Normalized instance information across cloud providers.

    All providers map their native instance data to this structure,
    allowing the monitor to work with any cloud without knowing
    provider-specific details.
    """

    id: str
    """Cloud-specific instance ID (e.g., Vast.ai instance ID, SkyPilot cluster name)."""

    name: str
    """Instance label/name. For our workers, this is 'codex-worker-{uuid}'."""

    status: str
    """Normalized status: 'running', 'init', 'stopped', 'error', 'unknown'."""

    gpu_name: str = ''
    """GPU model name, e.g., 'RTX 3090', 'RTX 3060'."""

    gpu_count: int = 0
    """Number of GPUs."""

    gpu_ram_mb: int = 0
    """GPU RAM in megabytes (e.g., 24576 for 24GB). Useful for filtering out
    low-VRAM variants like RTX 3060 Ti (8GB vs 12GB)."""

    ip: str = ''
    """Public IP address."""

    ssh_port: int = 0
    """SSH port number."""

    created_at: Optional[datetime] = None
    """When the instance was created/launched (UTC)."""

    cost_per_hour: float = 0.0
    """Cost in USD per hour."""

    raw: Dict = field(default_factory=dict)
    """Raw provider-specific data for debugging or advanced use."""


class CloudProvider(ABC):
    """
    Abstract base class for cloud VM providers.

    Implementations must provide methods for listing, launching, and
    destroying instances. The monitor uses this interface to manage
    GPU workers without knowing which cloud is being used.
    """

    @abstractmethod
    def list_instances(self, prefix: str = '') -> List[InstanceInfo]:
        """
        List all instances, optionally filtered by name prefix.

        Args:
            prefix: If provided, only return instances whose name starts
                    with this prefix (e.g., 'codex-worker-').

        Returns:
            List of InstanceInfo objects for matching instances.
        """
        ...

    @abstractmethod
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

        Args:
            name: Name/label for the instance (e.g., 'codex-worker-abc12345').
            yaml_path: Path to the task definition file (SkyPilot YAML or equivalent).
            envs: Environment variables to set on the instance.
            down: If True, auto-terminate when the task exits.
            idle_minutes_to_autostop: Minutes of idle time before auto-teardown.
            stream: If True, stream logs to stdout (for interactive use).

        Returns:
            Instance ID on success, None on failure.
        """
        ...

    @abstractmethod
    def destroy_instance(self, instance_id: str) -> bool:
        """
        Destroy/terminate an instance.

        Args:
            instance_id: The cloud-specific instance ID to destroy.

        Returns:
            True on success, False on failure.
        """
        ...
