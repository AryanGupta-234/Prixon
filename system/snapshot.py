"""Normalized system state (spec section 14).

This is internal structured data -- nothing in this file ever gets dumped
whole into an LLM prompt (spec sections 13/44 are explicit about that).
Consumers (brain/router.py, and later the anomaly detector / alert engine)
read specific fields off this, not the raw dict.

Every monitor degrades to None/empty on its own field rather than raising --
see collector.py -- so a snapshot is always constructable even on hardware
missing a sensor (no battery, no GPU, sandboxed/virtualized network), per
spec section 11's "do not assume GPU telemetry is available."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CpuState:
    usage_percent: Optional[float] = None
    per_core_percent: List[float] = field(default_factory=list)
    frequency_mhz: Optional[float] = None
    core_count: Optional[int] = None


@dataclass
class MemoryState:
    total_mb: Optional[float] = None
    used_mb: Optional[float] = None
    available_mb: Optional[float] = None
    percent: Optional[float] = None


@dataclass
class DiskState:
    total_gb: Optional[float] = None
    free_gb: Optional[float] = None
    used_percent: Optional[float] = None
    path: str = "/"


@dataclass
class NetworkAdapterState:
    name: str
    is_up: bool
    role: str  # "physical" | "virtual" | "loopback" | "unknown"


@dataclass
class NetworkState:
    adapters: List[NetworkAdapterState] = field(default_factory=list)
    online: Optional[bool] = None  # None = couldn't determine, not "offline"


@dataclass
class BatteryState:
    present: bool = False
    percent: Optional[float] = None
    charging: Optional[bool] = None
    seconds_remaining: Optional[int] = None


@dataclass
class GpuState:
    available: bool = False
    utilization_percent: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    unavailable_reason: Optional[str] = None


@dataclass
class SystemSnapshot:
    timestamp: float = field(default_factory=time.time)
    cpu: CpuState = field(default_factory=CpuState)
    memory: MemoryState = field(default_factory=MemoryState)
    disk: DiskState = field(default_factory=DiskState)
    network: NetworkState = field(default_factory=NetworkState)
    battery: BatteryState = field(default_factory=BatteryState)
    gpu: GpuState = field(default_factory=GpuState)
    # Per-monitor errors, kept out of band so a failed sensor doesn't fail
    # the whole snapshot -- see collector.py. Diagnostic use only (e.g.
    # `--debug`); not meant for the LLM prompt.
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu": vars(self.cpu),
            "memory": vars(self.memory),
            "disk": vars(self.disk),
            "network": {
                "adapters": [vars(a) for a in self.network.adapters],
                "online": self.network.online,
            },
            "battery": vars(self.battery),
            "gpu": vars(self.gpu),
            "errors": self.errors,
        }
