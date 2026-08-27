"""Assembles a full SystemSnapshot from the individual monitors.

Each monitor already degrades internally (returns an empty/default dataclass
rather than raising) -- this adds one more layer of isolation on top so a
genuinely unexpected exception in one monitor (a psutil API that behaves
differently on some platform this wasn't tested on, say) still leaves the
other fields populated instead of losing the whole snapshot. Per spec
section 11: "detect capabilities dynamically," never let one missing sensor
take down system awareness as a whole.
"""
from __future__ import annotations

from system import battery_monitor, cpu_monitor, disk_monitor, gpu_monitor, memory_monitor, network_monitor, process_monitor
from system.snapshot import SystemSnapshot


class SystemCollector:
    def snapshot(self, probe_internet: bool = True, cpu_interval: float = 0.0) -> SystemSnapshot:
        snap = SystemSnapshot()
        for field_name, fn in (
            ("cpu", lambda: cpu_monitor.read(interval=cpu_interval)),
            ("memory", memory_monitor.read),
            ("disk", disk_monitor.read),
            ("network", lambda: network_monitor.read(probe_internet=probe_internet)),
            ("battery", battery_monitor.read),
            ("gpu", gpu_monitor.read),
            ("processes", process_monitor.read),
        ):
            try:
                setattr(snap, field_name, fn())
            except Exception as exc:  # noqa: BLE001
                snap.errors[field_name] = str(exc)
        return snap


# Module-level default instance -- monitors are stateless enough that one
# shared collector is fine; nothing here caches results between calls.
_default_collector = SystemCollector()


def snapshot(probe_internet: bool = True, cpu_interval: float = 0.0) -> SystemSnapshot:
    return _default_collector.snapshot(probe_internet=probe_internet, cpu_interval=cpu_interval)
