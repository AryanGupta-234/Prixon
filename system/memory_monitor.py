"""Memory monitor (spec section 11). See cpu_monitor.py for why psutil
rather than the PowerShell diagnostic."""
from __future__ import annotations

from system.snapshot import MemoryState

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def read() -> MemoryState:
    if not _HAS_PSUTIL:
        return MemoryState()
    try:
        vm = psutil.virtual_memory()
        return MemoryState(
            total_mb=round(vm.total / (1024 * 1024), 1),
            used_mb=round(vm.used / (1024 * 1024), 1),
            available_mb=round(vm.available / (1024 * 1024), 1),
            percent=vm.percent,
        )
    except Exception:
        return MemoryState()
