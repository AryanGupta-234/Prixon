"""CPU monitor (spec section 11).

Uses psutil directly rather than the PowerShell diagnostic in
tools.SAFE_SCRIPTS["cpu_info"] -- that diagnostic is a one-shot, Windows-only,
~subprocess-launch-cost call meant for an explicit user "show me my CPU"
request. This needs to run cheaply and often (every routing decision, and
later every poll tick for baseline/anomaly detection), and psutil already
works cross-platform, which is also what let this be tested on this
non-Windows dev machine at all. tools.py's diagnostic is untouched and still
serves its original purpose.
"""
from __future__ import annotations

from system.snapshot import CpuState

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def read(interval: float = 0.0) -> CpuState:
    """interval=0.0 (default) returns the usage since the last call (or
    since psutil import) rather than blocking to sample -- fine for
    frequent routing checks. Pass a small interval (e.g. 0.1) for a one-off
    accurate read when nothing has called psutil.cpu_percent() recently."""
    if not _HAS_PSUTIL:
        return CpuState()
    try:
        per_core = psutil.cpu_percent(interval=interval, percpu=True)
        usage = sum(per_core) / len(per_core) if per_core else psutil.cpu_percent(interval=interval)
        freq = psutil.cpu_freq()
        return CpuState(
            usage_percent=round(usage, 1),
            per_core_percent=[round(c, 1) for c in per_core],
            frequency_mhz=round(freq.current, 0) if freq else None,
            core_count=psutil.cpu_count(logical=True),
        )
    except Exception:
        return CpuState()
