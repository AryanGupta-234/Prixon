"""Disk monitor (spec section 11). Defaults to the system drive (C:\\ on
Windows, / elsewhere) -- per-drive breakdown (spec's Get-PSDrive diagnostic
already covers "all drives" for an explicit user request) isn't needed for
routing decisions, just "is the primary drive under pressure."
"""
from __future__ import annotations

import os
import sys

from system.snapshot import DiskState

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _default_path() -> str:
    return "C:\\" if sys.platform.startswith("win") else "/"


def read(path: str = None) -> DiskState:
    path = path or _default_path()
    if not _HAS_PSUTIL:
        return DiskState(path=path)
    try:
        usage = psutil.disk_usage(path)
        return DiskState(
            total_gb=round(usage.total / (1024 ** 3), 1),
            free_gb=round(usage.free / (1024 ** 3), 1),
            used_percent=usage.percent,
            path=path,
        )
    except Exception:
        return DiskState(path=path)
