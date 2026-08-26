"""Battery monitor (spec section 11). Desktops and this sandbox report no
battery at all -- present=False is a legitimate, common answer, not an
error, so this never populates `errors` for that case."""
from __future__ import annotations

from system.snapshot import BatteryState

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def read() -> BatteryState:
    if not _HAS_PSUTIL:
        return BatteryState(present=False)
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return BatteryState(present=False)
        seconds = battery.secsleft if isinstance(battery.secsleft, int) and battery.secsleft >= 0 else None
        return BatteryState(
            present=True, percent=battery.percent, charging=battery.power_plugged, seconds_remaining=seconds,
        )
    except Exception:
        return BatteryState(present=False)
