"""GPU monitor (spec section 11): 'detect capabilities dynamically... do
not assume GPU telemetry is available.'

Only NVIDIA (via pynvml, optional dependency) is wired up here -- that's
the realistic case for local-model VRAM questions (spec section 20's
'can I run this model locally?'). Intel/AMD integrated graphics on the
target ASUS Vivobook have no equivalently simple cross-platform telemetry
API, so this reports unavailable with a clear reason rather than guessing.
Extend with another backend later without changing the GpuState shape.
"""
from __future__ import annotations

from system.snapshot import GpuState

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

_nvml_initialized = False
_nvml_init_failed_reason = None


def _ensure_nvml():
    global _nvml_initialized, _nvml_init_failed_reason
    if _nvml_initialized or _nvml_init_failed_reason is not None:
        return
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
    except Exception as exc:  # noqa: BLE001
        _nvml_init_failed_reason = f"nvmlInit failed: {exc}"


def read() -> GpuState:
    if not _HAS_PYNVML:
        return GpuState(available=False, unavailable_reason="pynvml not installed (no NVIDIA GPU support configured)")

    _ensure_nvml()
    if not _nvml_initialized:
        return GpuState(available=False, unavailable_reason=_nvml_init_failed_reason)

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuState(
            available=True,
            utilization_percent=float(util.gpu),
            vram_used_mb=round(mem.used / (1024 * 1024), 1),
            vram_total_mb=round(mem.total / (1024 * 1024), 1),
        )
    except Exception as exc:  # noqa: BLE001
        return GpuState(available=False, unavailable_reason=f"query failed: {exc}")
