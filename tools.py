"""Allow-listed Windows tools.

The LLM can request a tool by name, but it never supplies an arbitrary shell
command. Every executable action is resolved here from a fixed registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import os
import subprocess
import sys


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: str
    handler: Callable[..., "ToolResult"]


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: Any = None


def _windows_only() -> Optional[ToolResult]:
    if not sys.platform.startswith("win"):
        return ToolResult(False, "This tool is available only on Windows.")
    return None


def open_uri(uri: str) -> ToolResult:
    bad = _windows_only()
    if bad:
        return bad
    try:
        os.startfile(uri)  # type: ignore[attr-defined]
        return ToolResult(True, "Opened.")
    except Exception as exc:
        return ToolResult(False, f"I couldn't open that: {exc}")


def launch_process(executable: str, args: str = "") -> ToolResult:
    bad = _windows_only()
    if bad:
        return bad
    # executable is resolved from the allow-list; args are optional fixed
    # arguments supplied by the application/tool wrapper, not raw user text.
    try:
        cmd = [executable] + ([args] if args else [])
        subprocess.Popen(cmd, shell=False)
        return ToolResult(True, "Launched.")
    except Exception as exc:
        return ToolResult(False, f"I couldn't launch that: {exc}")


# Defense-in-depth beyond the dataset allow-list: even if a dataset row were
# ever misconfigured to point at one of these, close_process refuses
# outright rather than trusting the allow-list alone. Closing the wrong
# process is far more consequential than launching the wrong one, so this
# gets its own explicit guard rather than relying solely on "the dataset
# would never contain this."
_NEVER_CLOSE = {
    "explorer.exe", "svchost.exe", "csrss.exe", "winlogon.exe", "wininit.exe",
    "services.exe", "lsass.exe", "smss.exe", "dwm.exe", "system", "system idle process",
    "python.exe", "python3.exe", "pythonw.exe",  # never let it close its own interpreter
}


def close_process(executable: str) -> ToolResult:
    """Terminates all running processes whose image name matches
    `executable` (case-insensitive). Tries a graceful terminate() first,
    escalating to kill() only for processes still alive after a short
    grace period -- an abrupt kill() risks unsaved user data, whereas
    terminate() gives the app a chance to clean up first.

    Like launch_process, this function itself will run with whatever name
    it's given -- safety comes from the dataset only ever supplying a fixed,
    reviewed executable name per allow-listed target (spec section 46: the
    LLM never invents the process name, it only selects among existing
    dataset targets). _NEVER_CLOSE above is the extra guard in case that
    allow-list is ever wrong.
    """
    bad = _windows_only()
    if bad:
        return bad
    try:
        import psutil
    except ImportError:
        return ToolResult(False, "psutil is required to close applications and is not installed.")

    target = (executable or "").strip().lower()
    if not target or target in _NEVER_CLOSE:
        return ToolResult(False, f"Closing '{executable}' is not permitted.")

    matched = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if (proc.info.get("name") or "").lower() == target:
                matched.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not matched:
        return ToolResult(True, f"'{executable}' doesn't appear to be running.")

    for proc in matched:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, still_alive = psutil.wait_procs(matched, timeout=3)
    for proc in still_alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return ToolResult(True, f"Closed {len(matched)} process(es) named '{executable}'.")


def run_powershell(script: str) -> ToolResult:
    bad = _windows_only()
    if bad:
        return bad
    try:
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=20,
        )
        if p.returncode != 0:
            return ToolResult(False, p.stderr.strip() or "The command failed.")
        return ToolResult(True, "Done.", p.stdout.strip())
    except subprocess.TimeoutExpired:
        return ToolResult(False, "That took too long and timed out.")
    except Exception as exc:
        return ToolResult(False, f"Something went wrong: {exc}")


# Safe diagnostic scripts are fixed constants. The model cannot replace them.
SAFE_SCRIPTS: Dict[str, str] = {
    "list_processes": "Get-Process | Select-Object Name,Id,CPU | Sort-Object CPU -Descending | Select-Object -First 30 | ConvertTo-Json -Compress",
    "list_services": "Get-Service | Select-Object Status,Name,DisplayName | ConvertTo-Json -Compress",
    "network_status": "Get-NetAdapter | Select-Object Name,Status,LinkSpeed | ConvertTo-Json -Compress",
    "ip_config": "Get-NetIPConfiguration | Select-Object InterfaceAlias,IPv4Address,IPv6Address,DNSServer | ConvertTo-Json -Compress",
    "disk_usage": "Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | ConvertTo-Json -Compress",
    "os_info": "Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber | ConvertTo-Json -Compress",
    "battery_status": "Get-CimInstance Win32_Battery | Select-Object Name,BatteryStatus,EstimatedChargeRemaining,EstimatedRunTime | ConvertTo-Json -Compress",
    "cpu_info": "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,LoadPercentage | ConvertTo-Json -Compress",
    "memory_info": "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress",
    "startup_apps": "Get-CimInstance Win32_StartupCommand | Select-Object Name,Command,Location | ConvertTo-Json -Compress",
}


def diagnostic(name: str) -> ToolResult:
    script = SAFE_SCRIPTS.get(name)
    if not script:
        return ToolResult(False, "That diagnostic is not available.")
    return run_powershell(script)


TOOLS: Dict[str, ToolSpec] = {
    "open_uri": ToolSpec("open_uri", "Open an allow-listed Windows URI.", "low", open_uri),
    "launch_process": ToolSpec("launch_process", "Launch an allow-listed Windows application.", "low", launch_process),
    "close_process": ToolSpec("close_process", "Close an allow-listed running application.", "medium", close_process),
    "diagnostic": ToolSpec("diagnostic", "Run a fixed, read-only Windows diagnostic.", "low", diagnostic),
}


def get_tool(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)
