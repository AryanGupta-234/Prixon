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
    "diagnostic": ToolSpec("diagnostic", "Run a fixed, read-only Windows diagnostic.", "low", diagnostic),
}


def get_tool(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)
