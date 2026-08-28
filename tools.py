"""Allow-listed Windows tools with runtime environment resolution."""
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
    try:
        cmd = [executable] + ([args] if args else [])
        subprocess.Popen(cmd, shell=False)
        return ToolResult(True, "Launched.")
    except Exception as exc:
        return ToolResult(False, f"I couldn't launch that: {exc}")


_NEVER_CLOSE = {
    "explorer.exe", "svchost.exe", "csrss.exe", "winlogon.exe", "wininit.exe",
    "services.exe", "lsass.exe", "smss.exe", "dwm.exe", "system", "system idle process",
    "python.exe", "python3.exe", "pythonw.exe",
}


def close_process(executable: str) -> ToolResult:
    """Terminate a currently running process after safety checks."""
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


def _normalize_name(value: str) -> str:
    import re
    value = (value or "").lower().replace(".exe", "")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _process_score(hint: str, proc_name: str, exe_path: str = "", cmdline: str = "") -> float:
    """Score a running process against a human name without an app allow-list.

    Matching uses the actual runtime process name/path/command line. This
    makes aliases discoverable from the machine rather than maintained as
    hardcoded application mappings.
    """
    import difflib
    h = _normalize_name(hint)
    if not h:
        return 0.0
    candidates = [
        _normalize_name(proc_name),
        _normalize_name(os.path.basename(exe_path)),
        _normalize_name(exe_path),
        _normalize_name(cmdline),
    ]
    best = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        if h == candidate:
            best = max(best, 1.0)
        elif h in candidate or candidate in h:
            best = max(best, 0.90)
        else:
            # Compare individual words as well as the whole phrase so names
            # such as "visual studio code" can match a runtime path containing
            # "Microsoft VS Code" without a hardcoded alias table.
            hw = set(h.split())
            cw = set(candidate.split())
            if hw and cw:
                overlap = len(hw & cw) / max(1, len(hw | cw))
                best = max(best, overlap * 0.82)
            best = max(best, difflib.SequenceMatcher(None, h, candidate).ratio() * 0.78)
    return best


def find_running_app(name_hint: str) -> Optional[str]:
    """Resolve a human application name against the live process environment.

    No application-name dictionary is used. The resolver inspects process
    image names, executable paths and command lines at the moment of the
    request, then returns only a process that is actually running.
    """
    try:
        import psutil
    except ImportError:
        return None

    hint = (name_hint or "").strip()
    if not hint:
        return None

    scored = []
    for proc in psutil.process_iter(["name", "exe", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            exe = proc.info.get("exe") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
            score = _process_score(hint, name, exe, cmdline)
            if score >= 0.62:
                scored.append((score, name))
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def close_running_app(name_hint: str) -> ToolResult:
    """Resolve a currently running process dynamically, then close it."""
    bad = _windows_only()
    if bad:
        return bad
    resolved = find_running_app(name_hint)
    if not resolved:
        return ToolResult(True, f"I don't see anything matching '{name_hint}' currently running.")
    return close_process(resolved)


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
