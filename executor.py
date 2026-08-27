"""Policy + execution layer.

The model is never allowed to pass arbitrary shell text to this module.
Actions resolve to allow-listed targets in data_store and fixed tool wrappers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import sys

import config
from tools import ToolResult, close_process, close_running_app, diagnostic, launch_process, open_uri


class ExecutionResult:
    def __init__(self, ok: bool, message: str, data: Any = None):
        self.ok = ok
        self.message = message
        self.data = data


def needs_confirmation(risk: str) -> bool:
    return (risk or "low").lower() in config.CONFIRM_RISK_LEVELS


def _run_allowlisted(group, parameters: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    parameters = parameters or {}
    action = (group.action or "").lower()
    target = group.target

    # URI actions come from the dataset, never from the user/LLM.
    if group.uri and action in {"open_uri", "launch_process_or_uri"}:
        result = open_uri(group.uri)
        return ExecutionResult(result.ok, result.message, result.data)

    if action == "launch_process_or_uri" and group.executable:
        result = launch_process(group.executable, group.fixed_args)
        return ExecutionResult(result.ok, result.message, result.data)

    # Fixed executable actions from the dataset. The executable is parsed from
    # the stored command only; no user-generated shell is accepted.
    if action in {"open_app", "open_mmc", "open_folder", "open_shell_folder"}:
        executable = group.executable
        if not executable:
            return ExecutionResult(False, "That application target is not configured safely.")
        result = launch_process(executable, group.fixed_args)
        return ExecutionResult(result.ok, result.message, result.data)

    # close_app_dynamic: the ONLY close-application action now -- there is
    # no per-app dataset entry to look up an executable from. parameters
    # carries whatever main.py already resolved against real running
    # processes (tools.find_running_app) before the user was asked to
    # confirm, so the thing being closed is guaranteed to be the same thing
    # that was actually named in the confirmation prompt. Falls back to
    # resolving from the raw hint itself only if parameters somehow didn't
    # carry a pre-resolved name (e.g. a direct/test call into this
    # function bypassing main.py's flow).
    if action == "close_app_dynamic":
        resolved = (parameters or {}).get("resolved_process")
        if resolved:
            result = close_process(resolved)
        else:
            hint = (parameters or {}).get("app_name_hint") or ""
            result = close_running_app(hint)
        return ExecutionResult(result.ok, result.message, result.data)

    # Dataset diagnostic actions are mapped to a fixed diagnostic identifier.
    if action == "run_command" and target in config.SAFE_DIAGNOSTIC_TARGETS:
        result = diagnostic(config.SAFE_DIAGNOSTIC_TARGETS[target])
        return ExecutionResult(result.ok, result.message, result.data)

    return ExecutionResult(False, "That action is not enabled in the execution policy.")


def run(group, parameters: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    if not sys.platform.startswith("win"):
        return ExecutionResult(False, "This assistant action requires Windows.")
    return _run_allowlisted(group, parameters)
