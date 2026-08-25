"""Tool router + capability registry (spec sections 15-16).

All execution should pass through ToolRouter.dispatch() rather than callers
invoking executor.run() directly. executor.py keeps doing exactly what it
did before -- the allow-list / policy enforcement doesn't move here, and
this file has no execution logic of its own -- this just adds one funnel
point that attaches real verification and exposes a registry the LLM/planner
can reason over later without changing what's actually permitted to run.

CapabilityRegistry is derived from the same dataset data_store.py already
indexes, not a second hand-maintained list that can silently drift out of
sync with what's actually executable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config
import executor
import verification
from data_store import ActionGroup

# read_only / low_risk / state_changing / destructive / sensitive, per spec
# section 28. Nothing in the current 207-action catalog is destructive or
# sensitive -- this mapping exists so a future action type has a taxonomy
# to slot into instead of inventing one at the point of use.
_ACTION_CATEGORY = {
    "run_command": "read_only",
    "open_app": "low_risk",
    "open_uri": "low_risk",
    "launch_process_or_uri": "low_risk",
    "open_mmc": "low_risk",
    "open_folder": "low_risk",
    "open_shell_folder": "low_risk",
}


@dataclass
class Capability:
    target: str
    target_name: str
    intent: str
    action: str
    category: str
    risk: str
    verifiable: bool


class CapabilityRegistry:
    """What Jarvis can actually do -- built from data_store.ActionIndex.groups
    so it can never list a capability that isn't backed by a real, executable
    dataset entry."""

    def __init__(self, index):
        self._capabilities: Dict[str, Capability] = {}
        for target, group in index.groups.items():
            action = (group.action or "").lower()
            self._capabilities[target] = Capability(
                target=target, target_name=group.target_name, intent=group.intent,
                action=group.action, category=_ACTION_CATEGORY.get(action, "low_risk"),
                risk=group.risk, verifiable=(action == "run_command" or bool(group.executable)),
            )

    def get(self, target: str) -> Optional[Capability]:
        return self._capabilities.get(target)

    def has(self, target: str) -> bool:
        return target in self._capabilities

    def by_category(self, category: str) -> List[Capability]:
        return [c for c in self._capabilities.values() if c.category == category]

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self._capabilities.values():
            out[c.category] = out.get(c.category, 0) + 1
        return out


@dataclass
class DispatchResult:
    ok: bool
    message: str
    data: Any = None
    verification: Optional[verification.VerificationResult] = None


class ToolRouter:
    """Single funnel for execution: delegates the actual policy-enforced
    call to executor.run() unchanged, then attaches best-effort verification
    matched to exactly the action path executor.py took."""

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def dispatch(self, group: ActionGroup, parameters: Optional[Dict[str, Any]] = None) -> DispatchResult:
        exec_result = executor.run(group, parameters)
        if not exec_result.ok:
            return DispatchResult(False, exec_result.message, exec_result.data, None)

        # Mirrors executor._run_allowlisted's branching exactly, so
        # verification checks the thing that actually happened rather than
        # guessing from the action name alone.
        action = (group.action or "").lower()
        if group.uri and action in {"open_uri", "launch_process_or_uri"}:
            v = verification.verify_uri_open()
        elif action == "launch_process_or_uri" and group.executable:
            v = verification.verify_process_launch(group.executable)
        elif action in {"open_app", "open_mmc", "open_folder", "open_shell_folder"} and group.executable:
            v = verification.verify_process_launch(group.executable)
        elif action == "run_command" and group.target in config.SAFE_DIAGNOSTIC_TARGETS:
            v = verification.verify_diagnostic(exec_result.ok, exec_result.data)
        else:
            v = verification.VerificationResult(False, None, "no verification strategy for this action")

        return DispatchResult(exec_result.ok, exec_result.message, exec_result.data, v)
