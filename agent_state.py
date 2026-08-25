"""Canonical agent state -- single source of truth for what Jarvis currently
knows about the active task/goal and the computer.

Every subsystem should read/write through this object rather than keeping
its own parallel copy of "what's going on right now". Phase 1 keeps this
intentionally small (only the fields Tier 1 reference resolution and the
context envelope actually need); later phases extend it rather than
replacing it, so nothing built on top of it has to be rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    # What the user is currently trying to accomplish, if anything durable.
    active_goal: Optional[str] = None

    # Last successfully executed task -- this is what bare "it" / "that" /
    # "again" resolve to when there's no more specific antecedent.
    last_target: Optional[str] = None
    last_target_name: Optional[str] = None
    last_intent: Optional[str] = None

    # Apps this session has launched, most-recently-launched last. (There is
    # no close/kill tool in the current tool registry yet -- see README --
    # so this is tracked now so a future close_app tool has something to
    # resolve "close it" against without another state-layer change.)
    open_apps: List[str] = field(default_factory=list)

    # Populated by a future computer-state poller (section 14 of the spec).
    # Present now so ContextEngine has a stable field to read even before
    # anything writes to it.
    computer_state: Dict[str, Any] = field(default_factory=dict)

    execution_state: str = "idle"  # idle | awaiting_confirmation | executing

    def note_successful_task(self, target: str, target_name: str, intent: str):
        self.last_target = target
        self.last_target_name = target_name
        self.last_intent = intent
        if target_name and target_name not in self.open_apps:
            self.open_apps.append(target_name)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_goal": self.active_goal,
            "last_target": self.last_target,
            "last_target_name": self.last_target_name,
            "last_intent": self.last_intent,
            "open_apps": list(self.open_apps),
            "computer_state": self.computer_state,
        }
