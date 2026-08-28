"""Canonical agent state for active tasks, references, and computer context.

The state is intentionally small and explicit: it stores only short-lived
context needed to resolve follow-up references and current system facts.
Persistent learning remains the responsibility of memory.py and later
consolidation passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    active_goal: Optional[str] = None

    # Last successfully executed task.
    last_target: Optional[str] = None
    last_target_name: Optional[str] = None
    last_intent: Optional[str] = None

    # Most recent concrete application/entity mentioned or discovered by a
    # diagnostic. This is deliberately separate from last_target: asking
    # "is Spotify running?" executes a process-list diagnostic, but the next
    # "close it" should refer to Spotify, not to the diagnostic itself.
    last_referenced_app: Optional[str] = None
    last_referenced_app_hint: Optional[str] = None

    open_apps: List[str] = field(default_factory=list)
    computer_state: Dict[str, Any] = field(default_factory=dict)
    execution_state: str = "idle"

    def note_successful_task(self, target: str, target_name: str, intent: str):
        self.last_target = target
        self.last_target_name = target_name
        self.last_intent = intent
        if target_name and target_name not in self.open_apps:
            self.open_apps.append(target_name)

    def note_referenced_app(self, process_name: str, hint: str = ""):
        self.last_referenced_app = process_name
        self.last_referenced_app_hint = hint or process_name

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_goal": self.active_goal,
            "last_target": self.last_target,
            "last_target_name": self.last_target_name,
            "last_intent": self.last_intent,
            "last_referenced_app": self.last_referenced_app,
            "last_referenced_app_hint": self.last_referenced_app_hint,
            "open_apps": list(self.open_apps),
            "computer_state": self.computer_state,
        }
