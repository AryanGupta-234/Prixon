"""Canonical agent state for active tasks, references, world context and learned experience."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    active_goal: Optional[str] = None
    last_target: Optional[str] = None
    last_target_name: Optional[str] = None
    last_intent: Optional[str] = None
    last_referenced_app: Optional[str] = None
    last_referenced_app_hint: Optional[str] = None

    open_apps: List[str] = field(default_factory=list)
    computer_state: Dict[str, Any] = field(default_factory=dict)
    learned_context: Dict[str, Any] = field(default_factory=dict)
    execution_state: str = "idle"

    def note_successful_task(self, target: str, target_name: str, intent: str,
                             resolved_name: Optional[str] = None):
        concrete = resolved_name or target_name
        self.last_target = target
        self.last_target_name = concrete
        self.last_intent = intent

        intent_l = (intent or "").lower()
        target_l = (target or "").lower()
        is_close = intent_l in {"close_app", "close_application", "quit_app", "exit_app"} or "close_app" in target_l

        if is_close:
            victim = (resolved_name or target_name or "").lower()
            self.open_apps = [a for a in self.open_apps if a.lower() != victim]
            if self.last_referenced_app and self.last_referenced_app.lower() == victim:
                self.last_referenced_app = None
                self.last_referenced_app_hint = None
        elif concrete:
            if concrete not in self.open_apps:
                self.open_apps.append(concrete)
            self.last_referenced_app = concrete
            self.last_referenced_app_hint = concrete

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
            "learned_context": self.learned_context,
            "execution_state": self.execution_state,
        }
