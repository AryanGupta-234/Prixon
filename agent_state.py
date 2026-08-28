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
    # Keep the executable target and the concrete conversational entity
    # separate. "list_processes" is a capability; "Spotify" is an entity.
    last_referenced_app: Optional[str] = None
    last_referenced_app_hint: Optional[str] = None
    last_entity_name: Optional[str] = None
    last_entity_process: Optional[str] = None
    last_operation: Optional[str] = None
    reference_turn: int = -1
    turn_id: int = 0

    open_apps: List[str] = field(default_factory=list)
    computer_state: Dict[str, Any] = field(default_factory=dict)
    learned_context: Dict[str, Any] = field(default_factory=dict)
    execution_state: str = "idle"

    def begin_turn(self):
        """Advance conversational time and expire stale entity references."""
        self.turn_id += 1
        if self.last_entity_name and self.reference_turn >= 0 and self.turn_id - self.reference_turn > 4:
            self._clear_reference()

    def _clear_reference(self):
        self.last_referenced_app = None
        self.last_referenced_app_hint = None
        self.last_entity_name = None
        self.last_entity_process = None
        self.reference_turn = -1

    def note_successful_task(self, target: str, target_name: str, intent: str,
                             resolved_name: Optional[str] = None):
        concrete = resolved_name or target_name
        self.last_target = target
        self.last_target_name = concrete
        self.last_intent = intent
        self.last_operation = intent
        intent_l = (intent or "").lower()
        target_l = (target or "").lower()
        is_close = intent_l in {"close_app", "close_application", "quit_app", "exit_app"} or "close_app" in target_l
        if is_close:
            victim = (resolved_name or target_name or "").lower()
            self.open_apps = [a for a in self.open_apps if a.lower() != victim]
            if self.last_entity_process and self.last_entity_process.lower() == victim:
                self._clear_reference()
            elif self.last_entity_name and self.last_entity_name.lower() == victim:
                self._clear_reference()
        elif concrete:
            if concrete not in self.open_apps:
                self.open_apps.append(concrete)
            # Do not overwrite a concrete conversational entity with a tool
            # name. The caller can explicitly set it with note_referenced_app.
            if not self.last_entity_name:
                self.last_entity_name = concrete
            self.last_referenced_app = self.last_entity_process or self.last_entity_name
            self.last_referenced_app_hint = self.last_entity_name
            self.reference_turn = self.turn_id

    def note_referenced_app(self, process_name: str, hint: str = "", operation: str = "application_status"):
        """Record both the human-facing entity and its live process identity."""
        entity = (hint or process_name or "").strip()
        process = (process_name or "").strip() or None
        self.last_entity_name = entity or None
        self.last_entity_process = process
        self.last_referenced_app = process or entity or None
        self.last_referenced_app_hint = entity or process
        self.last_operation = operation
        self.reference_turn = self.turn_id
        if entity and entity not in self.open_apps:
            self.open_apps.append(entity)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_goal": self.active_goal,
            "last_target": self.last_target,
            "last_target_name": self.last_target_name,
            "last_intent": self.last_intent,
            "last_referenced_app": self.last_referenced_app,
            "last_referenced_app_hint": self.last_referenced_app_hint,
            "last_entity_name": self.last_entity_name,
            "last_entity_process": self.last_entity_process,
            "last_operation": self.last_operation,
            "reference_age_turns": self.turn_id - self.reference_turn if self.reference_turn >= 0 else None,
            "open_apps": list(self.open_apps),
            "computer_state": self.computer_state,
            "learned_context": self.learned_context,
            "execution_state": self.execution_state,
        }
