"""Compact situation model: the agent's current view of its environment."""
from __future__ import annotations
from typing import Any, Dict

def build(state, memory, patterns=None) -> Dict[str, Any]:
    """Build a small, model-facing situation snapshot without raw telemetry dumps."""
    ctx = {
        "active_goal": state.active_goal,
        "last_target": state.last_target_name,
        "last_intent": state.last_intent,
        "referenced_app": state.last_referenced_app,
        "referenced_app_hint": state.last_referenced_app_hint,
        "reference_age_turns": state.snapshot().get("reference_age_turns"),
        "tracked_apps": list(state.open_apps[-12:]),
        "computer": state.computer_state or {},
        "recent_events": [
            {"event": e.event_type, "target": e.target_name or e.target, "intent": e.intent, "success": e.success}
            for e in memory.episodes[-8:]
        ],
    }
    if patterns is not None:
        ctx["learned"] = patterns.context()
    return ctx
