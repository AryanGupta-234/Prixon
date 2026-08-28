"""Tier 1 deterministic conversational reference resolution.

This layer is intentionally entity/capability agnostic. It never contains a
list of applications or assumes that a particular product exists. It only
uses the current AgentState to resolve linguistic references such as `it`,
`that`, and `again` to the most recent concrete entity/operation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_state import AgentState

_AGAIN_PATTERN = re.compile(r"\b(?:again|repeat(?:\s+that)?|one\s+more\s+time|same\s+thing|do\s+that|check\s+again)\b", re.I)
_BARE_REFERENTS = {"it", "that", "this", "that one", "this one"}
_STATUS_FOLLOWUP = re.compile(r"^(?:what(?:'s|\s+is)\s+)?(?:it|that|this)\s+(?:still\s+)?(?:running|open|active)\??$", re.I)
_ACTION_VERBS = {
    "close", "quit", "exit", "stop", "shut", "open", "launch", "start",
    "show", "check", "inspect", "find", "restart", "enable", "disable",
    "delete", "remove", "pause", "resume", "terminate", "kill",
}


@dataclass
class Tier1Result:
    resolved: bool
    target: Optional[str] = None
    target_name: Optional[str] = None
    intent: Optional[str] = None
    reference: str = "none"
    confidence: float = 0.0
    reason: str = ""
    action_hint: Optional[str] = None


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _referent(state: AgentState) -> Optional[str]:
    """Return the concrete conversational entity, if one is currently focused."""
    return (state.last_entity_name or state.last_referenced_app_hint or state.last_target_name or "").strip() or None


def resolve(user_text: str, state: AgentState) -> Tier1Result:
    text = _normalize(user_text)
    if not text:
        return Tier1Result(False)

    entity = _referent(state)

    # Repeat the previous semantic operation against the previous concrete
    # entity. This is deliberately generic: the operation may be an app
    # status check, file inspection, network query, display operation, etc.
    if _AGAIN_PATTERN.search(text) and len(text.split()) <= 8:
        if entity and state.last_operation:
            return Tier1Result(
                True, target=state.last_target, target_name=entity,
                intent=state.last_operation, reference="recent_action", confidence=0.97,
                reason="repeated the previous operation for the current concrete entity",
            )
        if state.last_target:
            return Tier1Result(
                True, target=state.last_target, target_name=state.last_target_name,
                intent=state.last_intent or state.last_operation, reference="recent_action", confidence=0.9,
                reason="repeated the last successful task",
            )
        return Tier1Result(False, reason="no previous task to repeat")

    # Pronoun-only status follow-ups inherit the active entity and operation.
    if _STATUS_FOLLOWUP.match(text) and entity:
        return Tier1Result(
            True, target=state.last_target, target_name=entity,
            intent=state.last_operation or state.last_intent or "status_check",
            reference="recent_entity", confidence=0.95,
            reason="resolved the pronoun to the current concrete entity",
        )

    # Generic imperative/reference resolution. The verb is retained only as a
    # semantic hint; the capability is still selected by the normal layer.
    words = text.split()
    if entity and len(words) <= 6 and words:
        verb = words[0]
        remainder = text[len(verb):].strip()
        if verb in _ACTION_VERBS and remainder in _BARE_REFERENTS:
            return Tier1Result(
                True, target=state.last_target, target_name=entity,
                intent=state.last_operation or verb, reference="recent_entity", confidence=0.94,
                reason="resolved a generic action against the active concrete entity",
            )

    return Tier1Result(False)
