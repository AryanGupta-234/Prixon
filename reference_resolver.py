"""Tier 1 deterministic conversational reference resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_state import AgentState

_AGAIN_PATTERN = re.compile(r"\b(again|repeat that|one more time|same thing|do that|check again)\b")
_OPEN_VERBS = ("open", "launch", "start", "bring up", "pull up", "show")
_BARE_REFERENTS = {"it", "that", "this", "that one", "this one"}
_CLOSE_VERBS = ("close", "quit", "exit", "stop", "shut")
_STATUS_FOLLOWUP = re.compile(r"^(?:what(?:'s| is)\s+)?(?:it|that)\s+(?:still\s+)?(?:running|open|active)\??$", re.I)


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


def resolve(user_text: str, state: AgentState) -> Tier1Result:
    text = _normalize(user_text)
    if not text:
        return Tier1Result(False)

    # "check again" means repeat the previous OPERATION against the previous
    # CONCRETE ENTITY. Never use last_target alone: for an app-status query
    # last_target is often the generic list_processes capability, while the
    # entity is Spotify/VS Code/etc.
    if _AGAIN_PATTERN.search(text) and len(text.split()) <= 6:
        if state.last_entity_name and state.last_operation:
            if state.last_operation == "application_status":
                return Tier1Result(
                    True,
                    target=None,
                    target_name=state.last_entity_name,
                    intent="application_status",
                    reference="recent_action",
                    confidence=0.97,
                    reason="repeated the previous application-status operation for the same entity",
                    action_hint="list_processes",
                )
            if state.last_target:
                return Tier1Result(
                    True, target=state.last_target, target_name=state.last_entity_name or state.last_target_name,
                    intent=state.last_operation, reference="recent_action", confidence=0.9,
                    reason="repeated the previous operation for the current entity",
                )
        elif state.last_target:
            return Tier1Result(
                True, target=state.last_target, target_name=state.last_target_name,
                intent=state.last_intent, reference="recent_action", confidence=0.9,
                reason="repeated the last successful task",
            )
        return Tier1Result(False, reason="no previous task to repeat")

    # Natural follow-up status questions such as "what is it running?" or
    # "is it still running?" retain the active entity and query live state.
    if _STATUS_FOLLOWUP.match(text) and state.last_entity_name:
        return Tier1Result(
            True, target=None, target_name=state.last_entity_name,
            intent="application_status", reference="recent_entity", confidence=0.95,
            reason="resolved the pronoun to the current application entity",
            action_hint="list_processes",
        )

    # Follow-up commands such as "close it" resolve to the concrete app, not
    # the capability that happened to inspect it.
    words = text.split()
    if len(words) <= 4 and words and words[0] in _CLOSE_VERBS:
        remainder = text[len(words[0]):].strip()
        if remainder in _BARE_REFERENTS and state.last_entity_name:
            return Tier1Result(
                True,
                target=None,
                target_name=state.last_entity_name,
                intent="close_app",
                reference="recent_app_reference",
                confidence=0.98,
                reason="resolved the pronoun to the most recently referenced concrete application",
                action_hint="close_app_dynamic",
            )

    for verb in _OPEN_VERBS:
        prefix = f"{verb} "
        if text.startswith(prefix):
            remainder = text[len(prefix):].strip()
            if remainder in _BARE_REFERENTS or (
                remainder.endswith(" again")
                and remainder[: -len(" again")].strip() in _BARE_REFERENTS
            ):
                if state.last_entity_name:
                    return Tier1Result(
                        True, target=None, target_name=state.last_entity_name,
                        intent="open_app", reference="recent_entity", confidence=0.9,
                        reason="resolved the pronoun to the current application entity",
                        action_hint="open_app_dynamic",
                    )
                if state.last_target:
                    return Tier1Result(
                        True, target=state.last_target, target_name=state.last_target_name,
                        intent=state.last_intent, reference="recent_target", confidence=0.85,
                        reason="opened the last referenced target",
                    )
                return Tier1Result(False, reason="no previous target for 'it'")

    return Tier1Result(False)
