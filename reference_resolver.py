"""Tier 1 deterministic conversational reference resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_state import AgentState

_AGAIN_PATTERN = re.compile(r"\b(again|repeat that|one more time|same thing|do that)\b")
_OPEN_VERBS = ("open", "launch", "start", "bring up", "pull up", "show")
_BARE_REFERENTS = {"it", "that", "this", "that one", "this one"}
_CLOSE_VERBS = ("close", "quit", "exit", "stop", "shut")


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

    if _AGAIN_PATTERN.search(text) and len(text.split()) <= 6:
        if state.last_target:
            return Tier1Result(
                True, target=state.last_target, target_name=state.last_target_name,
                intent=state.last_intent, reference="recent_action", confidence=0.9,
                reason="repeated the last successful task",
            )
        return Tier1Result(False, reason="no previous task to repeat")

    # Follow-up commands such as "close it" should resolve to the most
    # recently referenced concrete app, not necessarily the last executable
    # task. Example: "is Spotify running?" executes list_processes, but
    # "close it" clearly refers to Spotify.
    words = text.split()
    if len(words) <= 4 and words and words[0] in _CLOSE_VERBS:
        remainder = text[len(words[0]):].strip()
        if remainder in _BARE_REFERENTS and state.last_referenced_app:
            return Tier1Result(
                True,
                target=None,
                target_name=state.last_referenced_app,
                intent="close_app",
                reference="recent_app_reference",
                confidence=0.96,
                reason="closed the most recently referenced running application",
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
                if state.last_target:
                    return Tier1Result(
                        True, target=state.last_target, target_name=state.last_target_name,
                        intent=state.last_intent, reference="recent_target", confidence=0.85,
                        reason="opened the last referenced target",
                    )
                return Tier1Result(False, reason="no previous target for 'it'")

    return Tier1Result(False)
