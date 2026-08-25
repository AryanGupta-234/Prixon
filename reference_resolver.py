"""Tier 1: deterministic, zero-LLM reference resolution.

Handles the class of request that doesn't need semantic reasoning at all --
"do that again", "same thing", bare "open it"/"open that" -- by reading
AgentState directly. This keeps the obvious cases near-instant and off the
LLM critical path; anything genuinely ambiguous (confidence not high, or no
antecedent) falls through to Tier 3 (nlu.resolve) unchanged.

Deliberately NOT handled here: "close it". The current tool registry
(tools.py / executor.py) has no close/kill-process capability at all, so
resolving "close it" to something we can't actually execute would just move
the failure from "the LLM didn't understand" to "Tier 1 confidently pointed
at a capability that doesn't exist" -- worse, not better. Add a real
close_app tool first, then extend this resolver.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_state import AgentState

_AGAIN_PATTERN = re.compile(r"\b(again|repeat that|one more time|same thing|do that)\b")
_OPEN_VERBS = ("open", "launch", "start", "bring up", "pull up", "show")
_BARE_REFERENTS = {"it", "that", "this", "that one", "this one"}


@dataclass
class Tier1Result:
    resolved: bool
    target: Optional[str] = None
    target_name: Optional[str] = None
    intent: Optional[str] = None
    reference: str = "none"
    confidence: float = 0.0
    reason: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def resolve(user_text: str, state: AgentState) -> Tier1Result:
    """Returns resolved=True only when there is a single, unambiguous
    deterministic answer grounded in AgentState. Anything else returns
    resolved=False so the caller falls through to Tier 2/3."""
    text = _normalize(user_text)
    if not text:
        return Tier1Result(False)

    # "do that again" / "same thing" / "repeat that" / bare "again"
    # -> re-run whatever the last *successful* task was.
    if _AGAIN_PATTERN.search(text) and len(text.split()) <= 6:
        if state.last_target:
            return Tier1Result(
                True, target=state.last_target, target_name=state.last_target_name,
                intent=state.last_intent, reference="recent_action", confidence=0.9,
                reason="repeated the last successful task",
            )
        return Tier1Result(False, reason="no previous task to repeat")

    # bare "open it" / "open that" / "launch it" -- but only when there's
    # truly nothing else in the sentence, so this never shadows a real
    # request like "open the wifi settings".
    words = text.split()
    for verb in _OPEN_VERBS:
        prefix = f"{verb} "
        if text.startswith(prefix):
            remainder = text[len(prefix):].strip()
            if remainder in _BARE_REFERENTS or (remainder.endswith(" again") and remainder[: -len(" again")].strip() in _BARE_REFERENTS):
                if state.last_target:
                    return Tier1Result(
                        True, target=state.last_target, target_name=state.last_target_name,
                        intent=state.last_intent, reference="recent_target", confidence=0.85,
                        reason="opened the last referenced target",
                    )
                return Tier1Result(False, reason="no previous target for 'it'")

    return Tier1Result(False)
