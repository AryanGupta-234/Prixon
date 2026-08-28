"""Context engine: deterministic references -> lexical -> semantic -> LLM.

The LLM tier receives a compact live-agent context envelope in addition to
ConversationState. This keeps conversational continuity grounded in both the
recent dialogue and the assistant's current task/computer state without
making the LLM responsible for reconstructing state from raw diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import goal_engine
import reference_resolver
import tier2
from agent_state import AgentState
from data_store import ActionGroup
from memory import UnifiedMemory
from nlu import NLUResult
from nlu import resolve as llm_resolve


@dataclass
class RoutedResult:
    result: NLUResult
    tier: str
    debug: Dict[str, Any]


def _group_for_action(groups: Dict[str, ActionGroup], action: str):
    wanted = (action or "").lower()
    for group in groups.values():
        if (group.action or "").lower() == wanted:
            return group
    return None


def _compact_agent_context(state: AgentState, memory: UnifiedMemory) -> Dict[str, Any]:
    """Build a small, LLM-friendly context envelope.

    Keep this deliberately compact: the detailed episodic log remains on
    disk, while the reasoning model gets only the facts useful for the next
    conversational turn. This makes references such as "what we were doing"
    and "the app I just checked" much more reliable without flooding the
    prompt with historical events.
    """
    recent = []
    for ep in memory.episodes[-8:]:
        recent.append({
            "event": ep.event_type,
            "target": ep.target_name or ep.target,
            "intent": ep.intent,
            "success": ep.success,
        })

    return {
        "active_goal": state.active_goal,
        "last_successful_target": state.last_target_name,
        "last_successful_intent": state.last_intent,
        "last_referenced_app": state.last_referenced_app,
        "last_referenced_app_hint": state.last_referenced_app_hint,
        "open_apps": state.open_apps[-12:],
        "computer_state": state.computer_state,
        "recent_events": recent,
    }


def route(user_text: str, candidates: List[Dict], state: AgentState, memory: UnifiedMemory,
          groups: Dict[str, ActionGroup], assistant_name: Optional[str] = None,
          broad_search: bool = False, semantic_index=None) -> RoutedResult:
    candidates = goal_engine.bias_candidates(candidates, state.active_goal, groups)

    tier1 = reference_resolver.resolve(user_text, state)
    if tier1.resolved:
        target = tier1.target
        if not target and tier1.action_hint:
            group = _group_for_action(groups, tier1.action_hint)
            if group:
                target = group.target
        if target:
            return RoutedResult(
                NLUResult(
                    match_target=target, confidence="high", reply="On it.",
                    intent=tier1.intent or "unknown", parameters={
                        "app_name_hint": tier1.target_name
                    } if tier1.action_hint == "close_app_dynamic" else {},
                    reference=tier1.reference,
                    raw={"tier1_reason": tier1.reason},
                ),
                "tier1",
                {"reference": tier1.reference, "confidence": tier1.confidence,
                 "reason": tier1.reason},
            )

    t2 = tier2.classify(user_text, candidates)
    if t2.resolved:
        result = NLUResult(
            match_target=t2.target, confidence="high", reply="On it.",
            intent=t2.intent or "unknown", parameters=t2.parameters or {}, reference="none",
            raw={"tier2_reason": t2.reason},
        )
        return RoutedResult(result, "tier2-lexical", {
            "tier1_reason": tier1.reason, "confidence": t2.confidence, "reason": t2.reason,
        })

    t2_semantic = tier2.Tier2Result(False, reason="semantic index not available")
    if semantic_index is not None and semantic_index.ready:
        semantic_candidates = semantic_index.search(user_text)
        t2_semantic = tier2.classify_semantic(user_text, semantic_candidates, groups)
        if t2_semantic.resolved:
            result = NLUResult(
                match_target=t2_semantic.target, confidence="high", reply="On it.",
                intent=t2_semantic.intent or "unknown", parameters=t2_semantic.parameters or {},
                reference="none", raw={"tier2_semantic_reason": t2_semantic.reason},
            )
            return RoutedResult(result, "tier2-semantic", {
                "tier1_reason": tier1.reason, "tier2_lexical_reason": t2.reason,
                "confidence": t2_semantic.confidence, "reason": t2_semantic.reason,
            })

    # Give Tier 3 the current agent state and a compact episodic tail through
    # the existing UnifiedMemory/ConversationState envelope. We overwrite the
    # ephemeral slot each turn; it is not persisted as a user preference.
    memory.conversation.slots["live_agent_context"] = _compact_agent_context(state, memory)
    result = llm_resolve(user_text, candidates, assistant_name, broad_search, memory.conversation)
    return RoutedResult(result, "tier3", {
        "raw": result.raw, "tier1_reason": tier1.reason, "tier2_lexical_reason": t2.reason,
        "tier2_semantic_reason": t2_semantic.reason,
    })
