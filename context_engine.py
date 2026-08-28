"""Context engine: references -> live environment -> semantic shortlist -> Qwen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import concepts
import config
import goal_engine
import reference_resolver
import tier2
import tools
from agent_state import AgentState
from data_store import ActionGroup
from memory import UnifiedMemory
from nlu import NLUResult
from nlu import resolve as llm_resolve
from system import system_agent


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


def _snapshot_context(snapshot) -> Dict[str, Any]:
    """Return only compact, model-facing live state."""
    if snapshot is None:
        return {"available": False}
    out: Dict[str, Any] = {
        "available": True,
        "cpu_percent": snapshot.cpu.usage_percent,
        "memory_percent": snapshot.memory.percent,
        "disk_free_gb": snapshot.disk.free_gb,
        "network_online": snapshot.network.online,
    }
    if snapshot.battery.present:
        out["battery_percent"] = snapshot.battery.percent
        out["battery_charging"] = snapshot.battery.charging
    if snapshot.gpu.available:
        out["gpu_percent"] = snapshot.gpu.utilization_percent
    if snapshot.processes.top_by_cpu:
        top = snapshot.processes.top_by_cpu[0]
        out["top_cpu_process"] = {"name": top.name, "cpu_percent": top.cpu_percent}
    return out


def _compact_agent_context(state: AgentState, memory: UnifiedMemory, patterns=None, user_text: str = "") -> Dict[str, Any]:
    snapshot = system_agent.latest_snapshot()
    state.computer_state = _snapshot_context(snapshot)
    recent_episodes = memory.episodes[-1:]
    learned = patterns.context() if patterns is not None else {}
    out: Dict[str, Any] = {"world_state": state.computer_state}
    if state.active_goal:
        out["active_goal"] = state.active_goal
    if state.last_entity_name:
        out["active_entity"] = state.last_entity_name
        out["entity_process"] = state.last_entity_process
        out["reference_age_turns"] = state.snapshot().get("reference_age_turns")
    if state.last_operation:
        out["active_operation"] = state.last_operation
    if state.open_apps:
        out["tracked_apps"] = state.open_apps[-6:]
    if recent_episodes:
        ep = recent_episodes[0]
        out["last_event"] = {"event": ep.event_type, "target": ep.target_name or ep.target, "success": ep.success}
    reliable = (state.learned_context or {}).get("reliable_actions") or []
    if reliable:
        out["reliable_actions"] = reliable[:3]
    habits = (learned or {}).get("habits") or []
    if habits:
        out["learned_habits"] = habits[:3]
    concept_hint = concepts.relevant_hint(user_text)
    if concept_hint:
        out["concept_hint"] = concept_hint
    return out


def _live_app_status(user_text: str, state: AgentState) -> Optional[RoutedResult]:
    """Answer named application status from live Windows process state."""
    hint = tier2.extract_running_app_hint(user_text)
    if not hint:
        return None
    process = tools.find_running_app(hint)
    state.note_referenced_app(process or hint, hint, operation="application_status")
    params = {"app_name_hint": hint, "resolved_process": process, "running": bool(process)}
    reply = f"Yes — {hint} is running." if process else f"No — I don't see {hint} running right now."
    return RoutedResult(
        NLUResult(match_target=None, confidence="high", reply=reply, intent="application_status",
                  parameters=params, reference="explicit", raw={"direct_live_app_status": True, "process": process}),
        "environment", {"reason": "direct live app-status resolution", "app": hint, "process": process},
    )


def route(user_text: str, candidates: List[Dict], state: AgentState, memory: UnifiedMemory,
          groups: Dict[str, ActionGroup], assistant_name: Optional[str] = None,
          broad_search: bool = False, semantic_index=None, patterns=None) -> RoutedResult:
    state.begin_turn()

    # First use live environment perception when the user explicitly names an
    # entity and asks for its current runtime state. This is dynamic: the
    # entity is discovered from the user's language and current processes.
    live = _live_app_status(user_text, state)
    if live is not None:
        return live

    candidates = goal_engine.bias_candidates(candidates, state.active_goal, groups)

    # Explicit conversational references beat semantic similarity, but only
    # resolve the linguistic reference. The capability itself remains a
    # normal catalog/semantic decision. This prevents reference handling from
    # becoming a hidden application command table.
    tier1 = reference_resolver.resolve(user_text, state)
    tier1_augmented_query = user_text
    if tier1.resolved and tier1.target:
        params = {}
        if tier1.target_name:
            params["entity_name"] = tier1.target_name
            params["app_name_hint"] = tier1.target_name
        return RoutedResult(
            NLUResult(match_target=tier1.target, confidence="high", reply="On it.", intent=tier1.intent or "unknown",
                       parameters=params, reference=tier1.reference, raw={"tier1_reason": tier1.reason}),
            "tier1", {"reference": tier1.reference, "confidence": tier1.confidence, "reason": tier1.reason})
    elif tier1.resolved and tier1.target_name:
        # The reference is known, but the requested operation is new. Feed the
        # concrete entity into semantic retrieval without inventing a target.
        tier1_augmented_query = f"{user_text} {tier1.target_name}"

    semantic_ready = bool(semantic_index is not None and semantic_index.ready)
    top_k = max(config.TOP_K_CANDIDATES, 1)
    expanded_query = concepts.expand_query(tier1_augmented_query)
    semantic_candidates = semantic_index.search(expanded_query, top_k=top_k) if semantic_ready else []
    if semantic_candidates:
        candidates = []
        for item in semantic_candidates[:top_k]:
            group = groups.get(item.get("target"))
            if group:
                candidates.append(group.to_candidate(item.get("score", 0.0)))
    elif config.LEGACY_LEXICAL_FALLBACK:
        candidates = candidates
    else:
        candidates = []

    if len(candidates) > top_k:
        candidates = candidates[:top_k]

    if not semantic_ready and semantic_index is not None and semantic_index.loading and not candidates:
        return RoutedResult(
            NLUResult(match_target=None, confidence="none",
                      reply="Still finishing startup — give me a few seconds and try again.",
                      intent="startup_warmup", parameters={}, reference="none",
                      raw={"semantic_loading": True}),
            "warmup", {"reason": "semantic index still loading, no candidates available"},
        )

    memory.conversation.slots["live_agent_context"] = _compact_agent_context(state, memory, patterns, tier1_augmented_query)
    result = llm_resolve(tier1_augmented_query, candidates, assistant_name, broad_search, memory.conversation)
    if tier1.resolved and tier1.target_name:
        result.parameters = {**result.parameters, "app_name_hint": tier1.target_name, "entity_name": tier1.target_name}
        if not result.reference or result.reference == "none":
            result.reference = tier1.reference
    return RoutedResult(result, "tier3-qwen-semantic", {
        "semantic_candidates": semantic_candidates[:5],
        "semantic_ready": semantic_ready,
        "reference_resolution": {
            "resolved": tier1.resolved,
            "entity": tier1.target_name,
            "reason": tier1.reason,
        },
        "raw": result.raw,
    })
