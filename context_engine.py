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

    # Environment status is answered from live state before semantic retrieval.
    live = _live_app_status(user_text, state)
    if live is not None:
        return live

    candidates = goal_engine.bias_candidates(candidates, state.active_goal, groups)

    # Explicit conversational references must beat semantic similarity.
    tier1 = reference_resolver.resolve(user_text, state)
    if tier1.resolved:
        target = tier1.target
        if not target and tier1.action_hint:
            group = _group_for_action(groups, tier1.action_hint)
            if group:
                target = group.target
        if target:
            params = {}
            if tier1.target_name:
                params["app_name_hint"] = tier1.target_name
            return RoutedResult(
                NLUResult(match_target=target, confidence="high", reply="On it.", intent=tier1.intent or "unknown",
                           parameters=params, reference=tier1.reference, raw={"tier1_reason": tier1.reason}),
                "tier1", {"reference": tier1.reference, "confidence": tier1.confidence, "reason": tier1.reason})

    semantic_ready = bool(semantic_index is not None and semantic_index.ready)
    top_k = max(config.TOP_K_CANDIDATES, 1)
    expanded_query = concepts.expand_query(user_text)
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

    memory.conversation.slots["live_agent_context"] = _compact_agent_context(state, memory, patterns, user_text)
    result = llm_resolve(user_text, candidates, assistant_name, broad_search, memory.conversation)
    return RoutedResult(result, "tier3-qwen-semantic", {
        "semantic_candidates": semantic_candidates[:5],
        "semantic_ready": semantic_ready,
        "raw": result.raw,
    })
