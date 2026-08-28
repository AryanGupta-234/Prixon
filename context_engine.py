"""Context engine: references -> live environment -> semantic -> grounded LLM reasoning."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    if snapshot is None:
        return {"available": False}
    return {
        "available": True,
        "age_seconds": round(max(0.0, time.time() - snapshot.timestamp), 1),
        "cpu": {"usage_percent": snapshot.cpu.usage_percent, "cores": snapshot.cpu.core_count, "frequency_mhz": snapshot.cpu.frequency_mhz},
        "memory": {"percent": snapshot.memory.percent, "available_mb": snapshot.memory.available_mb, "used_mb": snapshot.memory.used_mb, "total_mb": snapshot.memory.total_mb},
        "disk": {"free_gb": snapshot.disk.free_gb, "used_percent": snapshot.disk.used_percent},
        "network": {"online": snapshot.network.online},
        "battery": {"present": snapshot.battery.present, "percent": snapshot.battery.percent, "charging": snapshot.battery.charging},
        "gpu": {"available": snapshot.gpu.available, "utilization_percent": snapshot.gpu.utilization_percent, "vram_used_mb": snapshot.gpu.vram_used_mb, "vram_total_mb": snapshot.gpu.vram_total_mb},
        "top_cpu": [{"name": p.name, "pid": p.pid, "cpu_percent": p.cpu_percent} for p in snapshot.processes.top_by_cpu[:5]],
        "top_memory": [{"name": p.name, "pid": p.pid, "memory_mb": p.memory_mb} for p in snapshot.processes.top_by_memory[:5]],
    }


def _compact_agent_context(state: AgentState, memory: UnifiedMemory, patterns=None) -> Dict[str, Any]:
    snapshot = system_agent.latest_snapshot()
    state.computer_state = _snapshot_context(snapshot)
    recent = [{"event": ep.event_type, "target": ep.target_name or ep.target, "intent": ep.intent, "success": ep.success} for ep in memory.episodes[-8:]]
    learned = patterns.context() if patterns is not None else {}
    return {
        "active_goal": state.active_goal,
        "last_successful_target": state.last_target_name,
        "last_successful_intent": state.last_intent,
        "last_referenced_app": state.last_referenced_app,
        "last_referenced_app_hint": state.last_referenced_app_hint,
        "reference_age_turns": state.snapshot().get("reference_age_turns"),
        "tracked_apps": state.open_apps[-12:],
        "world_state": state.computer_state,
        "learned_experience": state.learned_context,
        "learned_patterns": learned,
        "recent_events": recent,
    }


def _live_app_status(user_text: str, state: AgentState) -> Optional[RoutedResult]:
    """Answer named application status from live Windows process state."""
    hint = tier2.extract_running_app_hint(user_text)
    if not hint:
        return None
    process = tools.find_running_app(hint)
    state.note_referenced_app(process or hint, hint)
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

    # Live environment queries are resolved before any action retrieval.
    live = _live_app_status(user_text, state)
    if live is not None:
        return live

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
                NLUResult(match_target=target, confidence="high", reply="On it.", intent=tier1.intent or "unknown",
                           parameters={"app_name_hint": tier1.target_name} if tier1.action_hint == "close_app_dynamic" else {},
                           reference=tier1.reference, raw={"tier1_reason": tier1.reason}),
                "tier1", {"reference": tier1.reference, "confidence": tier1.confidence, "reason": tier1.reason})

    # Semantic embeddings are now the primary retrieval mechanism. TF-IDF is
    # deliberately not consulted in normal operation.
    semantic_candidates = semantic_index.search(user_text, top_k=max(config.TOP_K_CANDIDATES, 12)) if semantic_index is not None and semantic_index.ready else []
    if semantic_candidates:
        t2_semantic = tier2.classify_semantic(user_text, semantic_candidates, groups)
        if t2_semantic.resolved:
            result = NLUResult(match_target=t2_semantic.target, confidence="high", reply="On it.", intent=t2_semantic.intent or "unknown", parameters=t2_semantic.parameters or {}, reference="none", raw={"semantic_reason": t2_semantic.reason})
            return RoutedResult(result, "tier2-semantic", {"semantic": semantic_candidates[:5], "confidence": t2_semantic.confidence, "reason": t2_semantic.reason})
        candidates = []
        for item in semantic_candidates[:max(config.TOP_K_CANDIDATES, 12)]:
            group = groups.get(item.get("target"))
            if group:
                candidates.append(group.to_candidate(item.get("score", 0.0)))
    elif config.LEGACY_LEXICAL_FALLBACK:
        t2 = tier2.classify(user_text, candidates)
        if t2.resolved:
            result = NLUResult(match_target=t2.target, confidence="high", reply="On it.", intent=t2.intent or "unknown", parameters=t2.parameters or {}, reference="none", raw={"legacy_lexical_reason": t2.reason})
            return RoutedResult(result, "tier2-legacy-lexical", {"reason": t2.reason})

    # Qwen is the semantic arbitrator when deterministic/reference/world
    # reasoning cannot confidently resolve the request.
    memory.conversation.slots["live_agent_context"] = _compact_agent_context(state, memory, patterns)
    result = llm_resolve(user_text, candidates, assistant_name, broad_search, memory.conversation)
    return RoutedResult(result, "tier3-semantic-brain", {"semantic_candidates": semantic_candidates[:5], "raw": result.raw})
