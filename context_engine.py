"""Context engine: references -> live environment -> semantic shortlist -> Qwen."""
from __future__ import annotations

import time
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
    """Only the fields Qwen actually reasons about. Previously this always
    included the full GPU block (even when there's no GPU, i.e. most of the
    time) and 3+3 top-process lists on every single call. Zero-value/absent
    blocks are now omitted entirely rather than shipped as empty structure --
    that structure costs real prompt-eval tokens for information that isn't
    there."""
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
    # Top processes are only relevant to "what's using my X" style
    # questions, which are answered by the environment-first path (Section
    # 9) before Qwen is ever called -- so Qwen itself rarely needs this.
    # Keep exactly one entry as a lightweight situational cue rather than
    # dropping it completely, instead of the previous 3+3.
    if snapshot.processes.top_by_cpu:
        top = snapshot.processes.top_by_cpu[0]
        out["top_cpu_process"] = {"name": top.name, "cpu_percent": top.cpu_percent}
    return out


def _compact_agent_context(state: AgentState, memory: UnifiedMemory, patterns=None, user_text: str = "") -> Dict[str, Any]:
    snapshot = system_agent.latest_snapshot()
    state.computer_state = _snapshot_context(snapshot)
    # Only the single most recent episode, and only when it exists -- Qwen
    # needs "what just happened", not a rolling log; CONTEXT_TURNS already
    # covers multi-turn conversational continuity separately.
    recent_episodes = memory.episodes[-1:]
    learned = patterns.context() if patterns is not None else {}
    out: Dict[str, Any] = {"world_state": state.computer_state}
    if state.active_goal:
        out["active_goal"] = state.active_goal
    if state.last_referenced_app:
        out["last_referenced_app"] = state.last_referenced_app
        out["reference_age_turns"] = state.snapshot().get("reference_age_turns")
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

    # Explicit references are deterministic and should win over semantic similarity.
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

    # Embeddings are retrieval only. They narrow the capability space, but
    # NEVER execute a command by similarity alone. Qwen remains the semantic
    # arbiter and sees the live situation model plus this shortlist.
    #
    # NOTE: this used to read `max(config.TOP_K_CANDIDATES, 12)`, which meant
    # tuning TOP_K_CANDIDATES down to 6 (see config.py) was a silent no-op --
    # the floor of 12 always won. On real hardware this was worth ~350-400
    # extra prompt tokens (2 examples x 6 extra candidates), which measurably
    # matters on this CPU's ~46-55 tok/s prompt-eval speed. Use the tuned
    # value directly; the query-expansion in concepts.py compensates for any
    # recall lost from a smaller shortlist.
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
        # Kept only for temporary A/B regression testing. Normal mode is off.
        candidates = candidates
    else:
        candidates = []

    # Safety net: never let a caller-supplied candidate list (e.g. a future
    # full_catalog() use, or LEGACY_LEXICAL_FALLBACK) balloon the Qwen prompt.
    # This is what actually caused the 60s Ollama timeouts -- see main.py.
    if len(candidates) > top_k:
        candidates = candidates[:top_k]

    if not semantic_ready and semantic_index is not None and semantic_index.loading and not candidates:
        # Embeddings are still warming up in the background and we have no
        # other way to narrow 208 actions down. Calling Qwen with an empty
        # allow-list can't succeed anyway -- say so instead of burning a
        # multi-second CPU round trip for nothing.
        return RoutedResult(
            NLUResult(match_target=None, confidence="none",
                      reply="Still finishing startup -- give me a few seconds and try again.",
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
