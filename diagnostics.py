"""Self-diagnostics (spec section 49).

This answers spec section 48's "system self-awareness" questions from
actual runtime state -- LLM provider status, local model status, memory
system, embeddings, tool router, system agent, voice, network, resource
availability -- rather than anything hallucinated (non-negotiable #21).

Deliberately a standalone module callable both as `python main.py
--healthcheck` and importable for a future "are you OK?" conversational
Tier 3 answer (spec section 48) without duplicating the checks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import config


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "degraded" | "unavailable"
    detail: str


def _check_llm_providers(router) -> List[CheckResult]:
    results = []
    for name, provider in router._providers.items():  # noqa: SLF001 -- diagnostics module, same trust boundary as router itself
        if name in router._exhausted:
            results.append(CheckResult(f"LLM: {provider.name}", "degraded", "quota exhausted this session"))
        elif provider.available():
            results.append(CheckResult(f"LLM: {provider.name}", "ok", "available"))
        else:
            reason = "no credentials configured" if name != "local" else "Ollama server not reachable / OLLAMA_MODEL not set"
            results.append(CheckResult(f"LLM: {provider.name}", "unavailable", reason))
    return results


def _check_memory(memory) -> CheckResult:
    try:
        count = len(memory.episodes)
        return CheckResult("Memory", "ok", f"{count} episodes loaded from {memory.episodic_path}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Memory", "unavailable", str(exc))


def _check_embeddings(semantic_index) -> CheckResult:
    if semantic_index is None:
        return CheckResult("Embeddings", "unavailable", "not initialized")
    if semantic_index.ready:
        return CheckResult("Embeddings", "ok", "ready")
    return CheckResult("Embeddings", "degraded", "still loading in background")


def _check_tool_router(registry) -> CheckResult:
    try:
        summary = registry.summary()
        return CheckResult("Tool router", "ok", str(summary))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Tool router", "unavailable", str(exc))


def _check_system_agent() -> CheckResult:
    from system import system_agent
    snap = system_agent.latest_snapshot()
    if snap is None:
        return CheckResult("System agent", "degraded", "not running or no poll completed yet")
    age = time.time() - snap.timestamp
    return CheckResult("System agent", "ok", f"last snapshot {age:.0f}s ago")


def _check_resource_pressure() -> CheckResult:
    from system import resource_policy, system_agent
    from system import collector as system_collector
    snap = system_agent.latest_snapshot() or system_collector.snapshot(probe_internet=True)
    advice = resource_policy.advise(snap)
    mem = snap.memory
    cpu = snap.cpu
    detail = f"RAM available: {mem.available_mb:.0f}MB, CPU: {cpu.usage_percent:.0f}%" if mem.available_mb is not None and cpu.usage_percent is not None else "readings unavailable"
    if advice.avoid_local:
        return CheckResult("Resource pressure", "degraded", f"elevated -- {advice.reason}")
    return CheckResult("Resource pressure", "ok", detail)


def _check_network() -> CheckResult:
    from system import system_agent
    from system import collector as system_collector
    snap = system_agent.latest_snapshot() or system_collector.snapshot(probe_internet=True)
    if snap.network.online is True:
        return CheckResult("Network", "ok", "online")
    if snap.network.online is False:
        return CheckResult("Network", "degraded", "appears offline")
    return CheckResult("Network", "degraded", "reachability unknown")


def _check_voice() -> CheckResult:
    import voice
    if voice.voice_available():
        return CheckResult("Voice", "ok", "ready")
    return CheckResult("Voice", "unavailable", "dependencies not installed")


def run(router, memory, semantic_index, registry) -> List[CheckResult]:
    """Runs every check and returns the results in a fixed, readable order.
    Each check is isolated -- one failing check reports itself as
    unavailable rather than crashing the whole health report (same
    philosophy as system/collector.py's per-monitor isolation)."""
    checks: List[CheckResult] = []
    for fn, args in (
        (_check_llm_providers, (router,)),
        (_check_memory, (memory,)),
        (_check_embeddings, (semantic_index,)),
        (_check_tool_router, (registry,)),
        (_check_system_agent, ()),
        (_check_resource_pressure, ()),
        (_check_network, ()),
        (_check_voice, ()),
    ):
        try:
            result = fn(*args)
            checks.extend(result if isinstance(result, list) else [result])
        except Exception as exc:  # noqa: BLE001
            checks.append(CheckResult(fn.__name__.lstrip("_"), "unavailable", f"check itself failed: {exc}"))
    return checks


def format_report(checks: List[CheckResult]) -> str:
    lines = [f"{config.ASSISTANT_NAME.upper()} HEALTH", ""]
    icon = {"ok": "✓", "degraded": "~", "unavailable": "✗"}
    for c in checks:
        lines.append(f"  {icon.get(c.status, '?')} {c.name}: {c.detail}")
    return "\n".join(lines)
