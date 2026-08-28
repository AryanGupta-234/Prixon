"""Provider chain + failover with resource-aware local/cloud routing."""
from __future__ import annotations

from typing import Dict, List, Optional

import config
from brain.local_provider import OllamaProvider
from brain.cloud_provider import CerebrasProvider, GroqProvider, HuggingFaceProvider
from brain.provider_base import LLMProvider
from system import collector as system_collector
from system import resource_policy
from system import system_agent

_QUOTA_ADVICE = {
    "cerebras": "Cerebras free tier resets daily",
    "groq": "Groq free tier resets daily",
    "huggingface": "Hugging Face included credits reset monthly and may be very limited",
}


class ModelRouter:
    def __init__(self, providers: Optional[List[LLMProvider]] = None):
        self._providers: Dict[str, LLMProvider] = {
            p.name: p for p in (providers or [OllamaProvider(), CerebrasProvider(), GroqProvider(), HuggingFaceProvider()])
        }
        self._working_provider: Optional[str] = None
        self._exhausted: set = set()

    def _provider_chain(self) -> List[str]:
        preferred = config.LLM_PROVIDER if config.LLM_PROVIDER in self._providers else None
        fallback_order = [p for p in config.PROVIDER_FALLBACK_ORDER if p in self._providers]
        ordered = ([preferred] if preferred else []) + [p for p in fallback_order if p != preferred]
        available = [p for p in ordered if self._providers[p].available() and p not in self._exhausted]

        advice = None
        if "local" in available:
            try:
                snap = system_agent.latest_snapshot() or system_collector.snapshot(probe_internet=True, cpu_interval=0.0)
                advice = resource_policy.advise(snap)
            except Exception:
                advice = None

            if advice is not None and advice.avoid_local:
                # Resource pressure is a preference signal, not a reason to
                # make Prixon brain-dead. If every remote provider is already
                # exhausted/unavailable, retain local as the final fallback.
                remote_available = [p for p in available if p != "local"]
                if remote_available:
                    available.remove("local")
                else:
                    advice = resource_policy.RoutingAdvice(
                        avoid_local=False,
                        prefer_local=True,
                        reason=f"local retained as last-resort brain despite {advice.reason}",
                    )
            elif advice is not None and advice.prefer_local and available[0] != "local":
                available.remove("local")
                available.insert(0, "local")

        if config.DEBUG:
            reason = advice.reason if advice else "n/a"
            print(f"[MODEL_ROUTER] chain={available} advice={reason}", flush=True)
        return available

    def call(self, system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
        chain = self._provider_chain()
        if not chain:
            if self._exhausted:
                tried = ", ".join(sorted(self._exhausted))
                raise RuntimeError(
                    f"Every currently configured provider ({tried}) is unavailable or out of quota. "
                    "Prixon can continue offline only when the local Ollama provider is reachable."
                )
            raise RuntimeError(
                "No LLM provider is configured. Run Ollama locally or configure a cloud provider in .env."
            )

        if self._working_provider in chain:
            chain.remove(self._working_provider)
            chain.insert(0, self._working_provider)

        last_exc = None
        for name in chain:
            provider = self._providers[name]
            try:
                result = provider.chat(system_prompt, user_message, max_tokens=max_tokens, temperature=temperature)
                self._working_provider = name
                if config.DEBUG:
                    print(f"[MODEL_ROUTER] used provider={name}", flush=True)
                return result
            except Exception as exc:
                last_exc = exc
                kind = provider.classify_error(exc)
                if config.DEBUG:
                    print(f"[MODEL_ROUTER] provider={name} failed kind={kind} error={exc}", flush=True)
                if kind == "auth":
                    raise RuntimeError(f"{name} rejected its credentials (check .env).") from exc
                if kind == "quota":
                    self._exhausted.add(name)
                continue

        raise RuntimeError(f"None of the available providers ({', '.join(chain)}) could handle the request right now. Last error: {last_exc}") from last_exc


_default_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    global _default_router
    if _default_router is None:
        _default_router = ModelRouter()
    return _default_router


def call_llm(system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
    return get_router().call(system_prompt, user_message, max_tokens=max_tokens, temperature=temperature)
