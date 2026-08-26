"""Provider chain + failover (spec sections 7-8, Slice 1 scope).

This replaces nlu.py's old free-function _call_llm()/_provider_chain() with
the exact same behavior over LLMProvider instances instead of functions:
same "try last-working provider first," same per-session quota-exhaustion
memory, same error messages main.py already knows how to display.

Resource-aware routing (spec section 9 -- RAM/CPU/GPU/network state
deciding local vs. cloud) is deliberately NOT here yet. This module only
asks "is this provider available and does it still have quota this
session?" -- Slice 2 adds a system-state check in front of this chain
without needing to change chat()'s call signature.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import config
from brain.local_provider import OllamaProvider
from brain.cloud_provider import CerebrasProvider, GroqProvider, HuggingFaceProvider
from brain.provider_base import LLMProvider

_QUOTA_ADVICE = {
    "cerebras": "https://cloud.cerebras.ai (free tier resets daily)",
    "groq": "https://console.groq.com (free tier resets daily)",
    "huggingface": "https://huggingface.co/settings/billing (free tier resets monthly, and it's a very small allowance)",
}


class ModelRouter:
    def __init__(self, providers: Optional[List[LLMProvider]] = None):
        # "local" first in preference order once it's actually available --
        # cheap/private/offline-capable requests should prefer it once
        # Slice 2 adds real task-difficulty and resource checks. For now,
        # config.LLM_PROVIDER still picks which one goes first, matching
        # existing behavior exactly.
        self._providers: Dict[str, LLMProvider] = {
            p.name: p for p in (providers or [OllamaProvider(), CerebrasProvider(), GroqProvider(), HuggingFaceProvider()])
        }
        self._working_provider: Optional[str] = None
        self._exhausted: set = set()

    def _provider_chain(self) -> List[str]:
        preferred = config.LLM_PROVIDER if config.LLM_PROVIDER in self._providers else None
        fallback_order = [p for p in config.PROVIDER_FALLBACK_ORDER if p in self._providers]
        ordered = ([preferred] if preferred else []) + [p for p in fallback_order if p != preferred]
        return [p for p in ordered if self._providers[p].available() and p not in self._exhausted]

    def call(self, system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
        chain = self._provider_chain()
        if not chain:
            if self._exhausted:
                tried = ", ".join(sorted(self._exhausted))
                raise RuntimeError(
                    f"Every configured provider ({tried}) is out of free quota for "
                    f"now. " + " / ".join(f"{p}: {_QUOTA_ADVICE[p]}" for p in sorted(self._exhausted) if p in _QUOTA_ADVICE)
                )
            raise RuntimeError(
                "No LLM provider is configured. Set at least one of CEREBRAS_API_KEY, "
                "GROQ_API_KEY, or HF_TOKEN in .env, or run a local Ollama server."
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
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                kind = provider.classify_error(exc)
                if kind == "auth":
                    raise RuntimeError(
                        f"{name} rejected its credentials (check the matching API key/token in .env)."
                    ) from exc
                if kind == "quota":
                    self._exhausted.add(name)
                # cold_start/unsupported/other/quota: fall through to the next provider.
                continue

        tried = ", ".join(chain)
        raise RuntimeError(
            f"None of the available providers ({tried}) could handle the request right now. Last error: {last_exc}"
        ) from last_exc


# Module-level singleton -- mirrors nlu.py's previous module-level session
# state (_working_provider, _exhausted_providers) so behavior across a
# single run is unchanged: one router, one memory of what's exhausted/working.
_default_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    global _default_router
    if _default_router is None:
        _default_router = ModelRouter()
    return _default_router


def call_llm(system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
    return get_router().call(system_prompt, user_message, max_tokens=max_tokens, temperature=temperature)
