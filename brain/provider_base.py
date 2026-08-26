"""LLMProvider abstraction (spec section 7).

Kept synchronous to match the rest of this codebase -- main.py's REPL loop,
data_store.py, memory.py, none of it is async, and nothing here needs to be.
The spec's example signature is `async def chat`; this uses a plain `chat`
method instead so it composes with what already exists rather than forcing
an asyncio rewrite of main.py just to satisfy an example signature. If a
genuinely async provider (e.g. a streaming local runtime) shows up later,
wrap it with `asyncio.run()` at the call site rather than making everything
upstream async for one provider's sake.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """Raised (or left to propagate as a plain Exception) by chat() on
    failure. router.py is the only thing that decides what to do about a
    failure -- providers just report it."""


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Cheap, local, no-network check: credentials configured, or a
        local runtime reachable. router.py uses this to build the provider
        chain -- it should never be expensive enough to matter per-call."""

    @abstractmethod
    def chat(self, system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
        """Returns the raw text content of the model's reply. Let the
        underlying exception propagate on failure -- router.py classifies it
        via classify_error() rather than the provider deciding what a
        failure means for the rest of the session."""

    def classify_error(self, exc: Exception) -> str:
        """One of: cold_start, unsupported, quota, auth, other.

        Default is the same string-sniffing nlu.py used before this existed
        as a class hierarchy -- kept as the shared default so every cloud
        provider doesn't have to reimplement it, but a provider whose SDK
        raises a distinct exception type for e.g. rate limits should
        override this with something more precise than message-sniffing.
        """
        msg = str(exc).lower()
        if "503" in msg or "loading" in msg or "currently loading" in msg:
            return "cold_start"
        if "not supported" in msg or "model_not_supported" in msg or "does not exist" in msg or "404" in msg:
            return "unsupported"
        if any(kw in msg for kw in (
            "429", "402", "rate limit", "rate_limit", "quota", "credit",
            "insufficient", "exceeded", "payment required", "too many requests",
        )):
            return "quota"
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg or ("invalid" in msg and "key" in msg) or ("invalid" in msg and "token" in msg):
            return "auth"
        return "other"
