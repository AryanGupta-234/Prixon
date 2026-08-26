"""Local LLMProvider via Ollama (spec section 5).

Scoped narrowly for this slice: a working local chat path behind the same
LLMProvider interface as the cloud providers, so brain/router.py can start
treating "local" as a real chain member. Resource-awareness (RAM/CPU/GPU
checks before deciding to use it -- spec section 9) is Slice 2's job, not
this one; `available()` here only checks that an Ollama server is actually
reachable, not whether the machine currently has headroom to run it well.
"""
from __future__ import annotations

from typing import Optional

import config
from brain.provider_base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "local"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self._checked_available: Optional[bool] = None

    def available(self) -> bool:
        """A short-timeout ping to Ollama's own /api/tags endpoint -- cheap
        enough to call once per provider-chain build, unlike actually
        loading a model. Cached per-instance so a dead server during one
        session doesn't get re-probed on every single turn; router.py
        creates a fresh provider chain per process anyway, so this resets
        naturally on restart."""
        if self._checked_available is not None:
            return self._checked_available
        if not self.model:
            self._checked_available = False
            return False
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            self._checked_available = resp.status_code == 200
        except Exception:
            self._checked_available = False
        return self._checked_available

    def chat(self, system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=60,  # local inference on a laptop CPU can be slow -- generous vs. the 30s cloud timeout
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return (data.get("message") or {}).get("content") or ""

    def classify_error(self, exc: Exception) -> str:
        # A local server that's simply not running/reachable is neither a
        # quota nor an auth problem -- router.py should just skip to the
        # next provider without remembering this as "exhausted for the
        # session" the way a cloud quota hit is remembered, since the local
        # server could come back at any time.
        msg = str(exc).lower()
        if "connection" in msg or "timeout" in msg or "timed out" in msg:
            return "unsupported"
        return super().classify_error(exc)
