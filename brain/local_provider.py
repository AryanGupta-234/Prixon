"""Local LLMProvider via Ollama with live availability checks."""
from __future__ import annotations

import time
from typing import Optional

import config
from brain.provider_base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "local"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self._checked_available: Optional[bool] = None
        self._checked_at = 0.0
        self._availability_ttl = 5.0

    def available(self) -> bool:
        """Check that Ollama is reachable *and the configured model exists*.

        Availability is intentionally short-lived. Ollama may start/stop or
        a model may be pulled while Prixon is already running, so a permanent
        per-process negative cache is incorrect for an agent living in a
        changing environment.
        """
        now = time.monotonic()
        if self._checked_available is not None and now - self._checked_at < self._availability_ttl:
            return self._checked_available
        if not self.model:
            self._checked_available = False
            self._checked_at = now
            return False
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            if resp.status_code != 200:
                ok = False
            else:
                models = resp.json().get("models", [])
                names = {str(m.get("name", "")) for m in models}
                ok = self.model in names or any(
                    name.split(":", 1)[0] == self.model.split(":", 1)[0] for name in names
                )
            self._checked_available = ok
        except Exception:
            self._checked_available = False
        self._checked_at = now
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
            timeout=config.OLLAMA_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        text = (data.get("message") or {}).get("content") or ""
        if not text.strip():
            raise RuntimeError("Ollama returned an empty model response")
        return text

    def classify_error(self, exc: Exception) -> str:
        msg = str(exc).lower()
        if any(x in msg for x in ("connection", "timeout", "timed out", "connectex", "refused")):
            return "unsupported"
        if "not found" in msg or "model" in msg and "pull" in msg:
            return "unsupported"
        return super().classify_error(exc)
