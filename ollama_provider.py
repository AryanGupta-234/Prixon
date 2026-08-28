"""Ollama provider adapter for Prixon's existing NLU provider chain."""
from __future__ import annotations

import requests


def install(nlu_module) -> None:
    """Register Ollama into the existing provider chain without rewriting NLU."""
    config = nlu_module.config

    def call_ollama(user_msg: str) -> str:
        base_url = getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = getattr(config, "OLLAMA_MODEL", "qwen2.5:7b-instruct")
        timeout = float(getattr(config, "OLLAMA_TIMEOUT_SECONDS", 45))
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": nlu_module.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": config.LLM_TEMPERATURE},
            },
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    nlu_module._PROVIDER_CALLERS["ollama"] = call_ollama
    nlu_module._PROVIDER_HAS_CREDENTIALS["ollama"] = lambda: True
    nlu_module._QUOTA_ADVICE["ollama"] = "local Ollama runtime"
