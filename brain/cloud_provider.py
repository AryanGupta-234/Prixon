"""Cloud LLMProvider implementations.

These are the exact same three providers nlu.py called directly before this
package existed (Cerebras, Groq, Hugging Face) -- moved here as classes
behind LLMProvider so brain/router.py can treat them uniformly and so a
future CloudProvider (or the LocalProvider in local_provider.py) slots in
without router.py caring which one it is. Behavior is unchanged: same
endpoints, same timeout, same HF multi-model fallback list, same
"remember the model that last worked" stickiness.
"""
from __future__ import annotations

import time
from typing import List, Optional

import config
from brain.provider_base import LLMProvider


class _OpenAICompatibleProvider(LLMProvider):
    """Cerebras and Groq both expose an OpenAI-compatible /chat/completions
    endpoint -- one base class covers either, just pointed at the right
    base_url/key/model. Plain HTTP (no extra SDK dependency) since the shape
    is simple and identical across both."""

    base_url: str = ""

    def _api_key(self) -> str:
        raise NotImplementedError

    def _model(self) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return bool(self._api_key())

    def chat(self, system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"},
            json={
                "model": self._model(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


class CerebrasProvider(_OpenAICompatibleProvider):
    name = "cerebras"
    base_url = config.CEREBRAS_BASE_URL

    def _api_key(self) -> str:
        return config.CEREBRAS_API_KEY

    def _model(self) -> str:
        return config.CEREBRAS_MODEL


class GroqProvider(_OpenAICompatibleProvider):
    name = "groq"
    base_url = config.GROQ_BASE_URL

    def _api_key(self) -> str:
        return config.GROQ_API_KEY

    def _model(self) -> str:
        return config.GROQ_MODEL


class HuggingFaceProvider(LLMProvider):
    """Unlike the single-shot providers above, Hugging Face's router can
    land on different underlying models depending on what's currently
    hosted, so this tries a short internal fallback list before giving up
    on HF as a whole (see config.HF_MODEL_FALLBACKS)."""

    name = "huggingface"

    def __init__(self):
        self._client = None
        self._working_model: Optional[str] = None

    def available(self) -> bool:
        return bool(config.HF_TOKEN)

    def _get_client(self):
        if self._client is None:
            if not config.HF_TOKEN:
                raise RuntimeError("HF_TOKEN is not set.")
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(api_key=config.HF_TOKEN, provider="auto")
        return self._client

    def chat(self, system_prompt: str, user_message: str, *, max_tokens: int, temperature: float) -> str:
        client = self._get_client()
        models: List[str] = ([config.HF_MODEL] if config.HF_MODEL else []) + [
            m for m in config.HF_MODEL_FALLBACKS if m != config.HF_MODEL
        ]
        if self._working_model in models:
            models.remove(self._working_model)
            models.insert(0, self._working_model)

        last_exc = None
        for model in models:
            for _attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=max_tokens, temperature=temperature,
                    )
                    self._working_model = model
                    return response.choices[0].message.content or ""
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    kind = self.classify_error(exc)
                    if kind == "cold_start":
                        time.sleep(8)
                        continue
                    if kind in ("unsupported", "quota"):
                        break  # try the next model in HF's own list
                    raise
        raise last_exc if last_exc else RuntimeError("No Hugging Face model available.")
