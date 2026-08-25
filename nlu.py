"""LLM-first conversational NLU and planning.

The LLM is the semantic brain. Retrieval only narrows the allow-list. The
model can understand natural English, references, synonyms, implied intent,
parameters and multi-turn context, but it can only select catalogued targets.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config

_hf_client = None
_working_hf_model = None
_working_provider = None  # remembers the last provider that actually worked
_exhausted_providers = set()  # providers confirmed quota-exhausted this session

_ALL_PROVIDERS = ("cerebras", "groq", "huggingface")

SYSTEM_PROMPT = r'''You are the reasoning and dialogue brain of a highly capable Windows personal assistant.

Your personality: calm, concise, observant, confident, natural spoken English, lightly witty when it fits.
Think like a futuristic computer companion, but do not imitate any copyrighted character's exact dialogue,
voice, catchphrases, or personality.

CORE PRINCIPLE
The user speaks naturally. You infer what they MEAN, not just what words they used.
The supplied candidates are an execution allow-list. You may select ONLY a candidate target from that list.
Never invent tools, URIs, executables, commands, or capabilities.

UNDERSTAND:
- synonyms: screen/display/monitor, wifi/wireless/internet/network, sound/audio/volume, app/program/application
- colloquialisms: "crank it up", "pull that up", "the thing for wifi", "what's eating my storage"
- indirect requests: "my eyes are killing me, dim it" -> display-related action if available
- typos and speech-to-text errors when meaning is obvious
- references: it, that, this, that one, same one, there, back, again, the thing we just opened
- ellipsis: "same thing, but bluetooth"; "make it louder"; "okay now the other one"
- intent differences: open vs inspect vs troubleshoot vs change vs launch vs close
- questions vs commands: "what's using my RAM?" is information, not a request to change RAM
- parameter extraction: percentages, amounts, names, paths, app names, directions, counts
- multiple requests: create a plan in your head, but select the FIRST executable action only

REFERENCE RESOLUTION
Use recent context aggressively but safely. If "it" or "that" has one obvious compatible antecedent,
resolve it. If there are two plausible targets, do NOT guess; return match_target=null and ask a short question.

NATURAL CONVERSATION
Do not require the user to phrase commands like a programmer. "Can you take me to the place where I change
my mouse?" should work. "My internet is acting weird" should prefer diagnostics if a diagnostic candidate exists.

SAFETY
Never infer permission for destructive or risky actions. The caller performs the final risk/confirmation check.
For low-risk actions, select normally. For medium/high risk, preserve the candidate risk and let the caller ask.

OUTPUT ONLY JSON with this exact schema:
{
  "match_target": "target string or null",
  "confidence": "high|medium|low|none",
  "intent": "short stable intent name",
  "parameters": {},
  "reference": "none|explicit|recent_target|recent_action",
  "reply": "3-16 word natural spoken response",
  "reason": "brief reasoning summary"
}
'''


@dataclass
class Turn:
    user: str
    target: Optional[str] = None
    target_name: Optional[str] = None
    intent: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reply: str = ""


@dataclass
class ConversationState:
    turns: List[Turn] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        recent = self.turns[-config.CONTEXT_TURNS:]
        return {
            "recent_turns": [
                {"user": t.user, "target": t.target, "target_name": t.target_name,
                 "intent": t.intent, "parameters": t.parameters, "reply": t.reply}
                for t in recent
            ],
            "active_slots": self.slots,
        }

    def remember(self, user: str, result: "NLUResult", target_name: Optional[str]):
        self.turns.append(Turn(user=user, target=result.match_target, target_name=target_name,
                               intent=result.intent, parameters=result.parameters, reply=result.reply))
        if result.match_target:
            self.slots["last_target"] = result.match_target
            self.slots["last_target_name"] = target_name
        self.slots.update(result.parameters)


@dataclass
class NLUResult:
    match_target: Optional[str]
    confidence: str
    reply: str
    intent: str = "unknown"
    parameters: Dict[str, Any] = field(default_factory=dict)
    reference: str = "none"
    raw: Dict[str, Any] = field(default_factory=dict)


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        if not config.HF_TOKEN:
            raise RuntimeError("HF_TOKEN is not set.")
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(api_key=config.HF_TOKEN, provider="auto")
    return _hf_client


def _classify_error(exc: Exception) -> str:
    """
    Distinguishes the failure modes that matter for deciding what to do
    next -- retry the same thing, move on to the next model/provider, or
    give up on this provider entirely for the rest of the session:
      - "cold_start"     -> model is warming up, worth a short retry
      - "unsupported"    -> this model isn't routable right now, skip it
      - "quota"          -> free-tier credits/rate limit exhausted
      - "auth"           -> bad/missing credentials, no point retrying
      - "other"          -> unknown, surface it
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


def _call_openai_compatible(base_url: str, api_key: str, model: str, user_msg: str) -> str:
    """
    Cerebras and Groq both expose an OpenAI-compatible /chat/completions
    endpoint, so one function covers either -- just point it at the right
    base_url/key/model. Kept as a plain HTTP call (no extra SDK dependency)
    since the shape is simple and identical across both providers.
    """
    import requests
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def _call_cerebras_raw(user_msg: str) -> str:
    if not config.CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is not set.")
    return _call_openai_compatible(config.CEREBRAS_BASE_URL, config.CEREBRAS_API_KEY, config.CEREBRAS_MODEL, user_msg)


def _call_groq_raw(user_msg: str) -> str:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return _call_openai_compatible(config.GROQ_BASE_URL, config.GROQ_API_KEY, config.GROQ_MODEL, user_msg)


def _call_huggingface_raw(user_msg: str) -> str:
    """
    Unlike the single-shot providers above, Hugging Face's router can land
    on different underlying models depending on what's currently hosted, so
    this tries a short internal fallback list before giving up on HF as a
    whole (see config.HF_MODEL_FALLBACKS).
    """
    global _working_hf_model
    client = _get_hf_client()
    models = ([config.HF_MODEL] if config.HF_MODEL else []) + [
        m for m in config.HF_MODEL_FALLBACKS if m != config.HF_MODEL
    ]
    if _working_hf_model in models:
        models.remove(_working_hf_model)
        models.insert(0, _working_hf_model)

    last_exc = None
    for model in models:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                    max_tokens=config.LLM_MAX_TOKENS, temperature=config.LLM_TEMPERATURE,
                )
                _working_hf_model = model
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                kind = _classify_error(exc)
                if kind == "cold_start":
                    time.sleep(8)
                    continue
                if kind in ("unsupported", "quota"):
                    break  # try the next model in HF's own list
                raise
    raise last_exc if last_exc else RuntimeError("No Hugging Face model available.")


_PROVIDER_CALLERS = {
    "cerebras": _call_cerebras_raw,
    "groq": _call_groq_raw,
    "huggingface": _call_huggingface_raw,
}

_PROVIDER_HAS_CREDENTIALS = {
    "cerebras": lambda: bool(config.CEREBRAS_API_KEY),
    "groq": lambda: bool(config.GROQ_API_KEY),
    "huggingface": lambda: bool(config.HF_TOKEN),
}

_QUOTA_ADVICE = {
    "cerebras": "https://cloud.cerebras.ai (free tier resets daily)",
    "groq": "https://console.groq.com (free tier resets daily)",
    "huggingface": "https://huggingface.co/settings/billing (free tier resets monthly, and it's a very small allowance)",
}


def _provider_chain() -> List[str]:
    """
    Preferred provider first (config.LLM_PROVIDER), then the rest of
    config.PROVIDER_FALLBACK_ORDER, skipping anything without credentials
    configured and anything already confirmed exhausted this session.
    """
    ordered = [config.LLM_PROVIDER] + [p for p in config.PROVIDER_FALLBACK_ORDER if p != config.LLM_PROVIDER]
    return [p for p in ordered if p in _PROVIDER_CALLERS and _PROVIDER_HAS_CREDENTIALS[p]() and p not in _exhausted_providers]


def _call_llm(user_msg: str) -> str:
    global _working_provider

    chain = _provider_chain()
    if not chain:
        if _exhausted_providers:
            tried = ", ".join(sorted(_exhausted_providers))
            raise RuntimeError(
                f"Every configured provider ({tried}) is out of free quota for "
                f"now. " + " / ".join(f"{p}: {_QUOTA_ADVICE[p]}" for p in sorted(_exhausted_providers))
            )
        raise RuntimeError(
            "No LLM provider is configured. Set at least one of CEREBRAS_API_KEY, "
            "GROQ_API_KEY, or HF_TOKEN in .env."
        )

    # Try the provider that last worked first, if it's still in the chain.
    if _working_provider in chain:
        chain.remove(_working_provider)
        chain.insert(0, _working_provider)

    last_exc = None
    for provider in chain:
        try:
            result = _PROVIDER_CALLERS[provider](user_msg)
            _working_provider = provider
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            kind = _classify_error(exc)
            if kind == "auth":
                raise RuntimeError(
                    f"{provider} rejected its credentials (check the matching "
                    f"API key/token in .env)."
                ) from exc
            if kind == "quota":
                _exhausted_providers.add(provider)
            # cold_start/unsupported/other/quota: all fall through to try
            # the next provider in the chain rather than failing outright.
            continue

    tried = ", ".join(chain)
    raise RuntimeError(
        f"None of the available providers ({tried}) could handle the request "
        f"right now. Last error: {last_exc}"
    ) from last_exc


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _candidate_view(candidates: List[Dict], broad: bool = False) -> List[Dict]:
    """Field set sent to the LLM per candidate.

    In shortlist mode (~10 candidates) examples/retrieval_score are cheap
    and help disambiguation, so they stay. In broad mode (up to all 207
    catalog entries -- this is what fires for anything TF-IDF can't match
    at all, like a stray "hi" reaching here) the SAME fields on 207 items is
    what produced a 21k-token request against an 8k TPM free-tier cap and
    got the whole call rejected with a 413. Broad mode strips examples
    (the single biggest per-item cost), drops retrieval_score (it's always
    0 in broad mode -- nothing was scored), and omits `risk` entirely when
    it's the default "low" rather than repeating it 200+ times. Measured:
    this took the worst-case (207-item) payload from ~20k estimated tokens
    down to ~5k.
    """
    if broad:
        out = []
        for c in candidates:
            item = {
                "target": c.get("target"), "name": c.get("target_name"),
                "intent": c.get("intent"), "action": c.get("action"),
            }
            risk = c.get("risk", "low")
            if risk != "low":
                item["risk"] = risk
            out.append(item)
        return out
    return [
        {
            "target": c.get("target"), "name": c.get("target_name"), "intent": c.get("intent"),
            "action": c.get("action"), "risk": c.get("risk", "low"),
            "retrieval_score": c.get("score", 0), "examples": c.get("examples", [])[:4],
        }
        for c in candidates
    ]


def resolve(user_text: str, candidates: List[Dict], assistant_name: Optional[str] = None,
            broad_search: bool = False, state: Optional[ConversationState] = None) -> NLUResult:
    state = state or ConversationState()
    context = {
        "assistant_name": assistant_name or config.ASSISTANT_NAME,
        "request": user_text,
        "conversation": state.snapshot(),
        "retrieval": "broad" if broad_search else "shortlist",
        "allow_list": _candidate_view(candidates, broad=broad_search),
    }
    prompt = (
        "Understand the user's request. Resolve references using recent conversation. "
        "Select at most ONE allow-listed target. Extract useful parameters, but never invent a capability.\n\n"
        # Compact, not indent=2 -- pretty-printing adds whitespace/newline
        # tokens the model gets zero information value from. On the broad
        # (207-item) case this alone was worth ~28% of the payload.
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    parsed = _extract_json(_call_llm(prompt))

    valid = {c.get("target") for c in candidates}
    target = parsed.get("match_target")
    confidence = str(parsed.get("confidence", "none")).lower()
    if confidence not in {"high", "medium", "low", "none"}:
        confidence = "none"
    if target not in valid:
        target = None
        confidence = "none"

    result = NLUResult(
        match_target=target,
        confidence=confidence,
        reply=str(parsed.get("reply") or ("On it." if target else "I'm not quite sure what you mean.")),
        intent=str(parsed.get("intent") or "unknown"),
        parameters=parsed.get("parameters") if isinstance(parsed.get("parameters"), dict) else {},
        reference=str(parsed.get("reference") or "none"),
        raw=parsed,
    )
    group = next((c for c in candidates if c.get("target") == target), None)
    state.remember(user_text, result, group.get("target_name") if group else None)
    return result
