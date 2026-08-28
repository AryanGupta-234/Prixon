"""LLM-first conversational NLU and planning.

The LLM is the semantic brain. Retrieval only narrows the allow-list. The
model can understand natural English, references, synonyms, implied intent,
parameters and multi-turn context, but it can only select catalogued targets.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from brain.router import call_llm

# Kept deliberately short: this is sent on EVERY Qwen call and this
# hardware's prompt-eval speed (~46-55 tok/s, CPU-only) makes every fixed
# token here a permanent latency tax paid by every request forever,
# regardless of whether that request needed it. A 7B instruct model already
# knows English synonyms, colloquialisms, and typo-tolerance; it does not
# need those re-taught in prose every call. What it DOES need spelled out
# every time are the few rules that are specific to THIS system and would
# otherwise be guessed inconsistently: the allow-list constraint, how to use
# live_agent_context, and the exact output schema.
#
# Anything that used to live here as static prose but genuinely varies by
# request (concept synonym groups, recent habits/sequences) now arrives
# per-request instead, via concepts.py and live_agent_context -- see
# context_engine.py. That keeps this fixed cost small while the system's
# effective vocabulary/knowledge can still grow, just as data instead of
# permanent prompt text.
SYSTEM_PROMPT = r'''You are a Windows assistant's reasoning brain. Natural, concise spoken replies.

Select at most one target from allow_list, or null if none fit. Never invent tools/commands.
Infer meaning, not literal words (synonyms, typos, indirect phrasing all count).
Questions ("what's using my RAM?") are information requests, not action requests.

live_agent_context (in conversation.active_slots) is assistant-owned ground truth: active goal,
last referenced app, tracked apps, world_state, recent event, learned habits, concept_hint.
Use it to resolve "it"/"that"/references. If ambiguous between 2+ targets, return match_target=null
and ask briefly instead of guessing. To close/quit "it", use last_referenced_app if one exists.

Never grant yourself permission for risky actions; the caller re-checks risk separately.

OUTPUT ONLY JSON:
{"match_target":"string or null","confidence":"high|medium|low|none","intent":"short name",
"parameters":{},"reference":"none|explicit|recent_target|recent_action",
"reply":"3-16 word natural reply","reason":"brief reason"}
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
            "retrieval_score": c.get("score", 0), "examples": c.get("examples", [])[:2],
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
        "Understand the user's request. Resolve references using recent conversation and live assistant context. "
        "Select at most ONE allow-listed target. Extract useful parameters, but never invent a capability.\n\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    if config.DEBUG:
        conversation_chars = len(json.dumps(context["conversation"], ensure_ascii=False))
        capability_chars = len(json.dumps(context["allow_list"], ensure_ascii=False))
        system_state_chars = len(json.dumps(context["conversation"].get("active_slots", {}).get("live_agent_context", {}), ensure_ascii=False))
        print(f"[CONTEXT] conversation_chars={conversation_chars} capability_chars={capability_chars} "
              f"system_state_chars={system_state_chars} candidate_count={len(candidates)} "
              f"total_prompt_chars={len(prompt)}", flush=True)
    parsed = _extract_json(call_llm(SYSTEM_PROMPT, prompt, max_tokens=config.LLM_MAX_TOKENS, temperature=config.LLM_TEMPERATURE))
    # Stash the exact input this output was produced from. This is consumed
    # (and stripped back out) by main.py's post-verification training logger
    # -- see training_log.py. It is never read by anything that interprets
    # NLUResult.raw as "the model's output" (match_target/confidence/etc are
    # all still read by exact key elsewhere), so this rides along harmlessly
    # for callers that don't know about it.
    parsed["_prompt_context"] = context

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
