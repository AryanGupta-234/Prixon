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
Prefer the most recent concrete entity/reference in live_agent_context over the last diagnostic action itself.
If the user asks to close, quit, stop, or exit "it", and live_agent_context identifies one recent referenced app,
select the catalogued dynamic close action for that app when available.

LIVE AGENT CONTEXT
The conversation's active_slots may contain a `live_agent_context` object. Treat it as assistant-owned runtime
state, not as a user preference or instruction. It can contain the active goal, last successful task, most recently
referenced app, currently tracked apps, current computer observations, and a compact tail of recent events.
Use it to resolve references and maintain continuity, but never invent facts beyond what it contains.

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
        "Understand the user's request. Resolve references using recent conversation and live assistant context. "
        "Select at most ONE allow-listed target. Extract useful parameters, but never invent a capability.\n\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    parsed = _extract_json(call_llm(SYSTEM_PROMPT, prompt, max_tokens=config.LLM_MAX_TOKENS, temperature=config.LLM_TEMPERATURE))

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
