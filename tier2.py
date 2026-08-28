"""Tier 2: cheap local intent classification and entity extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config

MIN_TOP_SCORE = 0.55
MIN_SCORE_GAP = 0.25

@dataclass
class Tier2Result:
    resolved: bool
    target: Optional[str] = None
    intent: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    parameters: Optional[Dict[str, Any]] = None

def classify(user_text: str, candidates: List[Dict]) -> Tier2Result:
    if not candidates:
        return Tier2Result(False, reason="no candidates")
    top = candidates[0]
    top_score = top.get("score", 0.0)
    if top_score < MIN_TOP_SCORE:
        return Tier2Result(False, reason=f"top score {top_score:.2f} below {MIN_TOP_SCORE} threshold")
    runner_up_score = candidates[1].get("score", 0.0) if len(candidates) > 1 else 0.0
    gap = top_score - runner_up_score
    if gap < MIN_SCORE_GAP:
        return Tier2Result(False, reason=f"ambiguous: gap {gap:.2f} below {MIN_SCORE_GAP} vs runner-up")
    return Tier2Result(True, target=top.get("target"), intent=top.get("intent"), confidence=min(0.95, top_score),
                       reason=f"unambiguous lexical match (score {top_score:.2f}, gap {gap:.2f} vs runner-up)",
                       parameters=extract_entities(user_text))

def classify_semantic(user_text: str, semantic_candidates: List[Dict], groups: Dict) -> Tier2Result:
    if not semantic_candidates:
        return Tier2Result(False, reason="no semantic candidates")
    top = semantic_candidates[0]
    top_score = top.get("score", 0.0)
    if top_score < config.SEMANTIC_MIN_SCORE:
        return Tier2Result(False, reason=f"semantic top score {top_score:.2f} below {config.SEMANTIC_MIN_SCORE} threshold")
    runner_up_score = semantic_candidates[1].get("score", 0.0) if len(semantic_candidates) > 1 else 0.0
    gap = top_score - runner_up_score
    if gap < config.SEMANTIC_MIN_GAP:
        return Tier2Result(False, reason=f"semantic ambiguous: gap {gap:.2f} below {config.SEMANTIC_MIN_GAP}")
    group = groups.get(top.get("target"))
    return Tier2Result(True, target=top.get("target"), intent=group.intent if group else None, confidence=min(0.9, top_score),
                       reason=f"unambiguous semantic match (score {top_score:.2f}, gap {gap:.2f} vs runner-up)",
                       parameters=extract_entities(user_text))

_PERCENT_RE = re.compile(r"\b(\d{1,3})\s?%|\b(\d{1,3})\s*percent\b")
_DIRECTION_WORDS = ("up", "down", "left", "right", "louder", "quieter", "brighter", "dimmer")

def extract_entities(text: str) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}
    low = text.lower()
    m = _PERCENT_RE.search(low)
    if m:
        entities["percent"] = int(m.group(1) or m.group(2))
    for word in _DIRECTION_WORDS:
        if re.search(rf"\b{word}\b", low):
            entities["direction"] = word
            break
    return entities

_CLOSE_VERBS_RE = re.compile(r"\b(close|quit|exit|kill|terminate|shut\s*down|stop)\b\s+(.*)", re.I)
_RUNNING_RE = re.compile(
    r"\b(?:is|are|does)\s+(?P<app>.+?)\s+(?:still\s+|currently\s+)?(?:running|open|active)\b|"
    # "check/chk/see/tell me if/whether X is/are running" AND the shorter
    # "check/chk if X running" (no is/are) that natural typing drops.
    r"\b(?:check|chk|see|tell\s+me)\s+(?:if|whether)\s+(?P<app2>.+?)\s+(?:(?:is|are)\s+)?(?:still\s+|currently\s+)?(?:running|open|active)\b|"
    r"\b(?:check|chk|see|tell\s+me)\s+(?:for)\s+(?P<app3>.+?)\s+(?:running|open|active)\b",
    re.I,
)
# Bare "spotify running?" / "spotify open?" -- deliberately a SEPARATE,
# stricter pass rather than folded into _RUNNING_RE. An earlier version
# anchored this at ^ inside the same alternation, but because Python tries
# alternatives left-to-right at each start position, the permissive bare
# branch matched at position 0 before the verb-anchored branches got a
# chance to match later in the string -- e.g. "can u chk if spotify running"
# wrongly captured "chk if spotify" as the app name instead of matching the
# check/chk branch on "spotify". Trying this only after _RUNNING_RE fails,
# and only against short question-like text, avoids that.
_BARE_RUNNING_RE = re.compile(r"^(?P<app>[a-z0-9 ]+?)\s+(?:still\s+|currently\s+)?(?:running|open|active)\s*\??$", re.I)
_BARE_STOPWORDS = {
    "if", "whether", "chk", "check", "see", "tell", "does", "is", "are", "can", "could",
    "would", "please", "u", "you", "your", "the", "my", "this", "that",
}
_APP_NAME_FILLER_WORDS = {
    "the", "my", "this", "that", "app", "application", "program", "process", "processes",
    "please", "for", "me", "now", "right", "down", "up", "bro", "man", "pls", "plz", "u", "can",
    "still", "currently", "also",
}

def extract_app_name_hint(text: str) -> Optional[str]:
    m = _CLOSE_VERBS_RE.search(text)
    if not m:
        return None
    return _clean_app_hint(m.group(2))

def extract_running_app_hint(text: str) -> Optional[str]:
    stripped = text.strip()
    m = _RUNNING_RE.search(stripped)
    if m:
        return _clean_app_hint(m.group("app") or m.group("app2") or m.group("app3"))
    m = _BARE_RUNNING_RE.match(stripped)
    if m:
        words = m.group("app").lower().split()
        # A genuine bare app-status query is short and has no leftover verb
        # words -- if any showed up, the sentence needed the verb-anchored
        # patterns above and just didn't match one of their forms; don't
        # guess by swallowing the whole sentence as an "app name".
        if 1 <= len(words) <= 4 and not (set(words) & _BARE_STOPWORDS):
            return _clean_app_hint(m.group("app"))
    return None

def _clean_app_hint(value: str) -> Optional[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in _APP_NAME_FILLER_WORDS]
    hint = " ".join(words).strip()
    return hint or None
