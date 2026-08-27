"""Tier 2: cheap, local, LLM-free classification.

Deliberately NOT a neural "small local model" -- that would mean shipping or
downloading real model weights (e.g. sentence-transformers) on first run,
which needs an internet connection and a few hundred MB this phase doesn't
assume the user wants yet. What's here is the honest version of "skip the
remote LLM when retrieval already made the answer obvious": it reuses the
TF-IDF scores data_store.py already computed for free and resolves only when
the top candidate is both strong in absolute terms and clearly ahead of the
runner-up. Swap the body of `classify()` for an embedding-model call later
without changing its signature if a real local model gets added.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# How much clearer the top candidate needs to be than the runner-up before
# Tier 2 trusts it without asking the LLM to adjudicate. Tuned conservatively
# on purpose: false confidence here means silently doing the wrong thing
# with no LLM in the loop to catch it, so this only fires when it's
# genuinely not close.
MIN_TOP_SCORE = 0.55
MIN_SCORE_GAP = 0.25


import config


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
    return Tier2Result(
        True, target=top.get("target"), intent=top.get("intent"),
        confidence=min(0.95, top_score),
        reason=f"unambiguous lexical match (score {top_score:.2f}, gap {gap:.2f} vs runner-up)",
        parameters=extract_entities(user_text),
    )


def classify_semantic(user_text: str, semantic_candidates: List[Dict], groups: Dict) -> Tier2Result:
    """Same shape as classify(), but scored on embedding cosine similarity
    instead of TF-IDF -- catches paraphrases that share no vocabulary at all
    ("make it quieter" vs "lower the volume"), which lexical Tier 2 can't.
    Returns not-resolved (never raises) if semantic_candidates is empty,
    which is exactly what embeddings.SemanticIndex.search() returns when the
    model isn't loaded -- so this degrades for free when embeddings aren't
    available."""
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
    return Tier2Result(
        True, target=top.get("target"), intent=group.intent if group else None,
        confidence=min(0.9, top_score),
        reason=f"unambiguous semantic match (score {top_score:.2f}, gap {gap:.2f} vs runner-up)",
        parameters=extract_entities(user_text),
    )
# NOTE: executor._run_allowlisted intentionally ignores the `parameters`
# argument today -- letting extracted free-text parameters reach a shell
# call is exactly what the allow-list architecture exists to prevent. This
# extractor exists so Tier 2/3 have somewhere to put percentages/counts/
# directions they notice, and so a future parameterized tool (e.g. "set
# volume to 40%") has real data to consume instead of nothing. Treat this
# as inert plumbing today, not a live feature -- nothing downstream reads it yet.
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
_APP_NAME_FILLER_WORDS = {
    "the", "my", "this", "that", "app", "application", "program", "process",
    "please", "for", "me", "now", "right", "down", "up", "bro", "man", "pls", "plz", "u", "can",
}


def extract_app_name_hint(text: str) -> Optional[str]:
    """Pulls a candidate app name out of a close/quit/exit-style request,
    independent of which tier resolved the intent -- used only when the
    matched action is close_app_dynamic (tools.close_running_app then
    resolves this hint against whatever's ACTUALLY running, so a rough
    extraction here is fine; it doesn't need to be exact, just close
    enough for the fuzzy matching downstream). Strips the close-verb and
    common filler words, keeps what's left. Returns None if nothing
    meaningful remains (e.g. bare 'close' with no target)."""
    m = _CLOSE_VERBS_RE.search(text)
    if not m:
        return None
    remainder = m.group(2)
    words = [w for w in re.findall(r"[a-z0-9]+", remainder.lower()) if w not in _APP_NAME_FILLER_WORDS]
    hint = " ".join(words).strip()
    return hint or None
