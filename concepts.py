"""Growable vocabulary of synonyms/antonyms/aliases -- as DATA, not prose.

Handoff spec Section 14 is explicit: Qwen should get compact structured
context, not a wall of instruction text. But the underlying need is real --
the user should be able to say "crank up the screen" today and "amp up the
monitor" next month and have Prixon recognize both without a human hand-
editing SYSTEM_PROMPT every time. The previous approach (spelling out
synonym lists in prose inside SYSTEM_PROMPT) doesn't scale: every new
synonym group permanently taxes every single request's prompt-eval time,
forever, whether or not that request needed it.

This module keeps that vocabulary in data/concept_aliases.json instead:
  - It is retrieval-time expansion, not always-on prompt text: a request
    about "screen" only pays for the "display" concept group; a request
    about Spotify pays nothing for it.
  - It can grow indefinitely (by hand, or later by pattern-learning
    promoting an observed correction into a permanent alias) without
    touching this file or SYSTEM_PROMPT.
  - It is inspectable and editable by a human at any time -- not weights,
    not a prompt string, just a JSON file of concept groups.

This is deliberately small and dumb by design: exact/substring word
matching, no embeddings, no LLM call. It only exists to widen the net that
FastEmbed casts before Qwen ever sees anything, and (optionally) to hand
Qwen a two-line hint about which canonical concept an ambiguous word maps
to. It is not a semantic engine; FastEmbed and Qwen remain that.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "concept_aliases.json")
_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_cache_mtime: float = -1.0

_DEFAULT_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "display": {
        "synonyms": ["screen", "display", "monitor", "brightness"],
        "antonyms": ["dark", "dim"],  # opposite-direction words still route to the same capability
    },
    "audio": {
        "synonyms": ["sound", "audio", "volume", "speaker", "mute"],
        "antonyms": ["quiet", "silent"],
    },
    "network": {
        "synonyms": ["wifi", "wireless", "internet", "network", "connection"],
        "antonyms": ["offline", "disconnected"],
    },
    "application": {
        "synonyms": ["app", "program", "application", "process", "software"],
        "antonyms": [],
    },
    "power": {
        "synonyms": ["battery", "charge", "power"],
        "antonyms": ["dead", "drained"],
    },
}


def _load() -> Dict[str, Any]:
    """Reload from disk when the file changes, so edits (manual or future
    pattern-promoted) take effect without restarting Prixon."""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = os.path.getmtime(_PATH)
        except OSError:
            mtime = -1.0
        if mtime == _cache_mtime and _cache:
            return _cache
        data = None
        try:
            with open(_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
        if not isinstance(data, dict) or not data:
            data = _DEFAULT_GROUPS
            _ensure_seeded(data)
        _cache = data
        _cache_mtime = mtime
        return _cache


def _ensure_seeded(data: Dict[str, Any]) -> None:
    """Write the default seed file once if none exists yet, so it's visible
    and editable on disk from the start rather than only living in this
    module's fallback constant."""
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        if not os.path.exists(_PATH):
            with open(_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def matched_groups(text: str) -> List[str]:
    """Which concept groups does this utterance touch? Cheap word-overlap
    check -- deliberately not fuzzy, so it stays fast and predictable."""
    words = set(_words(text))
    if not words:
        return []
    groups = _load()
    hits = []
    for name, spec in groups.items():
        vocab = set(w.lower() for w in (spec.get("synonyms", []) + spec.get("antonyms", [])))
        if words & vocab:
            hits.append(name)
    return hits


def expand_query(text: str) -> str:
    """Append canonical concept names for any matched group, so FastEmbed's
    retrieval query carries the capability's own vocabulary even when the
    user's literal words differ from it. Cheap, additive, never removes the
    user's original words."""
    hits = matched_groups(text)
    if not hits:
        return text
    return text + " " + " ".join(hits)


def relevant_hint(text: str) -> Dict[str, List[str]]:
    """A tiny, per-request slice of the vocabulary -- only the groups this
    utterance actually touched -- suitable for optionally handing Qwen two
    lines of disambiguation context instead of the whole file. Usually empty
    (most utterances don't touch any tracked concept at all)."""
    hits = matched_groups(text)
    if not hits:
        return {}
    groups = _load()
    return {name: groups[name].get("synonyms", []) for name in hits}


def add_alias(group: str, word: str, kind: str = "synonyms") -> None:
    """Append one new word to a concept group and persist it. This is the
    hook pattern-learning (Section 17) can eventually call when it notices
    the same correction happen repeatedly -- e.g. the user always says
    'the box' to mean 'the PC case' -- without editing any Python."""
    if kind not in {"synonyms", "antonyms"}:
        kind = "synonyms"
    data = dict(_load())  # _load() takes _lock itself -- must not hold it here too
    with _lock:
        group_data = dict(data.get(group, {"synonyms": [], "antonyms": []}))
        words = list(group_data.get(kind, []))
        word = word.strip().lower()
        if word and word not in words:
            words.append(word)
        group_data[kind] = words
        data[group] = group_data
        try:
            os.makedirs(os.path.dirname(_PATH), exist_ok=True)
            tmp = _PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _PATH)
        except Exception:
            return
        global _cache_mtime
        _cache_mtime = -1.0  # force reload next call
