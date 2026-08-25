"""Goal engine (spec sections 12-13), scoped to what the current tool
catalog can actually back.

The spec's canonical example -- "take me to Spotify": launch it if
installed, else open the web version -- has no analogue in this dataset.
All 207 targets are Windows system settings, MMC snap-ins, Explorer
shortcuts, and read-only PowerShell diagnostics (see data/windows_automation
_10000.jsonl). There are no third-party apps, no installed/not-installed
branching, and no web-URL fallback targets to fall back to. Building that
exact example against this catalog would be decorative, not functional --
so this doesn't pretend to.

What IS real and testable here: tagging each target with a coarse topic
(network, display, audio, power, ...) using the same keyword vocabulary
data_store.SYNONYMS is already keyed on, and using the topic of the last
successful task as AgentState.active_goal so a short, verb-less follow-up
in the same topic ("what about the wifi", "and the battery?") stays biased
toward that topic on the next retrieval pass instead of starting cold. This
is goal *continuation*, not goal-driven multi-strategy planning -- that
would need capabilities (app detection, web fallback) this build doesn't have.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from data_store import ActionGroup

# Reuses data_store.SYNONYMS' vocabulary as topic membership so a topic tag
# and the retrieval synonym expansion can't silently drift apart.
_TOPICS: Dict[str, tuple] = {
    "network": ("wifi", "wireless", "internet", "network", "vpn", "ethernet", "bluetooth"),
    # NOTE: deliberately no "screen" or bare "monitor" here -- the dataset's
    # own instruction templates tack "... screen" onto nearly every target
    # ("fonts screen", "character map screen"), and "Resource Monitor" is a
    # perf tool, not a display setting, so bare "monitor" isn't a reliable
    # display signal in this corpus either.
    "display": ("display", "brightness", "resolution", "night light"),
    "audio": ("sound", "audio", "volume", "speaker", "microphone"),
    "power": ("battery", "power", "sleep", "shutdown", "restart"),
    "storage": ("storage", "disk", "drive", "space", "recycle", "bin"),
    "security": ("security", "defender", "antivirus", "virus", "firewall", "password", "login", "sign-in"),
    "personalization": ("theme", "wallpaper", "background", "personalization", "dark", "colors"),
    # NOTE: no bare "user" -- it shows up incidentally ("Startup folder" ==
    # "open user Startup folder") without being about account management.
    "accounts": ("account", "family", "sign-in options"),
    "updates": ("update",),
    "devices": ("device", "printer", "camera", "webcam"),
    "accessibility": ("accessibility", "ease of access", "magnifier", "narrator"),
}

_topic_cache: Dict[str, Optional[str]] = {}


def _contains_word(text: str, word: str) -> bool:
    # Plain substring containment would match "power" inside "PowerShell"
    # (a dev tool, not a power-settings target) -- word-boundary matching
    # avoids that whole class of false positive.
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def _topic_for_group_data(target_name: str, phrasings: List[str]) -> Optional[str]:
    # The target's own name is a far stronger signal than incidental words in
    # example phrasings -- "Battery"'s phrasings happen to say "battery
    # screen", which without this weighting ties 1-1 against "power" on the
    # word "screen" and loses to "display" on dict insertion order. Weighting
    # name matches heavily fixes that without needing per-topic word lists
    # to be mutually exclusive (they aren't, and shouldn't have to be).
    name_low = target_name.lower()
    text_low = f"{target_name} {' '.join(phrasings)}".lower()
    best_topic, best_score = None, 0
    for topic, words in _TOPICS.items():
        score = (sum(3 for w in words if _contains_word(name_low, w))
                 + sum(1 for w in words if _contains_word(text_low, w)))
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic


def topic_for_group(group: ActionGroup) -> Optional[str]:
    if group.target not in _topic_cache:
        _topic_cache[group.target] = _topic_for_group_data(group.target_name, group.example_phrasings)
    return _topic_cache[group.target]


def bias_candidates(candidates: List[Dict], active_goal: Optional[str],
                     groups: Dict[str, ActionGroup], boost: float = 0.15) -> List[Dict]:
    """Nudges same-topic candidates up when there IS an active goal. Only
    ever tips close calls -- a strong signal for a different topic still
    wins, since `boost` is small relative to a real score gap."""
    if not active_goal:
        return candidates
    boosted = []
    for c in candidates:
        group = groups.get(c["target"])
        c = dict(c)
        if group and topic_for_group(group) == active_goal:
            c["score"] = round(c.get("score", 0.0) + boost, 4)
        boosted.append(c)
    boosted.sort(key=lambda c: c["score"], reverse=True)
    return boosted
