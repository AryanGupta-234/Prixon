"""Action catalog and lexical retrieval.

Retrieval is only candidate generation. The LLM performs semantic reasoning
against the returned allow-listed candidates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


def _tokenize(text: str):
    return re.findall(r"[a-z0-9_'-]+", text.lower())


STOP = {"the", "a", "an", "to", "for", "me", "my", "please", "can", "you", "i", "want", "do", "it", "on"}

SYNONYMS = {
    "wifi": "wireless internet network connection wi-fi",
    "wireless": "wifi internet network connection",
    "internet": "wifi wireless network connection online",
    "screen": "display monitor brightness visual",
    "display": "screen monitor brightness visual",
    "monitor": "screen display",
    "dim": "brightness screen display darker",
    "bright": "brightness screen display",
    "sound": "audio volume speaker",
    "audio": "sound volume speaker",
    "loud": "volume audio sound louder",
    "quiet": "volume audio sound quieter",
    "bluetooth": "wireless devices",
    "camera": "webcam video",
    "webcam": "camera video",
    "wallpaper": "background personalization desktop",
    "trash": "recycle bin deleted files",
    "bin": "recycle bin trash",
    "app": "application program software",
    "program": "app application software",
    "close": "exit quit stop",
    "open": "launch start show pull up bring up",
    "fix": "diagnose troubleshoot repair",
    "broken": "problem issue troubleshoot diagnose",
    "storage": "disk drive space capacity",
    "space": "storage disk drive capacity",
    "restart": "reboot relaunch",
    "shutdown": "power off turn off",
}

NORMALIZATIONS = {
    "wi fi": "wifi", "wi-fi": "wifi", "wify": "wifi", "blue tooth": "bluetooth",
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "lemme": "let me",
    "gimme": "give me", "kinda": "kind of", "pls": "please", "plz": "please",
    "u": "you", "ur": "your", "what's": "what is", "whats": "what is",
}


def normalize(text: str) -> str:
    out = text.lower().strip()
    # Whole-word normalization prevents chat shorthand like "u" from
    # corrupting normal words such as "launch" or "calculator".
    for src, dst in sorted(NORMALIZATIONS.items(), key=lambda kv: len(kv[0]), reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(src) + r"(?![a-z0-9])"
        out = re.sub(pattern, dst, out)
    return out


def expand(text: str) -> str:
    base = normalize(text)
    extras = []
    for key, synonyms in SYNONYMS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", base):
            extras.append(synonyms)
    return base + " " + " ".join(extras)


@dataclass
class ActionEntry:
    id: str
    instruction: str
    intent: str
    target: str
    target_name: str
    action: str
    uri: str = ""
    execution: str = ""
    verification: str = ""
    risk: str = "low"
    platform: str = "Windows"
    shell: str = ""


@dataclass
class ActionGroup:
    target: str
    target_name: str
    intent: str
    action: str
    uri: str
    execution: str
    verification: str
    risk: str
    shell: str = ""
    example_phrasings: List[str] = field(default_factory=list)
    executable: str = ""
    fixed_args: str = ""

    def to_candidate(self, score: float = 0.0) -> Dict:
        return {
            "target": self.target,
            "target_name": self.target_name,
            "intent": self.intent,
            "action": self.action,
            "uri": self.uri,
            "risk": self.risk,
            "score": round(float(score), 4),
            "examples": self.example_phrasings[:5],
        }


def _parse_execution(execution: str):
    """Extract only simple executable/fixed-argument forms from known dataset entries."""
    execution = (execution or "").strip()
    m = re.match(r'Start-Process\s+"([^"]+)"\s*$', execution, re.I)
    if m:
        return m.group(1), ""
    m = re.match(r'explorer\.exe\s+(.+)$', execution, re.I)
    if m:
        return "explorer.exe", m.group(1)
    if re.match(r"^[\w.-]+\.(?:exe|msc|cpl)$", execution, re.I):
        return execution, ""
    return "", ""


class ActionIndex:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or config.DATA_PATH
        self.entries: List[ActionEntry] = []
        self.groups: Dict[str, ActionGroup] = {}
        self._load()
        self._build_index()

    def _load(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                e = ActionEntry(
                    id=d.get("id", ""), instruction=d.get("instruction", ""), intent=d.get("intent", ""),
                    target=d.get("target", ""), target_name=d.get("target_name", ""), action=d.get("action", ""),
                    uri=d.get("uri", ""), execution=d.get("execution", ""), verification=d.get("verification", ""),
                    risk=d.get("risk", "low"), platform=d.get("platform", "Windows"), shell=d.get("shell", ""),
                )
                self.entries.append(e)
                if e.target not in self.groups:
                    exe, args = _parse_execution(e.execution)
                    self.groups[e.target] = ActionGroup(
                        target=e.target, target_name=e.target_name, intent=e.intent, action=e.action,
                        uri=e.uri, execution=e.execution, verification=e.verification, risk=e.risk,
                        shell=e.shell, executable=exe, fixed_args=args,
                    )
                grp = self.groups[e.target]
                if e.instruction and len(grp.example_phrasings) < 8:
                    grp.example_phrasings.append(e.instruction)

    def _build_index(self):
        corpus = [f"{e.instruction} {e.target_name} {e.intent}" for e in self.entries]
        self._vectorizer = TfidfVectorizer(
            lowercase=True, tokenizer=_tokenize, token_pattern=None, ngram_range=(1, 2),
            stop_words=list(STOP), min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or config.TOP_K_CANDIDATES
        q = self._vectorizer.transform([expand(query)])
        sims = cosine_similarity(q, self._matrix)[0]
        candidates = []
        seen = set()
        for idx in sims.argsort()[::-1]:
            score = float(sims[idx])
            if score <= 0:
                break
            e = self.entries[idx]
            if e.target in seen:
                continue
            seen.add(e.target)
            candidates.append(self.groups[e.target].to_candidate(score))
            if len(candidates) >= top_k:
                break
        return candidates

    def get_group(self, target: str):
        return self.groups.get(target)

    def full_catalog(self) -> List[Dict]:
        return [g.to_candidate() for g in sorted(self.groups.values(), key=lambda x: x.target_name.lower())]
