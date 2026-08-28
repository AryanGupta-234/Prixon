"""Action catalog.

The old TF-IDF candidate generator has been retired from the normal runtime
path. SemanticIndex/fastembed is now responsible for semantic retrieval;
ActionIndex remains the authoritative allow-list and execution metadata.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

import config


NORMALIZATIONS = {
    "wi fi": "wifi", "wi-fi": "wifi", "wify": "wifi", "blue tooth": "bluetooth",
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "lemme": "let me",
    "gimme": "give me", "kinda": "kind of", "pls": "please", "plz": "please",
    "u": "you", "ur": "your", "what's": "what is", "whats": "what is",
}


def normalize(text: str) -> str:
    out = text.lower().strip()
    for src, dst in sorted(NORMALIZATIONS.items(), key=lambda kv: len(kv[0]), reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(src) + r"(?![a-z0-9])"
        out = re.sub(pattern, dst, out)
    return out


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
    """Load and expose the executable capability catalog."""
    def __init__(self, data_path: str = None):
        self.data_path = data_path or config.DATA_PATH
        self.entries: List[ActionEntry] = []
        self.groups: Dict[str, ActionGroup] = {}
        self._load()

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

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Compatibility shim: semantic retrieval owns candidate generation."""
        return []

    def get_group(self, target: str):
        return self.groups.get(target)

    def full_catalog(self) -> List[Dict]:
        return [g.to_candidate() for g in sorted(self.groups.values(), key=lambda x: x.target_name.lower())]
