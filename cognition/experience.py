"""Lightweight, local experience learning.

This is deliberately not model fine-tuning. It extracts high-confidence,
non-sensitive operational patterns from Prixon's own successful/failed task
history and exposes a compact summary to the reasoning layer.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Optional


PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "learned_state.json")


class ExperienceModel:
    def __init__(self, path: Optional[str] = None):
        self.path = path or PATH
        self._lock = threading.Lock()
        self.successes = Counter()
        self.failures = Counter()
        self.transitions = Counter()
        self.recent = deque(maxlen=30)
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.successes.update(data.get("successes", {}))
            self.failures.update(data.get("failures", {}))
            self.transitions.update({tuple(k.split("\u001f", 1)): v for k, v in data.get("transitions", {}).items() if "\u001f" in k})
            self.recent.extend(data.get("recent", []))
        except Exception:
            pass

    def observe(self, event_type: str, target: str = "", target_name: str = "", success: Optional[bool] = None):
        key = (target_name or target or "").strip().lower()
        if not key or success is None:
            return
        with self._lock:
            if success:
                self.successes[key] += 1
            else:
                self.failures[key] += 1
            previous = self.recent[-1].get("target") if self.recent else None
            if previous and previous != key:
                self.transitions[(previous, key)] += 1
            self.recent.append({"event": event_type, "target": key, "success": bool(success), "timestamp": time.time()})
            self._save_locked()

    def _save_locked(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            transitions = {"\u001f".join(k): v for k, v in self.transitions.items()}
            data = {
                "successes": dict(self.successes),
                "failures": dict(self.failures),
                "transitions": transitions,
                "recent": list(self.recent),
                "updated_at": time.time(),
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def context(self) -> Dict[str, Any]:
        with self._lock:
            reliable = []
            for target, ok in self.successes.most_common(12):
                bad = self.failures.get(target, 0)
                total = ok + bad
                if total >= 3:
                    reliable.append({"target": target, "successes": ok, "failures": bad,
                                     "success_rate": round(ok / total, 2)})
            common_sequences = [
                {"from": a, "to": b, "observations": n}
                for (a, b), n in self.transitions.most_common(8) if n >= 2
            ]
            return {"reliable_actions": reliable, "common_action_sequences": common_sequences}
