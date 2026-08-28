"""Local pattern, preference, and procedural learning for Prixon.

Learns only from repeated, observable assistant interactions. It does not
fine-tune a model and does not treat a single utterance as a permanent fact.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Optional

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patterns.json")


class PatternMemory:
    def __init__(self, path: Optional[str] = None):
        self.path = path or PATH
        self._lock = threading.Lock()
        self.preferences: Counter[str] = Counter()
        self.habits: Dict[str, Dict[str, Any]] = {}
        self.procedures: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.preferences.update(data.get("preferences", {}))
            self.habits.update(data.get("habits", {}))
            self.procedures.update(data.get("procedures", {}))
        except Exception:
            pass

    def observe_preference(self, key: str, value: str, evidence: str = ""):
        if not key or not value:
            return
        with self._lock:
            token = f"{key.strip().lower()}={value.strip().lower()}"
            self.preferences[token] += 1
            self._save_locked()

    def observe_action(self, target: str, intent: str, success: bool, timestamp: Optional[float] = None):
        if not target:
            return
        now = timestamp or time.time()
        key = target.strip().lower()
        with self._lock:
            item = self.habits.setdefault(key, {"observations": 0, "successes": 0, "failures": 0, "hours": Counter()})
            item["observations"] += 1
            item["successes" if success else "failures"] += 1
            hour = str(time.localtime(now).tm_hour)
            hours = item.get("hours", {})
            hours[hour] = int(hours.get(hour, 0)) + 1
            item["hours"] = hours
            self._save_locked()

    def observe_transition(self, previous: Optional[str], current: Optional[str], success: bool):
        if not previous or not current or not success or previous == current:
            return
        key = f"{previous.strip().lower()} -> {current.strip().lower()}"
        with self._lock:
            item = self.procedures.setdefault(key, {"observations": 0, "successes": 0})
            item["observations"] += 1
            item["successes"] += 1
            self._save_locked()

    def _save_locked(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            data = {
                "preferences": dict(self.preferences),
                "habits": self.habits,
                "procedures": self.procedures,
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
            habits = []
            for target, item in self.habits.items():
                total = int(item.get("observations", 0))
                if total < 3:
                    continue
                successes = int(item.get("successes", 0))
                rate = successes / total if total else 0.0
                hours = item.get("hours", {})
                common_hour = max(hours, key=hours.get) if hours else None
                habits.append({"target": target, "observations": total, "success_rate": round(rate, 2), "common_hour": common_hour})
            procedures = [
                {"sequence": key, **value}
                for key, value in self.procedures.items()
                if int(value.get("observations", 0)) >= 2
            ]
            preferences = [k for k, count in self.preferences.most_common(12) if count >= 2]
            return {"stable_preferences": preferences, "habits": habits[:12], "procedures": procedures[:12]}
