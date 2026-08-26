"""Baseline learning (spec section 16).

Learns "normal" for this specific machine/user via running mean+variance
(Welford's algorithm -- numerically stable, O(1) memory per metric, no
need to keep a raw history window). Persisted to a small local JSON file so
the baseline survives restarts instead of relearning from zero every launch
-- spec section 16 explicitly describes this as a multi-day process ("during
the first days/weeks Prixon should learn..."), which a rebuild-every-run
baseline would defeat.

Deliberately NOT sent anywhere -- this file, like system_events generally,
stays local (spec sections 8, 45).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Optional

_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "system_baseline.json"
)

# Below this many observations, a baseline's mean/std aren't trustworthy
# enough to judge anomalies against -- spec section 16: "do not immediately
# treat high resource usage as abnormal." anomaly_detector.py checks this
# before using z-scores from a metric's baseline.
MIN_OBSERVATIONS_FOR_BASELINE = 30


@dataclass
class MetricBaseline:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations from the mean (Welford)

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        return self.variance ** 0.5

    @property
    def reliable(self) -> bool:
        return self.count >= MIN_OBSERVATIONS_FOR_BASELINE

    def z_score(self, value: float) -> Optional[float]:
        if not self.reliable or self.stddev == 0:
            return None
        return (value - self.mean) / self.stddev


class BaselineStore:
    """One MetricBaseline per named metric (e.g. 'cpu_usage_percent',
    'memory_percent'). Thread-safe since SystemAgent updates this from a
    background thread while the main loop / router may read it."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or _BASELINE_PATH
        self._metrics: Dict[str, MetricBaseline] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, d in raw.items():
                self._metrics[name] = MetricBaseline(**d)
        except Exception:
            pass  # missing/corrupt baseline file -> start learning fresh

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with self._lock:
                raw = {name: asdict(b) for name, b in self._metrics.items()}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(raw, f)
        except Exception:
            pass  # best-effort persistence, same policy as memory.py's episodic log

    def update(self, metric: str, value: Optional[float]):
        if value is None:
            return
        with self._lock:
            self._metrics.setdefault(metric, MetricBaseline()).update(value)

    def get(self, metric: str) -> MetricBaseline:
        with self._lock:
            return self._metrics.get(metric, MetricBaseline())
