"""Anomaly detection (spec section 17): threshold + duration + baseline,
never a bare instantaneous threshold.

'CPU=95% for 2 seconds -> ignore. CPU=95% for 10 minutes -> investigate' is
the concrete example this implements: a metric has to stay above threshold
continuously for at least MIN_DURATION before this emits anything at all,
tracked per-metric via a start-of-streak timestamp that resets the moment
the metric drops back down (spec section 19's 'CPU returns normal -> reset').

Severity additionally considers the baseline (spec section 17's 'unknown
process, unusual for this user' example) when the baseline is reliable
enough to judge against -- see baseline.MIN_OBSERVATIONS_FOR_BASELINE.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import config
from system.baseline import BaselineStore
from system.snapshot import SystemSnapshot

SEVERITIES = ("info", "low", "medium", "high", "critical")


@dataclass
class AnomalyEvent:
    metric: str
    severity: str
    value: float
    duration_seconds: float
    confidence: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    top_process: Optional[str] = None
    top_process_value: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric, "severity": self.severity, "value": self.value,
            "duration_seconds": round(self.duration_seconds, 1), "confidence": round(self.confidence, 2),
            "reason": self.reason, "timestamp": self.timestamp,
            "top_process": self.top_process, "top_process_value": self.top_process_value,
        }


class _MetricWatch:
    """Tracks one metric's current above-threshold streak."""
    def __init__(self):
        self.streak_start: Optional[float] = None
        self.last_emitted_severity: Optional[str] = None  # for alert-fatigue reset logic


class AnomalyDetector:
    def __init__(self, baseline: Optional[BaselineStore] = None):
        self.baseline = baseline or BaselineStore()
        self._watches: Dict[str, _MetricWatch] = {
            "cpu_usage_percent": _MetricWatch(),
            "memory_percent": _MetricWatch(),
            "disk_used_percent": _MetricWatch(),
        }

    def _check_metric(self, metric: str, value: Optional[float], threshold: float,
                       min_duration: float, now: float, top_process=None) -> Optional[AnomalyEvent]:
        watch = self._watches[metric]
        if value is None or value < threshold:
            watch.streak_start = None
            watch.last_emitted_severity = None
            return None

        if watch.streak_start is None:
            watch.streak_start = now
        duration = now - watch.streak_start

        if duration < min_duration:
            return None  # above threshold, but not persistent enough yet -- spec section 17's "2 seconds -> ignore"

        baseline = self.baseline.get(metric)
        z = baseline.z_score(value)
        # Severity climbs with how far past threshold, how long, and (when
        # the baseline is trustworthy) how unusual this is for THIS machine
        # -- spec section 17: "unknown for this user -> notify."
        if z is not None and z >= 3 and duration >= min_duration * 3:
            severity = "high"
        elif duration >= min_duration * 2:
            severity = "medium"
        else:
            severity = "low"

        confidence = min(0.95, 0.5 + duration / (min_duration * 4))
        reason = f"{metric} at {value:.0f} for {duration:.0f}s (threshold {threshold:.0f})"
        if z is not None:
            reason += f", {z:.1f} std devs from this machine's baseline"

        if watch.last_emitted_severity == severity:
            return None  # already alerted at this severity -- spec section 19 alert-fatigue prevention
        watch.last_emitted_severity = severity

        top_name = top_process.name if top_process else None
        top_value = top_process.cpu_percent if (top_process and metric == "cpu_usage_percent") else (
            top_process.memory_mb if (top_process and metric == "memory_percent") else None
        )
        return AnomalyEvent(metric=metric, severity=severity, value=value, duration_seconds=duration,
                             confidence=confidence, reason=reason,
                             top_process=top_name, top_process_value=top_value)

    def check(self, snapshot: SystemSnapshot) -> list:
        """Returns a list of newly-crossed AnomalyEvents (usually empty).
        Also feeds the baseline with this snapshot's values -- callers
        should call this once per poll, not speculatively."""
        now = snapshot.timestamp
        events = []

        top_cpu_proc = snapshot.processes.top_by_cpu[0] if snapshot.processes.top_by_cpu else None
        top_mem_proc = snapshot.processes.top_by_memory[0] if snapshot.processes.top_by_memory else None

        cpu = snapshot.cpu.usage_percent
        self.baseline.update("cpu_usage_percent", cpu)
        e = self._check_metric("cpu_usage_percent", cpu, config.ANOMALY_CPU_THRESHOLD_PERCENT,
                                config.ANOMALY_CPU_MIN_DURATION_SECONDS, now, top_process=top_cpu_proc)
        if e:
            events.append(e)

        mem = snapshot.memory.percent
        self.baseline.update("memory_percent", mem)
        e = self._check_metric("memory_percent", mem, config.ANOMALY_MEMORY_THRESHOLD_PERCENT,
                                config.ANOMALY_MEMORY_MIN_DURATION_SECONDS, now, top_process=top_mem_proc)
        if e:
            events.append(e)

        disk = snapshot.disk.used_percent
        self.baseline.update("disk_used_percent", disk)
        e = self._check_metric("disk_used_percent", disk, config.ANOMALY_DISK_THRESHOLD_PERCENT,
                                config.ANOMALY_DISK_MIN_DURATION_SECONDS, now)
        if e:
            events.append(e)

        return events
