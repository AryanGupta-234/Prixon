"""System Agent (spec sections 10, 15, 19, 41, 52): the background thread
that turns raw polling into meaningful events, without blocking the main
assistant loop.

Two independent dedup layers exist on purpose, for two different questions:
- anomaly_detector.AnomalyDetector: "has this metric's severity actually
  changed since I last looked?" (per-streak, resets when the metric drops)
- AlertCooldownTracker below: "have I already bothered the user about this
  alert_id recently?" (wall-clock cooldown, independent of whether the
  underlying streak is ongoing) -- spec section 19's alert_id/
  last_triggered/cooldown/repeat_count fields live here.

Modeled on embeddings.SemanticIndex's pattern: runs in a daemon background
thread, degrades to "not ready yet" rather than blocking anything, and the
rest of the app treats "agent not running" and "agent running but nothing
notable happened" identically.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import config
from system import collector as system_collector
from system.anomaly_detector import AnomalyDetector, AnomalyEvent
from system.baseline import BaselineStore
from system.snapshot import SystemSnapshot


@dataclass
class _AlertRecord:
    last_triggered: float
    repeat_count: int = 1
    acknowledged: bool = False


class AlertCooldownTracker:
    """Spec section 19/41: don't repeatedly alert about the same thing."""

    def __init__(self):
        self._alerts: Dict[str, _AlertRecord] = {}
        self._lock = threading.Lock()

    def should_alert(self, alert_id: str, cooldown_seconds: float) -> bool:
        now = time.time()
        with self._lock:
            record = self._alerts.get(alert_id)
            if record is None or (now - record.last_triggered) >= cooldown_seconds:
                self._alerts[alert_id] = _AlertRecord(last_triggered=now)
                return True
            record.repeat_count += 1
            return False

    def reset(self, alert_id: str):
        """Called when the underlying condition resolves -- spec section
        19's 'CPU returns normal -> reset,' so the next occurrence is
        treated as a fresh alert rather than counting toward the old one's
        cooldown."""
        with self._lock:
            self._alerts.pop(alert_id, None)


class SystemAgent(threading.Thread):
    def __init__(self, memory=None, on_event: Optional[Callable[[AnomalyEvent], None]] = None,
                 poll_interval: Optional[float] = None):
        super().__init__(daemon=True)
        self.memory = memory
        self.on_event = on_event
        self.poll_interval = poll_interval or config.SYSTEM_POLL_INTERVAL_SECONDS
        self.baseline = BaselineStore()
        self.detector = AnomalyDetector(self.baseline)
        self.cooldown = AlertCooldownTracker()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[SystemSnapshot] = None
        self._poll_count = 0
        self.ready = False  # True once at least one snapshot has been taken
        self._last_known_online: Optional[bool] = None  # carried across polls that skip the real probe

    def latest_snapshot(self) -> Optional[SystemSnapshot]:
        with self._lock:
            return self._latest

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                pass  # a single bad poll should never kill the background thread
            if self._poll_count % 12 == 0:  # ~ every minute at the default 5s interval
                self.baseline.save()
            self._stop_event.wait(self.poll_interval)

    def _poll_once(self):
        probe_internet = (self._poll_count % config.SYSTEM_NETWORK_PROBE_EVERY_N_POLLS) == 0
        snap = system_collector.snapshot(probe_internet=probe_internet, cpu_interval=0.0)

        # network_monitor.read() correctly reports online=None on polls that
        # skip the actual reachability probe (spec section 11: don't guess).
        # That's the right answer for a single read(), but for the AGENT's
        # cached state it just means "still whatever it last confirmed" --
        # without this, resource_policy/diagnostics would flicker between
        # 'online' and 'unknown' every few seconds even on a perfectly
        # stable connection, purely because most polls don't re-probe.
        if snap.network.online is None and self._last_known_online is not None:
            snap.network.online = self._last_known_online
        elif snap.network.online is not None:
            self._last_known_online = snap.network.online

        with self._lock:
            self._latest = snap
            self.ready = True
        self._poll_count += 1

        events = self.detector.check(snap)
        for event in events:
            alert_id = f"{event.metric}:{event.severity}"
            if self.cooldown.should_alert(alert_id, config.ALERT_COOLDOWN_SECONDS):
                self._dispatch(event)

        # Reset cooldowns for metrics that have returned to a non-anomalous
        # streak (spec section 19) -- the detector's own watch already
        # cleared its streak; mirror that here for the cooldown tracker.
        for metric, watch in self.detector._watches.items():  # noqa: SLF001 -- same module family, not a public API boundary yet
            if watch.streak_start is None:
                for sev in ("low", "medium", "high", "critical"):
                    self.cooldown.reset(f"{metric}:{sev}")

    def _dispatch(self, event: AnomalyEvent):
        if self.memory is not None:
            try:
                self.memory.record_event(
                    "system_anomaly", intent=event.metric, target=event.metric,
                    target_name=event.metric, success=None, parameters=event.to_dict(),
                )
            except Exception:
                pass  # memory persistence is best-effort elsewhere too; don't let this kill the agent
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass


# Module-level default agent, mirroring embeddings.SemanticIndex's pattern --
# main.py starts this once; anything else (brain/router.py) that wants a
# cheap cached snapshot without forcing a fresh synchronous poll reads
# through here and falls back to a direct collector.snapshot() if no agent
# has been started yet (e.g. in unit tests).
_default_agent: Optional[SystemAgent] = None


def start_default_agent(memory=None, on_event=None) -> SystemAgent:
    global _default_agent
    if _default_agent is None or not _default_agent.is_alive():
        _default_agent = SystemAgent(memory=memory, on_event=on_event)
        _default_agent.start()
    return _default_agent


def latest_snapshot() -> Optional[SystemSnapshot]:
    """Returns the last cached poll if the background agent is running,
    else None -- callers (brain/router.py) fall back to a fresh synchronous
    snapshot when this is None, exactly like before Slice 3 existed."""
    if _default_agent is not None and _default_agent.ready:
        return _default_agent.latest_snapshot()
    return None
