"""Unified memory manager.

Single source of truth for "context". Nothing else in the codebase should
keep its own parallel conversation history -- nlu.ConversationState (the
existing short-term buffer the LLM prompt is built from) is owned and
exposed through here rather than duplicated, and a structured, persisted
episodic log sits alongside it.

Phase 1 scope: working memory (delegated) + episodic memory with simple
recency/entity-overlap retrieval. Semantic (embedding-based) retrieval,
procedural memory, and preference memory are later phases -- the Episode
schema below already has the fields they'd need, so adding them later is
additive, not a rewrite.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from nlu import ConversationState, NLUResult

EPISODIC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "episodic_memory.jsonl"
)

# Superset of event types Phase 1 writes plus ones later phases will add
# (app_closed, goal_completed, strategy_success, ...) -- kept here as the
# single documented vocabulary so new event types don't get invented ad hoc.
EVENT_TYPES = {
    "task_started", "task_completed", "task_failed",
    "app_opened", "app_closed", "website_opened",
    "goal_created", "goal_completed", "goal_abandoned",
    "preference_detected", "strategy_success", "strategy_failure",
}


@dataclass
class Episode:
    event_id: str
    event_type: str
    intent: str
    target: str
    target_name: str
    parameters: Dict[str, Any]
    success: Optional[bool]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnifiedMemory:
    """Owns the LLM-facing conversation buffer plus the persisted episodic log."""

    def __init__(self, episodic_path: Optional[str] = None, max_episodes_in_ram: int = 200):
        self.conversation = ConversationState()
        self.episodic_path = episodic_path or EPISODIC_PATH
        self.episodes: List[Episode] = []
        self._load_episodes(max_episodes_in_ram)

    # ---- working memory (delegates to the existing ConversationState) ----
    def snapshot(self) -> Dict[str, Any]:
        return self.conversation.snapshot()

    def remember_turn(self, user_text: str, result: NLUResult, target_name: Optional[str]):
        self.conversation.remember(user_text, result, target_name)

    # ---- episodic memory ----
    def _load_episodes(self, limit: int):
        if not os.path.exists(self.episodic_path):
            return
        try:
            with open(self.episodic_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.episodes.append(Episode(**d))
        except Exception:
            # A corrupt/partial log should never prevent the assistant from
            # starting -- worst case, episodic memory starts empty this run.
            pass

    def record_event(self, event_type: str, intent: str = "", target: str = "",
                      target_name: str = "", parameters: Optional[Dict[str, Any]] = None,
                      success: Optional[bool] = None) -> Episode:
        episode = Episode(
            event_id=str(uuid.uuid4()), event_type=event_type, intent=intent,
            target=target, target_name=target_name, parameters=parameters or {},
            success=success, timestamp=time.time(),
        )
        self.episodes.append(episode)
        try:
            os.makedirs(os.path.dirname(self.episodic_path), exist_ok=True)
            with open(self.episodic_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass  # persistence is best-effort; the in-RAM copy still works this session
        return episode

    def relevant_episodes(self, target: Optional[str] = None, intent: Optional[str] = None,
                           limit: int = 5) -> List[Episode]:
        """Cheap recency + entity/goal-overlap scoring -- no embeddings needed
        at hundreds-of-episodes scale. Swap this out, not around, when
        semantic retrieval (spec section 20) lands."""
        now = time.time()
        scored = []
        for ep in self.episodes:
            score = 0.0
            if target and ep.target == target:
                score += 0.6
            if intent and ep.intent == intent:
                score += 0.3
            age_minutes = (now - ep.timestamp) / 60.0
            score += max(0.0, 0.1 - age_minutes * 0.001)
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def last_successful_task(self) -> Optional[Episode]:
        for ep in reversed(self.episodes):
            if ep.event_type == "task_completed" and ep.success:
                return ep
        return None
