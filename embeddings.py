"""Local semantic embeddings for Tier 2 (spec section 4).

TF-IDF Tier 2 (tier2.py) is fast, dependency-free, and verified correct
against all 10,000 dataset rows -- but it's bag-of-words: "make it quieter"
shares almost no tokens with "lower the volume", so it can't catch genuine
paraphrases without someone hand-writing every synonym into
data_store.SYNONYMS. This adds a real local embedding model (fastembed --
ONNX runtime, no torch, ~30MB) as an additional Tier 2 signal for exactly
that gap.

Design constraints this respects, all because a personal assistant should
never feel like it's hanging:
- Never blocks the app. Model download/load happens in a background thread;
  if it isn't ready yet when a request comes in, that request just falls
  through to TF-IDF Tier 2 / Tier 3 as before -- it does NOT wait.
- Never hammers a blocked/offline network on every launch. fastembed's own
  retry logic burns ~40s (3s+9s+27s backoff) before giving up on one failed
  download; that cost is paid once, then cached to disk with a cooldown.
- Degrades to "not available" cleanly if fastembed isn't installed, the
  model can't be downloaded, or ONNX runtime doesn't like this machine --
  never crashes the app either way.

NOTE: the download host (huggingface.co) was not reachable from the sandbox
this was built in -- confirmed by actually running it and watching it fail
after ~40s. So the happy path (real embeddings actually loading and scoring)
is UNTESTED here; only the degrade-to-unavailable path is verified. Try it
on your own machine -- if the model downloads successfully this should just
start working with zero other code changes. If it doesn't, `--debug` will
show `SEMANTIC` as unavailable and say why.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

import config

_STATUS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "embedding_status.json"
)
_RETRY_COOLDOWN_SECONDS = 60 * 60  # don't retry a failed download more than once/hour


def _load_status() -> Dict:
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_status(status: Dict):
    try:
        os.makedirs(os.path.dirname(_STATUS_PATH), exist_ok=True)
        with open(_STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status, f)
    except Exception:
        pass  # best-effort; worst case we just retry next launch


class SemanticIndex:
    """Wraps a fastembed model. Loads in a background thread; `.ready` stays
    False until (if ever) it succeeds. Safe to construct with no network."""

    def __init__(self, groups: Dict[str, "ActionGroup"]):  # noqa: F821 (data_store.ActionGroup)
        self._groups = groups
        self._model = None
        self._vectors: Optional[Dict[str, list]] = None
        self.ready = False
        self.unavailable_reason: Optional[str] = None

        status = _load_status()
        last_failed = status.get("last_failed_at", 0)
        if status.get("permanently_unavailable"):
            self.unavailable_reason = status.get("reason", "previously unavailable")
            return
        if last_failed and (time.time() - last_failed) < _RETRY_COOLDOWN_SECONDS:
            remaining_min = int((_RETRY_COOLDOWN_SECONDS - (time.time() - last_failed)) / 60)
            self.unavailable_reason = f"retry cooldown ({remaining_min}m left) after: {status.get('reason', 'unknown error')}"
            return

        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            self.unavailable_reason = "fastembed not installed (pip install fastembed)"
            _save_status({"permanently_unavailable": True, "reason": self.unavailable_reason})
            return

        try:
            model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        except Exception as exc:
            self.unavailable_reason = f"model load/download failed: {exc}"
            _save_status({"last_failed_at": time.time(), "reason": self.unavailable_reason})
            return

        try:
            targets = list(self._groups.keys())
            texts = [
                f"{self._groups[t].target_name}. " + " ".join(self._groups[t].example_phrasings[:4])
                for t in targets
            ]
            vectors = list(model.embed(texts))
            self._model = model
            self._vectors = dict(zip(targets, vectors))
            self.ready = True
            _save_status({"permanently_unavailable": False, "last_success_at": time.time()})
        except Exception as exc:
            self.unavailable_reason = f"corpus embedding failed: {exc}"
            _save_status({"last_failed_at": time.time(), "reason": self.unavailable_reason})

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Returns [] (never raises) if the model isn't ready yet -- callers
        should treat that exactly like "no semantic signal available"."""
        if not self.ready or self._model is None or not self._vectors:
            return []
        try:
            import numpy as np
            q_vec = list(self._model.embed([query]))[0]
            q_norm = float(np.linalg.norm(q_vec)) + 1e-9
            scored = []
            for target, vec in self._vectors.items():
                sim = float(np.dot(q_vec, vec) / (q_norm * (float(np.linalg.norm(vec)) + 1e-9)))
                scored.append((sim, target))
            scored.sort(reverse=True)
            return [{"target": t, "score": round(s, 4)} for s, t in scored[:top_k]]
        except Exception:
            return []
