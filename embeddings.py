"""Local semantic embeddings for Tier 2 (spec section 4).

TF-IDF Tier 2 is fast and dependency-free, but it is bag-of-words: paraphrases
such as "make it quieter" and "lower the volume" may share few tokens. This
module adds a local fastembed/ONNX embedding signal for that gap.

The embedding model loads in a background thread so startup never blocks. A
failed download/load is cached with a one-hour retry cooldown, but a missing
Python dependency is NOT treated as permanent: if fastembed is installed after
a previous run, the next launch will retry normally.
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
_RETRY_COOLDOWN_SECONDS = 60 * 60


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
        pass


class SemanticIndex:
    """Wraps a fastembed model.

    Loading happens in a background thread. ``ready`` remains False until the
    model and the action corpus have both been embedded successfully.
    """

    def __init__(self, groups: Dict[str, "ActionGroup"]):  # noqa: F821
        self._groups = groups
        self._model = None
        self._vectors: Optional[Dict[str, list]] = None
        self.ready = False
        self.unavailable_reason: Optional[str] = None
        self.loading = False

        status = _load_status()
        last_failed = status.get("last_failed_at", 0)

        # Do not honor the old ``permanently_unavailable`` flag. Earlier
        # versions wrote that flag when fastembed was missing, which meant
        # installing fastembed later could never recover without manually
        # deleting data/embedding_status.json.
        if last_failed and (time.time() - last_failed) < _RETRY_COOLDOWN_SECONDS:
            remaining_min = max(
                1, int((_RETRY_COOLDOWN_SECONDS - (time.time() - last_failed)) / 60)
            )
            self.unavailable_reason = (
                f"retry cooldown ({remaining_min}m left) after: "
                f"{status.get('reason', 'unknown error')}"
            )
            return

        self.loading = True
        threading.Thread(target=self._load, daemon=True, name="prixon-embeddings").start()

    def _load(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            self.loading = False
            self.unavailable_reason = "fastembed not installed (pip install fastembed)"
            # Dependency absence is recoverable. Never mark it permanent.
            _save_status({"last_failed_at": time.time(), "reason": self.unavailable_reason})
            return

        try:
            model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        except Exception as exc:
            self.loading = False
            self.unavailable_reason = f"model load/download failed: {exc}"
            _save_status({"last_failed_at": time.time(), "reason": self.unavailable_reason})
            return

        try:
            targets = list(self._groups.keys())
            texts = [
                f"{self._groups[t].target_name}. "
                + " ".join(self._groups[t].example_phrasings[:4])
                for t in targets
            ]
            vectors = list(model.embed(texts))
            self._model = model
            self._vectors = dict(zip(targets, vectors))
            self.ready = True
            self.loading = False
            self.unavailable_reason = None
            _save_status({
                "permanently_unavailable": False,
                "last_success_at": time.time(),
            })
        except Exception as exc:
            self.loading = False
            self.unavailable_reason = f"corpus embedding failed: {exc}"
            _save_status({"last_failed_at": time.time(), "reason": self.unavailable_reason})

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return semantic candidates, or [] while unavailable/loading."""
        if not self.ready or self._model is None or not self._vectors:
            return []
        try:
            import numpy as np
            q_vec = list(self._model.embed([query]))[0]
            q_norm = float(np.linalg.norm(q_vec)) + 1e-9
            scored = []
            for target, vec in self._vectors.items():
                sim = float(
                    np.dot(q_vec, vec)
                    / (q_norm * (float(np.linalg.norm(vec)) + 1e-9))
                )
                scored.append((sim, target))
            scored.sort(reverse=True)
            return [{"target": t, "score": round(s, 4)} for s, t in scored[:top_k]]
        except Exception:
            return []
