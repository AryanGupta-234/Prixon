"""Turns real, verified Qwen interactions into future fine-tuning data.

This is NOT training. It only accumulates data -- see Section 16 of the
handoff spec: "This should NOT immediately become training the model. It is
external memory around the model." Training itself is a separate, occasional,
opt-in step (scripts/kaggle_train.py), run manually or on a schedule against
whatever has accumulated here.

Why this file exists instead of just fine-tuning on data/windows_automation_
10000.jsonl: that file is `utterance -> action name` pairs, the same shape
used to seed FastEmbed/TF-IDF retrieval. Fine-tuning a model on that shape
teaches it "phrase X maps to command Y" -- i.e. re-creates the rigid
classifier behavior Section 1 explicitly says Prixon should NOT be. What is
actually worth fine-tuning on is the full reasoning shape: a real utterance,
the real situational context it was answered in, and the exact JSON schema
output (match_target/confidence/intent/parameters/reference/reply/reason)
that turned out to be CORRECT (independently verified, not just "didn't
crash"). That is what this module captures.

Quality bar, deliberately strict:
  - Only interactions that actually reached Qwen (tier == "tier3-qwen-
    semantic"). Tier0/1/2 and environment-first resolutions are already
    correct by construction (regex/live process check) -- there's no Qwen
    behavior to reinforce there.
  - Only interactions with confidence in {high, medium}. Qwen's own "low"/
    "none" calls are exactly the failures we don't want to teach it to repeat.
  - Only interactions where the resulting action was independently VERIFIED
    (verification.py actually confirmed it, not merely "no exception").
Feeding in anything less than that would let Prixon learn from its own
mistakes with high confidence, which is worse than not fine-tuning at all.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "qwen_training_log.jsonl")
_lock = threading.Lock()


def log_verified_interaction(user_text: str, tier: str, output_raw: Dict[str, Any], verified_ok: bool) -> None:
    if tier != "tier3-qwen-semantic" or not verified_ok:
        return
    confidence = str(output_raw.get("confidence", "none")).lower()
    if confidence not in {"high", "medium"}:
        return
    prompt_context = output_raw.get("_prompt_context")
    if not prompt_context:
        return
    output = {k: v for k, v in output_raw.items() if k != "_prompt_context"}
    record = {
        "timestamp": time.time(),
        "input": {"user_text": user_text, "context": prompt_context},
        "output": output,
        "verified_ok": True,
    }
    try:
        with _lock:
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            with open(PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging failure should never break the actual assistant turn


def count() -> int:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0
