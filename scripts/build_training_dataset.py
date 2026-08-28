"""Turn data/qwen_training_log.jsonl (see cognition/training_log.py) into a
ChatML-style instruction-tuning dataset ready for the Kaggle QLoRA kernel.

Deliberately a SEPARATE step from logging, so the format used for training
can change over time without touching the always-on runtime logger.

Run locally (no GPU needed):
    python scripts/build_training_dataset.py [--min-examples 200] [--out data/qwen_sft_dataset.jsonl]

This mirrors the exact prompt shape nlu.py sends at inference time (same
SYSTEM_PROMPT, same context->prompt construction), so what the model is
fine-tuned on matches what it will actually be asked at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nlu  # noqa: E402  (path insert above must run first)
from cognition import training_log  # noqa: E402


def build(min_examples: int = 1) -> list:
    examples = []
    try:
        with open(training_log.PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context = rec.get("input", {}).get("context")
                output = rec.get("output")
                if not context or not output:
                    continue
                user_prompt = (
                    "Understand the user's request. Resolve references using recent conversation and live "
                    "assistant context. Select at most ONE allow-listed target. Extract useful parameters, "
                    "but never invent a capability.\n\n"
                    + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
                )
                examples.append({
                    "messages": [
                        {"role": "system", "content": nlu.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
                    ]
                })
    except FileNotFoundError:
        pass
    if len(examples) < min_examples:
        print(f"Only {len(examples)} verified examples logged so far "
              f"(threshold: {min_examples}). Keep using Prixon to accumulate more "
              f"before training is worthwhile -- see cognition/training_log.py.")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-examples", type=int, default=200,
                         help="Warn (but still write) if fewer than this many examples exist.")
    parser.add_argument("--out", default=os.path.join("data", "qwen_sft_dataset.jsonl"))
    args = parser.parse_args()

    examples = build(args.min_examples)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} training examples to {args.out}")


if __name__ == "__main__":
    main()
