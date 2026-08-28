"""
diagnose_ollama_speed.py

Standalone benchmark for the actual bottleneck behind Prixon's timeouts: how
fast OLLAMA_MODEL really runs on THIS machine, at realistic prompt sizes --
not the misleading `ollama run "hello"` benchmark (near-zero prompt, one
output token, tells you almost nothing about a real 2000+ token request).

PHASE 1 sends progressively LARGER prompts with output pinned to 1 token,
which isolates prompt-processing ("prefill") speed at each size.

PHASE 2 pins the prompt small and sends progressively LONGER requested
outputs, which isolates generation ("decode") speed.

Both phases escalate small -> big and stop automatically the moment a call
exceeds --cutoff seconds, so a slow machine doesn't sit here for 10 minutes
proving what it's already proven at a smaller size.

Uses Ollama's own returned prompt_eval_count / eval_count and duration
fields for exact tokens/sec -- not an estimate.

Usage:
    python diagnose_ollama_speed.py
    python diagnose_ollama_speed.py --model qwen2.5:7b-instruct --cutoff 90
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import requests

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Roughly matches natural English prose density for this tokenizer family;
# good enough for building test prompts of a target size, not exact math.
_FILLER_SENTENCE = (
    "The user asked the assistant to check whether an application is currently "
    "running on this Windows computer and report back a short natural answer. "
)


def _build_prompt(target_chars: int) -> str:
    reps = max(1, target_chars // len(_FILLER_SENTENCE) + 1)
    return (_FILLER_SENTENCE * reps)[:target_chars]


def _call(base_url: str, model: str, system: str, user: str, num_predict: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 8192},
    }
    start = time.time()
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    wall = time.time() - start
    r.raise_for_status()
    data = r.json()
    data["_wall_seconds"] = wall
    return data


def _ns_to_s(ns: Optional[int]) -> float:
    return (ns or 0) / 1e9


def _report_row(label: str, prompt_chars: int, data: dict) -> dict:
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_s = _ns_to_s(data.get("prompt_eval_duration"))
    eval_count = data.get("eval_count", 0)
    eval_s = _ns_to_s(data.get("eval_duration"))
    total_s = _ns_to_s(data.get("total_duration")) or data.get("_wall_seconds", 0.0)
    prompt_tps = (prompt_eval_count / prompt_eval_s) if prompt_eval_s > 0 else 0.0
    eval_tps = (eval_count / eval_s) if eval_s > 0 else 0.0
    row = {
        "label": label, "prompt_chars": prompt_chars,
        "prompt_eval_count": prompt_eval_count, "prompt_eval_s": round(prompt_eval_s, 2),
        "prompt_tok_per_s": round(prompt_tps, 1),
        "eval_count": eval_count, "eval_s": round(eval_s, 2), "eval_tok_per_s": round(eval_tps, 1),
        "total_s": round(total_s, 2),
    }
    print(f"  {label:<10} prompt_tokens={prompt_eval_count:<6} prompt_eval={row['prompt_eval_s']:>6}s "
          f"({row['prompt_tok_per_s']:>7} tok/s)   output_tokens={eval_count:<4} "
          f"decode={row['eval_s']:>6}s ({row['eval_tok_per_s']:>7} tok/s)   total={row['total_s']:>6}s",
          flush=True)
    return row


SYSTEM = "You are a test harness. Reply with a single word."


def phase_prompt_scaling(base_url: str, model: str, cutoff: float, timeout: float) -> List[dict]:
    print("\n=== PHASE 1: prompt-processing speed at increasing prompt sizes (output pinned to 1 token) ===")
    sizes_chars = [200, 800, 2000, 4000, 8000, 16000, 32000]
    rows: List[dict] = []
    for chars in sizes_chars:
        prompt = _build_prompt(chars)
        label = f"~{chars}c"
        try:
            data = _call(base_url, model, SYSTEM, prompt, num_predict=1, timeout=timeout)
        except requests.exceptions.ReadTimeout:
            print(f"  {label:<10} TIMED OUT after {timeout}s -- stopping phase 1 here.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<10} FAILED: {exc}")
            break
        row = _report_row(label, chars, data)
        rows.append(row)
        if row["total_s"] > cutoff:
            print(f"  -> exceeded cutoff ({cutoff}s), stopping phase 1 here.")
            break
    return rows


def phase_output_scaling(base_url: str, model: str, cutoff: float, timeout: float) -> List[dict]:
    print("\n=== PHASE 2: generation speed at increasing output lengths (prompt pinned small) ===")
    prompt = _build_prompt(300)
    output_sizes = [10, 50, 100, 200, 400, 800]
    rows: List[dict] = []
    for n in output_sizes:
        label = f"{n}tok"
        try:
            data = _call(base_url, model, "Reply with natural filler text only, no explanations.",
                         prompt, num_predict=n, timeout=timeout)
        except requests.exceptions.ReadTimeout:
            print(f"  {label:<10} TIMED OUT after {timeout}s -- stopping phase 2 here.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<10} FAILED: {exc}")
            break
        row = _report_row(label, 300, data)
        rows.append(row)
        if row["total_s"] > cutoff:
            print(f"  -> exceeded cutoff ({cutoff}s), stopping phase 2 here.")
            break
    return rows


def recommend(prompt_rows: List[dict], output_rows: List[dict], target_budget: float) -> None:
    print("\n=== RECOMMENDATION ===")
    if not prompt_rows:
        print("  No successful calls at all -- check Ollama is running and the model is pulled.")
        return
    good_prompt = [r for r in prompt_rows if r["prompt_tok_per_s"] > 0]
    avg_prompt_tps = sum(r["prompt_tok_per_s"] for r in good_prompt) / max(1, len(good_prompt))
    good_eval = [r for r in output_rows if r["eval_tok_per_s"] > 0]
    avg_eval_tps = sum(r["eval_tok_per_s"] for r in good_eval) / max(1, len(good_eval)) if good_eval else 0.0
    print(f"  Measured prompt-processing speed: ~{avg_prompt_tps:.1f} tokens/sec")
    if avg_eval_tps:
        print(f"  Measured generation speed:        ~{avg_eval_tps:.1f} tokens/sec")
    if avg_prompt_tps > 0:
        # Leave ~40% of the budget for decode time so the estimate isn't
        # purely prefill-bound.
        safe_prompt_tokens = int(avg_prompt_tps * target_budget * 0.6)
        print(f"  For a ~{target_budget:.0f}s total budget, keep your input prompt under roughly "
              f"{safe_prompt_tokens} tokens (~{safe_prompt_tokens * 4} chars).")
    print("  Compare this against Prixon's real '[OLLAMA] prompt_chars=' log line (run Prixon with")
    print("  --debug) to see whether a given request actually fits your hardware's budget, and set")
    print("  OLLAMA_TIMEOUT_SECONDS / LLM_MAX_TOKENS / TOP_K_CANDIDATES in .env accordingly.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark real Ollama prompt/generation speed, small to big.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cutoff", type=float, default=60.0,
                     help="Stop escalating once a single call exceeds this many seconds "
                          "(default 60, matching Prixon's current OLLAMA_TIMEOUT_SECONDS).")
    ap.add_argument("--call-timeout", type=float, default=300.0,
                     help="Hard timeout per HTTP call so a stuck request doesn't hang forever (default 300s).")
    ap.add_argument("--budget", type=float, default=15.0,
                     help="Target end-to-end response budget in seconds for the recommendation (default 15).")
    args = ap.parse_args()

    print(f"Ollama at {args.base_url}, model={args.model}")
    print(f"Escalating small -> big, stopping automatically once a call exceeds {args.cutoff}s.")

    try:
        requests.get(f"{args.base_url}/api/tags", timeout=5)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCan't reach Ollama at {args.base_url}: {exc}")
        print("Start Ollama first and make sure the model is pulled (`ollama pull <model>`).")
        return

    prompt_rows = phase_prompt_scaling(args.base_url, args.model, args.cutoff, args.call_timeout)
    output_rows = phase_output_scaling(args.base_url, args.model, args.cutoff, args.call_timeout)
    recommend(prompt_rows, output_rows, args.budget)


if __name__ == "__main__":
    main()