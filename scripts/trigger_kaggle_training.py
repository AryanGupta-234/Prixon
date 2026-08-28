"""Runs LOCALLY. Automates one full training cycle against Kaggle's free GPU
via the real, documented Kaggle API (`pip install kaggle`) -- push dataset,
push+run kernel, poll until done, pull the resulting adapter back down.

This is the piece that can be put on a schedule (cron / Windows Task
Scheduler) to run "at intervals", e.g. weekly, as requested. IMPORTANT
CAVEAT: Google Colab's free tier has no equivalent public trigger API --
its only scheduling option is the interactive, Drive-based "Schedule"
button (you'd have to open the notebook UI yourself). Kaggle is the actual
free platform this can be automated against; that's why this pipeline
targets Kaggle, not Colab.

Requires one-time setup:
    pip install kaggle
    Place your kaggle.json API token at ~/.kaggle/kaggle.json
    (Kaggle account -> Settings -> API -> Create New Token)

Usage:
    python scripts/trigger_kaggle_training.py --min-examples 200

Safe to run repeatedly / on a schedule: it no-ops (with a clear message) if
there isn't enough new verified data yet, so a cron job calling this weekly
won't burn GPU hours training on the same handful of examples over and over.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

DATASET_SLUG_NAME = "prixon-training-log"
KERNEL_SLUG_NAME = "prixon-qwen-qlora"
STATE_PATH = os.path.join(REPO_ROOT, "data", "kaggle_training_state.json")


def _load_state() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_trained_example_count": 0}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _current_example_count(dataset_path: str) -> int:
    if not os.path.exists(dataset_path):
        return 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-examples", type=int, default=200,
                         help="Skip training if fewer than this many NEW verified examples "
                              "have accumulated since the last successful run.")
    parser.add_argument("--username", required=True, help="Your Kaggle username (for slugs).")
    parser.add_argument("--adapter-out", default=os.path.join(REPO_ROOT, "data", "adapters", "latest"))
    args = parser.parse_args()

    # 1. Rebuild the dataset from whatever has been verified-logged since
    #    Prixon last ran (cognition/training_log.py). Cheap, local, no GPU.
    dataset_path = os.path.join(REPO_ROOT, "data", "qwen_sft_dataset.jsonl")
    subprocess.run([sys.executable, os.path.join(HERE, "build_training_dataset.py"),
                     "--min-examples", "1", "--out", dataset_path], check=True)

    state = _load_state()
    current = _current_example_count(dataset_path)
    new_examples = current - state.get("last_trained_example_count", 0)
    if new_examples < args.min_examples:
        print(f"Only {new_examples} new verified examples since the last training run "
              f"(need {args.min_examples}). Skipping this cycle -- not worth a GPU session yet.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("pip install kaggle first, and place your API token at ~/.kaggle/kaggle.json")
        return

    api = KaggleApi()
    api.authenticate()

    # 2. Push the dataset (new version) that the kernel will read as input.
    dataset_dir = os.path.join(REPO_ROOT, "data", "_kaggle_dataset_staging")
    os.makedirs(dataset_dir, exist_ok=True)
    staged = os.path.join(dataset_dir, "qwen_sft_dataset.jsonl")
    with open(dataset_path, "rb") as src, open(staged, "wb") as dst:
        dst.write(src.read())
    dataset_meta = {"title": DATASET_SLUG_NAME, "id": f"{args.username}/{DATASET_SLUG_NAME}",
                     "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_meta, f)
    try:
        api.dataset_create_version(dataset_dir, "new verified interactions", quiet=False)
    except Exception:
        api.dataset_create_new(dataset_dir, public=False, quiet=False)

    # 3. Push the training kernel and let Kaggle run it on a free GPU.
    kernel_dir = os.path.join(REPO_ROOT, "data", "_kaggle_kernel_staging")
    os.makedirs(kernel_dir, exist_ok=True)
    with open(os.path.join(HERE, "kaggle_kernel_train.py")) as src, \
         open(os.path.join(kernel_dir, "kaggle_kernel_train.py"), "w") as dst:
        dst.write(src.read())
    kernel_meta = {
        "id": f"{args.username}/{KERNEL_SLUG_NAME}",
        "title": KERNEL_SLUG_NAME,
        "code_file": "kaggle_kernel_train.py",
        "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{args.username}/{DATASET_SLUG_NAME}"],
    }
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_meta, f)
    api.kernels_push(kernel_dir)
    print("Kernel pushed and running on Kaggle. Polling for completion...")

    # 4. Poll until the run finishes (Kaggle free GPU sessions can run up to
    #    ~12h; QLoRA on a few hundred short examples should be far quicker).
    slug = f"{args.username}/{KERNEL_SLUG_NAME}"
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        status = api.kernels_status(slug)
        state_str = getattr(status, "status", str(status))
        print(f"  status: {state_str}")
        if state_str in ("complete", "error", "cancelAcknowledged"):
            break
        time.sleep(60)

    if state_str != "complete":
        print(f"Training run ended with status={state_str}, not pulling an adapter.")
        return

    # 5. Pull the resulting adapter back down.
    os.makedirs(args.adapter_out, exist_ok=True)
    api.kernels_output(slug, path=args.adapter_out)
    print(f"Adapter pulled to {args.adapter_out}")

    state["last_trained_example_count"] = current
    state["last_run_at"] = time.time()
    _save_state(state)
    print("Remember: this adapter changes MODEL BEHAVIOR, not runtime speed (see chat discussion). "
          "Evaluate it against a held-out set of real requests before merging it into the local "
          "Ollama model -- see scripts/README.md for the merge/quantize step.")


if __name__ == "__main__":
    main()
