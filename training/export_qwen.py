#!/usr/bin/env python3
"""Export a trained Unsloth adapter to merged weights and/or GGUF."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def env_value(name: Optional[str], default: Any = None) -> Any:
    if name:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "training" / "config" / "training.yaml"))
    parser.add_argument("--adapter", required=True, help="Path to the trained adapter directory")
    parser.add_argument("--quantization")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    export_cfg = cfg["export"]
    quantization = args.quantization or str(env_value(export_cfg.get("gguf_quantization_env"), export_cfg.get("gguf_quantization_default")))
    merged_dir = resolve_path(str(env_value(export_cfg.get("merged_dir_env"), export_cfg.get("merged_dir_default"))))
    gguf_dir = resolve_path(str(env_value(export_cfg.get("gguf_dir_env"), export_cfg.get("gguf_dir_default"))))

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Install the cloud training requirements first.") from exc

    adapter = resolve_path(args.adapter)
    if not adapter.exists():
        raise FileNotFoundError(adapter)

    # adapter_config.json contains the base-model identity, so the exporter
    # does not duplicate the base model name in source code or CLI flags.
    print(f"Loading adapter: {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=int(cfg["model"].get("max_seq_length_default", 8192)),
        load_in_4bit=True,
    )

    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting merged model to: {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    print(f"Exporting GGUF ({quantization}) to: {gguf_dir}")
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quantization)
    print("Export complete.")


if __name__ == "__main__":
    main()
