#!/usr/bin/env python3
"""Cloud-first QLoRA SFT entry point for Prixon."""
from __future__ import annotations

import argparse
import json
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


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict) and row.get("messages"):
                yield row


def latest_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    checkpoints = []
    for child in output_dir.glob("checkpoint-*"):
        if child.is_dir():
            try:
                checkpoints.append((int(child.name.split("-")[-1]), child))
            except ValueError:
                continue
    return str(max(checkpoints, key=lambda item: item[0])[1]) if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prixon with QLoRA in the cloud")
    parser.add_argument("--config", default=str(ROOT / "training" / "config" / "training.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    cloud_cfg = cfg["cloud"]

    model_name = str(env_value(model_cfg.get("name_env"), model_cfg.get("name_default")))
    max_seq_length = int(env_value(model_cfg.get("max_seq_length_env"), model_cfg.get("max_seq_length_default")))
    load_in_4bit = bool_value(model_cfg.get("load_in_4bit", True)) and not args.no_4bit
    seed = int(env_value(train_cfg.get("seed_env"), train_cfg.get("seed_default", 3407)))

    train_file = resolve_path(str(env_value(data_cfg.get("train_env"), data_cfg.get("train_default"))))
    validation_file = resolve_path(str(env_value(data_cfg.get("validation_env"), data_cfg.get("validation_default"))))
    output_dir = resolve_path(str(env_value(train_cfg.get("output_dir_env"), train_cfg.get("output_dir_default"))))

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")
    if not validation_file.exists():
        raise FileNotFoundError(f"Validation dataset not found: {validation_file}")

    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install training/requirements.txt and the cloud training dependencies first.") from exc

    print(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["rank"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth" if bool_value(train_cfg.get("gradient_checkpointing", True)) else False,
        random_state=seed,
    )

    train_rows = list(load_jsonl(train_file))
    validation_rows = list(load_jsonl(validation_file))
    if not train_rows or not validation_rows:
        raise RuntimeError("Training and validation datasets must both contain examples.")

    train_dataset = Dataset.from_list(train_rows)
    validation_dataset = Dataset.from_list(validation_rows)

    sft_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["epochs"]),
        per_device_train_batch_size=int(train_cfg["train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        lr_scheduler_type=str(train_cfg["scheduler"]),
        optim=str(train_cfg["optimizer"]),
        report_to=str(train_cfg["report_to"]),
        seed=seed,
        dataset_text_field="text",
        max_length=max_seq_length,
    )

    try:
        sft_args = SFTConfig(**sft_kwargs)
    except TypeError:
        sft_kwargs.pop("max_length", None)
        sft_kwargs.pop("dataset_text_field", None)
        sft_args = SFTConfig(**sft_kwargs)

    def formatting_func(examples):
        return [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in examples["messages"]
        ]

    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        formatting_func=formatting_func,
    )

    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    resume_from = latest_checkpoint(output_dir) if args.resume else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    push = bool_value(env_value(cloud_cfg.get("push_to_hub_env"), cloud_cfg.get("push_to_hub_default", False)))
    hub_repo = str(env_value(cloud_cfg.get("hub_repo_env"), cloud_cfg.get("hub_repo_default", ""))).strip()
    if push:
        if not hub_repo:
            raise RuntimeError("Hub repository is required when cloud.push_to_hub is enabled.")
        os.environ.setdefault("HF_TOKEN", str(env_value(cloud_cfg.get("hf_token_env"), "")))
        trainer.push_to_hub(hub_repo)

    summary = {
        "model": model_name,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "hub_push": push,
        "hub_repo": hub_repo if push else None,
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
