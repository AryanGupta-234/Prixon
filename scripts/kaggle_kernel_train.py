"""Runs ON KAGGLE (not locally) -- QLoRA fine-tune of qwen2.5:7b-instruct on
Prixon's verified-interaction log, using a free T4 (16GB VRAM) GPU session.

This is pushed and executed remotely by scripts/trigger_kaggle_training.py.
It is not part of Prixon's runtime; it only ever runs occasionally, off-box,
per Section 26 Pass 13 ("Do NOT fine-tune after every interaction").

Expects a Kaggle dataset attached at /kaggle/input/prixon-training-log
containing qwen_sft_dataset.jsonl (built locally by
scripts/build_training_dataset.py -- see that file for the exact format:
ChatML {"messages":[{system},{user},{assistant}]} triples matching nlu.py's
real prompt shape).

Writes the resulting LoRA adapter to /kaggle/working/adapter/, which Kaggle
exposes as the kernel's output for trigger_kaggle_training.py to pull back
down afterward.
"""
import json
import os

# Unsloth is what makes QLoRA fine-tuning of a 7B model fit comfortably in a
# free T4's 16GB VRAM. Install first thing in the kernel -- Kaggle base
# images don't ship it.
os.system("pip install -q unsloth")

from unsloth import FastLanguageModel  # noqa: E402
from trl import SFTTrainer  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from datasets import load_dataset  # noqa: E402

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # pre-quantized, fastest to load on a free T4
DATASET_PATH = "/kaggle/input/prixon-training-log/qwen_sft_dataset.jsonl"
OUTPUT_DIR = "/kaggle/working/adapter"
MAX_SEQ_LEN = 2048  # generous headroom over the ~800-1000 token prompts nlu.py actually sends


def format_chatml(example):
    """Unsloth/TRL want a single rendered text field, not a raw messages list."""
    msgs = example["messages"]
    text = ""
    for m in msgs:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return {"text": text}


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(format_chatml)
    print(f"Training on {len(dataset)} verified real interactions.")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=3, learning_rate=2e-4, fp16=not model.config.torch_dtype == "bfloat16",
            bf16=model.config.torch_dtype == "bfloat16", logging_steps=10, save_strategy="epoch",
            optim="adamw_8bit", report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
        json.dump({"base_model": MODEL_NAME, "examples": len(dataset)}, f)
    print(f"Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
