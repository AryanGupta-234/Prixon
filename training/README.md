# Prixon bulk training + QLoRA pipeline

This branch contains the cloud-first training pipeline for turning Prixon's existing action data plus external agent/tool-use corpora into a configurable SFT dataset for Qwen-family models.

## Architecture alignment

Prixon's runtime separates NLU, context, memory, goal handling, execution, tool routing, and verification. The dataset builder mirrors the runtime contract instead of inventing an unrelated tool vocabulary. The local `data/windows_automation_10000.jsonl` corpus is read as the source of truth for Prixon capabilities, and the system prompt is loaded from the runtime NLU module by configuration. The existing tool router also derives capabilities from the same action catalog rather than a second manually maintained registry. 

## Dataset sources

The default YAML mix contains:

- Prixon's existing Windows automation corpus.
- xLAM function-calling (60K rows on Hugging Face).
- Hermes function-calling.
- NVIDIA Nemotron function-calling pivot data.
- NVIDIA Nemotron terminal-agent data (31K rows on the current release).
- The Qwen3 subset of Agent-Ark/Toucan-1.5M, capped by YAML so it is streamed rather than vendored.

Toucan-1.5M currently reports 1,646,546 total trajectories across configurations and includes multi-turn, multi-round, sequential and parallel tool calls. The dataset is Apache-2.0. citeturn951248search3

The current xLAM parsed source contains 60,000 training rows and is CC-BY-4.0. citeturn951248search0

The current NVIDIA Nemotron terminal dataset exposes 31,111 samples and is CC-BY-4.0. citeturn951248search1turn951248search2

## Build the corpus on a cloud GPU

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
python training/build_dataset.py
python training/evaluate_dataset.py training/output/prixon_train.jsonl training/output/prixon_validation.jsonl training/output/prixon_eval.jsonl
```

The builder streams external Hugging Face sources. Generated output is intentionally ignored by Git because it can become very large.

### Local Prixon data only

```bash
python training/build_dataset.py --no-external
```

### Select sources

```bash
python training/build_dataset.py --source prixon_action_catalog --source toucan_qwen3
```

### Change source sizes/weights

Edit only `training/config/dataset.yaml`, for example `max_examples`, `weight`, `enabled`, ratios, or output locations. Do not modify the Python pipeline for normal dataset experiments.

## QLoRA training

After the dataset build and validation:

```bash
python training/train_qwen.py
```

To resume from the latest checkpoint:

```bash
python training/train_qwen.py --resume
```

The base model, sequence length, LoRA settings, optimizer, learning rate, batch/accumulation settings, seed, paths, and Hub behavior come from `training/config/training.yaml` or its configured environment variables.

The default base model is `Qwen/Qwen2.5-7B-Instruct` and is only a configuration default, not embedded in the training implementation.

## Export for Ollama

After training, export the adapter to merged weights and GGUF:

```bash
python training/export_qwen.py --adapter training/runs/prixon-qwen/adapter
```

The GGUF quantization and export directories are configurable in `training/config/training.yaml`.

## Hugging Face authentication

For private/gated data or model pushes, provide credentials through environment variables such as `HF_TOKEN`; never put tokens in source code or YAML committed to the repository.

## Licensing

The repository does not vendor third-party raw datasets. Before publishing a derived dataset or model, review the current license/terms of every enabled source and preserve required attribution. xLAM is currently listed as CC-BY-4.0. citeturn951248search0 Nemotron terminal/function-calling releases are currently listed as CC-BY-4.0. citeturn951248search1turn951248search9 Toucan-1.5M is currently listed as Apache-2.0. citeturn951248search3
