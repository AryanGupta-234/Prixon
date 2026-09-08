# Prixon bulk training corpus

This branch builds a large post-training corpus without modifying `main` and without embedding machine-specific paths, credentials, provider URLs, or base-model choices in Python code.

The pipeline combines Prixon's existing action catalog with external agent/tool-use datasets and converts them into the same chat/tool contract used by the training stack.

## Why this matches Prixon's architecture

Prixon's runtime already separates NLU, context, memory, goal handling, execution, tool routing, and verification. The dataset builder therefore does **not** invent a second hardcoded tool registry. The local catalog is read from the existing JSONL data and converted into candidate-selection, negative, and reference-resolution examples.

Local examples can produce three tracks:

- `prixon_nlu`: select the correct allow-listed capability.
- `prixon_nlu_negative`: refuse to invent an unavailable capability.
- `prixon_context`: resolve references against recent conversation state.

External sources provide broader function calling, terminal-agent behavior, and multi-tool trajectories. All source identifiers, limits, weights, split ratios, paths, enablement, and output locations live in `training/config/dataset.yaml` and can be overridden through the configured environment variables.

## Current bulk mix

The configured mix uses:

- Prixon's existing Windows automation corpus.
- xLAM function-calling data.
- Hermes function-calling data.
- NVIDIA Nemotron function-calling pivot data.
- NVIDIA Nemotron terminal-agent data.
- The Qwen3 subset of Toucan-1.5M, capped by configuration rather than copied into this repository.

Toucan-1.5M contains about 1.65M trajectories across multiple configurations and is Apache-2.0 licensed; the configured Qwen3 subset is streamed rather than vendored locally. The source contains multi-turn and multi-tool trajectories, which is useful for agentic SFT. citeturn2search0turn2search1

NVIDIA's function-calling pivot dataset is 9,620 rows and CC-BY-4.0; its terminal counterpart is about 1.37 GB and also CC-BY-4.0. citeturn1search0turn1search2

## Build on a cloud GPU machine

```bash
pip install -r training/requirements.txt
python training/build_dataset.py
```

The builder streams external Hugging Face sources, so the raw external corpora do not need to be committed to Git.

Output:

```text
training/output/prixon_train.jsonl
training/output/prixon_validation.jsonl
training/output/prixon_eval.jsonl
training/output/manifest.json
```

Generated output is ignored by Git because the resulting corpus can be very large.

## Local Prixon data only

```bash
python training/build_dataset.py --no-external
```

## Selected sources only

```bash
python training/build_dataset.py --source prixon_action_catalog --source toucan_qwen3
```

## Generic external dataset conversion

For an external dataset whose chat trajectory is stored as a JSON/string field, the generic converter can normalize it without adding dataset-specific logic to Prixon's Python code:

```bash
python training/prepare_external.py \
  --dataset <DATASET_ID> \
  --subset <OPTIONAL_SUBSET> \
  --split train \
  --messages-field messages \
  --tools-field available_tools \
  --output training/cache/external.jsonl \
  --streaming
```

The values are intentionally command-line/config inputs rather than constants in the converter.

## Configuration

Edit `training/config/dataset.yaml` instead of changing Python code. The dataset pipeline is designed so the same code can be reused with different datasets, source weights, limits, assistant names, output locations, and train/validation/eval ratios.

Review each third-party dataset's license and terms before redistributing a derived corpus or model. The pipeline streams third-party data and does not vendor their raw files into this repository.
