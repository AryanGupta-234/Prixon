# Prixon bulk training corpus

This branch builds a large post-training corpus without modifying `main` and without embedding machine-specific paths, credentials, provider URLs, or base-model choices in Python code.

The pipeline combines Prixon's existing action catalog with external agent/tool-use datasets and converts the local catalog into the same NLU contract used by the runtime.

## Why this matches Prixon's architecture

`nlu.resolve()` receives a system prompt, the user's request, recent conversation state, and an allow-list of candidate targets. `training/build_dataset.py` reproduces that shape from the existing `data/windows_automation_10000.jsonl` catalog instead of inventing a separate tool API.

Local examples are expanded into three tracks when enabled in `training/config/dataset.yaml`:

- `prixon_nlu`: correct target selection from a candidate allow-list.
- `prixon_nlu_negative`: correct action intentionally removed, teaching the model not to invent unavailable capabilities.
- `prixon_context`: recent-target/reference resolution using Prixon's conversation-state shape.

External sources add broader function calling and agent behavior. Dataset IDs, source weights, split ratios, source enablement, streaming, paths, assistant name, and output paths are all controlled by YAML/environment variables.

## Build on a cloud machine

```bash
pip install -r training/requirements.txt
python training/build_dataset.py
```

Output defaults to:

```text
training/output/prixon_train.jsonl
training/output/prixon_validation.jsonl
training/output/prixon_eval.jsonl
training/output/manifest.json
```

Generated data is ignored by Git because external corpora can be very large. Build it on the same cloud instance used for fine-tuning so Hugging Face sources can be streamed directly.

## Local Prixon data only

```bash
python training/build_dataset.py --no-external
```

## Selected sources only

```bash
python training/build_dataset.py --source prixon_action_catalog --source hermes_function_calling
```

## Configuration

Edit `training/config/dataset.yaml` rather than Python source. Environment overrides include the names configured there, such as:

```text
ASSISTANT_NAME
ASSISTANT_DATA_PATH
PRIXON_DATASET_OUTPUT
PRIXON_DATASET_SEED
```

The default external mix currently references open agent/function-calling corpora. The much larger terminal-agent source is present but disabled by default because it is substantially heavier to stream and process. Enable it in YAML when desired.

Review each third-party dataset's license/terms before redistributing a derived corpus or model. The pipeline streams those datasets and does not vendor their raw files into this repository.
