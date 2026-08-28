# Occasional fine-tuning pipeline (Pass 13 — not the current priority)

This is **separate from Prixon's runtime** and should stay that way. Prixon's
speed problem (the 60s Ollama timeout) is fixed by prompt/context size, not
training — see `nlu.py`, `context_engine.py`, `concepts.py`. Nothing here
runs while Prixon is answering a request.

## The loop

```
Prixon runs normally
  -> cognition/training_log.py logs only VERIFIED, high/medium-confidence
     Qwen interactions, in the real inference schema (data/qwen_training_log.jsonl)
       |
       v  (run manually or on a schedule, e.g. weekly cron)
scripts/trigger_kaggle_training.py
  -> scripts/build_training_dataset.py   (local, no GPU: JSONL -> ChatML pairs)
  -> pushes dataset + kaggle_kernel_train.py to Kaggle via the real Kaggle API
  -> Kaggle runs QLoRA fine-tuning on a free T4 GPU (via Unsloth)
  -> pulls the resulting LoRA adapter back to data/adapters/latest/
       |
       v  (manual step, deliberately not automatic)
Evaluate the adapter against real held-out requests.
Merge + quantize (e.g. via llama.cpp) into a GGUF Prixon can load in Ollama.
```

## Why Kaggle, not Colab, for the automated part

Colab's free tier only has an interactive, Drive-based "Schedule" button —
opening the notebook UI yourself is unavoidable, and there's no public API
to trigger it. (Colab *Enterprise*, a paid GCP product, does have a real
scheduling API — but that's not the free tier.) Kaggle's `kaggle` CLI/Python
API genuinely supports pushing code, running it on a GPU, and pulling
results back, all from a script — that's what `trigger_kaggle_training.py`
uses. If you'd rather run manually in Colab occasionally, `kaggle_kernel_
train.py` will run there too (swap the Unsloth install line if needed) —
you'd just be doing steps 2-4 by hand in the notebook UI.

## Why the raw 10,012-example catalog isn't used for this

`data/windows_automation_10000.jsonl` is `utterance -> action name` pairs —
the same shape used to seed embeddings/TF-IDF. Fine-tuning on that shape
would teach Qwen "phrase X maps to command Y", i.e. the rigid classifier
behavior the handoff spec explicitly says Prixon should NOT be (Section 1).
`cognition/training_log.py` instead captures the full reasoning shape: a
real utterance, the real situational context, and a real *verified-correct*
JSON output — matching exactly what `nlu.py` asks Qwen for at inference
time. That's what's actually worth spending GPU hours on.

## Quality gates already built in

- Only interactions that reached Qwen (`tier3-qwen-semantic`) are logged —
  deterministic tiers (regex/live-process-check) have no Qwen behavior to
  reinforce.
- Only `confidence in {high, medium}` — Qwen's own low-confidence guesses
  are exactly what shouldn't be reinforced.
- Only independently **verified** outcomes (real process-launch/close
  checks — see `verification.py`), not "didn't raise an exception".
- `trigger_kaggle_training.py` no-ops if fewer than `--min-examples` (default
  200) *new* verified examples have accumulated since the last run, so a
  scheduled weekly cron job won't burn GPU hours retraining on the same
  handful of examples.

## What this does and doesn't fix

Fine-tuning changes what Qwen *knows implicitly* and how it *behaves* — it
does not change inference speed on this CPU-only machine. A LoRA-tuned
7B model runs at the same tok/s as the base model. The payoff here is a
model that (eventually) needs a shorter, less-explained prompt to behave
correctly — which *indirectly* helps latency — not a training-side latency
fix by itself.
