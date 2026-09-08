# Recent-history architecture review

Reviewed branch: `test/pre-training-audit`  
Compared with: `main` and `origin/training/agi-dataset`

## Confirmed findings

### Partial routing migration — medium risk

Recent commits replaced TF-IDF candidate selection with FastEmbed shortlist retrieval plus LLM resolution. `data_store.ActionIndex.search()` remains as an intentional no-op compatibility shim, and `main.py` still contains migration scaffolding describing the retired full-catalog fallback. `tier2.classify()` and `tier2.classify_semantic()` remain implemented but have no call sites in the runtime request path. Only its app/entity extraction helpers are live.

This is not currently an execution-security bug: the active path remains bounded by the action catalog. It is a maintenance risk because a future caller can mistake the no-op search shim for working retrieval.

### Duplicate tool-registry representations — low risk

`tools.TOOLS`/`get_tool()` define a static registry, while the active `CapabilityRegistry` derives capabilities from `ActionIndex.groups`. No runtime call site consumes `tools.get_tool()`. The latter is authoritative for dispatch, so the duplicate static mapping can become stale without a test catching it.

Removing it would alter an importable interface; it was not removed in this audit-only change set. New benchmark coverage instead treats catalog outcomes as external expectations.

### Training boundary duplication — high architectural risk

`origin/training/agi-dataset` contains a dedicated `training/` pipeline with its own configuration, dataset build, evaluation, export, and cloud training entry points. The runtime branch separately contains `cognition/training_log.py` and `scripts/` training helpers. `main.py` writes verified runtime interactions to the latter logger.

This conflicts with the intended branch separation and risks divergent data formats and accidental training coupling. Moving either implementation would be a cross-branch design decision, so it is documented rather than changed.

### Behavior changes without broad routing tests — medium risk

The recent history contains multiple reference/entity and semantic routing refactors. Before the audit branch, tests covered only experience and pattern persistence, not the request pipeline, unsafe requests, provider recovery, or verification outcomes. The audit regression tests and the new 200-case implementation-independent benchmark close the measurement gap, but the benchmark has not yet been run through a non-executing Prixon adapter.

### Tracked interpreter artifacts on main — low risk

`main` historically tracks `__pycache__/*.pyc`; this branch ignores and deletes them. This is repository hygiene inconsistency, not runtime behavior.

## Changes intentionally not made

No runtime behavior was changed as part of the history review or benchmark. The confirmed issues above require either compatibility decisions (removing public legacy helpers) or cross-branch ownership decisions (training pipeline consolidation). Neither should be changed solely to satisfy a benchmark.

