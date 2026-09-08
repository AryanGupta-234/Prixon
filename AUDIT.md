# Prixon pre-training production audit

Audit branch: `test/pre-training-audit`  
Starting commit: `0e2b2d79a73193f3c309ce540dbf3b17ce04979f`  
Python tested: 3.11.9 on Windows

## Architecture

`main.py` is the CLI entry point. It initializes the catalog (`ActionIndex`),
state, unified memory, experience/pattern stores, capability registry/tool
router, semantic index, and background system agent. `--healthcheck` uses
`diagnostics.py` and exits without entering the REPL.

The request path is: small talk -> `context_engine.route` -> direct live-app
status or deterministic reference resolution -> FastEmbed shortlist ->
`nlu.resolve` -> `brain.router` provider abstraction -> allow-listed group ->
confirmation policy -> `ToolRouter` -> executor/tools -> verification ->
state/memory/experience/pattern updates. The provider abstraction is
`LLMProvider`, with Ollama local and optional Cerebras, Groq, and Hugging Face
providers. Configuration is environment-backed in `config.py`; secrets are
read from environment and are not logged.

`data_store.py` owns the action catalog and normalizes user text. `embeddings.py`
loads FastEmbed in a background thread and returns a bounded semantic shortlist.
`context_engine.py` combines live system state, goal bias, references, memories,
and semantic candidates. `reference_resolver.py` resolves recent target/app
references; `goal_engine.py` supplies goal bias. `memory.py` owns conversation
state and a JSONL episodic log; `cognition/experience.py` and
`cognition/patterns.py` persist learning summaries. `executor.py` permits only
fixed catalog actions, `tools.py` contains fixed Windows wrappers, and
`verification.py` distinguishes confirmation from an unverified handoff.
`voice.py` is optional and falls back to text mode. `system/` polls computer
state and reports anomalies.

## Verification performed

- Requirements install was exercised in an isolated virtual environment. Core,
  embedding, and voice modules import successfully; Python 3.11 is compatible.
- `python main.py --healthcheck` passes. It loaded 10,012 examples and 208
  executable actions. Ollama, network, voice, system agent, memory, and tool
  registry were healthy in this environment; embeddings were still warming in
  the short healthcheck window.
- Normal text startup and clean exit pass. Voice is detected as available, but
  no microphone/audio interaction was initiated.
- `pytest -q`: **14 passed, 0 failed**.
- Startup/healthcheck wall time: approximately **2.2 s**. FastEmbed/model and
  real LLM latency are environment/model dependent and were not benchmarked
  against live user actions. The existing `test.py` benchmark can measure
  Ollama prefill/decode latency safely when desired.

The automated matrix covers normalization, empty and malformed LLM output,
invented targets/prompt-shell injection, unknown capabilities, fixed-argument
execution, references/repeated requests, corrupted and persisted memory,
failed and unavailable verification, and timeout/auth provider fallback.
Windows UI, filesystem, network, shutdown/restart, and voice cases are
catalogued but were not executed against the host because doing so would alter
the machine or require ambiguous live fixtures. Their execution boundary is
covered with mocks and allow-list tests.

## Fixed issues

1. Healthcheck crashed on Windows CP1252 consoles because its Unicode symbols
   could not be printed. CLI output now explicitly uses UTF-8 with a safe
   replacement fallback.
2. Tasks with unavailable or failed verification were recorded as successful,
   updated reference state, and could enter the future training log. They now
   remain unverified/failed and never become successful state or training data.
3. Repeated actions did not produce persisted experience transitions, despite
   the intended sequence feature and regression test.
4. A corrupt episodic-memory line stopped loading all later valid lines. Valid
   records now continue to load.
5. An authentication failure from one LLM provider stopped fallback and emitted
   a misleading Ollama-only error. The failed provider is now exhausted for the
   session and remaining configured providers are tried.

## Security boundaries and verification status

The model sees only a retrieved action allow-list. `nlu.resolve` rejects any
target not in that list, and `executor.py` maps catalog actions to fixed URI,
executable, or diagnostic identifiers. User-supplied shell text and executable
parameters are ignored. High/medium risk actions require confirmation;
`close_process` protects critical system processes.

Process launches/closes and fixed diagnostics have real verification. URI
handoffs are explicitly **unavailable** for verification because Windows does
not provide a process handle. Any action without a verification strategy is now
reported as unverified rather than successful. A remaining limitation is that
process-name verification can confirm an already-running process rather than a
new launch; launch-handle/PID verification would be stronger.

## Remaining issues and training readiness

- **High:** URI/settings actions cannot be independently verified. Training
  data should continue to exclude them unless an evidence source is added.
- **Medium:** Semantic retrieval depends on FastEmbed/model availability; while
  it warms or is unavailable, normal action resolution cannot safely proceed.
- **Medium:** The catalog provides limited safe support for arbitrary file
  search, port checks, slow-network diagnosis, and ambiguous filenames; these
  requests should not be represented as supported training capabilities until
  fixed tools and verification exist.
- **Medium:** Real destructive/system-changing scenarios (shutdown/restart)
  were intentionally not run on the audit host. They need disposable Windows
  VM end-to-end tests before any release or training-data collection.
- **Low:** Runtime modules use broad exception handling in several persistence
  and optional-dependency paths. They fail gracefully, but structured logging
  and telemetry are limited to console debug output.

Prixon is **not training-ready for verified action fine-tuning** until URI
verification and a disposable-VM E2E matrix exist, and until unsupported
filesystem/network capabilities are either implemented as fixed tools or
excluded from intended behavior. No model training was started.
