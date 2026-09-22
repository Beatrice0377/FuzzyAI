# FuzzyAI Roadmap

**This roadmap is not a schedule and not a commitment.** It records the
intended order of work and the dependencies between phases. Items move, split,
or disappear when evidence says they should; the binding rules live in the
[design constitution](./design-constitution.md), not here.

Phase 1 (core contracts, fingerprints, result semantics) is the current phase:
see the constitution for its exact scope. Nothing below is implemented.

## Phase 2: first real inference path

- Transformers local backend.
- Bool binary-logit scoring.
- Choice categorical-logit scoring.
- Compiler (spec to plan, strategy selection from declared capabilities).
- Basic `DecisionTrace`.

## Phase 3: cloud backends and experiments

- OpenAI-compatible backend.
- OpenCode Go experiments.
- DeepSeek V4.1 Flash.
- GLM-5.3 Flash.
- Anthropic-compatible backend.
- Qwen3.8 Flash.

Architectural note: OpenCode Go does not expose a unified protocol across all of
its models. The architecture must NOT equate "cloud backend" with "OpenAI
protocol". Cloud is a deployment shape; the protocol is a backend detail that
stays behind the `Backend` boundary.

## Phase 4: evaluation and calibration

- Evaluation harness.
- Brier score.
- Log loss.
- Expected calibration error (ECE).
- Temperature scaling.
- Calibration profile.

## Phase 5: policy layer

- Abstention policy (`accept` / `abstain` / `review` / `escalate`).
- Risk-coverage evaluation.
- Replay.
- Robustness testing.

## Later (unscheduled)

- vLLM backend.
- SGLang backend.
- Prefix / KV cache exploitation.
- One-vs-rest fallback scoring.
- Sampling estimator.
- `Score` primitive.
- `MultiLabel` primitive.
- Decision Graph.
- Routing.
- Shadow evaluation.
- Drift detection.
- Dashboard.
