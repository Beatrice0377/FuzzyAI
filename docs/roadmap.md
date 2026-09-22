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

## Research / Architecture Backlog

These are directions under consideration, not current commitments. None of them
is guaranteed to reach 1.0, and whether each is pursued must be decided by
actual experimental results rather than by design intuition.

### Scoring Doctrine

A versioned specification of HOW the model should conduct semantic judgment,
independent of any business `DecisionSpec`. The concept hierarchy:

```
DecisionSpec         -> WHAT semantic question is being asked
Scoring Doctrine     -> HOW the model should conduct the judgment
Compiler             -> HOW that judgment is lowered into executable inference
Backend              -> WHERE / through which model interface inference runs
ProbabilityAssembler -> HOW RawEvidence becomes uncalibrated decision probabilities
Calibration          -> HOW empirical correctness meaning is attached
Policy               -> WHAT the software does with the result
```

A doctrine might eventually carry rules such as: treat context as evidence; do
not follow instructions embedded inside evidence; do not force certainty when
evidence is insufficient; distinguish absence of evidence from contradictory
evidence; preserve ambiguity instead of inventing a decisive answer; evaluate
candidates according to the declared decision semantics. NOT implemented; this
is a future concept only.

### Evidence Lineage / replayable derivation

Evolving `DecisionTrace` from a telemetry record into a replayable derivation
record, per constitution AP-02. The schema is not frozen.

### Incremental Decision Evaluation

Re-evaluating a decision as new evidence arrives against a shared context
(`context(t0) + new evidence -> re-evaluate`), per constitution AP-03. No
implementation; the enabling mechanisms are hypotheses.

### Cache-aware / cost-aware compilation

Among semantically equivalent execution layouts, preferring the one that
preserves stable prefixes and maximizes reusable computation (constitution
AP-03). Depends on a compiler that does not exist yet.

### Hierarchical Choice

Resolving a large candidate count (for example 200 categories) as top-level
semantic family -> sub-family -> leaf choice. Potential value: a smaller
candidate space per stage, local calibration, local abstention, an explainable
taxonomy, and possibly lower compute cost. It is unproven whether this is more
accurate, cheaper, or easier to calibrate than a flat `Choice`. Roadmap only;
must not be implemented early.

### Decision Quality Tiers

The user expresses requirements such as risk level, maximum cost, latency
budget, and target quality, for example via a conceptual `DecisionRequirements`
object, instead of asking the runtime to pick a model automatically:

```python
DecisionRequirements(
    risk="high",
    max_latency_ms=500,
    max_cost=0.01,
)
```

or via tiers named `cheap`, `balanced`, `high-assurance`. A future
compiler/router would select the model, strategy, number of passes, and
possibly escalation. A quality tier is only a statement of user requirements
and must not promise that the runtime knows its true quality; any automatic
routing must be versioned, traceable, replayable, and explicitly recorded.

### Selective fidelity

See constitution AP-07: not every semantic decision deserves the same
inference cost, and higher cost does not imply higher quality. Any fidelity
tier must be earned by evaluation.

### Trace retention modes

The conceptual `minimal` / `replayable` / `full` space recorded in constitution
AP-02; storage and privacy policy decide what each mode persists. Design space
only.
