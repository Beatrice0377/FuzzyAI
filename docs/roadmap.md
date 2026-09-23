# Probvenance Roadmap

**This roadmap is not a schedule and not a commitment.** It records the
intended order of work and the dependencies between phases. Items move, split,
or disappear when evidence says they should; the binding rules live in the
[design constitution](./design-constitution.md), not here.

Phase 1 (core contracts, fingerprints, result semantics) and the Phase 2A
Bool vertical slice, including the Phase 2A.1 scoring-validity diagnostics,
are implemented: see the constitution for exact scope. Phase 2A.2 is a
completed validation experiment, not a feature. Phase 2B / 2B.1 add the
experimental direct categorical Choice runtime and its cross-model validation,
and Phase 2C.0 closed scoring-continuation correctness plus compiler and
assembler provenance. Anything below not marked as delivered is not
implemented.

The project was renamed from FuzzyAI to Probvenance before any package release,
to avoid a naming collision and to better reflect the probability plus
provenance focus.

## Phase 2: first real inference path

Phase 2A delivered the Bool path end-to-end, Phase 2A.1 added the
scoring-validity diagnostics, and Phase 2A.2 ran the semantic signal
validation experiment. Phase 2B delivered the experimental direct categorical
Choice path and Phase 2B.1 replicated it on a second model family. The
remaining Phase 2 items below are still future.

- Transformers local backend. Delivered (Phase 2A): logits-only
  `TransformersBackend` behind the optional `transformers` extra.
- Bool binary-logit scoring. Delivered (Phase 2A): `assemble_bool_probability`
  (two-way softmax over the two verbalizer-token logits).
- Compiler (spec to plan, strategy selection from declared capabilities).
  Delivered (Phase 2A) for Bool: `BoolCompiler`.
- Basic `DecisionTrace`. Delivered (Phase 2A): `DecisionTrace` and
  `build_decision_trace`, wired through the thin `Probvenance` facade.
- Scoring-validity diagnostics and execution fingerprint. Delivered (Phase
  2A.1): every `DecisionTrace` carries `ScoringDiagnostics` (including
  `verbalizer_mass`, a full-vocabulary quantity independent of the restricted
  `probability_true`) and an `execution_fingerprint` identifying the execution
  environment and rendering configuration the plan actually ran under. No
  threshold and no auto-rejection is implemented; see `ScoringValidityPolicy`
  under Phase 5.
- Semantic signal validation. Completed (Phase 2A.2): a three-model
  mechanism-validation experiment on the binary scoring position (27
  formulations x 80 probes = 2160 probes per model, one forward pass each,
  `bfloat16`, batch size 1; record in
  `experiments/semantic_signal/REPORT.md`). It is an experiment record, not a
  capability claim and not a benchmark. Observed under those conditions:
  `verbalizer_mass` separates scoring-position failures from semantic
  failures (one of the three models left the decision position for 462 / 720
  probes under one doctrine while the other two stayed healthy), the
  contrast groups ordered correctly for MiniCPM5-2B and Qwen3.5-2B but not
  for LFM2.5-1.2B, the `insufficient` ladder rung is the systematic weak
  point (below `weak_negative` in 6 of 9 model x doctrine cells), and label
  families are not interchangeable without declared semantics. The
  Choice-readiness gate was assessed per model: met by a specific named
  configuration (MiniCPM5-2B or Qwen3.5-2B with `yes_no` / `true_false` under
  a decision-position-keeping doctrine), NOT met by the mechanism in general;
  the three-way candidate space itself remains unmeasured.
- Choice categorical-logit scoring (Phase 2B, 2B.1). Implemented and
  experimentally validated: the direct categorical runtime exists, and
  representation sensitivity was replicated on two model families. Design and
  measured outcome in `docs/choice-semantics.md` and
  `experiments/choice_signal/REPORT.md`. Remaining Choice work (one-vs-rest,
  multi-token scoring labels, open-set handling) stays future.

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

Calibration identity depends on probability-semantics formulation and source
identity, not on task or model identity alone; see
`docs/probability-semantics-identity.md`. The calibration semantics, dataset
identity, binding, profile identity, and evaluation contract are designed in
`docs/calibration-semantics.md`. The Phase 4A data foundation (ground truth,
binding, observation, dataset, and their identities and fingerprints) and the
pre-calibration evaluation foundation (the declared evaluation source cohort,
the metric-eligible dataset projection with explicit exclusion accounting, and
the winner-correctness Brier and exact log-loss metrics) are implemented.
`CalibrationProfile`, fitting, the remaining metrics, profile matching, and
runtime profile application do not exist, and `predicted_correctness` remains
`None` for every result the runtime can produce.

- Evaluation harness (started: declared evaluation source cohort, metric-eligible
  dataset projection, Brier, log loss).
- Expected calibration error (ECE).
- Temperature scaling.
- Calibration profile.

Constraint recorded by Phase 2B and replicated by Phase 2B.1 (do not implement it
here): a future `CalibrationProfile` must bind to probability-semantics-relevant
scoring representation identity, not merely to task or model identity. Two
executions that hold the `ChoiceDecision` fixed but use a different scoring
representation produce different uncalibrated distributions (measured: up to 0.31
total variation on `Qwen/Qwen3.5-2B` and 0.57 on `openbmb/MiniCPM5-2B`), so a
calibration fit to one representation does not transfer to another. At minimum
the binding should cover the model and revision, the decision family, the scoring
strategy, the doctrine, the compiler version, and the scoring representation or
plan family. It should NOT be bound to a single plan fingerprint, which would be
too fine-grained and would break on any irrelevant plan change. The formulation
identity and family definitions are designed in
`docs/probability-semantics-identity.md` (Phase 2C-Design) and materialized as
runtime fingerprints in Phase 2D. Calibration itself remains future work.

## Phase 5: policy layer

- Abstention policy (`accept` / `abstain` / `review` / `escalate`).
- Scoring validity policy (`ScoringValidityPolicy`, future): a policy object
  that consumes the diagnostics Phase 2A.1 now only measures. Inputs:
  `verbalizer_mass`, the top token, and model- or task-specific empirical
  distributions. Outputs: accept, reject, or warn. Phase 2A.1 deliberately
  implements NO threshold and NO auto-rejection, because no experiment
  justifies a threshold that is stable across models, tokenizers, chat
  templates, verbalizers, and prompts.
- Risk-coverage evaluation.
- Replay. Today's `DecisionTrace` is replay-oriented provenance: it records
  what a future replay would need but does not snapshot backend or tokenizer
  code, so strict replayability remains an open question this item must close.
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
candidates according to the declared decision semantics. The first concrete
doctrine now exists: `BINARY_SEMANTIC_JUDGMENT_V1` (`src/probvenance/doctrine.py`)
covers the binary case. Doctrines for other decision types remain future
concepts only.

### Evidence Lineage / replay-oriented derivation

Evolving `DecisionTrace` from a telemetry record toward a replay-oriented
derivation record, per constitution AP-02. The trace is replay-oriented
provenance: it records what a future replay would need, but it does not
snapshot backend or tokenizer code, so strict replayability remains an open
question, not a present property. The schema is not frozen.

### Incremental Decision Evaluation

Re-evaluating a decision as new evidence arrives against a shared context
(`context(t0) + new evidence -> re-evaluate`), per constitution AP-03. No
implementation; the enabling mechanisms are hypotheses.

### Cache-aware / cost-aware compilation

Among semantically equivalent execution layouts, preferring the one that
preserves stable prefixes and maximizes reusable computation (constitution
AP-03). Depends on cache-aware machinery that no compiler implements yet.

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
only. Note that the `replayable` label names a future retention mode, not a
present property of `DecisionTrace`: today's trace is replay-oriented
provenance and snapshots no backend or tokenizer code, so strict replayability
is not claimed at any retention level yet.
