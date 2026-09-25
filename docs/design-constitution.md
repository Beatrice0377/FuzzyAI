# Probvenance Design Constitution

Status: **early development**. The Bool vertical slice and the direct
categorical Choice slice are implemented. Calibration is implemented as an
explicit, offline fit plus an explicit runtime application; automatic
calibration and automatic profile selection are not implemented.

This is the highest-level, longest-lived document in the repository. It is the
binding design constraint for all future work in Probvenance. Code, tests, docs, and
APIs that contradict this constitution are wrong, even if they pass review. When
a future change alters a rule here, this document must be amended first.

---

## 1. Purpose and Core Principle

Probvenance is a **provider-agnostic probabilistic decision runtime**. It turns
language models into decision components whose outputs are evaluable,
calibratable, and trackable as semantic probabilities.

The problem it addresses: LLMs are routinely wired into program logic as if a
raw text blob or an unexamined score were a trustworthy probability. The result
is systems that cannot say what a score means, cannot reproduce a past decision,
and cannot separate "the model was unsure" from "the program should decline to
act". Probvenance gives that space a narrow, deterministic core.

The core principle, stated once and enforced everywhere:

> **The LLM handles semantic uncertainty. The program handles deterministic policy.**

Probvenance is NOT:

- a chat framework,
- an agent framework,
- merely a structured-output wrapper.

If a proposed feature fits one of those descriptions better than the
description above, it does not belong in Probvenance.

---

## 2. Glossary (normative)

These five terms have exactly the meanings below. Every public type, docstring,
error message, and doc must use them this way. Each definition includes an
explicit "what it is NOT" clause because the failure mode this project fights is
terminology drift.

### Probability

The decision distribution produced by **some scoring strategy**. A probability
is always relative to the strategy and evidence that generated it.

A full uncalibrated decision distribution is conditional not only on the semantic
question and the declared candidate set, but also on the concrete scoring
representation, meaning which scoring label each candidate is bound to. Holding a
`ChoiceDecision` completely fixed and changing only that binding moved the
distribution by up to 0.31 and 0.57 total variation on two model families
(Phase 2B and Phase 2B.1, `experiments/choice_signal/REPORT.md`). A probability
therefore has plan-relative semantics: the same decision compiled into a
different scoring representation is a different plan, and its numbers are not
automatically comparable with the original. Probvenance records that difference
rather than hiding it.

**What it is NOT:** a probability is not automatically a real-world correctness
rate. `0.9` does not mean "90% likely to be right" unless and until calibration
has established that mapping.

### Certainty

Describes ONLY how concentrated a probability distribution is, measured by
entropy and probability margin. It is a mathematical property of a distribution.

**What it is NOT:** certainty is not a correctness probability, not a quality
score, and not trustworthiness. It must **never** be called "confidence" in any
public name, field, doc, or message. The related field is named
`concentration`, deliberately not "confidence".

### Calibration

Using a ground-truth dataset to map raw model evidence, or raw probabilities,
onto empirically meaningful probabilities.

**What it is NOT:** calibration is not implemented in Phase 1, and it is not a
renaming trick. No amount of post-processing vocabulary turns an uncalibrated
score into a calibrated one.

### Predicted correctness

A statement about how likely a specific decision is to be correct. It may only
be expressed **after valid calibration**. Uncalibrated results carry
`predicted_correctness = None`.

**What it is NOT:** it is FORBIDDEN to label a max softmax, a logit, an entropy,
or an LLM's self-reported confidence as "predicted correctness". Any of those
presented under that name is a constitution violation.

### Abstention

The decision to not act on a model output: `accept`, `abstain`, `review`, or
`escalate`. Abstention belongs to the **policy layer**, not the model. A
Decision describes uncertainty; a Policy decides what to do about it.

**What it is NOT:** abstention is not a model output and not a special label the
runtime injects on the model's behalf. The model must not silently make business
decisions.

---

## 3. Normative Invariants

Each invariant is testable. Tests in this repository should cite invariant
numbers.

### Probability and result semantics

- **INV-01 (bounds).** `BoolResult.probability_true` is in `[0, 1]`. Every value
  in `ChoiceResult.probabilities` is in `[0, 1]`.
- **INV-02 (normalization).** The values of `ChoiceResult.probabilities` sum to
  1 within an absolute tolerance of `1e-9`.
- **INV-03 (deterministic tie-break).** `ChoiceResult.value` is the deterministic
  argmax of `probabilities`. On a tie, the entry appearing **first in the
  original candidate ordering** wins. This is a normative rule, not an
  implementation detail: two runs with the same probabilities must return the
  same `value`.
- **INV-04 (uncalibrated means None).** Every Phase 1 result has
  `predicted_correctness = None` and `calibrated = False`. Setting
  `predicted_correctness` while `calibrated=False` is an error
  (`InvalidProbabilityError`).
- **INV-05 (certainty math).** `Certainty.entropy` is **normalized** entropy in
  `[0, 1]`, computed with the natural log and normalized by `log(n)`, where
  `0` means fully concentrated and `1` means uniform. `Certainty.margin` is
  `top1 - top2` and lies in `[0, 1]`. `concentration = 1 - normalized_entropy`
  expresses peakedness and is deliberately not named "confidence". The Bool case
  is computed by treating the distribution as `[p, 1 - p]`. For a
  single-outcome distribution, normalized entropy is `0.0` and margin is the
  single probability.
- **INV-06 (no confidence alias).** No public name, field, doc, or message in
  this project may use "confidence" to mean certainty, probability, or predicted
  correctness.

### Determinism and fingerprints

- **INV-07 (semantic determinism).** The same semantic input must ALWAYS produce
  the same fingerprint. Fingerprints are derived from values only, never from
  Python object memory identity or `repr`.
- **INV-08 (type whitelist).** Canonicalization accepts only `None`, `bool`,
  `int`, `float`, `str`, `list`, and `dict[str, ...]` (recursively). `set`,
  `tuple`, callables, arbitrary Python objects, and automatic `datetime`
  coercion are forbidden and are rejected with `FingerprintError`.
- **INV-09 (finite floats only).** `NaN`, `Infinity`, and `-Infinity` are
  rejected with `FingerprintError`; they must never enter a deterministic JSON
  fingerprint. `-0.0` is normalized to `0.0`.
- **INV-10 (key-order independence).** `dict` keys are sorted during
  canonicalization, so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` produce the
  SAME context fingerprint.
- **INV-11 (literal Unicode).** Unicode is emitted literally and the UTF-8 bytes
  are hashed; no `\uXXXX` escaping. No Unicode normalization is performed:
  distinct code-point sequences are distinct inputs and may produce distinct
  fingerprints.
- **INV-12 (fingerprint definition).** A fingerprint is the SHA-256 hex digest
  of the canonical JSON string.
- **INV-13 (choice order is semantics-bearing).** The explicit candidate order
  of a `ChoiceDecision` IS preserved and DOES enter the fingerprint. This
  contrasts with INV-10 on purpose: unordered `context` object keys do not
  affect the fingerprint, but candidate order does, because order changes the
  decision's semantics (and the tie-break of INV-03 depends on it).

### Capability and backend boundary

- **INV-14 (explicit capabilities).** `BackendCapabilities` is explicit data,
  never `hasattr` probing. Its fields are exactly: `binary_token_logits`,
  `categorical_token_logits`, `token_logprobs`, `batching`, `prefix_cache`,
  `constrained_decoding`. A capability describes what a backend CAN do, never
  what we GUESS it can do.
- **INV-15 (no silent fallback).** The future compiler selects a scoring strategy
  only from declared capabilities. There is NO automatic fallback in Phase 1.
  Insufficient capability must raise `UnsupportedCapabilityError` rather than
  silently changing the probability semantics.
- **INV-16 (backend ignorance).** `Backend` is a narrow Protocol: a
  `capabilities` property plus `execute(plan: InferencePlan) -> RawEvidence`. A
  backend knows NOTHING about `BoolDecision`, `ChoiceDecision`, results, or
  certainty. It only runs an already-compiled plan.
- **INV-17 (provider independence of the core).** `InferencePlan` is
  provider-independent. Provider-specific knobs (for example
  `openai_temperature`, `anthropic_thinking`, `vllm_tensor_parallel`) must never
  appear in the core layer; they belong to future backend config.
- **INV-18 (evidence is not probability).** `RawEvidence` is raw model output
  BEFORE conversion into a `DecisionResult`. It is explicitly NOT a probability
  and NOT a calibrated probability. Its optional `metadata` is a controlled
  `dict[str, JSONValue]` extension point, never `dict[str, Any]`.

### Scoring position and restricted probability

- **INV-19 (restricted probability never stands alone).** A restricted candidate
  probability (for example `BoolResult.probability_true`) is conditional on the
  model's next token being one of the declared candidates. It must not be
  exposed through a `DecisionTrace` without the accompanying full-vocabulary
  candidate-space mass, because on its own it reads as evidence that the model
  was actually choosing among the candidates when it may not have been.
  Enforcement: `DecisionTrace.scoring_diagnostics` is a required field, and the
  mass is derived from full-vocabulary normalization rather than from the
  candidate logits alone. The categorical path generalises the same rule with
  `ChoiceScoringDiagnostics.scoring_label_mass`, so a direct categorical Choice
  result may not be traced without it. That is the same rule over a wider
  candidate set, so it does not get a new invariant number. `verbalizer_mass`
  (binary) and `scoring_label_mass` (categorical) are the same
  candidate-label-set mass equation under their respective scoring
  representations, and the categorical form reduces to the binary form at two
  scoring labels. Both measure how much of the raw next-token distribution landed
  on the declared scoring representation; neither measures whether the semantic
  candidate set is correct, exhaustive, or contains the true answer.
- **INV-20 (semantic candidate identity is not scoring-label identity).** A
  `ChoiceResult` is keyed by semantic candidate names and never by scoring
  labels, and no public result surface exposes a scoring label as a candidate.
  Enforcement: the assembler maps label-space probabilities onto the plan's
  `candidate_mapping` names, and `tests/test_choice_assembler.py` asserts the
  probability keys are semantic names under both a natural and a permuted
  binding.
- **INV-21 (the scoring representation is in the plan fingerprint, not the
  decision fingerprint).** Changing the candidate-to-label assignment while
  holding the `ChoiceDecision` fixed must leave the decision fingerprint
  unchanged and must change the plan fingerprint. Enforcement:
  `InferencePlan.fingerprint` covers `label_scheme_id` and `candidate_mapping`,
  `tests/test_choice_compiler.py` asserts both halves of the rule, and
  `experiments/choice_signal/REPORT.md` section 2 records six permutations of one
  case producing one distinct decision fingerprint and six distinct plan
  fingerprints.
- **INV-22 (evidence label order is provenance).** `RawEvidence` labels for
  direct categorical scoring must equal the plan's declared `targets` exactly,
  in the same order. Matching only the label set and reordering internally is
  forbidden, because label order, logit order, and the candidate mapping
  together form the provenance of a result. Enforcement:
  `assemble_choice_probability` compares `evidence.labels` against
  `plan.targets` directly and raises, and `tests/test_choice_assembler.py`
  asserts that permuted labels with legal-looking pairs are rejected rather than
  reordered.

### Probability formulation identity

These five invariants are frozen as normative design rules. Phase 2D
materialized both formulation identities in the runtime, which gives INV-25 and
INV-27 real enforcement points. INV-23, INV-24 and INV-26 still bind the design
rather than a code path, because no pooling, calibration, or source-identity API
exists yet; each invariant below states its own current enforcement level.

- **INV-23 (cross-formulation probabilities are not interchangeable).**
  Probabilities from different probability formulation identities must not be
  silently treated as interchangeable, directly pooled, or assumed to share
  calibration. Explicit cross-formulation evaluation is allowed when both
  identities and the comparison method are preserved. Enforcement: design
  contract only. Phase 2D made the identities this rule refers to derivable,
  which is a prerequisite for enforcement, not enforcement itself. The
  explicit-comparison clause is exactly what
  `experiments/choice_signal/REPORT.md` does when it reports total variation
  between two formulation identities while preserving both.
- **INV-24 (formulation family membership is not automatic compatibility).**
  Formulation family membership does not imply automatic interchangeability,
  automatic pooling, or calibration compatibility. The word is `automatic`:
  explicit comparison remains permitted, so this rule must never be read as
  "cannot be compared". Enforcement: design contract only. The runtime exposes a
  formulation-family fingerprint but applies no pooling or compatibility policy
  to it.
- **INV-25 (formulation identity excludes instance evidence).** Probability
  formulation identity excludes instance evidence, meaning the `question` and
  `context` of a decision, while including the semantic outcome space and the
  scoring representation. Evidence decides the probability value; it does not
  decide what the probability means. Enforcement: runtime-enforced by
  construction. `probability_formulation_payload` is built from plan fields only,
  so `question` and `context` cannot enter it, and
  `tests/test_probability_identity.py` asserts both their absence and that two
  decisions differing only in evidence share one formulation fingerprint.
- **INV-26 (unknown identity values are explicit unknowns).** Unknown
  probability-identity values are explicit unknowns and must not match, default
  to, or be interpreted as concrete values. An absent model revision is neither a
  wildcard nor equal to a concrete revision, and an absent scoring-relevant
  rendering key is not equal to a concrete setting. Enforcement: design contract
  only; the source axis has no runtime identity yet. Phase 2D.1 did not weaken
  this rule: a built-in doctrine version is now a real declared plan field that
  the formulation identity commits, so it is no longer an unknown case at all,
  while every genuinely absent value remains an explicit unknown rather than a
  default.
- **INV-27 (assembler identity and version are part of formulation identity).**
  The probability assembler identity and version are part of probability
  formulation identity, because two assemblers under one scoring strategy may
  apply different normalizations and so produce different probability semantics.
  Enforcement: execution-verified. `resolve_probability_assembler` runs a plan
  only when its `(strategy, assembler_id, assembler_version)` matches a known
  implementation and raises `UnsupportedAssemblerError` otherwise, so a recorded
  `assembler_id` cannot describe a transformation the runtime did not perform.
  `tests/test_assembler_dispatch.py` covers the valid, unknown,
  mismatched-strategy, and unsupported-version cases; Phase 2D additionally
  commits the assembler identity and version inside the formulation fingerprint,
  so a changed assembler is visible there too.

### Probability comparability vocabulary (normative, descriptive only)

Two independent axes describe how two probabilities relate:

```text
FormulationRelation:  exact | family | decision_family | different
SourceRelation:       exact | different | unknown
```

A `(FormulationRelation, SourceRelation)` pair states a fact about the two
identities. It deliberately does not answer `can_compare`, `can_pool`, or
`can_share_calibration`; those need a use-case-specific policy and evidence, and
no such policy exists yet. The axes are independent, so neither pair is globally
"more comparable" than another: `(family, exact)` and `(exact, different)` differ
in kind, not in degree. There is deliberately no `compatible` relation. This
vocabulary is normative in the design documents and is not implemented as a
runtime enum or API.

---

## 4. Architecture Principles

The invariants in section 3 are frozen, testable contracts of the Phase 1 code:
each one has a passing test or a runtime enforcement point. The Architecture
Principles below are different in kind. They are long-lived design rules that
future work must obey, and several of them are NOT yet testable in Phase 1 and
have no code enforcement. Nothing in this section should be read as a claim
that any principle is already verified; an AP number cited by a future design
or test means "this rule governs the design", not "this rule is currently
enforced".

### AP-01 Model-driven semantic judgment; deterministic system mechanics.

The model may own semantic judgment, the expression of semantic ambiguity, and
the interpretation of semantic evidence. The program must own probability
semantics, scoring strategy selection, capability negotiation, calibration,
fallback rules, policy, abstention, routing, versioning, and replay. Model-driven
does NOT mean model-controlled.

The model must never decide on its own:

- whether its output is calibrated,
- the probability semantics of its output,
- which scoring strategy to use,
- whether to fall back to another provider,
- business abstention decisions.

The responsibility chain:

```
Model              -> semantic judgment
Compiler           -> inference mechanics
Backend            -> execution
Probability layer  -> semantics
Calibration        -> empirical mapping
Policy             -> action
```

The abstention boundary deserves its own statement. The model must not reason
"because certainty is low, I will abstain". If "should we abstain" is itself
modelled, it must be an explicit separate `DecisionSpec` whose output is only
another semantic input, and the abstention decision is still executed by the
policy, in the program.

### AP-02 Derived decisions should preserve sufficient provenance for audit and replay.

Decision lineage is not the same thing as an observability log. An ordinary log
may carry only the model, the latency, and the probability and still be useful.
A lineage exists to support replay, audit, debugging, regression testing,
calibration validation, and scoring-strategy comparison, so it must record the
derivation dependencies of a decision. The intended future lineage graph:

```
DecisionSpec
  -> Scoring Doctrine version
  -> Compiler version
  -> InferencePlan
  -> Rendered model input
  -> Backend / model / model revision
  -> RawEvidence
  -> Probability transformation
  -> Calibration profile
  -> DecisionResult
  -> Policy outcome (if a policy layer is used)
```

A future `DecisionTrace` must not be merely a telemetry record; it should
evolve into a replayable derivation record. This round does NOT freeze the
`DecisionTrace` schema. A probability transformation must not become an
unexplainable black box sitting between `RawEvidence` and a `DecisionResult`:
the transformation applied must itself be part of the lineage. How much raw
detail to persist (full provider payloads, prompts, logits) is a storage and
privacy policy question, so the future Trace Retention Modes (conceptually
`minimal`, `replayable`, `full`) are recorded here as a design space, not as
an implementation.

### AP-03 When semantics are identical, prefer incremental and cache-preserving execution layouts.

The long-term scenario shape is `context(t0) + new evidence -> re-evaluate`:
conversations, incident streams, agent trajectories, monitoring events, and
fraud event streams all accumulate evidence against a shared context rather
than restarting from zero. Future building blocks that may make this practical
include context ancestry, fingerprints, KV cache, prefix cache, and previous
inference state.

The same principle has a cache-preserving compilation half: when several
execution layouts are semantically equivalent, a future compiler may prefer the
layout that preserves stable prefixes and maximizes reusable computation, and
may weigh shared-prefix length, number of model passes, batchability, cache
locality, provider caching characteristics, estimated token cost, and estimated
latency. None of this is implemented in Phase 1.1.

### AP-04 Optimization may not silently change probability semantics.

The trade-off order in the later section of this document puts semantic
correctness above performance optimization, and this principle applies that
ordering at optimization boundaries: a cache or cost optimization must never
alter decision semantics. If an optimization would change probability semantics,
it is not an optimization; it must be an explicit, versioned, lineage-recorded
change. No optimizer exists in this round.

### AP-05 Public capability/performance/quality claims require explicit evidence status.

Every capability, performance, quality, calibration, or provider-support claim
about this project must carry an explicit evidence status in
[`./claims.md`](./claims.md). The four statuses:

- `[V] VERIFIED`: backed by reproducible evidence in the repository.
- `[E] EXPERIMENTAL`: observed in experiments, not yet a stable guarantee.
- `[H] HYPOTHESIS`: a design expectation, awaiting evaluation.
- `[R] ROADMAP`: planned work, not a capability.

A roadmap item is never a capability claim.

### AP-06 Provider/model changes that affect probability semantics are provenance-relevant changes.

Changing the model, the model revision, the scoring doctrine version, the
compiler version, or the calibration profile can move probability semantics.
Each of those changes is therefore provenance-relevant and must be recorded in
the decision lineage. In particular, a calibration profile must not be bound
merely to a model name; it must be versioned against the artifacts that
actually affect probability semantics. The concrete schemas for these version
identifiers are not frozen in this round.

### AP-07 Not every semantic decision deserves the same inference cost.

A low-risk classification and a high-risk interpretation may warrant different
inference fidelity. Possible future shapes include a single-pass cheap
decision, multiple evidence probes, larger-model escalation, ensembles, and
sampling. The caution is part of the principle: higher computational cost does
NOT automatically imply higher semantic quality, and any quality tier must be
supported by evaluation rather than assumed. This is the "selective fidelity"
principle.

### AP-08 The compiler declares the representation; the backend validates executability.

Whether a scoring label resolves to one token is only decidable against a
concrete tokenizer, chat template, and rendered continuation, so a compiler
cannot both choose tokenization-valid labels and stay provider-independent. The
responsibility is therefore split: the compiler chooses the scoring
representation (candidate order, scoring label strings, the candidate-to-label
mapping, required capabilities, prompt structure) from the decision, the
doctrine, the versioned label scheme, and the declared capabilities, without
access to a tokenizer, model, chat template, token id, CUDA, or backend-specific
rendering; the backend resolves those labels in the real continuation and
rejects an unexecutable representation explicitly. Resolution is exact: a label
counts as a single-token continuation only when tokenizing the prefix plus the
label reproduces the original prefix token sequence followed by exactly one
additional token. A net increase of one token is not sufficient, because a
retokenized prefix can leave the final token unrelated to the original last
position, in which case reading its logit would not be evidence about the label.
Neither side may silently repair the other: the compiler never skips or
re-assigns a label, and the backend never falls back to another strategy,
truncates, sums multi-token logits, or generates. A future tokenizer-aware
resolver would be an explicit, recorded artifact, not an implicit behaviour.

---

## 5. Architecture and Layer Boundaries

The layering below is the proven design. Phase 2A wired the Bool path
end-to-end: `BoolCompiler`, `TransformersBackend` (optional extra), and
`assemble_bool_probability` and `assemble_choice_probability` now connect
`BoolDecision` and `ChoiceDecision` to their results plus a `DecisionTrace`,
under the `Probvenance` facade. Assembler selection is execution-truthful: the
facade runs only the implementation named by the plan's
`(strategy, assembler_id, assembler_version)` declaration and rejects a plan
that names no known implementation, so a recorded `assembler_id` cannot
describe a transformation the runtime did not perform. Calibration is
implemented as an explicit offline fit plus an explicit runtime application;
automatic calibration and the remaining backends remain future work.

```
DecisionSpec
    -> Compiler                 (BoolCompiler and ChoiceCompiler implemented)
    -> InferencePlan
    -> Backend                  (Protocol; TransformersBackend implemented)
    -> RawEvidence
    -> ProbabilityAssembler     (Bool binary and Choice categorical restricted softmax)
    -> DecisionResult
    -> Calibration              (explicit application implemented; automatic application future)
    -> Policy                   (out of scope by design)
```

`ProbabilityAssembler` is the single name for the layer that turns `RawEvidence`
into an uncalibrated decision probability distribution. An earlier draft called
this layer `Scoring / calibration`, which was too broad: it merged two layers
with different obligations. They are separate layers now.

### ProbabilityAssembler

```
RawEvidence
    -> uncalibrated decision probability distribution
```

It does NOT:

- perform model inference;
- decide business policy;
- perform empirical calibration;
- claim that its probability is a correctness probability.

Two assemblers exist. `assemble_bool_probability` turns a two-label logits
`RawEvidence` into an uncalibrated `BoolResult` via a numerically stable
two-way softmax over exactly the two verbalizer-token logits.
`assemble_choice_probability` turns a multi-label logits `RawEvidence` into an
uncalibrated `ChoiceResult` via a numerically stable N-way softmax over the
candidate scoring-label logits. Both are uncalibrated.

### Calibration

```
uncalibrated probability/evidence
    + ground-truth-derived calibration profile
    -> empirically meaningful calibrated information
```

Calibration is the only layer permitted to attach empirical correctness meaning
(INV-04). It is implemented for an explicit caller-driven fit and an explicit
caller-driven runtime application, and it still requires a caller to supply the
profile: nothing selects or applies one automatically. `Probability != predicted
correctness` holds at every layer.

| Layer | Phase 1 status | Responsibility | Must not know about |
|---|---|---|---|
| `DecisionSpec` (`BoolDecision`, `ChoiceDecision`) | implemented | Declare WHAT semantic decision to make: the question, the candidates (ordered), the context. Provide `fingerprint`. | Models, providers, tokens, scoring. |
| Compiler | implemented (Bool and Choice) | Choose a scoring strategy from declared capabilities; lower a spec into a plan. Raise `UnsupportedCapabilityError` when no declared capability supports the needed strategy. | Business policy, results. |
| `InferencePlan` | abstraction only | Provider-independent description of the inference to run, including `ScoringStrategy`. Fingerprintable. | Any provider-specific knob (INV-17). |
| `Backend` (Protocol) | protocol; one local implementation | Declare `capabilities` explicitly; `execute(plan)` and return raw output. | Decisions, results, certainty (INV-16). |
| `RawEvidence` | abstraction only | Carry raw model output (`EvidenceKind`) before any conversion; optional `dict[str, JSONValue]` metadata. | Probability semantics (INV-18). |
| `ProbabilityAssembler` | implemented (Bool and Choice) | Turn `RawEvidence` into an uncalibrated decision probability distribution. | Model inference, business policy, empirical calibration, and any claim that probability is a correctness probability. |
| Calibration | implemented (explicit only) | Map uncalibrated probability/evidence plus a caller-supplied ground-truth-derived calibration profile onto empirically meaningful calibrated information (INV-04). | Model inference, business policy, and any automatic selection or application. |
| `DecisionResult` (`BoolResult`, `ChoiceResult`, `Certainty`) | implemented | Report the probability distribution, certainty, `predicted_correctness=None`, `calibrated=False`. | What to do about the answer. |
| Policy | out of scope by design | Map a result plus risk tolerance onto `accept` / `abstain` / `review` / `escalate`. | (Consumes results; owns abstention.) |

### Phase 1 scope

Implemented in Phase 1: `DecisionSpec` (`BoolDecision`, `ChoiceDecision`), the
result model, `BackendCapabilities`, the `Backend` Protocol, `InferencePlan` and
`RawEvidence` abstractions, the fingerprint system, and the error taxonomy.
Phase 1 defines ONLY the `BoolDecision` and `ChoiceDecision` primitives;
`Score`, `MultiLabel`, `Rank`, `Preference`, and `Compare` are roadmap only.

Phase 2A added the first real inference path (Bool): `ScoringDoctrine` and
`BINARY_SEMANTIC_JUDGMENT_V1`, `BoolCompiler`, `assemble_bool_probability`,
`DecisionTrace` / `build_decision_trace`, the `Probvenance` / `Evaluation` facade,
and the optional-extra `TransformersBackend` (imported from
`probvenance.backends.transformers`, not re-exported from the package root). Phase
2B added the experimental direct categorical Choice path (`ChoiceCompiler`,
`CATEGORICAL_SEMANTIC_JUDGMENT_V1`, `assemble_choice_probability`,
`ChoiceScoringDiagnostics`), and Phase 2C.0 added compiler and assembler
provenance to plans and traces. The error taxonomy gains
`UnsupportedDecisionError`, `ScoringLabelError`, and `VerbalizerError` as
further `ProbvenanceError` subclasses. Abstention remains unimplemented;
calibration is partially implemented (one offline L2-regularized logistic
method, offline profile application, a post-calibration evaluation foundation,
exact profile identity, serialization and store, explicit profile eligibility
and selection, a non-authoritative profile catalog with snapshot identity,
serialization and an exact catalog store, and an explicit runtime application
that requires a caller-declared unique profile). What remains unimplemented is
automatic selection and automatic application; no runtime path picks or applies
a profile on its own.

Public API (Phase 1 core):

```
BoolDecision, Choice, ChoiceDecision,
BoolResult, ChoiceResult, DecisionResult, Certainty,
normalized_entropy, probability_margin,
BackendCapabilities, Backend,
InferencePlan, ScoringStrategy, RawEvidence, EvidenceKind,
JSONValue, canonical_json, fingerprint,
ProbvenanceError, InvalidDecisionError, InvalidProbabilityError,
UnsupportedCapabilityError, FingerprintError
```

Fingerprint API surface: `canonical_json(value)`, `fingerprint(value)`,
`BoolDecision.fingerprint`, `ChoiceDecision.fingerprint`, and
`InferencePlan.fingerprint`.

Error taxonomy: `ProbvenanceError` is the base; `InvalidDecisionError`,
`InvalidProbabilityError`, `UnsupportedCapabilityError`, and `FingerprintError`
are its Phase 1 subclasses.

### Execution Provenance

Four identities are kept distinct, because collapsing them loses information
that audit and regression work need:

- **Decision fingerprint** (INV-07, INV-12): what semantic question is being
  judged.
- **Plan fingerprint**: what the compiler produced from that decision.
- **Execution fingerprint**: the execution environment and rendering
  configuration the plan ACTUALLY ran under. Its payload is a deterministic
  JSON-compatible object covering the plan fingerprint, backend type and
  implementation version, model identifier and revision, tokenizer identifier
  and revision, runtime version, dtype, rendering config, input fingerprint, and
  the resolved verbalizer token ids. It is never built from a Python `repr`.
- **Trace id**: which single execution this was. It is per-execution identity and
  is deliberately NOT any fingerprint: running the same plan twice yields two
  trace ids and one execution fingerprint.

Rendering configuration that affects probability semantics (a model's thinking
mode, for example) belongs in the execution fingerprint and the trace, and
deliberately NOT in `InferencePlan`, which stays provider-independent (INV-17).
That is what keeps INV-17 honest: backend configuration does not enter the core
plan, but it cannot silently disappear either.

A `DecisionTrace` is a **replay-oriented provenance record**, not a strictly
replayable execution snapshot. It records what is needed to interpret and compare
an execution, but it does not snapshot backend or tokenizer code, so strict
replayability remains an open question (section 8).

Phase 2A and 2A.1 public API additions on top of the Phase 1 core: `Probvenance`,
`Evaluation`, `BoolCompiler`, `ScoringDoctrine`, `DecisionTrace`,
`ScoringDiagnostics`, `diagnose_bool_evidence`, `UnsupportedDecisionError`,
`VerbalizerError`, and `TransformersBackend` (optional `transformers` extra).
Choice scoring is implemented (Phase 2B direct categorical Choice, refined in
Phases 2B.1 and 2C).

Phase 1 non-goals (binding): no HTTP, no OpenAI/Anthropic/vLLM/SGLang, no
automatic routing, no decision graph, no dashboard, server, agent, RAG,
database, telemetry, or web UI. Phase 1 also had no calibration algorithm, which
Phase 4C later supplied as an explicit L2-regularized logistic fit; what stays
out of scope is automatic calibration. Local model loading left
the non-goals list with Phase 2A: it exists only as the logits-only
`TransformersBackend` behind the optional `transformers` extra, and the base
package still has zero runtime dependencies.

---

## 6. Policy vs Model Boundary

The model layer describes uncertainty. The policy layer acts on it.

A `DecisionResult` says: here is the distribution, here is how concentrated it
is, here is (still uncalibrated) whether we can claim predicted correctness. It
never says "ship it", "refund it", or "ask a human". Those verbs belong to a
Policy that chooses among `accept`, `abstain`, `review`, and `escalate`.

Consequences:

- The runtime must not silently abstain, silently retry, or silently downgrade a
  decision based on a threshold nobody configured. Silent behavior change is a
  semantics change (see INV-15).
- Thresholds, risk tolerance, and escalation paths are application concerns.
  Probvenance may later ship policy primitives (Phase 5), but they will be explicit
  objects, never hidden defaults inside the model path.
- "The model was unsure" and "the program declined to act" are different events
  and must remain representable separately.

---

## 7. Design Trade-off Priority Order

When two goals conflict, this order decides. It is not a suggestion.

```
semantic correctness > interface clarity > testability > future extensibility > current convenience
```

- **Semantic correctness** first: never trade a wrong-but-pleasant number (an
  uncalibrated score dressed as correctness, an implicit fallback that changes
  distribution semantics) for anything else.
- **Interface clarity** second: a narrow, honest API beats a broad, clever one.
  Forbidden vocabulary ("confidence") and forbidden shapes (`dict[str, Any]`)
  live here.
- **Testability** third: every invariant in section 3 must be testable; if a
  design makes an invariant untestable, the design loses.
- **Future extensibility** fourth: we design for Phases 2-5, but we do not
  implement them early or pretend they exist.
- **Current convenience** last: ergonomics never justify violating the layers
  above.

---

## 8. Open Questions Deferred to Later Phases

Recorded as open. None of these has a Phase 1 answer, and Phase 1 must not
smuggle one in.

- Which scoring strategies the Phase 2 compiler will support, and how
  `ScoringStrategy` and `EvidenceKind` members get finalized.
- How multi-token labels aggregate into a single categorical probability
  (sum, product, or max of token-level signals) in the Transformers backend.
- The `DecisionTrace` schema and what it must capture to make a decision
  strictly replayable. Phase 2A and 2A.1 define a replay-oriented trace (section
  5); what a stricter snapshot-based replay would additionally require is still
  open.
- ~~The calibration algorithm(s) and the serialized shape/versioning of a
  calibration profile (Phase 4).~~ Settled by Phase 4C: one explicit
  L2-regularized logistic method, a versioned canonical JSON serialization, and
  an exact identity-verified store. Which additional methods to add, and how to
  choose among them, remains open.
- How temperature scaling interacts with fingerprinting: does a calibrated
  result inherit the plan fingerprint, gain a profile fingerprint, or both?
- Whether normalized entropy and margin should be recomputed on calibrated
  probabilities, raw probabilities, or both, and how the two are distinguished.
- Batching and prefix-cache semantics per backend, and which of them are
  capabilities versus mere configuration.
- The provider-specific backend config surface (where `openai_temperature` and
  friends finally live) and how it is kept out of core types.
- Numeric representation of probabilities (precision, float type) across
  backends, and how the `1e-9` tolerance interacts with it.
- Fingerprint schema versioning: what happens to stored fingerprints when the
  canonical form of a Phase 1 type changes.
- The shape of abstention and risk-coverage evaluation in Phase 5.
- How the scoring doctrine is versioned and fingerprinted, and which of those
  identifiers enters the lineage.
- How a compiler version or fingerprint is recorded and stabilised across
  releases.
- How a calibration profile is versioned against the artifacts that actually
  affect probability semantics, rather than a bare model name.
- Which trace retention mode(s) exist, and what each mode must persist for a
  decision to be replayable.
- What incremental decision state, if any, must be part of lineage versus what
  can be recomputed from fingerprints.
- Whether AP-03 and AP-04 will eventually need testable invariants of their
  own.
- Whether the runtime should detect that a model was not at a decision point
  (its next token was not one of the scored candidates) and refuse, or whether
  refusal belongs entirely to the Phase 5 policy layer. Detection and refusal
  are separable: an experiment showed the same plan yielding `P(True) = 0.3479`
  or `0.9951` depending only on the chat-template rendering mode, so this
  question is about honesty of the number, not about performance.
- Whether rendering configuration that demonstrably affects probability
  semantics (a model's thinking mode and similar template switches) must be
  promoted into a versioned, lineage-recorded artifact instead of remaining
  untracked backend configuration. Today such a change alters the probability
  while leaving the plan fingerprint untouched.
- When a `verbalizer_mass` threshold may become an automatic scoring-validity
  decision, and where it would live. Today `ScoringDiagnostics` only measures and
  records. A future `ScoringValidityPolicy` would take the mass, the top token,
  and model- or task-specific empirical distributions and return accept, reject,
  or warn. No threshold is chosen now, because no experiment justifies one that
  is stable across models, tokenizers, chat templates, verbalizers, and prompts.
