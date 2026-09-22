# FuzzyAI Design Constitution

Status: **early development**, Phase 1.

This is the highest-level, longest-lived document in the repository. It is the
binding design constraint for all future work in FuzzyAI. Code, tests, docs, and
APIs that contradict this constitution are wrong, even if they pass review. When
a future change alters a rule here, this document must be amended first.

---

## 1. Purpose and Core Principle

FuzzyAI is a **provider-agnostic probabilistic decision runtime**. It turns
language models into decision components whose outputs are evaluable,
calibratable, and trackable as semantic probabilities.

The problem it addresses: LLMs are routinely wired into program logic as if a
raw text blob or an unexamined score were a trustworthy probability. The result
is systems that cannot say what a score means, cannot reproduce a past decision,
and cannot separate "the model was unsure" from "the program should decline to
act". FuzzyAI gives that space a narrow, deterministic core.

The core principle, stated once and enforced everywhere:

> **The LLM handles semantic uncertainty. The program handles deterministic policy.**

FuzzyAI is NOT:

- a chat framework,
- an agent framework,
- merely a structured-output wrapper.

If a proposed feature fits one of those descriptions better than the
description above, it does not belong in FuzzyAI.

---

## 2. Glossary (normative)

These five terms have exactly the meanings below. Every public type, docstring,
error message, and doc must use them this way. Each definition includes an
explicit "what it is NOT" clause because the failure mode this project fights is
terminology drift.

### Probability

The decision distribution produced by **some scoring strategy**. A probability
is always relative to the strategy and evidence that generated it.

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

---

## 4. Architecture and Layer Boundaries

The layering below is the proven design. It is **not yet wired end-to-end**; the
Compiler and all real backends are future work.

```
DecisionSpec  --(Compiler, future)-->  InferencePlan  -->  Backend (Protocol)  -->  RawEvidence
                                                                                        |
                                                          (future scoring/calibration)  v
                                                                                  DecisionResult
```

| Layer | Phase 1 status | Responsibility | Must not know about |
|---|---|---|---|
| `DecisionSpec` (`BoolDecision`, `ChoiceDecision`) | implemented | Declare WHAT semantic decision to make: the question, the candidates (ordered), the context. Provide `fingerprint`. | Models, providers, tokens, scoring. |
| Compiler | future | Choose a scoring strategy from declared capabilities; lower a spec into a plan. Raise `UnsupportedCapabilityError` when no declared capability supports the needed strategy. | Business policy, results. |
| `InferencePlan` | abstraction only | Provider-independent description of the inference to run, including `ScoringStrategy`. Fingerprintable. | Any provider-specific knob (INV-17). |
| `Backend` (Protocol) | protocol only | Declare `capabilities` explicitly; `execute(plan)` and return raw output. | Decisions, results, certainty (INV-16). |
| `RawEvidence` | abstraction only | Carry raw model output (`EvidenceKind`) before any conversion; optional `dict[str, JSONValue]` metadata. | Probability semantics (INV-18). |
| Scoring / calibration | future | Convert `RawEvidence` into a `DecisionResult`; calibration only after Phase 3+ evidence exists. | Policy decisions. |
| `DecisionResult` (`BoolResult`, `ChoiceResult`, `Certainty`) | implemented | Report the probability distribution, certainty, `predicted_correctness=None`, `calibrated=False`. | What to do about the answer. |
| Policy | out of scope by design | Map a result plus risk tolerance onto `accept` / `abstain` / `review` / `escalate`. | (Consumes results; owns abstention.) |

### Phase 1 scope

Implemented in Phase 1: `DecisionSpec` (`BoolDecision`, `ChoiceDecision`), the
result model, `BackendCapabilities`, the `Backend` Protocol, `InferencePlan` and
`RawEvidence` abstractions, the fingerprint system, and the error taxonomy.
No real model backend exists yet. Phase 1 defines ONLY the `BoolDecision` and
`ChoiceDecision` primitives; `Score`, `MultiLabel`, `Rank`, `Preference`, and
`Compare` are roadmap only.

Public API (Phase 1):

```
BoolDecision, Choice, ChoiceDecision,
BoolResult, ChoiceResult, DecisionResult, Certainty,
normalized_entropy, probability_margin,
BackendCapabilities, Backend,
InferencePlan, ScoringStrategy, RawEvidence, EvidenceKind,
JSONValue, canonical_json, fingerprint,
FuzzyAIError, InvalidDecisionError, InvalidProbabilityError,
UnsupportedCapabilityError, FingerprintError
```

Fingerprint API surface: `canonical_json(value)`, `fingerprint(value)`,
`BoolDecision.fingerprint`, `ChoiceDecision.fingerprint`, and
`InferencePlan.fingerprint`.

Error taxonomy: `FuzzyAIError` is the base; `InvalidDecisionError`,
`InvalidProbabilityError`, `UnsupportedCapabilityError`, and `FingerprintError`
are its Phase 1 subclasses.

Phase 1 non-goals (binding): no real model loading, no HTTP, no
transformers/OpenAI/Anthropic/vLLM/SGLang, no automatic routing, no decision
graph, no calibration algorithm, no dashboard, server, agent, RAG, database,
telemetry, or web UI.

---

## 5. Policy vs Model Boundary

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
  FuzzyAI may later ship policy primitives (Phase 5), but they will be explicit
  objects, never hidden defaults inside the model path.
- "The model was unsure" and "the program declined to act" are different events
  and must remain representable separately.

---

## 6. Design Trade-off Priority Order

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

## 7. Open Questions Deferred to Later Phases

Recorded as open. None of these has a Phase 1 answer, and Phase 1 must not
smuggle one in.

- Which scoring strategies the Phase 2 compiler will support, and how
  `ScoringStrategy` and `EvidenceKind` members get finalized.
- How multi-token labels aggregate into a single categorical probability
  (sum, product, or max of token-level signals) in the Transformers backend.
- The `DecisionTrace` schema and what it must capture to make a decision
  replayable.
- The calibration algorithm(s) and the serialized shape/versioning of a
  calibration profile (Phase 4).
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
