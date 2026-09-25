**English** | [简体中文](README.zh-CN.md)

# Probvenance

**Status: early development.** The deterministic core, a Bool vertical slice
with a real local Hugging Face backend, and direct categorical Choice inference
for the closed-set, single-label, single-token scoring path are implemented.
The Phase 4B winner-correctness evaluation foundation is implemented, as is the
`CalibrationProfile` identity foundation: the profile artifact commits its
binding, ground-truth semantics, target, input-score, method, fitted-parameter,
and training-dataset provenance, and fails closed on a concrete taxonomy
contradiction or an exact binding mismatch. One supported scalar fitting method
is implemented: `fit_l2_logistic_selected_probability` fits an L2-regularized
logistic map of the selected probability onto winner correctness. Offline
profile application and the post-calibration evaluation foundation are
implemented: one exact profile can be applied to one compatible evaluation
dataset to produce an immutable predicted-winner-correctness artifact that
records both the derived winner-correctness target label and the produced score
per row, and post-calibration Brier, exact log loss, companion diagnostics,
equal-width reliability, and the binned absolute-gap aggregate consume that one
artifact.
Explicit runtime-linked profile application is implemented: a caller may apply
one exact compatible profile to one uncalibrated runtime `Evaluation` to obtain
a calibrated result and a trace that mirror the profile identity. A profile also
has a versioned canonical JSON serialization and an identity-verified loader:
the loader reconstructs the nested binding and ground-truth-semantics
identities and re-verifies every fingerprint rather than trusting the document.
An exact content-addressed directory store is implemented: it persists a profile
under a path derived only from its fingerprint schema version and exact
fingerprint, and retrieval requires both values, restores the artifact through
the identity-verified loader with the requested identity as an independent
expected pin, and performs no matching or fallback. Explicit runtime profile
selection is implemented: `select_calibration_profile_for_runtime` takes one
uncalibrated evaluation and an explicit caller-supplied tuple of candidate
profiles and returns the single eligible profile, raises an explicit no-eligible
error when none is eligible, and raises an ambiguity error when more than one
distinct profile is eligible, with no tie-break. Selection is authorization, not
recommendation: it is storage-agnostic, applies nothing, and uses neither
quality metrics nor method support as a preference. An explicit in-memory profile
catalog and non-authoritative runtime discovery are implemented: a
`CalibrationProfileCatalog` is built from a caller-supplied tuple of profiles,
and `discover_calibration_profile_references_for_runtime` returns deterministic
exact profile references whose discovery metadata matches the same runtime
eligibility projection as selection, while authorizing, loading, selecting, and
applying nothing. A catalog also has an exact fingerprinted snapshot identity
(`canonical_payload` / `fingerprint`) and deterministic, identity-verified
serialization and loading (`serialize_calibration_profile_catalog` /
`load_calibration_profile_catalog`): the snapshot commits the ordered references
plus the Binding and target/input discovery projection, and loading recomputes
the canonical payload and catalog fingerprint rather than trusting the embedded
hash. An exact content-addressed directory store for those snapshots is
implemented (`DirectoryCalibrationProfileCatalogStore`): a snapshot is persisted
under a path derived only from the catalog store layout version, the catalog
fingerprint schema version, and the exact catalog fingerprint, and retrieval
requires both values, restores the snapshot through the identity-verified catalog
loader with the requested identity supplied as an independent expected pin, and
does no lifecycle, latest/default, alias, enumeration, or fallback matching. A
catalog snapshot is non-authoritative discovery metadata: its fingerprint proves
the snapshot identity, not that referenced profiles exist, that store artifacts
are intact, or that the snapshot metadata still matches the real profiles.
Automatic runtime profile selection, profile registries, binding-based lookup,
catalog lifecycle (latest, active, default, or production channels), store
enumeration, quality ranking, and a signed distribution do not exist, and
`predicted_correctness` remains `None` for every runtime result that no caller
has explicitly calibrated. Abstention and every other Choice strategy are not
implemented.

**Phase 4C is complete.** The calibration layer now spans the whole
identity-verified chain, from runtime evaluation and ground-truth observation
through fitting dataset contracts, exact profile fitting, offline application
and post-calibration evaluation, profile identity, serialization, and exact
profile storage, explicit unique-or-fail selection, non-authoritative catalog
discovery, catalog snapshot identity, catalog serialization, and exact catalog
snapshot storage. Every boundary in that chain is exact, explicitly gated, and
fails closed. Catalog lifecycle (latest, active, default, or production
channels and supersession), registries, automatic store or catalog
synchronization, store enumeration, quality ranking, a signed distribution,
automatic selection, and automatic calibration are later-phase directions
rather than unfinished Phase 4C work; there is no Phase 4C.12 for lifecycle
convenience.

Probvenance is a provider-agnostic probabilistic decision runtime. It turns language
models into evaluable, calibratable, trackable semantic-probability decision
components.

The problem: LLMs get wired into program logic as if a raw text blob or an
unexamined score were a trustworthy probability. Systems end up unable to say
what a score means, unable to reproduce a past decision, and unable to separate
"the model was unsure" from "the program should decline to act". Probvenance puts a
narrow, deterministic core under that space.

It is not a chat framework, not an agent framework, and not merely a
structured-output wrapper.

## Core principle

> **The LLM handles semantic uncertainty. The program handles deterministic policy.**

## Currently working: the Bool and Choice vertical slices

Two end-to-end paths exist today: the Bool slice and the experimental direct
categorical Choice slice. A thin `Probvenance` facade orchestrates each pipeline in
a fixed order and adds no fallbacks of its own. Since Phase 2A.1 the Bool slice
also reports scoring-validity diagnostics and an execution fingerprint on every
trace:

```
BoolDecision -> BoolCompiler -> InferencePlan -> TransformersBackend
   -> RawEvidence -> assemble_bool_probability -> BoolResult -> DecisionTrace
```

The backend is an optional extra. The base package has zero runtime
dependencies; installing the extra is what pulls in `torch` and `transformers`:

```bash
uv sync --extra transformers    # group: torch>=2.7, transformers>=4.53
```

```python
from probvenance import BoolDecision
from probvenance.backends.transformers import TransformersBackend
from probvenance.runtime import Probvenance

backend = TransformersBackend(
    "Qwen/Qwen3-0.6B",  # a local HF causal LM
    # Qwen3 starts a reasoning block first otherwise; see the note below.
    chat_template_kwargs={"enable_thinking": False},
)
ai = Probvenance(backend=backend)

evaluation = ai.evaluate_with_trace(
    BoolDecision(
        question="Is this a delivery issue?",
        context="My package never arrived.",
    )
)

print(evaluation.result.probability_true)
print(evaluation.trace.scoring_diagnostics.verbalizer_mass)
print(evaluation.trace.execution_fingerprint)
```

`ai.evaluate(decision)` is the same pipeline returning only the `BoolResult`
(`probability_true`, `certainty`, `predicted_correctness`, `trace_id`).

`ai.evaluate_with_trace(decision)` returns an `Evaluation` (the result plus its
`DecisionTrace`). The trace records the decision and plan fingerprints, the
scoring strategy, the doctrine id, the resolved verbalizer token ids, the raw
evidence, the input fingerprint, the backend type, the latency, the
`ScoringDiagnostics`, and the execution fingerprint. The trace is
replay-oriented provenance: it records what a future replay would need, but
it does not snapshot backend or tokenizer code, so strict replayability
remains an open question.

What `probability_true` is: a conditional, restricted probability,
`P(True | next token is one of the two scored verbalizer tokens)`, computed as
a numerically stable two-way softmax over exactly two verbalizer-token logits,
equal to `sigmoid(l_true - l_false)`. It cannot tell you whether the model
intended to choose among those candidates at all. What it is not: a
full-vocabulary probability, a real-world event probability, a
prediction-accuracy figure, or a calibrated number. `calibrated` stays `False`
and `predicted_correctness` stays `None`.

Because the restricted number alone cannot show whether the model was at a
decision point, every trace also carries an independent full-vocabulary
quantity, `trace.scoring_diagnostics.verbalizer_mass`:
`P(next token is one of the two scored verbalizer tokens)`, with the stable
log form `log_verbalizer_mass = logsumexp([l_true, l_false]) -
logsumexp(all_vocab_logits)`. The interpretation rule: `probability_true =
0.75` together with `verbalizer_mass = 0.000001` means the model internally
prefers `yes` over `no` but was almost certainly not about to output either,
so `0.75` must not be read as "75% inclined to answer yes" unless its
restricted nature is stated. The diagnostics carry no verdict: Phase 2A.1
applies no threshold and performs no auto-rejection.

The backend loads a local causal LM, runs it under `eval()` and
`torch.inference_mode()`, prefers CUDA and falls back to CPU, and reads only
the final-position logits. It never calls `generate()` and never parses
generated text. Verbalizers default to `yes`/`no` and must each resolve to
exactly one distinct scoring token after the real chat-templated prefix;
otherwise a `VerbalizerError` is raised. There is no silent fallback, no
truncation, no multi-token summing, and no sampling fallback.

`probability_true` is only meaningful if the model is genuinely at a decision
position, meaning its next token really is one of the two verbalizers. Some
models (thinking or reasoning models) start a reasoning block instead: with the
wrong chat-template mode the model's top token is its reasoning opener and both
verbalizer tokens sit in the far tail, so the two-way softmax renormalizes tail
noise into a number that looks plausible. The demo above therefore renders with
`chat_template_kwargs={"enable_thinking": False}`. The diagnostics make this
visible: the measured difference is recorded in
[docs/claims.md](docs/claims.md), where the same plan and question gave
`P(True) = 0.9988` with `verbalizer_mass = 0.954228` (top token `'yes'`, id
9693, probability 0.953119) when thinking was disabled, and `P(True) = 0.5116`
with `verbalizer_mass = 0.000000` (top token `'<think>'`, id 151667,
probability 0.999699) when thinking was enabled. Phase 2A.1 does not detect
this automatically and rejects nothing, because no threshold has experimental
support across models, tokenizers, chat templates, verbalizers, and prompts;
refusing to answer would be a policy action that does not exist yet.

Every evaluation carries four distinct identities, each answering a different
question:

- decision fingerprint: what semantic question is being judged
- plan fingerprint: what the compiler produced
- execution fingerprint: the execution environment and rendering
  configuration the plan actually ran under
- trace id: which single execution this was

Every evaluated probability is additionally traceable to an explicit probability
formulation identity and formulation-family identity (see
`docs/probability-semantics-identity.md`); neither names a model, a tokenizer, or
an input.

The execution fingerprint payload covers the plan fingerprint, backend type,
backend implementation version, model identifier, model revision, tokenizer
identifier, tokenizer revision, runtime version, dtype, rendering config,
input fingerprint, and the resolved positive/negative token ids. Running the
same plan twice yields two trace ids and one execution fingerprint. Rendering
configuration that affects probability semantics (for example a model's
thinking mode) enters the execution fingerprint and the trace, but
deliberately not the provider-independent `InferencePlan`.

Decision context is rendered into the user prompt as evidence only and never
into the system prompt. That is a structural placement rule, not a claim of
prompt-injection safety.

`BoolCompiler` and `ChoiceCompiler` are mutually exclusive: `BoolCompiler`
rejects a `ChoiceDecision` with `UnsupportedDecisionError`, and `ChoiceCompiler`
rejects a `BoolDecision`. The experimental Choice runtime is described under
[What exists today](#what-exists-today).

Phase 2A.2 used this slice's diagnostics in a semantic signal validation
experiment: does the binary scoring position carry a semantic signal at all,
and which part of the measurement comes from the model, the doctrine, and the
label family? It probed three local causal LMs under fixed conditions (one
forward pass per probe, no ground truth) and recorded what it observed. The
record lives in
[experiments/semantic_signal/REPORT.md](experiments/semantic_signal/REPORT.md):
an experiment record, not a capability claim and not a benchmark. One
concrete outcome of the round was a diagnostics fix: the overshoot clamp in
`src/probvenance/diagnostics.py` is now a relative tolerance (see
[docs/claims.md](docs/claims.md)).

## Planned API (not implemented)

The convenience surface below is still the **planned API**. There is no
`ai.bool` or `ai.choice` entry point; the working calls are
`ai.evaluate(BoolDecision(...))` and `ai.evaluate(ChoiceDecision(...))`.

```python
risk = ai.bool("Is this transaction suspicious?", context=transaction)

route = ai.choice(
    "Which team should handle this ticket?",
    choices={
        "billing": "Payment and billing issues",
        "shipping": "Delivery and logistics",
        "returns": "Returns and refunds",
    },
    context=ticket,
)
```

## Probability, certainty, predicted correctness: not the same thing

- **Probability**: the decision distribution produced by some scoring strategy.
  It is not automatically a real-world correctness rate. Today's Bool
  probability is a restricted (conditional) probability over the two
  verbalizer-token logits, not a full-vocabulary distribution; the trace's
  `verbalizer_mass` is the independent full-vocabulary companion quantity.
- **Certainty**: how concentrated that distribution is (entropy, margin). A
  mathematical property, not a correctness probability. We never call it
  "confidence".
- **Predicted correctness**: only expressible after valid calibration. Every
  result produced today has `predicted_correctness = None` and
  `calibrated = False`. A max softmax, a logit, an entropy, or an LLM saying
  "I'm 95% sure" is none of these things.

The full normative definitions live in the
[design constitution](docs/design-constitution.md).

## What exists today

The Phase 1 deterministic core:

- `BoolDecision` and `ChoiceDecision` decision specs
- The result model: `BoolResult`, `ChoiceResult`, `Certainty`
- `BackendCapabilities` (explicit data) and the narrow `Backend` Protocol
- `InferencePlan` and `RawEvidence` abstractions
- The deterministic fingerprint system (canonical JSON + SHA-256)
- The error taxonomy rooted at `ProbvenanceError`

The Phase 2A Bool path:

- `ScoringDoctrine` and the default `BINARY_SEMANTIC_JUDGMENT_V1` doctrine
- `BoolCompiler` (a pure planner: it never computes probabilities)
- `assemble_bool_probability` (two-way softmax over the binary token logits)
- `TransformersBackend` (optional `transformers` extra, logits only)
- `DecisionTrace` / `build_decision_trace` (its `trace_id` is never derived
  from a fingerprint)
- The thin `Probvenance` / `Evaluation` facade

The Phase 2A.1 scoring-validity increment:

- `ScoringDiagnostics` / `diagnose_bool_evidence` (full-vocabulary statistics:
  `verbalizer_mass`, top token id, probability, and best-effort text; no
  threshold, no verdict)
- The execution fingerprint on `DecisionTrace` (two executions of one plan:
  two trace ids, one execution fingerprint)

```
BoolDecision --(BoolCompiler)--> InferencePlan --(TransformersBackend)--> RawEvidence
                                                                             |
                              assemble_bool_probability     (calibration: explicit only; see below)
                                                                             v
                                                BoolResult + DecisionTrace (via Probvenance)
```

The Phase 2B direct categorical Choice path:

- `ChoiceCompiler` and the `CATEGORICAL_SEMANTIC_JUDGMENT_V1` doctrine, with the
  versioned `categorical-labels-v1` label scheme
- `CandidateLabelMapping` (semantic candidate to scoring label, deliberately
  carrying no token id, so the plan stays provider-independent)
- `assemble_choice_probability` (N-way softmax over the candidate logits) plus
  `ChoiceScoringDiagnostics` (`scoring_label_mass`, top token, per-candidate token
  probabilities)
- N-way scoring-label resolution and validation in `TransformersBackend`
- `ChoiceResult` keyed by semantic candidate names, with ties resolved on
  semantic candidate order

```
ChoiceDecision --(ChoiceCompiler)--> InferencePlan --(TransformersBackend)--> RawEvidence
                                                                                 |
                        assemble_choice_probability    (calibration: explicit only; see below)
                                                                                 v
                                             ChoiceResult + DecisionTrace (via Probvenance)
```

This path is experimental. It is closed-set, assumes the caller supplies
mutually exclusive candidates, supports single-label decisions only, requires
every scoring label to be exactly one token, has been studied only at small N,
and is **uncalibrated** by default (`predicted_correctness` is `None` unless a
caller explicitly applies a profile). It has no
open-set guarantee: when the candidate set omits the true topic, the model still
answers in-set and `scoring_label_mass` does not detect it.

Other Choice strategies (one-vs-rest, sampling, multi-token scoring labels),
automatic calibration, and abstention remain future work.

## Evidence and claims

Performance, quality, calibration, and provider-support claims are tracked by
evidence status in [docs/claims.md](docs/claims.md). The project makes no
benchmark, performance, or model-support claims today: the Bool path runs
against a local Hugging Face causal LM, but no model has been evaluated
against ground truth. Phase 2A.2 ran a semantic signal validation experiment
over the binary scoring position against three local Hugging Face causal LMs;
its record lives in
[experiments/semantic_signal/REPORT.md](experiments/semantic_signal/REPORT.md)
and documents observed signal behaviour under the conditions it states. That
is an experiment record, not a capability claim and not a benchmark.
Phase 2B ran a direct categorical Choice experiment over label permutations,
irrelevant candidate addition, description paraphrase, taxonomy overlap, and
out-of-set probes against a local model, and Phase 2B.1 replicated it on a second
model family under the same frozen case set; the record lives in
[experiments/choice_signal/REPORT.md](experiments/choice_signal/REPORT.md).
Hypotheses and roadmap items are labelled as such in the claims register,
rather than presented as capabilities.

## Development

The project's own dev commands (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

The `transformers` backend tests skip automatically unless
`uv sync --extra transformers` has been run; they never download models and
never need a GPU.

## Documentation

- [docs/design-constitution.md](docs/design-constitution.md): binding design
  constraints, glossary, and normative invariants. Read this before proposing
  anything.
- [docs/roadmap.md](docs/roadmap.md): intended phases. Not a schedule, not a
  commitment.
- [docs/claims.md](docs/claims.md): the evidence status of every capability claim ([V] verified, [E] experimental, [H] hypothesis, [R] roadmap).

## Non-goals

There is still no HTTP, no OpenAI/Anthropic/vLLM/SGLang integration, no cloud
backend, no automatic routing, no decision graph, no automatic calibration, no
abstention policy, and no dashboard, server, agent, RAG, database, telemetry,
or web UI. This project makes no benchmark or performance claims, and claims no
model quality or support.

## License

MIT. See [LICENSE](LICENSE).
