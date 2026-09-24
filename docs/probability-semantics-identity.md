# Probability Semantics Identity and Formulation Families

Status:

```text
Probability identity semantics:            design contract frozen
Runtime ProbabilityFormulationFingerprint: implemented
Runtime FormulationFamilyFingerprint:      implemented
Comparability policy:                      not implemented
Calibration:                               partially implemented (one offline
                                           L2-regularized logistic method, plus
                                           offline profile application and a
                                           post-calibration evaluation
                                           foundation; no runtime application)
```

Both runtime identities are derived from an `InferencePlan` alone and are exposed
as read-only plan properties and on `DecisionTrace`; see
`src/probvenance/probability_identity.py`. They are derived values and are
deliberately not part of any fingerprint payload. No comparability policy and no
pooling policy exist: the identities say which formulation
produced a probability, never whether two probabilities may be treated alike.
Where this document shows compiler and assembler identifiers and versions they
match the runtime. The doctrine identity matches too: a plan carries
`doctrine_id` and `doctrine_version` as separate declared fields, both enter the
formulation identity, and no version is ever inferred from the id string. This document answers which probabilities may be treated as
the same kind of object, and which merely come from the same decision.

## 1. Motivation

Probvenance already carries four identities:

```text
Decision Fingerprint    WHAT semantic question is asked
Plan Fingerprint        the exact compiled plan
Execution Fingerprint   the exact execution configuration and rendered input
Trace ID                one execution instance
```

Those four are not enough for the questions the project will face next:

```text
Calibration        which executions may learn one correctness mapping together?
Eval aggregation   which results may be averaged?
Regression         did the numbers change because the code changed, or because
                   the probability object changed?
Replay             can this exact execution be reproduced?
Shadow evaluation  are these two numbers even the same kind of number?
```

The tempting shortcuts are all wrong:

```text
same Decision Fingerprint  => comparable probability        (false)
same model                 => comparable probability        (false)
Plan Fingerprint           => calibration identity          (too fine)
Execution Fingerprint      => calibration identity          (too concrete)
```

Two measured facts from earlier phases force this document to exist:

- Bool: probability semantics depends on the model and the formulation
  (`experiments/semantic_signal/REPORT.md`).
- Choice: the semantic winner can stay stable while the full uncalibrated
  categorical distribution moves materially when only the scoring-label
  representation changes (`experiments/choice_signal/REPORT.md`).

So the project needs a concept that answers:

> Do these two probability outputs share the same probability semantics, such
> that they may be interpreted together?

This document defines that concept. It does not implement it.

## 2. Existing identities and why each is insufficient

Read against the real payloads.

### Decision Fingerprint

```json
{"v": 1, "kind": "bool", "question": "...", "context": "..."}
{"v": 1, "kind": "choice", "question": "...", "context": "...",
 "choices": [{"name": "...", "description": "..."}]}
```

It expresses what is being asked and over which alternatives. It does not
contain the scoring strategy, the doctrine, the scoring labels, the
candidate-to-label mapping, the compiler formulation, the model, or the
rendering mode. It is far too coarse to describe a probability.

It is also evidence-specific: `question` and `context` are inside it.

### Plan Fingerprint

```json
{"v": 3, "kind": "inference_plan",
 "decision_fingerprint": "...", "strategy": "...",
 "prompt": "...", "system_prompt": "...", "targets": ["A", "B", "C"],
 "positive_verbalizer": null, "negative_verbalizer": null,
 "doctrine_id": "...", "label_scheme_id": "...",
 "candidate_mapping": [{"candidate_index": 0, "candidate_name": "...",
                        "candidate_description": "...", "scoring_label": "A"}],
 "required_capabilities": {"...": true}}
```

It expresses one exact compiled plan. Two things in it are evidence-specific:
`decision_fingerprint` commits to `question` and `context`, and the rendered
`prompt` and `system_prompt` carry their text. So two questions under one
formulation have two different plan fingerprints. It is provider-independent
(no model, tokenizer, or rendering), which is correct, but it is the wrong
granularity for grouping: a calibration profile keyed on a plan fingerprint
would give almost every sample its own profile.

It also omits something it should not: the identity of the probability
assembler. The plan names the strategy, which implies the transformation class,
but it never names the transformation independently of the strategy, so a
change to the assembler under a stable strategy would not be visible here.

### Execution Fingerprint

```json
{"v": 2, "kind": "execution",
 "plan_fingerprint": "...", "backend_type": "...", "backend_version": "...",
 "model": "...", "model_revision": null, "tokenizer": "...",
 "tokenizer_revision": null, "runtime_version": "...", "dtype": "...",
 "rendering_config": {"...": "..."}, "input_fingerprint": "...",
 "positive_token_id": -1, "negative_token_id": -1,
 "resolved_target_token_ids": [["A", 123], ["B", 456], ["C", 789]]}
```

It expresses the actual execution environment, the actually rendered input, and
the resolved scoring token ids. The binary token id fields are `-1` when unused,
as in this categorical example. It is the right identity for replay, provenance,
and audit. It is more concrete than the plan and includes instance evidence
through `input_fingerprint`. It is not a natural calibration grouping key.

### Trace ID

One execution instance. It is not a probability-semantics identity at all.

### Conclusion

```text
One fingerprint cannot serve exact provenance, probability-space identity,
formulation-family grouping, and calibration grouping at once. The purposes are
different and the granularities conflict.
```

## 3. Probability semantics versus probability value

**Probability semantics** is the mapping from raw model evidence to an outcome
distribution under a declared strategy, scoring representation, and
probability assembler.

The separation this document is built on:

```text
Instance evidence (question, context) changes the probability VALUE.
It does not change what the probability MEANS.
```

```text
P(shipping | package is missing)     different value
P(shipping | package arrived late)   different value
                                     same interpretation
```

If `question` or `context` entered a formulation identity, nearly every
inference would form its own family and the identity would be useless for
grouping. Therefore:

```text
probability formulation identity EXCLUDES question and context.
```

The decision fingerprint already records the evidence. There is no loss.

The consequence is the plan-relative semantics already named in
`docs/choice-semantics.md`: a probability belongs to a formulation, not to a
decision alone.

## 4. Outcome-space identity

### Bool

The outcome space is fixed:

```text
Omega = {False, True}
```

The semantic outcomes do not vary between `BoolDecision` instances. What varies
is the scoring representation (the verbalizer pair), handled in section 5.

### Choice

The outcome space is the ordered candidate set, and it is semantics-bearing:

```text
P(billing | {billing, shipping, returns})
!=
P(billing | {billing, shipping, returns, technical})
```

Measured across Phase 2B and Phase 2B.1: adding a fourth candidate moved the
distribution by up to 0.0153 total variation on `Qwen/Qwen3.5-2B`, and on
`openbmb/MiniCPM5-2B` it flipped one case entirely (total variation 0.9629).
The outcome space is therefore part of the formulation identity, and it
includes:

```text
ordered candidate names
candidate descriptions
closed-set assumption
single-label / mutually-exclusive assumption
```

**Candidate order.** The runtime outcome space is ordered, because order
decides the deterministic tie-break and therefore `ChoiceResult.value` (INV-03).
As a runtime object, `[billing, shipping, returns]` and `[returns, billing,
shipping]` are not the same outcome space, so order is included.

This does not forbid order-independent reasoning. The restricted distribution
is also a function over semantic candidate names, and two such functions can be
compared order-independently by aligning on names. That comparison is a
property of the values, computed by the Eval layer, not a property of the
identity. The document keeps both facts apart on purpose:

```text
runtime deterministic semantics    order-dependent   (tie-break, value)
mathematical distribution identity order-independent (a map over names)
```

**Candidate descriptions.** A description participates in the prompt and shapes
how the model interprets a candidate. Phase 2B.1 measured a description
paraphrase moving the distribution by up to 0.3560 total variation on
MiniCPM5-2B with the candidate name unchanged, and the largest effect landed on
a `technical` case rather than the paraphrased candidate. The conservative rule
is adopted:

> If a description participates in the prompt and affects how the model
> interprets a candidate, it is probability-semantics-relevant.

So descriptions are included in the outcome-space identity.

### No automatic paraphrase equivalence

Probvenance must not attempt to decide that two descriptions mean the same thing.
That would require another semantic model and would put an unaccountable
judgement inside an identity primitive. If a caller wants several
descriptions to sit in one taxonomy family, the mechanism is an explicit
caller-supplied taxonomy identity (sections 9 and 10), not automatic inference.

## 5. Scoring representation identity

The semantic outcome space and the scoring representation are different
objects, and the project already froze the underlying separation: INV-20 states
that semantic candidate identity is not scoring-label identity, and INV-21
states that the scoring representation lives in the plan fingerprint rather
than the decision fingerprint.

```text
semantic candidate set     billing, shipping, technical
scoring representation     billing->A, shipping->B, technical->C
```

### 5.1 Candidate-to-label mapping (Choice)

The same semantic candidate set admits different representations:

```text
billing->A, shipping->B, technical->C
billing->C, shipping->A, technical->B
```

Same decision, same candidates, same descriptions, and yet Phase 2B.1 measured
the full distribution moving by up to 0.3082 total variation on
`Qwen/Qwen3.5-2B` (Phase 2B, reproduced in 2B.1) and 0.4672 on
`openbmb/MiniCPM5-2B` (Phase 2B.1) purely under label permutation. The exact
formulation identity must distinguish these two representations.

### 5.2 Verbalizer family (Bool)

The binary path has the same structure one level down. Its scoring
representation is the verbalizer pair, so different pairs are different
probability objects even when the semantic outcome space is unchanged:

```text
"yes" / "no"
"true" / "false"
"A" / "B"
```

Phase 2A.2 measured that the binary label family is not a neutral variable:
polarity consistency held at 0.92 to 1.00 across families, but the A/B label
family behaved differently from the word families and interacted with the
model. A surface-token-bound bias appeared: the model's preference attached to
the literal token (`true` versus `Yes`) rather than only to the polarity.

The conservative rule is therefore adopted:

```text
yes/no, true/false, and A/B are DIFFERENT exact formulation identities.
They may share a formulation family (same binary token logits mechanism).
```

They are not merged by default. A caller who believes two verbalizer families
are interchangeable must carry evidence and an explicit pooling rule, exactly
as for Choice label permutations.

### What the scoring representation identity includes

```text
for Choice:   label scheme id and version, candidate-to-label mapping
              (candidate index, candidate name, candidate description,
              scoring label)
for Bool:     positive and negative verbalizer strings
```

It deliberately does not include the resolved token ids (section 8).

## 6. Exact formulation identity

### Name

```text
ProbabilityFormulationFingerprint
```

The name describes the probability formulation. It avoids any suggestion of
empirical calibration, correctness, or quality. In prose this document calls it
the **exact formulation identity**. Here "exact" means exact in representation,
not exact in evidence: it is question-independent by design (section 3), and two
different questions over the same candidate set share it.

### Definition

> A probability formulation identity describes the declared mathematical and
> representational conditions under which a probability was produced: the
> outcome space it ranges over, the scoring representation that maps outcomes to
> model-scored labels, the strategy and probability assembler that turn model
> evidence into the distribution, the doctrine and compiler versions that fixed
> the formulation, and the semantic assumptions the caller declared.

It is provider-independent and evidence-independent in the narrow sense of
section 3. It sits between the decision fingerprint (too coarse, too
evidence-specific) and the execution fingerprint (too concrete).

### Field-by-field decisions

| Dimension | Decision | Reason |
|---|---|---|
| decision family (bool / choice) | include | A bool `[p, 1-p]` and an N-way distribution are different objects with different outcome spaces. Since Phase 4A the family is DECLARED by `InferencePlan` (`decision_family`) and is no longer inferred from the scoring strategy; the plan validates the declared family against an explicit table of legal implemented-strategy/family combinations. |
| scoring strategy | include | `binary_token_logits`, `categorical_token_logits`, `token_logprobs`, and future `one_vs_rest`, `sampling`, `pairwise` are different mathematical sources even when they emit `{"A": 0.7, "B": 0.3}`. |
| probability assembler id and version | include | The assembler defines the probability transformation (restricted N-way softmax, binary verbalizer softmax, future OVR normalization or pairwise aggregation). Since Phase 2C.0 it is named independently of the strategy on the plan (`assembler_id`, `assembler_version`) and in the plan fingerprint. |
| doctrine id and version | include | The doctrine fixes how the model is asked to interpret candidates. It shapes the formulation. |
| compiler id and version | include | The compiler chooses the representation. A version bump can change the formulation, so it must be visible. |
| semantic outcome space | include | Section 4. Different candidate sets are different probability objects. |
| candidate order | include (inside the outcome space) | Order decides the tie-break and `value` (INV-03). See section 4 for the order-independent comparison caveat. |
| candidate descriptions | include (inside the outcome space) | Measured to affect the distribution. |
| scoring representation: Choice mapping plus label scheme | include | Section 5.1. |
| scoring representation: Bool verbalizer pair | include | Section 5.2. The verbalizer pair IS the binary scoring representation. |
| closed-set assumption | include | It is a semantic claim about the outcome space, not an implementation detail. |
| single-label / mutually-exclusive assumption | include | It changes what the distribution means. |
| question | exclude | Instance evidence: changes the value, not the interpretation (section 3). Already in the decision fingerprint. |
| context | exclude | Same. |
| model id | exclude, source axis | Probability source, not formulation. See section 8. Keeps the formulation provider-independent, consistent with the plan-compiler boundary. |
| model revision | exclude, source axis | Same. |
| tokenizer and revision | exclude, source axis | Affects single-token feasibility and token priors, but it is an execution and source property. Feasibility is validated at execution time by the backend (AP-08). |
| rendering config (scoring-relevant subset) | exclude, source axis | Section 8. Thinking on versus off changes scoring-position validity, so it is probability-semantics-relevant, but it is an environment property. |
| resolved scoring token ids | exclude, execution-bound | Already in the execution fingerprint (INV-22 makes their order provenance). Duplicating them would create a second provenance layer for one fact. |
| dtype | exclude, execution-bound | Execution detail. Numerically relevant, handled by the execution fingerprint for replay. |
| backend type and version | exclude, execution-bound | Execution detail. Already in the execution fingerprint. |
| required capabilities | exclude | An execution precondition, not a statement about probability semantics. |
| input and rendered-input fingerprints | exclude, execution-bound | Replay concern, already in the execution fingerprint. |

### The three layers, and how axis separation works

Realized probability behavior depends on three layers:

```text
Formulation axis
    provider-independent probability transformation semantics: the outcome
    space, the scoring representation, the strategy, the assembler, the
    doctrine, and the compiler versions that fixed them.
Source axis
    model, tokenizer, and rendering properties that influence realized
    probabilities without being part of the provider-independent formulation.
Instance evidence
    the question and context that produce a particular value.
```

The rule that assigns a dimension to a layer:

```text
Does it describe the provider-independent transformation from model evidence to
a distribution? It belongs to the formulation axis.
Does it describe properties of the concrete model or rendering environment that
change realized probabilities? It belongs to the source axis.
Does it only fix the conditioning event for one inference? It is instance
evidence, and it is excluded from both identity axes.
```

Splitting model and rendering onto the source axis is required by the plan
compiler boundary and INV-17, which keep the compiled formulation
provider-independent. The consequence is strict: the formulation identity alone
is never a complete statement about what a probability is. Every comparison and
every calibration binding must compose the formulation axis with the source
axis. The formulation fingerprint is the provider-independent half of a pair,
not the whole probability identity.

## 7. Formulation families

### Name

```text
FormulationFamilyFingerprint
```

### What it is

A deliberately coarser grouping over the mechanism class. It answers:

> These executions used the same intended probability-generating mechanism, at
> the same arity, even if the concrete representation instance differed.

A family includes:

```text
decision family
scoring strategy
probability assembler id and version
doctrine id and version
compiler id and version
label scheme id
arity (number of candidates)
```

It excludes candidates, candidate names, candidate descriptions, and the
candidate-to-label assignment. Those belong to the exact identity.

Worked example: `billing->A, shipping->B, technical->C` and `billing->C,
shipping->A, technical->B` have

```text
same Decision Fingerprint
same Formulation Family Fingerprint
different ProbabilityFormulationFingerprint
different Plan Fingerprint
```

**Why arity is included.** Phase 2B.1 reports that `scoring_label_mass` is not
directly comparable across `N` (`experiments/choice_signal/REPORT.md`,
section 12), and reports the N=3 and N=5 drift separately. Arity changes the
restricted softmax support and the candidate-set cardinality, so it is part of
the declared mechanism configuration.

**What arity does not do.** Two N=3 sets with different candidate names are the
same family and different exact identities. Family membership never substitutes
for outcome-space equality.

### What it is not

This is the principle this document exists to freeze:

> Being in the same formulation family does not prove that probability values
> are interchangeable, poolable, or calibratable together.

A family records an intended mechanism class. It does not record empirical
probability behavior. Phase 2B.1 is the counterexample inside one family:
permuting the representation, which keeps the mechanism identical, already
produced substantial total-variation drift. Family membership is a starting
point for a calibration pooling decision, never the decision.

## 8. Model, tokenizer, and rendering: the source axis

The formulation identity is provider-independent. The model is not part of it.
Two options were considered.

**Option A: put the model inside the exact formulation identity.** Consequence:
the same plan compiled for two models would carry two different formulation
identities, the family could no longer be a pure mechanism class, and the
provider-independent plan would gain a provider-dependent twin. Rejected.

**Option B (adopted): keep the source on a separate axis.** The source identity
is a declared, structured record, not a third fingerprint this round. It
consists of:

```text
model
model revision
tokenizer
tokenizer revision
rendering semantics (the scoring-relevant subset)
```

Call it the **probability source identity**. It is in practice already pinned
by the execution fingerprint, and this document does not propose a new
fingerprint for it. A calibration binding composes the formulation identity
(or family) with it (section 9).

### Rendering semantics

Only rendering configuration that can change scoring semantics belongs here.
The known member today is `enable_thinking`. The rules:

```text
Only keys with a documented effect on scoring-position validity are included.
An arbitrary provider config dict is never hashed wholesale.
The subset is represented as a canonical, key-sorted JSON object.
Unknown or absent is not the same as a default value (section 9).
```

Phase 2A measured the effect directly: with thinking enabled the next token was
a reasoning tag, both verbalizer logits sat in the far tail, and the restricted
softmax still produced a plausible looking value; disabling thinking moved the
same question and plan from about 0.35 to about 0.99.

### Why token ids are not in the source axis

Resolved scoring token ids are execution provenance and already live in the
execution fingerprint (INV-22). Calibration can bind to the tokenizer identity
instead. Adding token ids to a second identity would duplicate a fact that
already has one home.

## 9. Calibration binding implications

This section designs binding requirements only. It implements nothing:
`CalibrationProfile`, temperature scaling, ECE, Brier, a calibration store, and
profile matching are all out of scope (section 14).

A calibration binding is a composition, not a new fingerprint:

```text
Calibration Binding
  = ProbabilityFormulationFingerprint (or FormulationFamilyFingerprint)
  + probability source identity
  + task/domain evidence
```

### Must bind

```text
decision family
scoring strategy
probability assembler id and version
doctrine id and version
compiler id and version
scoring representation identity (label scheme and mapping, or verbalizer pair)
outcome-space identity (ordered candidate names, descriptions, arity)
model id
model revision
tokenizer id and revision
rendering semantics (scoring-relevant subset)
```

### Should consider

```text
caller-supplied taxonomy id and version
domain / task identity
```

### Open

The pooling granularity (a formulation family versus a declared plan family),
revision grouping, cross-arity calibration, and description-paraphrase
equivalence are listed once, in section 16.

### Caller-supplied identity is provenance, not proof

If a caller declares `taxonomy_id="support-routing-v3"`, Probvenance records that
declaration. It must not treat the matching string as proof that two candidate
sets are semantically equivalent. A declaration is an assertion by the caller,
carried as provenance. The same caution applies to a declared rendering default.

### Unknown values

```text
revision=None is an explicit unknown, not a wildcard.
```

Two executions with `revision=None` and `revision=abcdef` are not confirmed to
be the same source. They are not matched by default. Likewise:

```text
an absent enable_thinking is not equal to enable_thinking=False.
```

For probability-semantics-relevant configuration, unknown is not equal to
default.

## 10. Comparability is not a single boolean

Comparability depends on the purpose. Debugging, regression, calibration, and
aggregate metrics ask different questions of the same two numbers. The design
therefore refuses a single `comparable = True/False`. It also refuses a single
strength ladder, because formulation and source are independent axes: a value
that climbs one axis cannot represent a difference on the other.

Instead the Eval layer reports a relation tuple derived from the identity facts:

```text
FormulationRelation
    exact            same ProbabilityFormulationFingerprint
    family           same FormulationFamilyFingerprint, different exact identity
    decision_family  same decision family only
    different        otherwise

SourceRelation
    exact            same source identity
    different        different source identity
    unknown          a source value is an explicit unknown (section 9)
```

The two axes are independent, so neither ordering is the "stronger" one.
`(family, exact)` and `(exact, different)` are simply different relationships,
and which is appropriate depends on the question being asked. A family is
model-independent by construction, so `family` on the formulation axis must
never be read as cross-model comparable; example 11.3 shows exactly that trap.

This is vocabulary for evaluation reports. It is not a runtime predicate, and no
automatic policy may be added on top of it: `if fp1 == fp2: comparable = True`
must not exist, because it would hide the two independent relationships behind
one bit. There is deliberately no `compatible` relation this round: that word
would need an evidence-backed definition of compatibility, which does not yet
exist.

## 11. Worked examples

### 11.1 Same Choice, label permutation

Decision `billing / shipping / technical`, `billing->A, shipping->B,
technical->C` versus `billing->C, shipping->A, technical->B`.

| Identity | Same? | Why |
|---|---|---|
| Decision Fingerprint | same | question, context, and choices unchanged |
| Plan Fingerprint | different | the mapping and rendered prompt lines changed |
| ProbabilityFormulationFingerprint | different | the scoring representation changed |
| FormulationFamilyFingerprint | same | same mechanism, same arity |
| Execution Fingerprint | different | different resolved token ids and rendered input |
| Trace ID | different | different executions |

Relation tuple: `formulation_relation = family`, `source_relation = exact`. This
is the case Phase 2B.1 measures when it reports total variation between two
label permutations: an explicit cross-formulation evaluation that preserves both
identities and the comparison method, which P1 permits.

### 11.2 Same formulation, different evidence

Context A: the package arrived. Context B: the package is missing.

| Identity | Same? | Why |
|---|---|---|
| Decision Fingerprint | different | context differs |
| ProbabilityFormulationFingerprint | same | the formulation is evidence-independent |
| FormulationFamilyFingerprint | same | same mechanism |
| probability value | different | the conditioning event differs |
| Execution Fingerprint | different | rendered input differs |

### 11.3 Same formulation, different model

`Qwen/Qwen3.5-2B` versus `openbmb/MiniCPM5-2B`, same decision and same
representation.

| Identity | Same? | Why |
|---|---|---|
| ProbabilityFormulationFingerprint | same | the model is not part of the formulation |
| FormulationFamilyFingerprint | same | same mechanism |
| source identity | different | different model |
| calibration sharing | not allowed by default | the model is a calibration-relevant source |

Relation tuple: `formulation_relation = exact`, `source_relation = different`.
Neither this tuple nor example 11.1's `(family, exact)` is inherently "more
comparable"; which one is usable depends on the question being asked.

Measured: Phase 2B.1 found mean total variation 0.0334 versus 0.0514 at N=3 and
0.0277 versus 0.0381 at N=5 between the two models under the same frozen
protocol, which is exactly why formulation equality alone cannot authorize
pooling.

### 11.4 Thinking on versus off

Same decision, same plan, same model, `enable_thinking` true versus false.

| Identity | Same? | Why |
|---|---|---|
| Decision Fingerprint | same | same question and context |
| Plan Fingerprint | same | rendering is backend-only and is not in the plan |
| ProbabilityFormulationFingerprint | same | rendering is not part of the formulation |
| FormulationFamilyFingerprint | same | same mechanism |
| source identity | different | rendering semantics differ |
| calibration sharing | not allowed by default | scoring-position validity differs |

This example is the reason the plan fingerprint cannot be a calibration key:
the two executions are indistinguishable at the plan level and completely
different in probability semantics.

### 11.5 Candidate description paraphrase

Same candidate name, description changed from `Payment, charges, and invoices`
to `Problems involving invoices, charges, or payments`.

| Identity | Same? | Why |
|---|---|---|
| ProbabilityFormulationFingerprint | different | descriptions are probability-semantics-relevant |
| FormulationFamilyFingerprint | same | same mechanism |
| Decision Fingerprint | different | choices include descriptions |

Measured: winner preserved 5 of 5 on both models, but total variation up to
0.3560 on MiniCPM5-2B (and 0.0115 on Qwen3.5-2B), so the distributions are not
interchangeable.

## 12. Proposed canonical payloads

Proposals only, for a future fingerprint class; none of this is implemented in
`src/probvenance/`. These payloads follow the existing fingerprint discipline:
schema version, canonical JSON, SHA-256, JSON-compatible values only (no
`repr(object)`, no callables, no sets, no automatic datetime conversion). The
compiler and assembler identifiers and versions shown are the real ones the
runtime now carries on `InferencePlan` and `DecisionTrace` (`bool-compiler` /
`choice-compiler` and `binary-restricted-softmax` /
`categorical-restricted-softmax`), reused here so the proposal matches reality.

### Exact formulation identity (Choice)

```json
{
  "v": 1,
  "kind": "probability_formulation",
  "decision_family": "choice",
  "strategy": "categorical_token_logits",
  "assembler": {"id": "categorical-restricted-softmax", "version": 1},
  "doctrine": {"doctrine_id": "categorical-semantic-judgment-v1", "version": 1},
  "compiler": {"id": "choice-compiler", "version": 1},
  "outcome_space": {
    "arity": 3,
    "candidates": [
      {"name": "billing", "description": "Payment, charges, and invoices"},
      {"name": "shipping", "description": "Delivery and logistics"},
      {"name": "technical", "description": "Technical problems"}
    ],
    "closed_set": true,
    "single_label": true
  },
  "scoring_representation": {
    "label_scheme_id": "categorical-labels-v1",
    "mapping": [
      {"candidate_index": 0, "candidate_name": "billing",
       "candidate_description": "Payment, charges, and invoices",
       "scoring_label": "A"},
      {"candidate_index": 1, "candidate_name": "shipping",
       "candidate_description": "Delivery and logistics",
       "scoring_label": "B"},
      {"candidate_index": 2, "candidate_name": "technical",
       "candidate_description": "Technical problems",
       "scoring_label": "C"}
    ]
  }
}
```

### Exact formulation identity (Bool)

```json
{
  "v": 1,
  "kind": "probability_formulation",
  "decision_family": "bool",
  "strategy": "binary_token_logits",
  "assembler": {"id": "binary-restricted-softmax", "version": 1},
  "doctrine": {"doctrine_id": "binary-semantic-judgment-v1", "version": 1},
  "compiler": {"id": "bool-compiler", "version": 1},
  "outcome_space": {"outcomes": ["false", "true"], "closed_set": true,
                    "single_label": true},
  "scoring_representation": {"positive_verbalizer": "yes",
                             "negative_verbalizer": "no"}
}
```

The verbalizers are shown at the code defaults. Casing is semantics-bearing
(section 5.2), so a payload must record the strings exactly as used.

### Formulation family identity

The same Choice decision, with the representation removed:

```json
{
  "v": 1,
  "kind": "formulation_family",
  "decision_family": "choice",
  "strategy": "categorical_token_logits",
  "assembler": {"id": "categorical-restricted-softmax", "version": 1},
  "doctrine": {"doctrine_id": "categorical-semantic-judgment-v1", "version": 1},
  "compiler": {"id": "choice-compiler", "version": 1},
  "label_scheme_id": "categorical-labels-v1",
  "arity": 3
}
```

This payload is identical under all six label permutations, which is the point,
and equality of this payload is explicitly not probability equality.

### Version bump rules

```text
assembler_version  bumps only when the mathematics of RawEvidence -> uncalibrated
                   probability changes (normalization, aggregation, or the
                   probability transformation). A refactor, a performance
                   optimization, or an equivalent implementation does not bump it.
compiler_version   bumps when the compiler changes prompt construction, label
                   mapping semantics, doctrine lowering, or strategy selection in
                   a way that affects the formulation. An equivalent internal
                   refactor does not bump it.
```

These rules exist so the provenance fields stay meaningful: a version is a claim
about probability semantics, not about code churn.

### Naming considered

```text
ProbabilitySemanticsFingerprint        rejected: "semantics" reads as empirical
                                        meaning and invites calibration reading
FormulationFingerprint                 rejected: too vague about exact vs family
ProbabilityFormulationFingerprint      adopted (exact)
ScoringFormulationFingerprint          rejected: "scoring" narrows it to the
                                        label layer, losing outcome space
FormulationFamilyFingerprint           adopted (family)
```

## 13. Frozen design invariants

These are frozen into `docs/design-constitution.md` as INV-23 through INV-27.
They are normative design rules; only INV-27 has a runtime enforcement point,
because no runtime API implements probability-identity comparison yet.

```text
P1  Probabilities from different probability formulation identities must not be
    silently treated as interchangeable, directly pooled, or assumed to share
    calibration. Explicit cross-formulation evaluation is allowed when both
    identities and the comparison method are preserved.

P2  Formulation family membership does not imply automatic interchangeability,
    automatic pooling, or calibration compatibility.

P3  Probability formulation identity excludes instance evidence (question and
    context) while including the semantic outcome space and the scoring
    representation.

P4  Unknown identity values (an absent revision, an absent scoring-relevant
    rendering key) are explicit unknowns and must not match, default to, or be
    interpreted as any concrete value.

P5  The probability assembler identity and version are part of probability
    formulation identity.
```

P1 generalizes INV-19, which already requires that a restricted probability be
read together with its candidate-space mass. P1 forbids implicit interchange,
blind pooling, and default shared calibration; it does not forbid explicit
comparison, which is exactly what Phase 2B.1 does when it reports total
variation between two formulation identities while preserving both. P3 is the
rule that keeps the formulation evidence-independent. P5 closes the gap that a
plan names the strategy but not the transformation; the runtime carries a real
assembler identity and version AND executes only when the declared tuple matches
a known implementation, so that provenance is execution-verified rather than
declarative. That stronger support is why it is now frozen as INV-27.

## 14. Non-goals

This round does not:

```text
implement CalibrationProfile, temperature scaling, ECE, Brier, a calibration
  store, or profile matching
implement any fingerprint class in src/probvenance/
add a public API
change runtime probability behavior (Phase 2C.0 later added compiler and
  assembler provenance to the plan fingerprint and trace, and made the
  continuation check exact; no accepted probability value changed)
define an automatic comparability policy
define automatic paraphrase equivalence
run any model
```

## 15. Verification

Recorded for this round.

```text
src/ and tests/ changes      none (git diff --stat -- src tests is empty)
fingerprint classes added    none
public API added             none
CalibrationProfile added     none
GPU used                     no
uv run pytest                499 passed
uv run ruff check .          All checks passed
uv run mypy src              Success: no issues found in 17 source files
```

The measured numbers cited in this document are `[E]` EXPERIMENTAL under their
named conditions in `docs/claims.md`. Nothing here is `[V]` VERIFIED.

## 16. Open questions

1. What exactly is the pooling granularity: a formulation family, or a
   separately defined plan family?
2. Should arity be part of the family, or its own grouping axis?
3. May model revisions ever be grouped, and on what evidence?
4. How should absent versus explicitly default rendering keys be treated across
   backends?
5. Is cross-arity calibration meaningful at all?
6. May description-paraphrase equivalence ever be caller-declarable without a
   semantic model?
7. Should the probability assembler identity be a declared constant or derived
   from code?
8. Should `taxonomy_id` be a core binding field or purely optional?

## 17. Relationship to the other documents

- `docs/design-constitution.md` holds the invariant and principle numbers
  referenced here: INV-03 (deterministic tie-break, section 4), INV-17
  (provider independence, section 6), INV-19 (restricted probability travels
  with its mass, section 13), INV-20 and INV-21 (semantic identity versus
  scoring representation, and where the representation lives, section 5),
  INV-22 (evidence label order is provenance, section 8), and AP-08 (the
  compiler declares the representation, the backend validates executability,
  section 6). This document proposes no invariant changes; its proposed
  invariants live in section 13 pending review.
- `docs/choice-semantics.md` defines Choice probability semantics and the
  scoring representation. This document generalizes the identity question
  across Bool and Choice and does not restate that content.
- `docs/claims.md` records evidence status.
- The terms "plan family" and "formulation fingerprint" were previously left
  undefined and explicitly deferred (`docs/roadmap.md`, Phase 4, and
  `docs/choice-semantics.md` open question 9). This document defines the
  formulation identity and family, so it fills a known gap rather than
  contradicting earlier text.
