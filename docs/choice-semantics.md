# Choice Probability Semantics

Design and exploration record for Phase 2B.

Status: **direct categorical Choice is implemented and experimentally
exercised (Phase 2B). Every other strategy discussed here remains design only.**
Enforcement today covers exactly one path: direct categorical, closed-set,
single-label, single-token scoring labels, small N. Read this together with
`design-constitution.md` (probability semantics, INV-19 to INV-22, AP-08) and
`claims.md` (what is actually verified, and under which conditions). A design
statement in this document is not a capability claim, and no part of the
one-vs-rest, sampling, or open-set discussion is enforced by the runtime.

## Phase 2B outcome

What the round actually settled, and what it deliberately left open.

Frozen as runtime invariants, each with a test or an enforcement point:

- semantic candidate identity is not scoring-label identity (INV-20);
- the scoring representation lives in the plan fingerprint, not the decision
  fingerprint (INV-21);
- evidence label order is provenance, so permuted labels are rejected rather
  than reordered (INV-22);
- the compiler declares the representation and the backend validates
  executability (AP-08);
- INV-19 was widened to cover `candidate_mass` for the categorical path, with
  no new number because it is the same rule over a wider candidate set;
- the Choice tie-break resolves on semantic candidate order, which Phase 1
  already covered for results and therefore also needs no new number.

Left as experimental robustness hypotheses, never promoted to invariants:

- label permutation stability of the restricted distribution;
- candidate addition stability;
- description paraphrase stability.

Measured outcome, recorded in full in `experiments/choice_signal/REPORT.md`:
the semantic winner was stable across all six label permutations at N=3 (90/90)
and across five permutations at N=5 (75/75), while the distribution SHAPE moved
(mean total variation 0.0334 and 0.0277, worst case 0.3082). Adding an absent
candidate and paraphrasing a description left the winner unchanged. With a
candidate set that omitted the true topic the model still answered in-set at
0.677 to 0.893 with `candidate_mass` near 0.99, so the round confirms that the
direct categorical path has no open-set guarantee and that `candidate_mass`
measures candidate-space occupancy rather than candidate-set correctness. No
pass threshold was set anywhere.

The question this document exists to answer:

> When we read an N-way distribution out of a language model's next-token
> logits, what exactly does each number depend on, which candidate set is it
> conditioned on, which changes are semantic changes, and which are only
> changes of scoring representation?

Phase 2A.2 established that binary next-token scoring is model- and
formulation-dependent, and that a named configuration passed the readiness gate
while the mechanism itself did not. The naive move, taking `2 logits` and
mechanically widening it to `N logits` and calling the result a Choice
probability, would discard every lesson from that round. This document is the
attempt to pay those lessons forward before writing any code.

---

## 1. Scope

In scope:

- the probability semantics of a closed-set, mutually exclusive, single-label
  Choice decision;
- the separation between a semantic candidate and the scoring label used to
  read a logit for it;
- the candidate-to-label mapping, and where it must be recorded;
- `candidate_mass` as the N-way generalisation of `verbalizer_mass`;
- the failure modes that follow from the probability being conditioned on a
  declared candidate set;
- a proposed implementation contract and experiment matrix for the next round.

Out of scope (see section 17): multi-token scoring labels, one-vs-rest,
sampling estimators, open-set decisions, automatic `other` injection, semantic
de-duplication, calibration, cloud backends, and any change to the existing
Bool path.

## 2. Definitions

**Semantic candidate.** One alternative of the decision as the caller means it:
a business concept such as `billing`, `shipping`, or `returns`. A candidate is
the thing a user cares about. It may be several words long, and its surface
form may tokenize unstably.

**Scoring label.** The short surface string whose token the backend reads a
logit for: `A`, `B`, `C`. A scoring label is an execution detail. It is not part
of the decision's meaning, and it must never appear in a result.

**Candidate-label mapping.** The declared bijection between the ordered semantic
candidates and the scoring labels, with the resolved token id for each label:
candidate `billing` reads the logit of the token that `A` resolves to.

**Candidate set.** The ordered set of semantic candidates actually declared for
this decision. It is finite and closed.

**Restricted categorical probability.** The conditional probability of a
semantic candidate given that the model's next token is one of the declared
scoring labels. It is the quantity `ChoiceResult.probabilities` is intended to
carry.

**Candidate mass.** The full-vocabulary probability mass that the next token
lands on the union of the declared scoring labels. It is the N-way
generalisation of `verbalizer_mass`, and it is a diagnostic, not part of the
decision distribution.

## 3. One-sentence definition

> Choice probabilities are conditional probabilities over an explicitly
> declared semantic candidate set under a specific scoring representation.

中文：

> Choice 概率是在明确声明的语义候选集合，以及特定 scoring representation
> 条件下得到的条件分布。

The consequences of that sentence are the rest of this document:

```text
P(billing) = 0.71
```

means "conditional on the next token being one of the declared scoring labels,
0.71 of that restricted mass sits on the token we mapped to billing, under this
model, doctrine, prompt, and mapping". It does not mean that billing is 71%
likely to be the true answer in the world.

This is consistent with the existing constitution rule that a probability is
not a correctness probability. It adds a second qualifier: a Choice probability
is not even an unconditional model probability, because it is normalized inside
a candidate set the caller chose.

## 4. Mathematical semantics

### 4.1 Restricted categorical distribution

Let the semantic candidate tuple in candidate order be

```text
C = (c_1, ..., c_N),  N >= 2
```

and let the mapping assign to candidate `c_i` a scoring label whose token id, at
the scoring position, has logit `l_i`. Then

```text
P(c_i | next token in scoring-label set) = exp(l_i) / sum_j exp(l_j)
```

This is the first definition to analyse and the working candidate. It is
`softmax` restricted to N declared entries, which is exactly the N-way
generalisation of the existing two-way Bool softmax.

Numerically it must never be evaluated in the raw exponent form. Subtract the
maximum first:

```text
m = max_j l_j
P(c_i) = exp(l_i - m) / sum_j exp(l_j - m)
```

which is stable for arbitrarily large or small logits, exactly as the Bool path
already is.

### 4.2 Candidate mass

```text
candidate_mass = sum_j exp(l_j) / sum_{v in V} exp(l_v)
```

where `V` is the full vocabulary. In log space, using the full-vocabulary
log-sum-exp the backend already computes for the Bool path:

```text
log_candidate_mass = logsumexp(l_1, ..., l_N) - logsumexp(V)
candidate_mass     = exp(log_candidate_mass)
```

`candidate_mass` reduces exactly to `verbalizer_mass` when `N = 2`. The existing
diagnostics module already computes the pieces: a candidate log-sum-exp over N
entries instead of two, minus the same vocabulary log-sum-exp.

### 4.3 Relationship to the Bool path

| | Bool | Choice |
|---|---|---|
| outcome space | `{False, True}` | `{c_1, ..., c_N}` |
| scoring labels | two verbalizers | N labels from a versioned label scheme |
| restricted distribution | two-way softmax | N-way softmax |
| candidate-space diagnostic | `verbalizer_mass` | `candidate_mass` |
| result carrier | `BoolResult.probability_true` | `ChoiceResult.probabilities` |

Direct categorical Choice at `N = 2` reduces mathematically to the same
restricted-normalization and candidate-space-mass equations used by Bool
binary-token scoring. This is a **mathematical degeneration, not a type
identity**: it does not mean `BoolDecision` and `ChoiceDecision` should be
merged, nor that their compiler contracts should be unified. It is, however, the
reason the Bool path's failure modes should be assumed to recur here until an
experiment says otherwise.

### 4.4 What the number is not

`P(c_i)` is not:

- an unconditional next-token probability (it is renormalized inside the
  candidate set);
- a real-world event probability;
- an accuracy figure;
- a calibrated number;
- comparable across two different candidate sets, even when the same candidate
  appears in both (section 10.2);
- evidence that the model was about to emit any candidate token at all (that is
  what `candidate_mass` is for).

## 5. Candidate and label separation

**Proposed hard design principle: a semantic candidate is not a scoring label.**

They should be separate objects even when they happen to be equal strings, and
the runtime should never assume they are.

Why the separation is required, not merely tidy:

1. **Length and tokenization.** Semantic candidates such as `account security`
   or `technical support` are multi-word, and their tokenization is unstable
   across models, casing, and leading whitespace. A scoring label has to be
   short and reliably one token.
2. **Prior contamination.** Any string a model sees carries a language prior.
   `billing` carries business-corpus prior; `A` carries almost none. Binding the
   scoring token to the semantic word means the measurement is partly a
   measurement of that word's prior.
3. **Auditability.** A trace that records `logits = (2.1, 0.5, -1.0)` is
   worthless without the mapping: there is no way to tell whether `A` meant
   billing or shipping. Recording the mapping is what makes a stored result
   interpretable later.
4. **Measurability of the bias.** Only when the two are separated can an
   experiment permute the label assignment while holding the candidate order
   fixed, and thereby measure token-level label bias on its own (section 15.3).
   If candidate text and scoring label were fused, that variable could not be
   isolated at all.
5. **Stability of the decision identity.** The decision fingerprint should
   change when the candidates change, and not when the execution labels change.
   That is only expressible if the two are distinct (section 7).

### 5.1 Who chooses the representation, and who validates it

A first draft of this design asked the compiler to select, from a label pool,
labels that resolve to exactly one token "under the current tokenizer and render
context", while also declaring the compiler and plan to be provider-independent.
Those two requirements are contradictory: single-token validity is only
decidable by running a concrete tokenizer over a concrete rendered continuation.
The boundary is therefore frozen as:

> **Compiler chooses the scoring representation. Backend validates whether that
> representation is executable under the concrete model, tokenizer, and
> rendering environment.**

In split form:

```text
ChoiceDecision
  -> ChoiceCompiler
  -> semantic candidate <-> scoring-label string mapping
  -> InferencePlan
  -> TransformersBackend
  -> render the actual model input
  -> resolve each scoring label in the actual continuation context
  -> validate single-token and pairwise-distinct token ids
  -> execute
```

The compiler must not access:

```text
tokenizer
model
chat template
token ids
CUDA
backend-specific rendering
```

The compiler decides, from the `ChoiceDecision`, the doctrine, the versioned
label scheme, and the declared backend capabilities:

```text
semantic candidate order
semantic candidate descriptions
scoring label strings
candidate <-> scoring-label mapping
required capabilities
prompt and doctrine structure
```

The compiler must not know:

```text
whether "A" is token id 123
whether "A" splits into two tokens under some tokenizer
whether a leading space changes the tokenization
```

The backend, in the actual rendered continuation, validates that each scoring
label is exactly one token and that the token ids are pairwise distinct. If any
label is not one token, resolves to nothing, or collides with another, execution
fails explicitly. It must not silently swap a label, jump to another pool,
truncate, take the first token, sum multi-token logits, fall back to
one-vs-rest, or fall back to generation.

For Phase 2B: **an invalid representation is an explicit failure.**

No tokenizer-aware resolver is introduced this round. There will be no
`ScoringLabelResolver`, `ModelAwareCompiler`, `TokenizerAwareCompiler`, or label
search engine. If experiment later shows that a fixed label scheme is not
compatible enough, that is the time to design an explicit, recorded resolver
artifact, not now.

## 6. Candidate-label mapping contract

Proposed shape (conceptual, not implemented):

```text
CandidateLabelMapping:
    candidate_index: int
    candidate_name: str
    candidate_description: str | None
    scoring_label: str
```

The mapping is deliberately provider-independent, so it carries no
`scoring_token_id`. A token id does not exist until a concrete tokenizer renders
a concrete input, so it belongs to execution provenance rather than to the
compiled plan (section 7).

It must carry, at minimum:

- **the semantic candidate name**, matching `ChoiceDecision.choice_names`;
- **the semantic candidate description**, so a stored trace can explain what
  the candidate meant without the caller's original code;
- **the scoring label text**, the execution-only surface string;
- **the resolved scoring token id**, because the label string alone is
  ambiguous across tokenizers;
- **the mapping order**, which follows semantic candidate order.

Invariants it should satisfy:

- one entry per candidate, no more and no fewer (completeness);
- entries in `ChoiceDecision` candidate order, and the candidate names equal
  `choice_names` exactly (order preservation);
- every scoring label non-empty and unique within the mapping;
- the mapping is a deterministic function of the `ChoiceDecision`, the label
  scheme version, and `N` alone, with no tokenizer input;
- the mapping is recorded in the plan and enters the plan fingerprint.

Token-level properties are validated by the backend at execution time rather
than asserted by the mapping: each scoring label resolves to exactly one token
in the *actual* rendered continuation (not merely via a bare
`tokenizer.encode(label)`), and the resolved token ids are pairwise distinct.

## 7. Identity and lineage rules

Four identities exist and must not be conflated.

**Decision fingerprint** describes the semantic question: question, context,
and the ordered candidates with their descriptions. It is produced by
`ChoiceDecision` today and already includes candidate order.

**Plan fingerprint** describes the compiled plan: which strategy, which prompt,
which scoring labels, which candidate-label mapping, which doctrine. It changes
when the mapping changes.

**Execution fingerprint** describes the environment and rendering the plan
actually ran under: model, revision, tokenizer, dtype, render configuration,
the actually rendered input, and the resolved token ids.

**Trace id** is per-execution identity. Two executions of the same plan produce
one execution fingerprint and two trace ids.

What the mapping must enter:

| identity | mapping enters? | why |
|---|---|---|
| decision fingerprint | no | the mapping is not part of the question |
| plan fingerprint | **yes** | a plan that reads label `A` for billing is a different plan from one that reads `A` for shipping |
| rendered input | **yes** | the prompt declares which label means which candidate |
| `RawEvidence` interpretation | **yes** | the evidence carries scoring-space labels; the mapping is what turns them into candidates |
| `DecisionTrace` | **yes** | without it, stored logits cannot be read back |
| execution fingerprint | indirectly, via the plan | plus the resolved token ids |

`InferencePlan` must not store resolved token ids even for execution
convenience. A plan is a compile-time artifact and a resolved id is an execution
artifact. Resolved ids belong in backend evidence metadata, in diagnostics, in
the trace, and in the execution fingerprint.

Proposed invariant for the implementation round:

> The same `ChoiceDecision` compiled with two different candidate-to-label
> assignments produces the **same decision fingerprint** and **different plan
> fingerprints**.

This is the cleanest test that the semantic/scoring separation is real rather
than nominal.

## 8. Result semantics

### 8.1 What the user sees

```python
result.probabilities
# {"billing": 0.71, "shipping": 0.20, "returns": 0.09}
```

Not:

```python
# {"A": 0.71, "B": 0.20, "C": 0.09}
```

**Proposed frozen rule: scoring labels are execution representation, not result
semantics.** The mapping is applied inside the probability layer, and nothing
downstream of it exposes label text. A result keyed by `A`/`B`/`C` would be
uninterpretable without side information, and would silently make the public
contract depend on the label scheme.

### 8.2 `value` and tie-breaking

`ChoiceResult.value` remains the semantic argmax, computed over semantic
candidate probabilities, never over label strings and never over label order.
Ties are resolved by the original semantic candidate order, exactly as today.

This must hold under any label assignment. If `billing -> C`, `shipping -> A`,
`returns -> B`, a tie between billing and shipping is still broken in favour of
billing, because billing comes first in `ChoiceDecision`. The scoring
representation must not be able to change the public outcome. This is worth
freezing as an invariant.

### 8.3 Existing validation still applies

The existing `ChoiceResult` rules remain: at least two entries, each probability
in `[0, 1]`, the set summing to one within the existing tolerance, `value`
belonging to the key set, and `value` being an argmax of the distribution.

## 9. Diagnostics

### 9.1 `candidate_mass`

`candidate_mass` answers one question: **did the model's next-token
distribution actually put mass on the tokens we declared as the candidate
space?**

It does not answer: whether the model chose the right candidate; whether the
candidate set was the right one; whether the candidates are exhaustive; whether
the decision is correct.

The Phase 2A.1 finding generalises directly. A high restricted probability with
a near-zero `candidate_mass` is renormalized tail noise, and the number must not
be read as inclination.

### 9.2 `candidate_mass` is not certainty

These two are orthogonal and the distinction should be stated in the
implementation docs with both examples:

```text
candidate_mass = 0.99, distribution = [0.34, 0.33, 0.33]
  the model is firmly in the candidate space, and undecided inside it

candidate_mass = 0.001, distribution = [0.99, 0.005, 0.005]
  the restricted distribution looks very peaked, and the model is almost
  certainly not about to emit any declared candidate token; the 0.99 is
  renormalized tail noise
```

Certainty (`entropy`, `margin`) describes only concentration of the restricted
distribution. It says nothing about whether that distribution is grounded.

### 9.3 Why `candidate_mass` does not generalise across `N`

```text
candidate_mass = 0.9
```

does not mean the same thing at `N = 2` and at `N = 20`. A larger label set has
more opportunities to accumulate mass by accident, and the labels themselves
carry different priors. Therefore:

> No universal runtime threshold of the form `candidate_mass > 0.5 => valid`
> may be introduced.

This is the same conclusion Phase 2A.1 reached for `verbalizer_mass`, and it
holds more strongly here. Whether `candidate count`, `candidate mass`, the
outside top token, and entropy can be combined into a defensible policy is a
research question recorded in section 18, not a Phase 2B deliverable.

### 9.4 Proposed diagnostics shape

Minimal proposal, staying close to the existing `ScoringDiagnostics`:

```text
ChoiceScoringDiagnostics:
    candidate_mass: float
    candidate_full_vocab_probabilities: tuple[float, ...]
    top_token_id: int
    top_token_probability: float
    top_token_text: str | None = None
```

Notes:

- `candidate_full_vocab_probabilities` are full-vocabulary probabilities, one
  per candidate, in candidate order. They are **not** the restricted
  probabilities: the restricted ones are the result and live only in
  `ChoiceResult`. Storing both is not two sources of truth, because they are
  different quantities. They are cheap to compute from the row already in hand.
- `top_token_id` / `top_token_probability` are retained because a high
  `candidate_mass` does **not** guarantee the top token is a candidate: with
  `A = 0.2`, `B = 0.2`, `C = 0.2` and an outsider at `0.3`, `candidate_mass` is
  `0.6` while the top token is outside the candidate set. The diagnostic pair is
  what makes that visible.
- `candidate_mass` is the N-way form of `verbalizer_mass`. Whether the Bool
  spelling is kept as an `N = 2` alias or renamed is an implementation-round
  decision; the general concept name should be `candidate_mass`.

## 10. Failure modes

### 10.1 Scoring-position failure

The model is not at the decision position (its next token is a reasoning tag, a
role token, or a word continuation). `candidate_mass` approaches zero while the
restricted distribution stays plausible-looking. Observed in Phase 2A.2: LFM2.5
under the evidence-oriented doctrine left the decision position on 462 of 720
probes. Assume this recurs, and assume it is model- and doctrine-dependent.

### 10.2 Candidate-set dependence

```text
P(billing | {billing, shipping, returns})
!=
P(billing | {billing, shipping, returns, account deletion})
```

Neither is an intrinsic property of `billing`. Adding a candidate renormalizes
everything, and the model's reading of the whole prompt changes as well. A
Choice probability is always conditioned on the declared set, and the set must
therefore be part of what a result is relative to.

### 10.3 Non-exhaustive candidate sets

If the true answer is outside the set, the restricted softmax still returns a
normalized distribution and still names a winner. `candidate_mass` may reveal
part of the problem, but a high `candidate_mass` does not prove the taxonomy was
exhaustive. The restricted softmax is structurally incapable of returning
"none of these". This is why `candidate_mass` must not be presented as an
open-set detector.

### 10.4 Non-exclusive candidates

`billing` and `refund` are not necessarily mutually exclusive. The math will
still normalize if a caller passes overlapping candidates; only the semantics
are wrong. The runtime cannot detect this and must not pretend to.

### 10.5 Taxonomy overlap

If two candidates overlap (`billing` and `payment issue`), the model may split
its mass between them, depressing both. This is **not** a robustness property
that a model should pass. It is a **taxonomy-overlap sensitivity**: a
characterisation of how a bad or overlapping taxonomy distorts the restricted
distribution. Such a taxonomy may already violate the caller's obligation to
declare mutually exclusive candidates.

The observation worth recording is that a Choice probability is not an intrinsic
per-candidate probability; it is relative to a taxonomy and its granularity.
Automatic semantic de-duplication is an ontology problem and is out of scope.

### 10.6 Label token prior

The model has its own prior over the label tokens themselves. This is exactly
the variable that the candidate/label separation makes measurable, and exactly
what the label permutation experiment is for.

### 10.7 Multi-token scoring labels

If a label does not resolve to exactly one token, the label has no single
logit, and any substitute (first token, mean logit, raw product of token
probabilities) silently changes the probability semantics.

## 11. Closed-set assumptions

A Choice decision, in this design, requires the caller to declare:

1. **mutual exclusivity** - the candidates are interpretations of one decision,
   not a set of independent booleans;
2. **single-label** - exactly one candidate applies, never several;
3. **closure** - the candidate set is the intended answer space, and the caller
   accepts that a true answer outside it cannot be returned.

The runtime cannot verify any of the three. They are declarations by the caller,
and they should be documented as such. Whether they belong in the constitution
as an explicit obligation on the caller is an open question (section 18).

On `other` and open sets: the runtime must **not** auto-inject an `other`
candidate. If a caller needs one, it is an explicit semantic candidate in the
first version. Three longer-term directions are recorded but not chosen:

1. caller explicitly includes an `other` candidate;
2. a separate `BoolDecision`, "is any declared candidate appropriate?";
3. a future open-set scoring strategy.

## 12. Multi-token policy

For the first Choice implementation:

- **scoring labels must resolve to exactly one token** in the actual
  continuation context, and all candidate token ids must be distinct. This is
  the direct generalisation of the existing verbalizer validation, and the same
  error type applies.
- **multi-token scoring labels are not implemented.** Multi-token label
  likelihood is a separate future scoring strategy with its own mathematical
  definition. The first implementation must not quietly use the first token,
  average the token logits, or multiply token probabilities without recording
  and justifying a length normalisation.
- **semantic candidates may be multi-token without limit.** The restriction is
  on the internal scoring label, never on the business name. `account security`
  is a perfectly good candidate.

## 13. Direct categorical vs one-vs-rest

`DirectCategoricalTokenLogits` treats the candidates as one mutually exclusive
N-way decision: a single scoring position, one label per candidate, one softmax.
Its probability semantics are the ones in section 4.

One-vs-rest treats each candidate as an independent judgement, normally by a
separate binary scoring call per candidate, and then combines the results. Those
per-candidate scores are not a categorical distribution; normalizing them
afterwards is a modelling choice, not an identity.

Therefore `method` must distinguish them, and the existing
`ScoringStrategy.CATEGORICAL_TOKEN_LOGITS` value is the correct label for this
strategy. A future one-vs-rest implementation must not reuse it. "choice" alone
is not an adequate `method` string, because the probability semantics differ.

### 13.1 Where direct categorical scoring applies

Appropriate:

```text
mutually exclusive, single-label, closed-set, small N
```

such as "which team should handle this ticket".

Not appropriate, and belonging to other primitives:

```text
select all that apply        -> MultiLabel
open-world classification    -> open-set / abstention policy
ranking                      -> Rank
continuous judgement         -> Score
```

## 14. Implementation contract for Phase 2B (delivered)

A minimal vertical slice, mirroring the Bool path exactly:

```text
ChoiceDecision
  -> ChoiceCompiler
  -> InferencePlan (strategy = categorical_token_logits)
  -> TransformersBackend
  -> RawEvidence
  -> ChoiceProbabilityAssembler
  -> ChoiceResult
  + ChoiceScoringDiagnostics
  + DecisionTrace
```

Component notes:

- **`ChoiceCompiler`** maps a `ChoiceDecision` plus the backend capabilities
  plus a doctrine into a plan. It must raise `UnsupportedCapabilityError` when
  the categorical capability is absent, and must never fall back to another
  scoring strategy.
- **`InferencePlan`** already has `targets: tuple[str, ...]`, unused by the Bool
  path, and it is the natural carrier for the ordered scoring labels. The
  candidate-label mapping needs to be expressible in the plan as well, since it
  is semantics-bearing; it should be a structured, readable field (for example a
  `candidate_mapping` tuple), never an opaque JSON blob, and the plan fingerprint
  must cover it. `mapping[i].scoring_label` must equal `targets[i]`.
- **`RawEvidence`** stays in scoring space:

  ```text
  kind   = LOGITS
  labels = ("A", "B", "C")
  values = (l_A, l_B, l_C)
  metadata: vocab_logsumexp, top_token_id, top_token_logit,
            resolved_target_token_ids, ...
  ```

  The backend must not know about semantic candidates. The assembly step is what
  applies the mapping. The evidence labels must match the plan's declared target
  order *exactly*: matching only the label set and reordering internally is
  forbidden, because label order, logit order, and the candidate mapping
  together form the provenance. Any order mismatch fails explicitly.
- **`ChoiceProbabilityAssembler`** is pure: validate the evidence labels against
  the mapping, validate the mapping, validate finite logits, compute the
  restricted N-way softmax, compute `candidate_mass`, apply the mapping to
  produce semantic probabilities, compute certainty, choose the semantic argmax
  with candidate-order tie-breaking, and return an uncalibrated `ChoiceResult`.
- **Certainty** carries over unchanged: normalised entropy over N outcomes,
  `margin = top1 - top2`. Note that normalised entropy is comparable across
  different `N` only as a concentration measure, not as a like-for-like
  comparison; this is worth recording, not worth redesigning now.

The label scheme: the compiler maintains a versioned, ordered scheme
(`categorical-labels-v1 = ("A", "B", "C", ...)`) and assigns the first `N`
labels to the candidates in candidate order. The mapping is therefore
deterministic from the `ChoiceDecision`, the scheme version, and `N` alone, with
no tokenizer input, and it is recorded in the plan. The scheme version enters
the compiler version, the plan, the fingerprint, and the trace.

Whether those labels are executable is not the compiler's question. The backend
resolves them against the real continuation and fails explicitly if any label is
not exactly one token or if two labels collide (section 5.1). The compiler never
skips a label and never re-assigns one, because that would let execution
silently alter plan semantics.

## 15. Experiment matrix (executed)

The purpose is not an accuracy benchmark. It is to observe how the numbers move
when the representation, the candidate set, or the wording changes.

### 15.1 Three-way classification

Support routing with obvious cases per category:

```text
billing, shipping, technical
```

For each category, several synthetic contexts that obviously belong to it.
Record: semantic argmax, `candidate_mass`, restricted distribution, certainty.

### 15.2 Five-way

```text
billing, shipping, returns, technical, account
```

Purpose: observe what `N` does to `candidate_mass`, permutation drift, entropy,
and ranking. Not an accuracy target.

### 15.3 Label permutation

With `N = 3`, the full `3! = 6` permutations are cheap and should all be run for
each case, holding the semantic candidate order and the prompt entry order
fixed, changing only which candidate each label is bound to:

```text
run 1:  billing -> A, shipping -> B, returns -> C
run 2:  billing -> C, shipping -> A, returns -> B
...
```

This is the experiment that isolates token-label bias. For `N = 5` (`5! = 120`),
do not run the full set: use identity, reverse, cyclic shift, and two or three
deterministic pseudo-random permutations with a recorded seed.

Metrics, all simple and dependency-free:

```text
argmax preservation rate      fraction of cases whose semantic argmax survives
mean total variation distance TV(P, Q) = 0.5 * sum_i |P_i - Q_i|, in [0, 1]
rank preservation             fraction of candidate pairs whose order survives
max absolute probability drift
```

### 15.4 Candidate perturbations

```text
irrelevant addition    add "account deletion" to a 3-way set; observe whether
                       the winner and ranking survive and how mass moves
description paraphrase rewrite "Payment and billing issues" as "Problems
                       involving charges, invoices, or payments"
taxonomy overlap       add "payment issue" next to "billing"; observe mass split
```

### 15.5 What counts as a result

Exact probability invariance is **not** expected. Token priors make drift
inevitable. The properties worth measuring are the ones above: whether the
semantic argmax survives, whether the ranking roughly survives, and how large
the drift is.

## 16. Frozen invariants vs robustness hypotheses

To be frozen as runtime invariants after the implementation round (and only
then):

1. the restricted distribution is an N-way softmax over the declared scoring
   logits, computed stably;
2. `candidate_mass` is computed from full-vocabulary normalisation;
3. the mapping is complete and order-preserving over the candidates, and each
   `mapping[i].scoring_label` equals `targets[i]`;
4. scoring labels are unique within the mapping, and the backend resolves each
   to exactly one token with pairwise-distinct ids, failing explicitly
   otherwise;
5. semantic probabilities sum to one;
6. `ChoiceResult` never exposes scoring labels;
7. the mapping enters the plan fingerprint;
8. two different label assignments give the same decision fingerprint and
   different plan fingerprints;
9. tie-breaking uses semantic candidate order, never label order;
10. `candidate_mass` lives in diagnostics, never in `ChoiceResult`;
11. scoring a Choice decision adds no extra forward pass;
12. `RawEvidence` labels match the plan's declared target labels exactly, in
    order, with no internal reordering;
13. an unexecutable scoring representation fails explicitly, and no strategy,
    label, or pool fallback occurs.

To remain experimental robustness hypotheses, with no threshold attached:

```text
label permutation robustness
candidate addition robustness
description paraphrase robustness
obvious-category directional behaviour
candidate-set sensitivity
```

Taxonomy overlap is deliberately not in that list. It characterises a bad
taxonomy rather than testing a model, so it is reported as a failure mode
(section 10.5), not as a property a model is expected to satisfy.

These become part of the semantic unit test suite only once data exists to say
what a defensible bound is. No numeric gate may be set before measurement.

## 17. Non-goals

Not in the next implementation round:

```text
multi-token scoring labels
one-vs-rest
sampling estimators
open-set decisions
automatic other injection
semantic de-duplication
large-N optimisation
batching
calibration
cloud backends
any change to the Bool path
```

## 18. Open questions

1. Whether `candidate count`, `candidate mass`, outside-top-token probability,
   and restricted entropy can be combined into a defensible scoring-validity
   policy, and at which layer that policy belongs.
2. Whether the closed-set obligations (exclusive, single-label, closed) should
   become an explicit documented obligation on the caller in the constitution,
   rather than a note in this document.
3. Whether a future round should let the compiler select labels using
   tokenizer knowledge, as an explicit recorded resolver artifact, given that
   this round freezes declaration-only compilation.
4. Whether `verbalizer_mass` should be renamed to `candidate_mass` for the Bool
   path, or retained as an `N = 2` alias.
5. Whether normalised entropy is a defensible cross-`N` comparison, or whether
   `N` must always be reported alongside it.
6. Whether per-candidate full-vocabulary probabilities belong in the standard
   trace or only in a fuller retention mode.
7. What the correct treatment is when a label in the fixed scheme does not
   resolve to a single token for a given tokenizer, beyond today's explicit
   failure.
8. Whether a future open-set strategy should be a Bool guard, an explicit
   `other` candidate, or a genuinely different scoring strategy.
