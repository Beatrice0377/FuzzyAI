# Calibration Semantics, Dataset Identity, and Evaluation Contract

Status:

```text
Calibration semantics:     design proposed (this document)
CalibrationObservation:    design only, not implemented
CalibrationBinding:        design only, not implemented
CalibrationDataset:        design only, not implemented
CalibrationProfile:        design only, not implemented
Calibration runtime:       not implemented
predicted_correctness:     None for every result the runtime can currently produce
```

This document designs the semantics of calibration. It implements nothing. There
is no `CalibrationProfile`, no fitting, no reliability metric, and no profile
matching in the repository, and this document does not add any.

This document is a design proposal and makes no verified claim. Every measured
number it cites (the Phase 2B.1 total-variation figures in section 1, and the
`scoring_label_mass` observations in sections 6.2, 13.3, and 20.4) is `[E]`
EXPERIMENTAL under its named conditions, as recorded in `docs/claims.md`. Nothing
here is `[V]` VERIFIED.

The question it answers:

> How does Probvenance move from an uncalibrated semantic probability produced by
> a model to correctness information supported by empirical data, without
> confusing different formulations, sources, tasks, or ground-truth semantics?

It does not answer which fitting algorithm to use. Algorithm choice is future
work (section 13).

## 1. Motivation

The runtime today produces probabilities with explicit, auditable provenance. A
`BoolResult` carries `probability_true`; a `ChoiceResult` carries a distribution
over ordered semantic candidates. Both are uncalibrated, both carry
`predicted_correctness = None`, and both carry `calibrated = False`.

That is deliberate. A restricted probability over two verbalizer logits, or a
restricted softmax over N categorical label logits, answers "which outcome does
the model favour". It does not answer "how often is that answer right".

Phase 2B.1 measured why that gap cannot be closed by vocabulary. Holding a
`ChoiceDecision` completely fixed and changing only the internal scoring-label
representation moved the emitted distribution by up to 0.31 total variation on
`Qwen/Qwen3.5-2B` and 0.57 on `openbmb/MiniCPM5-2B`, on two model families. A
number that moves when the representation moves is not a property of the task.

Calibration is the layer entitled to attach empirical correctness meaning. To do
that honestly it needs three things the runtime does not have on its own: a
declared population, real ground truth, and a binding that says exactly which
outputs a fitted mapping may be applied to.

## 2. Terminology

Three concepts that must never be merged.

### 2.1 Uncalibrated decision probability

What the runtime produces today.

```text
Bool:   P(True), restricted to the declared verbalizer pair
Choice: P(candidate_i), restricted to the declared scoring labels
```

Definition: the semantic decision distribution produced by the declared scoring
strategy and probability assembler for a specific decision, source, and evidence.

It is NOT:

```text
a real-world frequency
an empirical correctness rate
the probability the model answers correctly
a calibrated probability
predicted correctness
```

A restricted probability with `max = 0.95` does not mean there is a 95 percent
chance the answer is right. It means the model, under this formulation, places
0.95 of its candidate-restricted mass on that outcome.

### 2.2 Calibrated probability

A probability that has been mapped onto empirically meaningful values using
ground-truth data, within a stated population.

Two distinct senses exist and this document keeps them apart (section 4):

```text
distribution calibration:  recalibrate the whole probability distribution
correctness calibration:   map to a probability that the selected answer is right
```

Neither is implemented. Neither is implied by the other.

### 2.3 Predicted correctness

Definition:

> For a decision already produced, an empirical estimate, learned from historical
> ground-truth observations, that the result is the correct answer within the
> applicable population.

Example:

```text
ChoiceResult:  shipping = 0.82
Calibration:   predicted_correctness = 0.71
```

The two numbers answer different questions. `0.82` is the model's restricted
preference under a formulation. `0.71` is a statement about how often answers
like this were right, in a named population, under a named profile.

`predicted_correctness` is always `None` until a profile exists and matches
(section 12). It may never be populated from a max softmax, a logit, an entropy,
a margin, `scoring_label_mass`, or a model's self-reported confidence. The
constitution forbids this by name.

## 3. Calibration target

Two targets are possible. They are related but not the same problem.

### 3.1 Winner correctness

```text
selected = the result's selected semantic value
Y_correct = 1 if selected == ground_truth else 0
```

The calibrator learns a mapping from an uncalibrated quantity (section 13) to an
estimate of `P(Y_correct = 1)`. This is the target behind `predicted_correctness`.

### 3.2 Class probability calibration

For each candidate `c_i`:

```text
P(c_i)  versus  1[ground_truth == c_i]
```

This is a reliability question about the whole distribution, evaluated per class
and globally. It is what multiclass ECE and classwise reliability curves measure.

These are different calibration problems:

```text
winner correctness:   one binary score per observation, about the selected value
class probability:    an N-way distribution per observation, about every class
```

A model can be well calibrated on winner correctness while being badly calibrated
on class probabilities, and the reverse. They need different observations,
different metrics, and different sample sizes.

### 3.3 What winner correctness is not

Winner correctness calibration does not re-rank candidates, does not change the
selected value, and does not change `value` or `probabilities`. It attaches a
separate correctness estimate to an already produced result. The uncalibrated
distribution is left exactly as it was.

## 4. Recommended first target

Recommended: **winner correctness first**, full distribution calibration later.

Reasons, in the order they weighed:

```text
1. The public contract already exists. BoolResult and ChoiceResult already carry
   predicted_correctness as a first-class field with documented semantics. Winner
   correctness gives that field a defined meaning without inventing a new one.

2. It directly supports the policy layer. Future abstention and review decisions
   need "how likely is this answer right", not "was the whole distribution
   renormalized". Predicted correctness is the natural policy input, with the
   caveat that policy also needs discrimination and a calibrated scalar does not
   by itself supply it (section 15).

3. It is a binary problem. One scalar input, one binary target, one calibrator.
   Multiclass distribution calibration must preserve arity, ordering, and
   candidate identity, and is far easier to get subtly wrong.

4. Sample size. Per-class calibration needs enough observations per candidate to
   estimate each class, which multiplies the data requirement by the arity.
   Winner correctness needs enough observations overall.

5. It composes with the measurement discipline. Winner correctness can be
   evaluated with Brier, log loss, and reliability curves on a held-out split
   without committing to a full distributional model.
```

Multiclass distribution calibration is not rejected. It is deferred so that the
first implementation can be small enough to audit. This is a recommendation for
sequencing, not a claim that winner correctness is sufficient forever.

## 5. Ground-truth semantics

Ground truth is first-class provenance, not a bare label.

A Choice ground truth is never a scoring label. It is always a semantic candidate:

```text
recorded:  ground_truth = "shipping"
never:     ground_truth = "B"
```

`B` is a scoring-label identity that only exists relative to a tokenizer and a
rendering. A dataset that stores `B` is storing an execution artifact and calling
it a task label.

### 5.1 Bool ground truth

A Bool label must answer the same semantic proposition as the `DecisionSpec`:

```text
question = "Does this request concern billing?"
ground_truth = True means the request does concern billing
```

Two Bool datasets whose questions are differently worded are not automatically
the same population. "Does this concern billing" and "Is this a payment problem"
are different propositions, and observations from them must not be pooled merely
because both are Bool. If a caller wants to pool them, that is a declared claim
about their equivalence, and it belongs in binding and evidence (section 9), not
in an implicit assumption.

### 5.2 Choice ground truth

Ground truth must belong to the semantic outcome space of the decision. It is
recorded with the candidate name, and with the taxonomy it came from when the
caller declares one.

### 5.3 Ground-truth provenance

A bare label is not enough to fit or evaluate a profile. An observation must be
able to state, at least conceptually:

```text
label_source        who or what produced the label
labeling_rule       by what rule (human adjudication, downstream outcome, log)
taxonomy_id/version which candidate taxonomy the label belongs to
adjudicated         whether disagreements were resolved
ambiguity_policy    how unresolved cases were handled
```

Without label provenance, an observation is not fit-eligible (section 8.3).

## 6. Ordinary error versus taxonomy miss

Two outcomes that look similar in a confusion matrix and are not the same thing.

### 6.1 Ordinary model error

```text
ground_truth is IN the declared candidate set
the model selected a different in-set candidate
```

This is a classification error. It belongs in winner-correctness fitting as
`Y_correct = 0`.

### 6.2 Taxonomy or support miss

```text
ground_truth is NOT in the declared candidate set
```

The model still produced an in-set answer, and Phase 2B.1 measured that it does so
with high probability: with a candidate set missing the true topic, both tested
models returned an in-set winner with `scoring_label_mass` near 1.0. Nothing in
the restricted distribution signalled that the answer was out of scope.

A taxonomy miss is not evidence that the model chose wrongly among the declared
candidates, because the right candidate was never available. Counting it as an
ordinary error would penalize the model for a defect in the candidate set.

The distinction is recorded, not collapsed:

```text
ordinary_error:    ground_truth in candidate set, selection wrong
taxonomy_miss:     ground_truth not in candidate set
```

Which of these enters a winner-correctness fitting population is a deliberate
choice, made and recorded. The conservative default is that winner-correctness
fitting uses ordinary errors and correct answers, and that taxonomy misses are
retained as observations but excluded from the accuracy estimate, with the
exclusion counted and reported. A taxonomy miss is a signal about the candidate
set, and it should be visible rather than silently absorbed.

Two consequences of that default must be stated, because they are easy to miss.
First, the estimate it produces is conditional on the candidate set being
adequate: it is the model's accuracy among the cases the taxonomy can represent,
and it is not the overall accuracy of the deployed system. Second, the runtime
cannot detect a taxonomy miss at inference time (section 13.3), so the deployment
population contains undetectable misses while the profile was fitted without them.
A profile fitted only on in-set truth is therefore applied to a population that
includes cases where the system is wrong, which biases `predicted_correctness`
upward exactly where it matters. Excluding misses from fitting does not remove
them from deployment. This document does not propose a detection rule and does not
introduce a threshold; it records the limitation so that a future profile does not
silently inherit it.

Candidate naming is not decided by this document. `out_of_support`,
`taxonomy_miss`, and `invalid_for_closed_set` are all reasonable; choosing one is
a naming decision reserved for implementation, not a semantic question.

## 7. Ambiguous or unresolved ground truth

Sources of ambiguity:

```text
multiple annotators disagree
ground truth unresolved at fit time
multiple valid labels
```

The Choice contract is closed-set, single-label, and mutually exclusive. An
unresolved or multi-valid label does not fit that contract, and it must not be
silently compressed into one of the valid labels.

Options considered:

```text
exclude                drop the observation from fitting
mark unresolved        keep it, flagged, never counted as correct or incorrect
require adjudication   refuse to fit on it until a single label is decided
```

Recommended: **require adjudication for inclusion, with unresolved observations
retained and flagged**.

Reasoning. A winner-correctness label is binary by definition, so an observation
whose truth is genuinely multiple-valid has no correct `Y_correct` and cannot be
fit. Silently picking one valid label would fabricate ground truth. Dropping it
without a trace would hide how much of the population was ambiguous, which is
itself important (a taxonomy with 30 percent ambiguous cases is a taxonomy
problem). So unresolved observations are recorded with an explicit ambiguous
status, excluded from the binary fit, and counted in reporting.

Implementing multi-label truth is explicitly out of scope. If multi-label truth
becomes necessary, it is a separate target with its own observation shape, not a
patch to this one.

## 8. CalibrationObservation

A conceptual structure. Not implemented.

It must be able to answer: what did the model output, what was the truth, was it
correct, which formulation produced it, which source produced it, and which task
population it belongs to.

Fields are grouped by role. The grouping matters because not every field is part
of identity.

### 8.1 Required measurement

```text
decision_family                  bool or choice
semantic_outcome_space           ordered candidate names + descriptions, or [False, True]
uncalibrated_probabilities       the restricted distribution as produced
selected_value                   the semantic value the runtime selected
ground_truth                     the semantic ground truth (never a scoring label)
correct                          derived: selected_value == ground_truth
observation_status               correct | ordinary_error | taxonomy_miss | ambiguous
```

`selected_value` is the value after the runtime's deterministic tie-break
(section 17.3). The observation records the selection the system actually made; it
never re-runs the tie-break.

### 8.2 Binding metadata

The fields that decide which profile may be applied. See section 9.

```text
probability_formulation_fingerprint
model_id
model_revision
tokenizer_id
tokenizer_revision
rendering_semantics
domain_or_task_declaration      when the caller supplies one
taxonomy_id/version             when the caller supplies one
```

`decision_family` and the family fingerprint are not repeated here. The decision
family is already inside `probability_formulation_fingerprint`, and a family
fingerprint is not a matching key (section 9.8, section 12.3).

### 8.3 Ground-truth provenance (identity-bearing)

Required for fit-eligibility (section 5.3, P10), and therefore part of the
observation identity rather than lineage decoration:

```text
label_source        who or what produced the label
labeling_rule       by what rule
adjudicated         whether disagreements were resolved
ambiguity_policy    how unresolved cases were handled
taxonomy_id/version when the caller declares a taxonomy
```

A profile fitted against human-adjudicated labels and a profile fitted against
downstream-outcome labels are different artifacts even when the recorded labels
agree, so these fields may not be dropped from the observation identity.

### 8.4 Audit metadata

Recorded for lineage, not used as identity:

```text
trace_id, execution_fingerprint    which execution produced the result
timestamp
dataset_provenance
```

Audit metadata answers "where did this come from". Binding metadata answers "what
may be applied to this". Ground-truth provenance answers "what does the label
mean". Conflating them is how a calibration profile ends up bound to a timestamp.

The brief's caution applies: do not mechanically place every field into a primary
key. The four groups above are the discipline.

## 9. CalibrationBinding

A `CalibrationBinding` expresses which class of probability outputs a profile or
observation is about. It is not a profile and not an algorithm.

### 9.1 Why not the Execution Fingerprint

The execution fingerprint is too fine. It commits the actually rendered input, the
resolved scoring token ids, and therefore the instance evidence. Almost every
observation differs in it, so a key that strict matches nothing twice and is
useless as a population key.

### 9.2 Why not the Plan Fingerprint

The plan fingerprint also commits the question, the context, and the rendered
prompt. Two observations whose only difference is the ticket text have different
plan fingerprints. Same problem, one level up.

### 9.3 The composition

Consistent with `docs/probability-semantics-identity.md` section 9:

```text
CalibrationBinding
  = probability formulation identity
  + probability source identity
  + task/domain evidence
```

A binding is a composition of existing identities plus declarations. It is not a
new fingerprint and it does not replace the formulation or family fingerprints.
Why the composition is three parts and not one: the formulation fingerprint
deliberately excludes the model, the task, and the instance evidence. The model
belongs to the source axis by design, task and domain sit in the binding per
`docs/probability-semantics-identity.md` section 9, and INV-25 excludes instance
evidence from the formulation identity. Calibration is where those independently
separated axes must be brought back together, because empirical correctness
genuinely depends on all three.

### 9.4 Must bind

```text
probability_formulation_fingerprint
model_id
model_revision
tokenizer_id
tokenizer_revision
rendering_semantics (scoring-relevant subset)
```

The decision family and the outcome space are inside
`probability_formulation_fingerprint` and are not repeated. The family fingerprint
is not a matching key, because a family match does not grant pooling (INV-24,
section 12.3).

### 9.5 Should bind

```text
domain_or_task_declaration
taxonomy_id and taxonomy_version
```

### 9.6 Optional metadata

```text
fit timestamp
operator identity
source system reference
```

### 9.7 Open

```text
whether a declared plan family may serve as a coarser binding for pooling
cross-revision reuse within one model
cross-arity calibration
description-paraphrase equivalence
```

Not decided here. The pooling and revision questions map to the open questions in
section 23; the cross-arity and description-paraphrase questions remain open in
`docs/probability-semantics-identity.md` section 16. This document does not add
them as further open questions of its own.

### 9.8 Do not duplicate formulation fields

If a binding already carries `probability_formulation_fingerprint`, it must not
also carry compiler id, assembler version, doctrine version, and the mapping.
Those fields are inside the formulation identity, and copying them creates a
second source of truth that can drift.

The distinction is:

```text
identity key        the fingerprint, used for matching
audit projection    a human-readable rendering of what the fingerprint commits
```

An audit projection is allowed, and useful, as long as it is derived from the
fingerprint and never used for matching. It is a view, not a key.

### 9.9 Caller declarations are provenance, not proof

If a caller declares `task_id = "support-routing"` or
`taxonomy_id = "support-routing-v3"`, Probvenance records the declaration. A
matching string is not proof that two populations are statistically
interchangeable, and it is not proof that two candidate sets are semantically
equivalent. The declaration is an assertion carried as provenance.

## 10. CalibrationDataset identity

A `CalibrationDatasetFingerprint` answers: which exact observations was this
profile fitted on?

It is not the binding. The two answer different questions:

```text
CalibrationBinding             which outputs is this profile for
CalibrationDatasetFingerprint  which observations was it fitted on
```

A profile fitted on a small, unrepresentative dataset under a correct binding is
still a bad profile, and only the dataset identity makes that visible.

### 10.1 Ordering

The dataset identity is a canonical identity over the observation multiset. Row
order in a file is not semantically meaningful, so:

```text
same observations, different row order  ->  same dataset fingerprint
```

### 10.2 Multiplicity

Duplicates must not be collapsed by a set operation. A repeated observation can
mean a genuinely repeated event, a repeated sample, or a data bug, and those are
not the same population. Preserving multiplicity is what lets a reader distinguish
"one observation" from "the same observation a thousand times".

The proposed mechanism, designed but not implemented:

```text
1. compute a canonical fingerprint for each observation
2. sort the fingerprints
3. hash the sorted list WITH multiplicity preserved
```

A set-based hash would lose the multiplicity and is rejected. A multiset hash over
the sorted list keeps ordering-independence and multiplicity at the same time.

### 10.3 What the dataset identity does not say

It says which observations were used. It does not say they are representative, it
does not say the split was held out, and it does not say the labels were good.
Those are separate claims with separate evidence (section 11).

### 10.4 Observation identity

A future `CalibrationObservationFingerprint` would commit one observation's
measurement and provenance: the decision or input identity, the uncalibrated
probabilities, the selected semantic value, the ground truth, its label
provenance (section 8.3), the formulation identity, and the source metadata. It is
not implemented this round, and the dataset mechanism in 10.2 depends on its
canonical form being well defined.

The Trace ID is not that identity. A trace id identifies one execution instance.
The same deterministic replay produces a different trace id, and one semantic
observation can arise from several executions. An observation identity must be a
function of the measurement and its provenance, never of when or how many times it
was computed. Trace id and execution fingerprint are therefore audit metadata
(section 8.4), not the observation identity.

## 11. Data splits

Frozen discipline:

> Calibration fitting data and final calibration evaluation data must be distinct,
> unless an explicit resampling or cross-validation procedure with recorded
> provenance is being used.

Fitting a calibrator and then reporting its reliability metric on the same
observations is not calibration evidence. It measures the calibrator's ability to
memorize its own training set.

Conceptual splits:

```text
train / fitting        fit the calibrator parameters
validation             model selection among candidate calibrators
test / evaluation      the reported quality number
```

Any reported calibration quality statement must name the population and the split
that produced it. A number without a split is not a result.

No cherry-picking. If several calibration methods are tried, the selection
procedure is recorded, and the final reported quality comes from a split that was
not used to select the method. Selecting the best method and then advertising its
quality on the same held-out set without recording the selection is not evidence.

## 12. Profile identity, matching, and mismatch behavior

### 12.1 Profile is not a result

```text
CalibrationProfile   a reusable fitted artifact
predicted_correctness  a value produced by applying a profile to one result
```

They are not one object. A profile outlives any single result, and a result is
never the fitting artifact.

### 12.2 Profile identity

A profile's identity is composed of everything that determines what its fitted
numbers mean:

```text
CalibrationProfile identity
  = binding
  + calibration target
  + method_id and method_version
  + fitted parameters
  + training dataset fingerprint
```

The binding alone is not the profile identity: one binding can carry several
profiles, for different targets, methods, or fitting datasets. Two profiles that
differ in any component are different artifacts and are never interchangeable.

### 12.3 Matching is exact by default

Default rule: a profile applies only when its binding matches the observation's
binding exactly.

Two cases are distinguished and are not the same:

```text
caller supplies a profile whose binding does not match  ->  explicit error
no profile exists for this binding                      ->  predicted_correctness = None
```

The first is a mistake the caller made and must be surfaced. The second is
ordinary: most results will have no profile. A mismatch is never quietly treated
as "no profile exists", because that would hide a misapplied artifact.

Explicitly forbidden silent behaviors:

```text
find the nearest profile
fall back to a formulation-family profile
fall back to a global model-level calibration
use a profile fitted on a different rendering mode
```

These may be researched later, but never as a silent substitution. A wrong
calibration applied silently is worse than no calibration, because it manufactures
false correctness information.

### 12.4 Missing profile

When no profile applies:

```text
predicted_correctness = None
```

and, once the runtime has the field, an explicit uncalibrated status. It must
never be replaced by:

```text
the max probability
an entropy-derived confidence
a margin-derived score
```

This preserves the current public contract, which already returns `None`.

## 13. Calibration method identity

No method is implemented and no library is chosen. Candidate classes to keep in
view:

```text
Platt / logistic scaling
temperature scaling
isotonic regression
histogram / binning methods
```

Whatever is chosen, a profile must record:

```text
method_id
method_version
fitted parameters
```

Algorithm identity is provenance. A profile whose method is unknown cannot be
audited, and a profile whose version changed without a version bump can be
silently misapplied.

### 13.1 Temperature scaling caveat

Temperature scaling is a natural fit for logits or a multiclass distribution,
because it rescales the whole vector. Winner-correctness calibration is a scalar
problem: one score in, one correctness estimate out. Temperature scaling is
popular and is therefore the reflex choice, but it is not obviously the right
first method for a scalar target. A scalar calibrator (Platt scaling or isotonic
regression) is at least as plausible for the winner-correctness target. The choice
belongs to implementation, with evidence.

### 13.2 Calibrator inputs

Three possibilities:

```text
A. selected probability only   max_i P(c_i)
B. richer deterministic features   entropy, margin, scoring_label_mass
C. the raw vector distribution   the whole restricted distribution
```

Different choices produce different calibration semantics. A calibrator trained on
the selected probability alone answers "given this selected probability, how often
right". A calibrator trained on entropy and margin as well answers a richer
question but needs more data and is harder to audit.

Recommendation for the first calibrator: **use the selected probability as the
primary input, and keep the other diagnostics out of the model initially**. Use
`entropy`, `margin`, and `scoring_label_mass` as evaluation slicing dimensions
(does calibration hold across low and high mass), not as silent model features.

Two limitations of that minimal recommendation must be stated, because they trade
one kind of correctness for another. First, a calibrator over the maximum of a
restricted distribution inherits the restriction: a high maximum with a low
`verbalizer_mass` or `scoring_label_mass` is not a decision at all, and a profile
fitted on the selected probability alone cannot correct for scoring-position
failures within a binding, because the mass is deliberately not an input and not a
threshold (section 16.2). Second, for a Bool distribution the margin is a
deterministic function of the selected probability (`margin = |2p - 1|`), so
slicing a Bool result by margin adds no information; the mass is the diagnostic
that carries information the selected probability does not. Neither limitation is
resolved here; they are recorded so the first implementation does not pretend the
scalar is sufficient.

Whatever is chosen, the input feature set must be versioned and bound into the
profile. A calibrator whose inputs are not recorded cannot be reproduced.

### 13.3 `scoring_label_mass` is not correctness

Phase 2B.1 measured that out-of-set inputs still produce `scoring_label_mass`
near 1.0. The mass measures how much probability the declared scoring labels hold,
not whether the candidate set is right and not whether the answer is correct. It
is at most a candidate feature for a future calibrator, never a certainty signal
and never a correctness proxy.

## 14. Metric semantics

Metrics evaluate a calibrator. They do not produce correctness information.

### 14.1 Brier score

For winner correctness, with `p_i` the predicted correctness estimate and
`y_i` in `{0, 1}`:

```text
Brier = mean((p_i - y_i)^2)
```

Range `[0, 1]`, lower is better. It is a proper scoring rule.

Caveat: Brier is affected by both calibration and discrimination (refinement). A
model with poor discrimination but a well-fitted calibration can achieve a
moderate Brier by predicting something close to the base rate for everything. So
Brier is not a pure calibration metric, and it must not be described as one.

### 14.2 Log loss

For binary correctness:

```text
log_loss = -mean( y * log(p) + (1 - y) * log(1 - p) )
```

Properties: proper scoring rule, punishing on confident errors far more than
Brier. The numerical treatment of `p = 0` and `p = 1` is a real implementation
concern and is explicitly deferred. This document does not fix an epsilon policy.

### 14.3 Expected calibration error

ECE is not "the calibration metric". It is one estimator with several free
choices, and its value depends on all of them:

```text
binning scheme (equal-width, equal-mass)
bin count
bin boundaries
sample size
```

Two ECE numbers computed with different configurations are not comparable, and a
single ECE number without its configuration is not evidence.

Critically, ECE is also sensitive to sample size in ways users forget: with few
observations per bin, ECE is dominated by noise, and it can be driven to near zero
by overfitting the bins if the evaluation data was also used for fitting.

### 14.4 ECE identity

The metric configuration is itself provenance:

```text
ECE, bins = 10, equal-width
```

and

```text
ECE, bins = 15, equal-mass
```

are different metric configurations and must not be compared as if they were one
number. Any reported ECE carries its configuration.

### 14.5 Reliability curve

Human-auditable output:

```text
predicted probability bin  versus  observed correctness rate
```

It shows whether the model is overconfident or underconfident and where. No
plotting is implemented.

### 14.6 Multiclass metrics

For future full-distribution calibration:

```text
multiclass Brier
multiclass log loss
classwise reliability
top-label calibration
```

The distinction to preserve:

```text
winner-correctness evaluation   one binary problem
full-distribution evaluation    an N-way problem
```

They are not interchangeable and must not be reported under one name.

## 15. Calibration is not discrimination

Calibration and discrimination are different properties and must not be conflated.

> Calibration cannot manufacture discrimination.

If a model outputs `0.99` for every input, a calibrator can in principle map
`0.99` to the population base rate, producing a well-calibrated overall number.
The model still cannot tell one input from another. A calibration procedure that
reports good reliability for this model is measuring the population base rate, not
the model's ability to discriminate.

Consequences:

```text
a calibration report must state discrimination alongside calibration
a well-calibrated but non-discriminating model is not a useful decision component
"calibrated" is never a substitute for "discriminates"
```

## 16. Observation validity

Three gates, designed but not implemented.

### 16.1 Structural validity

```text
a valid Result
a valid Trace
a known formulation identity
ground truth compatible with the decision family
```

### 16.2 Scoring-position validity

Phase 2A.1 and 2B.1 added diagnostics: `verbalizer_mass` for Bool and
`scoring_label_mass` for Choice. These let a reader see whether the model was
positioned to emit a scoring label at all.

They must NOT become an inclusion threshold. In particular:

```text
mass > 0.5  ->  include        FORBIDDEN as an automatic rule
```

The diagnostics belong in observation metadata and evaluation slices. A threshold
that silently decides which ground-truth observations count is exactly the kind of
hidden policy the project forbids. If a future validity policy uses a threshold,
it is an explicit, versioned, evidence-backed policy object, not a magic number in
a fitting routine.

### 16.3 Semantic validity

A calibration fit that produces good reliability numbers does not prove the model
has a semantic signal for the task. Calibration operates on the outputs the model
already produces; it cannot create information the model does not carry. Section
15 is the formal version of this caution.

## 17. Correctness semantics

### 17.1 Bool

```text
Y_correct = 1 if selected_value == ground_truth else 0
```

Simple and unambiguous, provided the ground truth answers the same proposition
(section 5.1).

### 17.2 Choice

```text
Y_correct = 1 if selected_semantic_candidate == ground_truth_semantic_candidate
```

Never compare scoring labels. `selected == "shipping"` is correct; `selected ==
"B"` is an execution artifact.

### 17.3 Ties

The runtime already resolves ties deterministically: the first candidate in
declared semantic order among those with maximal probability wins (`ChoiceResult`
uses a strict greater-than comparison while iterating in candidate order; INV-03).

A calibration observation records the value the runtime actually selected. It
never re-runs the tie-break, never consults a different ordering, and never treats
a tie as ambiguous for correctness purposes. The selection is a fact of the
execution.

### 17.4 Out-of-set

See section 6. A taxonomy miss is not an ordinary error and is flagged as such.

## 18. Pipeline and online application

### 18.1 Full conceptual pipeline

```text
DecisionSpec
  -> Compiler
  -> InferencePlan
  -> Backend
  -> RawEvidence
  -> ProbabilityAssembler
  -> Uncalibrated DecisionResult
  -> (offline, with ground truth) CalibrationObservation
  -> CalibrationDataset
  -> CalibrationProfile
  -> Calibrated correctness information
  -> Policy
```

Ground truth is not part of the online inference pipeline. It exists in the
offline evaluation and fitting path. A system that required ground truth at
inference time would not be doing inference.

### 18.2 Online application

```text
DecisionResult
  + a matching CalibrationProfile
  -> predicted_correctness
```

If no profile matches, `predicted_correctness = None`.

### 18.3 Abstention boundary

Calibrated predicted correctness is an input to a future policy, not the policy.
This document does not implement or define `accept`, `abstain`, `review`, or
`escalate`.

```text
Calibration  !=  Policy
```

Risk-coverage and selective accuracy are future evaluation work with their own
thresholds, and are not defined here.

## 19. Claims discipline

A calibration claim is allowed only when all of the following exist:

```text
a fitted calibration profile
a held-out evaluation split
a documented population and binding
```

Then, and only then, may a statement scope to that profile and population:

```text
"This named configuration was evaluated for calibration on dataset X under
binding Y."
```

Forbidden:

```text
"Probvenance probabilities are calibrated."
"Choice probabilities are accurate."
"predicted_correctness is trustworthy in general."
```

Even a profile with excellent reliability supports only a scoped statement about
that profile, that dataset, and that binding. Outside the fitted population and
binding, the honest answer is unknown, and unknown beats silent reuse.

This is the calibration-specific application of AP-05 (claims require evidence
status) and AP-06 (semantics-affecting changes are provenance-relevant).

## 20. Worked examples

### 20.1 Bool, correct

```text
Decision:    "Does this concern billing?"
Formulation: binary_token_logits, yes/no, doctrine binary-semantic-judgment-v1
Evidence:    ticket text
Output:      P(True) = 0.82, selected = True
Ground truth: True
correct = 1
```

The value `0.82` is an uncalibrated restricted probability. It is not the
predicted correctness `0.82`. Only after a profile fitted on a matching binding
learns that outputs near `0.82` were correct about 71 percent of the time does a
`predicted_correctness` near `0.71` become meaningful, and even then only for that
binding and population.

### 20.2 Choice, correct

```text
Decision:     "Which team owns this ticket?"
Candidates:   billing / shipping / technical (ordered)
Output:       billing = 0.10, shipping = 0.72, technical = 0.18
selected = shipping
Ground truth: shipping
correct = 1
```

`0.72` is a restricted Choice probability over three scoring labels. It is not a
correctness probability. The two numbers in `shipping = 0.72` and a later
`predicted_correctness = 0.68` answer different questions.

### 20.3 Choice, wrong

```text
selected = shipping
ground truth = technical
correct = 0
observation_status = ordinary_error
```

This is the wrong-selection case that enters a winner-correctness fitting
population as `Y_correct = 0`.

### 20.4 Out-of-set

```text
Candidates:   billing / shipping / technical
Ground truth: account
```

The true answer is outside the declared closed set. The model will still return an
in-set winner with high `scoring_label_mass` (measured in Phase 2B.1). This is a
taxonomy miss:

```text
observation_status = taxonomy_miss
```

It is retained, flagged, and excluded from the winner-correctness accuracy
estimate by default, with the exclusion counted. It is not silently recorded as an
ordinary error.

### 20.5 Cross-formulation

```text
Plan A: billing -> A, shipping -> B, technical -> C
Plan B: billing -> C, shipping -> A, technical -> B
```

These have the same formulation family (same strategy, compiler, assembler,
doctrine, label scheme, arity) and different probability formulation
fingerprints. Phase 2B.1 measured that the distributions differ. By default they
do not enter the same calibration dataset, and a profile fitted on one is not
applied to the other. A deliberate cross-formulation comparison is allowed when
both identities and the method are recorded (INV-23), but it is never implicit.

### 20.6 Cross-source

```text
Same exact formulation.
Source A: Qwen3.5-2B
Source B: MiniCPM5-2B
```

Same formulation identity, different source. Even with identical candidate sets,
the model is a binding dimension, so the two do not share a profile by default.
AP-06 is explicit that a profile must not be bound merely to a model name, and the
converse holds too: a profile fitted on one model is not valid for another by
default.

### 20.7 Rendering mode

```text
Same model, same formulation, same source revision.
Rendering A: enable_thinking = False
Rendering B: enable_thinking = True
```

Phase 2A measured that with thinking enabled the next token is a reasoning token,
both verbalizer logits sit in the tail, and the restricted softmax renormalizes
tail noise into a plausible-looking number. Rendering semantics are a binding
dimension, and an absent `enable_thinking` is not equal to
`enable_thinking = False` (INV-26). The two rendering modes never share a profile.

## 21. Proposed future invariants

Proposed for review. Not written into `docs/design-constitution.md`. Numbering
continues the design document's P series (P1 to P5 live in
`docs/probability-semantics-identity.md` section 13). These are calibration-layer
invariants only, and none of them repeats an INV-23..27 rule.

```text
P6  Predicted correctness requires a matching profile. An uncalibrated result's
    predicted_correctness is None, and no transform may populate it without a
    fitted profile whose binding matches the observation's binding.

P7  A calibration profile's identity is its binding, its calibration target, its
    method identity and version, its fitted parameters, and its training dataset
    identity. Two profiles differing in any component are different artifacts and
    are never interchangeable.

P8  A calibration profile records its fitting dataset identity and its
    fitting/evaluation split. A calibration quality statement without a held-out
    population is not evidence.

P9  Profile matching is exact by default. An unmatched observation yields no
    predicted correctness, never a fallback to a nearest, family, or global
    profile.

P10 Ground-truth semantics are first-class provenance. An observation without
    label provenance and a resolution status is not fit-eligible.
```

Two calibration consequences of the frozen invariants are deliberately not
repeated as new invariants here: that pooling across formulation identities
requires an explicit, evidence-backed declaration, and that family membership does
not grant pooling. Those are INV-23 and INV-24, and section 20.5 records what they
mean for calibration.

## 22. Non-goals

This round designs semantics only. It does not implement, and it does not decide
the eventual API for:

```text
CalibrationProfile, CalibrationObservation, CalibrationBinding
CalibrationDatasetFingerprint, CalibrationObservationFingerprint,
  CalibrationProfileFingerprint
any calibration fitting algorithm or library
temperature scaling, Platt scaling, isotonic regression, binning
Brier, log loss, ECE, or reliability-curve runtime code
profile matching runtime
a calibration store
abstention, review, escalate, or any policy
risk-coverage or selective accuracy
multi-label ground truth
open-set detection
```

It also does not change `src/probvenance/`. The runtime is untouched.

## 23. Open questions

Ten, and only ten. Solved questions have been removed.

```text
1.  Sequencing: winner-correctness first, or multiclass first. Recommended
    winner-correctness (section 4); the execution order is still open.
2.  Minimum sample size for a profile, and how to report the statistical
    uncertainty of a fitted calibrator.
3.  Class-conditional (per-candidate) correctness profiles: worth having, or
    does winner correctness suffice.
4.  Domain/task identity: caller-declared only, or also inferred, and how it
    binds when a caller supplies nothing.
5.  Taxonomy identity: is taxonomy_version a binding dimension or audit metadata,
    and who owns the version.
6.  Cross-formulation pooling: what evidence is sufficient to permit it
    deliberately.
7.  Cross-revision reuse: may a profile transfer across revisions of one model,
    and under what evidence.
8.  Calibration input features for the first calibrator (selected probability
    alone, or selected probability plus diagnostics).
9.  Dataset fingerprint mechanics: the canonical form of a single observation
    fingerprint, and whether duplicates ever need more than multiplicity.
10. A standard ECE configuration for reporting, or acceptance that ECE is always
    reported with its configuration and never compared across configurations.
```

## 24. Relationship to the other documents

```text
docs/design-constitution.md          defines Calibration, Predicted correctness,
                                     INV-23..27, AP-05, AP-06. This document
                                     applies them to the calibration layer.
docs/probability-semantics-identity.md  defines the formulation and family
                                     identities and the source axis, and its
                                     section 9 gives the binding composition this
                                     document builds on.
docs/choice-semantics.md             defines Choice probability semantics,
                                     including the out-of-set behaviour that
                                     section 6 relies on.
docs/claims.md                       records evidence status. This document is a
                                     design proposal and makes no verified claim.
docs/roadmap.md                      holds the Phase 4 and Phase 5 items; this
                                     document is their semantic contract.
experiments/choice_signal/REPORT.md  is the measurement behind sections 6 and 20.5.
```
