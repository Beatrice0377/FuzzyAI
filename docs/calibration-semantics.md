# Calibration Semantics, Dataset Identity, and Evaluation Contract

Status:

```text
Calibration semantics:     frozen (this document)
CalibrationObservation:    implemented (src/probvenance/calibration.py)
CalibrationBinding:        implemented (src/probvenance/calibration.py)
CalibrationDataset:        implemented (src/probvenance/calibration.py)
CalibrationProfile:        not implemented
Evaluation cohort:         implemented (src/probvenance/calibration_evaluation.py):
                           declared evaluation source cohort, full observation
                           statuses retained, cohort fingerprint v1
Evaluation dataset:        implemented (src/probvenance/calibration_evaluation.py):
                           metric-eligible projection of one declared cohort,
                           dataset fingerprint v2 with explicit exclusion
                           accounting
Evaluation metrics:        pre-calibration winner-correctness Brier and exact
                           natural-log log loss over the metric-eligible
                           projection implemented
                           (src/probvenance/calibration_evaluation.py); result
                           fingerprint v2 commits source cohort identity and
                           exclusion accounting
Companion diagnostics:     implemented (src/probvenance/calibration_evaluation.py):
                            winner-correctness companion diagnostics artifact
                            (empirical correctness rate, mean selected
                            probability, empirical constant Brier reference)
                            with result fingerprint v1
Reliability summary:       implemented (src/probvenance/calibration_evaluation.py):
                            pre-calibration equal-width winner-correctness
                            reliability summary (binning id equal-width v1,
                            reliability id winner-reliability-curve v1, result
                            fingerprint v1); plotting and equal-mass binning
                            remain unimplemented
Binned absolute-gap
aggregate:                 implemented (src/probvenance/calibration_evaluation.py):
                            derived ECE-form aggregate over one exact
                            WinnerReliabilityResult (aggregate id
                            winner-correctness-equal-width-binned-absolute-gap
                            v1, result fingerprint v1); it is a binned
                            absolute-gap diagnostic, NOT a calibration-error
                            claim; the ordinary calibration-error
                            interpretation remains ungranted
Fitting algorithms:        not implemented
Calibration runtime:       not implemented
predicted_correctness:     None for every result the runtime can currently produce
```

This document originated as a design document. The Phase 4A data foundation
portions it specified (ground truth, binding, observation, dataset, and their
identities and fingerprints, in `src/probvenance/calibration.py`) are now
implemented. The Phase 4A evaluation foundation it specified (the evaluation
dataset contract and the winner-correctness Brier metric, in
`src/probvenance/calibration_evaluation.py`) is now implemented as well, and
the evaluation pipeline now commits cohort provenance: metric exclusion is
provenance. `CalibrationProfile`, fitting, the remaining metrics, and runtime
profile application remain unimplemented, and this document does not add any.

This document is in part a design proposal. The deterministic data model and
the evaluation foundation carry `[V]` VERIFIED claims in `docs/claims.md`,
backed by tests in `tests/test_calibration.py` and
`tests/test_calibration_evaluation.py`. Every measured number it cites (the
Phase 2B.1 total-variation figures in section 1, and the `scoring_label_mass`
observations in sections 6.2, 13.3, and 20.4) is `[E]` EXPERIMENTAL under its
named conditions, and the not-yet-implemented portions (fitting, profiles,
runtime application) remain design statements without verified claims.

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

Implemented as `CalibrationObservation` in `src/probvenance/calibration.py`.
It answers: what did the model output, what was the truth, was it correct,
which formulation produced it, which source produced it, and which task
population it belongs to.

Fields are grouped by role. The grouping matters because not every field is part
of identity.

An observation may only be constructed through `from_evaluation`. Before any
provenance is derived, the result and the trace must carry matching runtime
linkage identities: the runtime stamps the same trace id onto both the
assembled result and the reported trace, and the pair is checked under the
supported construction contract. A pair whose linkage identities do not match
is rejected rather than repaired or warned about, and a result without a trace
id cannot be verified against the trace at all, so it is rejected too. Direct
field construction of an observation is rejected for the same reason: without
the runtime-issued trace there is no linkage identity to check.
`dataclasses.replace` reconstruction is rejected for the same reason again:
it rebuilds the observation through the direct constructor. The linkage check
is linkage CONSISTENCY, not content attestation: it protects against
accidental pairing of objects carrying different runtime linkage identities,
and it does not prove that a caller-constructed result's probabilities were
emitted by the supplied trace.
Lower-level Python escape hatches such as `object.__new__`, `copy`, and
`pickle` are not supported construction paths and are not defended against.

### 8.1 Required measurement

```text
decision_family                  bool or choice
semantic_outcome_space           ordered candidate names + descriptions, or [False, True]
uncalibrated_probabilities       the restricted distribution as produced
selected_value                   the semantic value the runtime selected
ground_truth                     the semantic ground truth (never a scoring label)
correct                          derived: selected_value == ground_truth
observation_status               resolved | taxonomy_miss | unresolved
```

The runtime status enum is exactly the `CalibrationObservationStatus` members
`RESOLVED` / `TAXONOMY_MISS` / `UNRESOLVED` shown above. The finer vocabulary
`correct | ordinary_error | taxonomy_miss | ambiguous` used elsewhere in this
document (sections 6 and 20) is a conceptual resolution vocabulary for
reasoning about fitting populations, not the runtime enum: the runtime folds
the correct/ordinary-error distinction into the derived `correct` boolean on
`RESOLVED` observations, and represents ambiguity as `UNRESOLVED`.

`selected_value` is the value after the runtime's deterministic tie-break
(section 17.3). The observation records the selection carried by the supplied
runtime-linked result; it never re-runs the tie-break. Linkage does not attest
execution origin (section 17.3).

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

A binding is a composition of existing identities plus declarations. Its
`.fingerprint` is an internal, versioned representation of that composite
identity (the runtime exposes and consumes it for identity and diagnostics);
it does not replace the component formulation or family fingerprints.
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

The dataset fingerprint answers: which exact observations was this profile
fitted on? In this document `CalibrationDatasetFingerprint` is conceptual
vocabulary for that identity: it is implemented as the `fingerprint` property
of `CalibrationDataset`, not as a separate class.

It is not the binding. The two answer different questions:

```text
CalibrationBinding             which outputs is this profile for
dataset fingerprint            which observations was it fitted on
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

The implemented mechanism (`CalibrationDataset.fingerprint`):

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

The implemented `CalibrationObservation.canonical_payload()` commits one
observation's measurement and provenance: the decision or input identity, the
uncalibrated probabilities, the selected semantic value, the full ground-truth
record including its label provenance (section 8.3), the formulation identity,
and the source metadata.

The Trace ID is not that identity. A trace id identifies one execution instance.
The same deterministic replay produces a different trace id, and one semantic
observation can arise from several executions. An observation identity must be a
function of the measurement and its provenance, never of when or how many times it
was computed. Trace id and execution fingerprint are therefore audit metadata
(section 8.4), not the observation identity.

### 10.5 Pooling rule

A `CalibrationDataset` pools observations only when they share the same
probability population AND the same ground-truth semantics:

```text
same CalibrationBinding canonical payload          probability population identity
same GroundTruthSemanticsIdentity                  what the ground-truth target MEANS
all observations fit-eligible                      eligibility
```

`GroundTruthSemanticsIdentity` is derived from the ground-truth provenance and
commits the labeling rule, the ambiguity policy, and the ground-truth taxonomy
pair. It deliberately excludes `label_source`: a specific label producer is not
part of the statistical target, so different label sources (for example
different annotators) may coexist in one dataset when the labeling semantics
match, while their observation identities remain distinct because
`label_source` stays in the observation identity through the full provenance.
Full `GroundTruthProvenance` equality is never required for pooling.

The probability binding and the ground-truth semantics stay orthogonal: they
remain separate classes, and the semantics identity is never placed inside
`CalibrationBinding`. Conceptually a fitting population is
`CalibrationBinding + GroundTruthSemanticsIdentity`.

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

Dependent observations are recorded here as a methodology constraint, not as a
dataset identity primitive:

> When observations carry an explicit dependence or group identity, train,
> validation, and test procedures must not silently split one declared dependency
> group across partitions when doing so would create leakage.

Core calibration code must not infer groups from case-id prefixes, ladder names,
string conventions, or observation similarity. Phase 4C.0 does not add a group
identity field to `CalibrationObservation`, because no general data contract for
it exists yet. Experiment harnesses may enforce group-aware partitioning
separately, using their own explicit grouping.

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
  = CalibrationBinding
  + GroundTruthSemanticsIdentity
  + calibration target ID and calibration target version
  + method_id and method_version
  + fitted parameters
  + training dataset fingerprint
```

The calibration target is an explicit versioned identity, not a prose
placeholder: as of Phase 4C.0 the single implemented target is
`winner_correctness` version 1 (`WINNER_CORRECTNESS_TARGET_ID` and
`WINNER_CORRECTNESS_TARGET_VERSION`). That identity is shared by every
winner-correctness evaluation artifact and is owned by no single metric. A
target with the same ID but a different version is a different target
semantics identity, not the same target under a new label.

The target identity is owned by the calibration foundation module, not by the
evaluation layer, because the profile identity above composes it with
`CalibrationBinding` and `GroundTruthSemanticsIdentity`. Those two already live
in the calibration foundation, so keeping the target there lets profile fitting
consume all three without the calibration foundation depending on the
evaluation layer, which itself depends on the calibration foundation.

The binding alone is not the profile identity: one binding can carry several
profiles, for different targets, methods, or fitting datasets. The ground-truth
semantics identity is included because the same probability population does not
imply the same meaning of correctness: labels established under different
labeling rules or ambiguity policies measure different statistical targets, so
two datasets under one binding but with different ground-truth semantics must
never feed one profile identity. Two profiles that differ in any component are
different artifacts and are never interchangeable. The profile itself remains
unimplemented.

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

Exact binding match is also not relaxed by formulation-family membership. The
probability formulation family does not prove that calibration transfers: the
Phase 2 identity work is evidence FOR exact binding, not permission for family
matching. A relaxed match would require its own explicit, identity-bearing
policy and supporting evidence.

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

### 12.5 Profile application provenance

Frozen requirement, before any profile exists:

```text
No supported path may produce calibrated = True or a non-None
predicted_correctness without inspectable calibration-profile identity and
version provenance.
```

A calibrated result must be auditable back to the exact profile artifact that
produced it, otherwise a downstream consumer cannot tell which fitted numbers,
binding, ground-truth semantics, target, method, and training population the
value came from. This round records the requirement only. The storage location
is deliberately NOT frozen yet: it may end up on the calibrated result's own
provenance, on a separate calibration-application artifact, or as a
`DecisionTrace` extension, and the design constitution still leaves that open.
`predicted_correctness` remains `None` for every result the runtime can
currently produce.

### 12.6 Taxonomy compatibility precondition for fitting and application

Two taxonomy declarations exist and they answer different questions. The
`CalibrationBinding` taxonomy is part of the declared probability population.
The `GroundTruthSemanticsIdentity` taxonomy is part of what the labels mean.
They are not the same concept and neither axis is removed or merged.

A fitting population can currently hold a binding taxonomy of `support` v3
beside a ground-truth taxonomy of `legacy-support` v1 and remain structurally
valid, because Phase 4A recorded orthogonal identities. That is not sufficient
once a profile claims to map scores from one population onto correctness labels
from another taxonomy. An explicit contradiction must not enter profile
semantics unexplained.

Frozen rule for the FIRST profile and fitting implementation:

```text
Case A  both taxonomy IDs concrete and equal
        potentially compatible; if both versions are concrete they must also be
        equal, and a concrete version mismatch is an explicit mismatch

Case B  both taxonomy IDs concrete and different
        profile fitting and application are NOT allowed by the initial
        contract; fail closed

Case C  one or both taxonomy IDs unknown (None)
        equality is never invented and the pair is never called proven
        compatible; the pair stays structurally representable because the
        taxonomy declaration is optional provenance

Case D  same taxonomy ID but only one version concrete
        the unknown version is never inferred to equal the known one; this is
        an unresolved relationship, not an explicit contradiction, and neither
        side is silently rewritten
```

Worked examples of the intended rule:

```text
binding taxonomy support v3   + ground-truth taxonomy support v3
  -> no explicit contradiction

binding taxonomy support v3   + ground-truth taxonomy support v2
  -> explicit version mismatch; initial profile fitting fails closed

binding taxonomy support v3   + ground-truth taxonomy legacy-support v1
  -> cross-taxonomy relationship; initial profile fitting fails closed

binding taxonomy None         + ground-truth taxonomy support v3
  -> unknown relationship, not proof of equality; never claimed compatible
```

The rule constrains profile fitting and application only. It does NOT
retroactively invalidate `CalibrationObservation`, the evaluation cohort, or
`CalibrationDataset`: those are provenance and data artifacts and may
legitimately record a mismatch, so their admission is unchanged.

Explicit cross-taxonomy fitting or application requires an identity-bearing
mapping contract; that contract is not implemented yet and no speculative
mapping framework is created here. Because no fitter or profile exists yet,
this rule is frozen in design only. A tested compatibility helper is deferred
until it has a real caller, so that no dead policy code is added.

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

### 13.4 Method selection remains open

Phase 4C.0 does not select, implement, or canonize a calibrator. Platt scaling,
temperature scaling, isotonic regression, and histogram or binning methods all
remain candidates, and none is preferred here. The current design notes already
argue that winner correctness is a scalar calibration problem, so temperature
scaling is not obviously the correct first method. Selecting the first fitter
belongs to a later implementation round with a concrete mathematical contract.

One specific claim must NOT be documented as fact: that an exact log-loss
objective combined with a `p = 1` sample forces a temperature-scaling parameter
toward zero. That is not a valid general contract, because different calibrator
parameterizations behave differently at the endpoints. The only rule frozen here
is the one in 13.5.

### 13.5 Fitting objective and endpoint policy are semantics-bearing

Frozen rule, before any fitter exists:

```text
A calibration method's fitting objective and its numerical endpoint policy are
semantics-bearing method configuration. They must be explicit and versioned.
```

Consequences that bind a future fitter:

```text
an evaluation metric is never silently reused as a training objective
epsilon, clipping, smoothing, finite-endpoint substitution, and regularization
are never introduced silently; if a fitter needs any of them, they belong to the
method identity/configuration and must be documented and tested
```

The reason is the same as for the evaluation metrics: a number produced under an
undocumented objective and endpoint policy cannot be reproduced, and a silent
substitution changes what the fitted artifact means without changing its version.

## 14. Metric semantics

Metrics evaluate declared probability-like scores against declared empirical
targets. They do not produce correctness information: the correctness labels
come from the declared ground-truth semantics, and the metric only compares a
score with them.

### 14.0 Pre-calibration evaluation and post-calibration evaluation

The same formula (for example Brier, section 14.1) can be run against two
different input scores, and the two runs mean different things. The runs are
distinguished by input-score identity:

- Pre-calibration evaluation evaluates the uncalibrated selected semantic
  probability (input-score identity `uncalibrated-selected-probability`)
  against the derived winner-correctness label. It is implemented in
  `src/probvenance/calibration_evaluation.py` as
  `evaluate_uncalibrated_winner_brier` over a `CalibrationEvaluationDataset`.
  It compares the raw score against the derived winner-correctness label on the
  evaluated rows and is a baseline.
- Post-calibration evaluation would evaluate `predicted_correctness` (the
  score a future calibration profile would produce) against the same derived
  winner-correctness label. It does not exist yet: no `CalibrationProfile`
  exists, and the runtime keeps `predicted_correctness = None` and
  `calibrated = False`.

The pre-calibration baseline must NOT be presented as calibration-quality
evidence: no calibrator was involved in producing it, so it cannot show how
much, or whether, calibration helps. Comparing a pre-calibration run with a
post-calibration run is exactly how the effect of a future calibrator would be
measured, and the comparison is only meaningful when both runs use the same
dataset identity (the `CalibrationEvaluationDataset` fingerprint) and differ
only in input-score identity.

`CalibrationEvaluationDataset` records a declared split role (`validation` or
`test`) and a declared split id. These are provenance, not proof: they record
which role the caller assigned to the split and do NOT prove statistical
independence from any data used for future fitting. Independence discipline is
the caller's responsibility (section 11).

### 14.0.1 Declared source cohort and metric-eligible projection

The evaluation pipeline is conceptually:

```text
declared evaluation source cohort
  (retain all observation statuses)
  -> CalibrationEvaluationCohort
  (deterministic metric-eligibility projection, exclusions explicitly
   accounted)
  -> CalibrationEvaluationDataset
  -> Brier / exact log loss
```

`CalibrationEvaluationCohort` is the FULL set of `CalibrationObservation` rows
supplied as one declared evaluation split, BEFORE metric eligibility is
applied. It may contain fit-eligible resolved rows, taxonomy-miss rows,
unresolved rows, and resolved-but-unadjudicated rows; those rows are retained
as real rows and explicitly accounted, never silently filtered. A cohort is
NOT a fitting dataset, NOT a metric result, and NOT proof that the caller
supplied every real deployment event: it records the source cohort supplied
to the evaluation harness, and it cannot prove that a caller did not discard
events before cohort construction. The experiment discipline is: construct
the cohort at ingestion, before metric-eligibility projection.

The projection is a deterministic, mutually exclusive partition of the cohort
rows with fixed precedence:

```text
if status == TAXONOMY_MISS:                                    -> taxonomy_miss
elif status == UNRESOLVED:                                     -> unresolved
elif status == RESOLVED and provenance.adjudicated is not True -> unadjudicated_resolved
else:                                                          -> eligible
```

The partition satisfies
`source_count == eligible_count + taxonomy_miss_count + unresolved_count +
unadjudicated_resolved_count`, with no overlapping counts and no
double-counting (an unresolved row is counted exactly once, as unresolved,
even when its provenance is also unadjudicated).

Three counts must never be conflated:

- `source_count`: the number of rows in the declared source cohort.
- `evaluated_count` (the metric artifact's `count`): the number of rows
  actually scored by the metric, that is, the eligible rows.
- the exclusion accounting (`taxonomy_miss_count`, `unresolved_count`,
  `unadjudicated_resolved_count`): coverage information about which rows were
  NOT scored and why.

The metric value is a statement about the admitted (eligible) population
only. A taxonomy miss is not an ordinary incorrect prediction (section 6):
it is a resolved ground truth outside the declared semantic outcome space,
and it is recorded here as exclusion accounting, not folded into the metric
and not reported as a taxonomy-miss-rate metric. The counts are provenance
and accounting in this design; they are not a new metric.

Because metric exclusion is provenance, the dataset fingerprint (version 2)
commits the source cohort fingerprint, the split metadata, the eligible
observation fingerprints, and the full exclusion accounting. Two cohorts
with identical scored rows but different taxonomy-miss, unresolved, or
unadjudicated exclusions therefore never collapse to the same evaluation
dataset fingerprint or the same metric artifact fingerprint. Worked
contrast: a source cohort with 1000 rows, 100 evaluated, and 900 taxonomy
misses, versus another source cohort with 100 rows, 100 evaluated, and 0
taxonomy misses, produce the same Brier value and the same evaluated count,
but they do NOT share an artifact fingerprint: the cohort fingerprints
differ, the dataset fingerprints differ, and the metric artifact
fingerprints differ.

Limitation, stated conservatively: the cohort records what was SUPPLIED to
the evaluation harness. It does not prove complete deployment coverage and
does not guarantee that no rows were omitted upstream of cohort
construction.

### 14.1 Brier score

For winner correctness, with `p_i` the evaluated probability-like score (the
uncalibrated selected semantic probability for a pre-calibration run, or
`predicted_correctness` for a post-calibration run; see section 14.0) and
`y_i` in `{0, 1}`:

```text
Brier = mean((p_i - y_i)^2)
```

Range `[0, 1]`, lower is better.

Propriety is a property of the interpretation, not of the formula alone. Brier is
a proper scoring rule when its input is read as a probability forecast of the
scored binary target, that is, when `p_i` is claimed to estimate
`P(Y_correct = 1)`; propriety then says a forecaster minimizes the expected score
by reporting its true belief. It says nothing about whether a particular input
already carries that meaning. Phase 4B deliberately does not grant that
interpretation to the uncalibrated selected semantic probability (section 14.0):
there the same formula is used as a pre-calibration diagnostic baseline against
winner correctness, its value does not establish that the raw score already means
predicted correctness, and it must not be reported as calibration-quality
evidence.

Caveat: Brier is affected by both calibration and discrimination (refinement). A
model with poor discrimination but a well-fitted calibration can achieve a
moderate Brier by predicting something close to the base rate for everything. So
Brier is not a pure calibration metric, and it must not be described as one.

### 14.2 Log loss

For binary correctness:

```text
log_loss = -mean( y * log(p) + (1 - y) * log(1 - p) )
```

Phase 4B implements this metric for the pre-calibration winner-correctness
baseline, using the natural logarithm with an exact boundary policy:

```text
p = 1, y = 1  -> 0
p = 0, y = 0  -> 0
p = 0, y = 1  -> +infinity
p = 1, y = 0  -> +infinity
```

There is no clipping, no epsilon, and no smoothing. A clipped implementation
would replace an exact mathematical result with an arbitrary finite value, which
would fold a numerical implementation policy into the metric semantics; this
project does not permit that silent change. A clipped variant would require an
explicit, identity-bearing boundary configuration, and none exists.

Positive infinity is a legitimate metric result, not an invalid input. Because
canonical JSON forbids non-finite numbers, the result fingerprint encodes the
value structurally (a discriminator plus an optional finite number) instead of
serializing a non-finite number, so an infinite result stays fingerprintable and
mathematical positive infinity is never confused with a missing value.

Properties: a proper scoring rule under the same forecast interpretation as
section 14.1, punishing confident errors far more sharply than Brier. As in
section 14.1, this pre-calibration run does not grant that interpretation to the
raw uncalibrated selected semantic probability, so the metric is a diagnostic
baseline and not calibration-quality evidence.

### 14.2.1 Companion winner-correctness diagnostics

Alongside Brier and exact log loss, the evaluation module produces one
companion diagnostics artifact for the pre-calibration winner-correctness
target: `WinnerCorrectnessDiagnosticsResult`, built by
`evaluate_uncalibrated_winner_diagnostics(dataset)` over the same
metric-eligible projection. It commits three derived quantities, each with its
own identity:

```text
empirical winner-correctness rate   correct_count / count
mean selected probability           mean of the uncalibrated selected
                                    semantic probabilities p_i
empirical constant Brier reference  mean((q - y_i)^2) with q the
                                    empirical correctness rate
```

The empirical winner-correctness rate is ONE quantity. On this binary
winner-correctness target, `mean(Y_correct)` is simultaneously the empirical
base rate of `Y_correct` and the ordinary decision accuracy on the evaluated
rows. The artifact therefore carries exactly one numeric field for it
(`empirical_correctness_rate`); no duplicate `accuracy` or `base_rate` metric
identity exists. It is a population-level rate over the evaluated rows, never a
per-example predicted probability for a new observation, and it is never
written into `predicted_correctness`.

The mean selected probability is aggregate raw-score behaviour: the average of
the uncalibrated selected semantic probabilities over the same evaluated rows.
It stays the uncalibrated selected semantic probability in aggregate form; it
is not confidence, not predicted correctness, and not a calibrated probability.

The empirical constant Brier reference is a hindsight, in-sample, descriptive
reference point. It answers exactly one question: what Brier value would a
constant score equal to this evaluated population's empirical correctness rate
obtain on these same evaluated rows? It is derived from the outcomes of the
same evaluation dataset it describes, so it is not an operational predictor
available before observing the evaluation outcomes, and it is not a
training-derived baseline. It must not be presented as a naive production
baseline, a best baseline, an expected deployment baseline, or a deployable
baseline of any kind. The endpoint behaviour is deliberate: an all-correct
population has `q = 1` and reference `0`, and an all-wrong population has
`q = 0` and reference `0`. That is exactly what a hindsight prevalence
reference does, and it is why the reference must not be turned into a skill
score, a relative improvement, a percentage better, or any winner-versus-loser
verdict. No constant base-rate log loss, entropy reference, or cross-entropy
reference is derived from it.

No calibration-gap, overconfidence, or underconfidence metric is derived from
the difference between the mean selected probability and the empirical
correctness rate. A single global mean difference cannot describe calibration:
two populations can share the same mean and the same rate while having
completely different conditional reliability structure. A structured
equal-width reliability summary (section 14.5) is now implemented for that
question, and the ECE estimator FORM is now implemented as a derived binned
absolute-gap diagnostic over that summary (section 14.5.1); the ordinary
calibration-error interpretation of that number remains ungranted.

Together the three diagnostics help interpret a Brier value on the evaluated
rows: the rate says how often the winner was correct, the mean selected
probability says how large the raw scores were on average, and the constant
reference says what a prevalence-matched constant score would have scored on
the same rows. None of the three is calibration evidence: the rate carries no
score information, the mean score carries no correctness structure, and the
reference is a hindsight in-sample quantity rather than a forecast property.

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

The equal-width ECE estimator FORM is now implemented, but only as a derived
binned absolute-gap diagnostic over the reliability summary (section 14.5.1).
The formula is the estimator form; the granted semantics are narrower than the
name "ECE" suggests, and the ordinary calibration-error interpretation is not
granted by this implementation.

### 14.5 Reliability curve

A structured, pre-calibration equal-width reliability summary IS implemented
(`src/probvenance/calibration_evaluation.py`): the artifact
`WinnerReliabilityResult`, built only by
`evaluate_uncalibrated_winner_reliability(dataset, bin_count=...)`, reports
for each raw selected-probability region the observation count, the
winner-correctness count, the mean uncalibrated selected probability, the
empirical correctness rate, and the sorted member observation fingerprints.
Plotting remains unimplemented: the artifact is structured data, not a chart,
and no matplotlib, plotly, PNG, or SVG output exists.

The binning policy is equal-width only (`equal-width`, semantic version 1).
For `B = bin_count` and `i = 0 .. B-1` the frozen interval contract is:

```text
bin 0:      [0/B, 1/B)
bin i:      [i/B, (i+1)/B)
last bin:   [(B-1)/B, 1]
```

The lower bound is inclusive and the upper bound is exclusive, EXCEPT the
final upper bound `1` is inclusive. So `p = 0` falls in the first bin, `p = 1`
falls in the last bin, and an exact interior boundary `p = i/B` belongs to bin
`i`, not to bin `i-1`. Boundary ownership is decided by comparing the stored
probability value against the mathematical rational boundaries `i/B` (exact
`fractions.Fraction.from_float` comparison), never by binary floating
multiplication; a rational boundary with no exact float representation (for
example `1/3`) is handled by that exact rational comparison, and the stored
float falls deterministically on one side of it.

Empty bins are retained, never omitted: `len(bins) == bin_count` always
holds, and an empty bin carries `count = 0`, `correct_count = 0`, both
statistics as `null`, and no members. `null` means "no observation fell in
this interval"; it is never NaN and never a fake `0.0`, because an empty
interval is not the same thing as an interval whose observed rate is zero.

Equal-mass (quantile) binning remains unimplemented future work: it needs its
own tie, duplicate-score, and deterministic-partition contract before it can
carry an identity. No `scheme` parameter that only half-works is exposed.

A reliability summary is NOT ECE. It reports per-region descriptions; it
computes no weighted gap aggregate, stores no per-bin gap or
calibration-error field, and derives no ECE number in any form. ECE needs its
own metric identity, version, and configuration contract.

#### 14.5.1 Binned absolute-gap aggregate (ECE-form, derived)

A derived aggregate over one exact `WinnerReliabilityResult` IS implemented
(`src/probvenance/calibration_evaluation.py`): the artifact
`WinnerBinnedAbsoluteGapResult`, built only by
`evaluate_winner_binned_absolute_gap(reliability)`, with aggregate id
`winner-correctness-equal-width-binned-absolute-gap` version 1 and result
fingerprint version 1. The pipeline order is strictly linear, never two
parallel derivations from the dataset:

```text
CalibrationEvaluationCohort
  -> CalibrationEvaluationDataset
    -> WinnerReliabilityResult
      -> WinnerBinnedAbsoluteGapResult
```

The aggregate never re-bins observations, never re-extracts probabilities,
never re-assigns bin indices, and never re-implements boundary logic: the
source reliability summary is the only binning truth source, and the
evaluator takes no `bin_count` argument because the source reliability
partition is accepted as truth. For each non-empty bin `b` of the source
summary, in fixed bin-index order:

```text
absolute_gap_b   = abs(mean_selected_probability_b
                       - empirical_correctness_rate_b)
weighted_term_b  = (count_b / count) * absolute_gap_b
value            = math.fsum(weighted_term_b for non-empty bins)
```

This is mathematically the conventional equal-width ECE estimator form,
frozen as the sample-weighted absolute discrepancy between raw selected-score
bin means and empirical winner-correctness rates. Empty bins contribute zero
mass: they are skipped, their `null` statistics are never treated as `0`, no
fake observed gap is constructed, and the presence of an empty bin is not an
error. The value is guaranteed finite and within `[0, 1]`; a non-finite
statistic, a negative weighted term, or an out-of-range aggregate means the
upstream reliability invariant is broken and fails closed with
`InvalidDecisionError`.

The artifact commits the source reliability fingerprint and payload schema
version, the reliability and binning identities with versions and the bin
configuration, the evaluation dataset and source cohort fingerprints with
their payload versions, `source_count`, the evaluated `count`, the three
exclusion counts, the target, the input-score identity, and the value. It
stores no per-bin gaps, no largest gap, no worst bin, no direction, and no
overconfidence or underconfidence counts. Two reliability artifacts that
happen to produce the same numeric value carry different reliability
fingerprints and therefore different aggregate fingerprints: numeric equality
is not provenance identity. Different bin configurations (`B = 5` versus
`B = 10`) produce different reliability and aggregate fingerprints, because
ECE-form estimators depend on bin configuration.

The interpretation boundary is the point of the naming. The same formula does
NOT equal the same semantic claim: the ordinary calibration-error
interpretation of an ECE number requires interpreting the score as a
probability forecast of `Y_correct`, and this round does not grant that. The
input is the raw selected semantic probability, not `predicted_correctness`,
not a confidence, and not a calibrated probability. The absolute value
carries NO direction, so no overconfidence or underconfidence inference is
possible from it. The reliability summary remains the primary bin evidence;
the aggregate discards substantial information (all per-bin structure) into
one number, and it inherits every caveat of section 14.3: different bin
configurations are non-comparable, sample-size sensitivity remains, and
equal-mass binning remains unimplemented. This is not a calibration-quality
claim about any model.

The interpretation stays pre-calibration. The x-axis score is the
UNCALIBRATED selected semantic probability, not `predicted_correctness`, not
`P(Y_correct = 1)`, not a confidence, and not a calibrated probability. A bin
showing mean raw score `0.8` with an observed rate of `0.6` is a description
of that raw-score region; it does not license naming that a "20-point
overconfidence", because the raw score does not carry
correctness-probability semantics.

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

Three gates. Structural validity has real runtime enforcement now; the other
two remain design statements.

### 16.1 Structural validity

```text
a valid Result
a valid Trace
a known formulation identity
ground truth compatible with the decision family
```

Implemented: `CalibrationObservation` construction and `CalibrationDataset`
pooling enforce structural validity at runtime, through fit eligibility
(status plus adjudication) and explicit dataset rejection (including the
ground-truth semantics check).

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
a fitting routine. No threshold exists in the runtime.

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

A calibration observation records the selection carried by the supplied
runtime-linked result. It never re-runs the tie-break, never consults a
different ordering, and never treats a tie as ambiguous for correctness
purposes. The recorded selection is honored as recorded; the linkage check
attests linkage consistency, not the execution origin of the payload.

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
population as `Y_correct = 0`. Here `ordinary_error` is conceptual vocabulary
for an in-taxonomy wrong selection, not the runtime status enum (which is
`resolved | taxonomy_miss | unresolved`): at runtime this observation carries
status `resolved`.

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

The Phase 4A data foundation (ground truth, binding, observation, dataset, and
their identities and fingerprints) and the Phase 4A evaluation foundation (the
evaluation dataset contract and the winner-correctness Brier metric) are
implemented and are no longer listed here. This document still does not
implement, and does not decide the eventual API for:

```text
CalibrationProfile, CalibrationProfileFingerprint
any calibration fitting algorithm or library
temperature scaling, Platt scaling, isotonic regression, binning
post-calibration metric runs (predicted_correctness as the input score)
log loss, ECE, or reliability-curve runtime code
profile matching runtime
a calibration store
abstention, review, escalate, or any policy
risk-coverage or selective accuracy
multi-label ground truth
open-set detection
```

Scoring-position and semantic validity (section 16) also remain design
statements without runtime enforcement.

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
