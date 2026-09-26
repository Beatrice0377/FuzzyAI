# Calibration Transportability under Probability-Measurement Shift

## Research Specification v1

Status: frozen research specification v1
Implementation status: not started

This document is a research specification. It defines the problem, the
measurement semantics, the experimental separation, the estimands, the metrics,
the evidence discipline, the research questions, the pilot gate, and the scope
boundaries for a follow-up research track. It is not a design for code that
exists, and it does not authorize any implementation.

The word **frozen** here means that R1 and R2 are expected to follow this
research definition by default; if the research design has to change, it must
change through an explicit revision and an explicit version of this document,
never through silent drift. Frozen here is **not** a core artifact schema
version, a public API version, a `CalibrationProfile` version, or a runtime
contract version. Those live in the frozen Phase 4C contracts and are unrelated
to this file.

---

## 0. Scope and non-scope

### 0.1 In scope for this specification

- The problem definition of cross-measurement calibration transportability.
- The statistical objects, estimands, and metrics that future experiments
  must report.
- The separation between pilot evidence and confirmatory evidence.
- The research questions and the research gate.
- The boundaries between this research track and the frozen Phase 4C runtime.

### 0.2 Out of scope for this specification

This document deliberately does not define, and no implementation may be
started for:

- any new scorer or probability formulation;
- a fixed-decision runtime API or any other research runtime;
- any research artifact class or experiment-run class;
- a transport audit, a transport graph implementation, or a compatibility
  registry or store;
- cross-identity selector behavior or automatic compatibility inference;
- statistical certification code, confidence bounds, or error control;
- policy, abstention, review routing, or any deployment lifecycle concept.

Section 21 records the boundary against the frozen Phase 4C contracts, and
Section 20 records the non-claims this research must not make.

---

## 1. Working thesis (North Star)

This specification freezes one working thesis:

> **A fitted calibration map is an empirical artifact tied to a declared
> probability measurement and fitting population, not an intrinsic property of a
> model. Reusing it across measurement protocols is a directional, population-
> and metric-dependent compatibility claim that requires independent evidence.**

In Chinese:

> 拟合得到的校准函数，是绑定于某个明确定义的概率测量及其拟合总体的经验工件，
> 而不是模型本身的固有属性。跨测量协议复用它，是一个有方向、依赖总体与评价
> 准则、且需要独立证据支持的统计兼容性命题。

Two commitments follow directly from the thesis:

1. **Declared measurement identity is the default fail-closed authorization
   boundary.** In the frozen Phase 4C runtime, a calibration map is selectable
   only within the exact declared measurement identity it was fitted under.
   Nothing in this research track relaxes that boundary in production.
2. **Empirical compatibility is a research result, not a runtime entitlement.**
   Showing that a map transports under some measured condition does not by
   itself authorize reusing it in the runtime. Turning evidence into production
   policy is a separate, later decision.

### 1.1 Identity is not compatibility

The research track freezes the slogan:

> **Identity != compatibility.**

Exact declared measurement identity is the conservative default boundary.
Whether two different measurement identities are empirically compatible is a
question that must be answered with independent evidence, and the answer is
allowed to be "yes, under these conditions", "no", or "unknown". The research
goal explicitly permits the eventual finding that an exact operational identity
is more conservative than empirical compatibility. That is a legitimate result,
not a failure.

The slogan names **declared measurement identity**, not "all provenance
identity". The frozen runtime selector's eligibility is not decided jointly by
every field of profile identity; it is decided by an exact binding match plus
the declared target and input-score identities. The research track must respect
that precise boundary rather than restate it loosely.

---

## 2. The statistical objects: frozen-decision vs end-to-end

This is the highest-priority semantic fork in the specification.

### 2.1 The frozen Phase 4C object that must NOT be reused

The frozen Phase 4C runtime measures this object:

```
the measurement chooses its own winner
        -> selected probability
        -> winner correctness
```

Concretely, the frozen target is `winner_correctness` and the frozen input score
is the `uncalibrated-selected-probability`. The measurement's own winner defines
the item's correctness label.

A frozen-decision transport experiment is **not** this statistical object. It
must not reuse `winner_correctness`, and it must not reuse the
`uncalibrated-selected-probability` input identity.

### 2.2 The frozen-decision research object

A frozen-decision experiment fixes an externally chosen anchor decision first:

```
an externally / pre-frozen anchor decision D_i
        -> measurement protocol A scores D_i
        -> measurement protocol B scores D_i
        -> the SAME anchor correctness Y_i
```

Therefore:

```
Y_i = 1[D_i is correct]
S_i^(M) = measurement M's score assigned to the SAME frozen anchor decision D_i
```

Crucially, even when measurement `M`'s own winner differs from `D_i`, the
experiment must take `M`'s score for `D_i`. It must never substitute `M`'s own
winner score.

### 2.3 The end-to-end object (secondary)

In an end-to-end condition, each protocol is left to choose and score its own
winner:

```
protocol A: chooses its own winner, scores it, gets its own winner-correctness label
protocol B: chooses its own winner, scores it, gets its own winner-correctness label
```

so the anchors and labels may differ:

```
D_i^A != D_i^B
Y_i^A != Y_i^B
```

This condition is useful, but it is not a pure measurement-shift estimand, and
it must never be silently merged with the frozen-decision condition.

### 2.4 Both conditions must always be named

No table, figure, or claim may present a cross-formulation comparison as
"cross-formulation calibration" without stating which condition produced it.
Every reported result must be explicitly labeled:

```
Frozen-decision
vs
End-to-end
```

---

## 3. Research-only target and input-score semantics

The frozen-decision experiment needs its own semantic identifiers. This
specification freezes the following research-layer names:

```
target:
  id:      fixed-decision-correctness
  version: 1

input score:
  id:      fixed-decision-semantic-probability
  version: 1
```

These identifiers must satisfy all of the following:

1. They must not reuse `winner_correctness`.
2. They must not reuse `uncalibrated-selected-probability`.
3. They must clearly express that the score refers to an externally frozen
   semantic decision, not necessarily to the measurement protocol's own winner.
4. They are, for now, research-specification semantic identifiers only.
5. This specification does not add any public constant, enum, core class, or
   fingerprint schema for them. They will be materialized only if and when a
   future, separately-authorized implementation phase needs them.

If the source tree already contains a more suitable, non-conflicting canonical
name, a future revision may adopt it, provided the four principles above still
hold. Until then, the identifiers above are the research contract.

---

## 4. Anchor decision requirements

The anchor decision `D_i` is the defining object of the frozen-decision
condition, so its construction requirements are strict.

### 4.1 Required properties

`D_i` must be:

- **pre-frozen**: selected and recorded before any scoring, not derived from the
  compared scores;
- **protocol-independent relative to the compared A/B measurements**: its
  selection mechanism must not observe or use the scores, winner decisions,
  confidence outputs, calibration outputs, or any other measurement outcomes
  produced by measurement A or measurement B. It must not depend on any
  measurement outcome that only becomes observable after A or B has run;
- **immutable during scoring**: scoring protocol A and protocol B must observe
  exactly the same `D_i`.

`protocol-independent` here has a precise, non-mystical meaning: the
anchor-generation mechanism must not depend on `A` scores, `B` scores, `A`
winner, `B` winner, `A` confidence, `B` confidence, `A`/`B` calibration output,
or any other outcome of the compared measurements. The anchor must be selected,
recorded, and frozen before those outcomes are observed.

#### 4.1.1 Anchor independence is about the selection mechanism, not the value

Anchor independence is a property of the anchor-**selection procedure**, not a
requirement that the realized anchor value differ from the winners produced by
the compared measurement protocols. It is **not** required that `D_i` differ
from A's winner or B's winner.

After `D_i` has been frozen, it may coincidentally equal:

- the winner selected by A;
- the winner selected by B;
- the winners selected by both A and B; or
- neither protocol's winner.

Any such coincidence is valid and MUST NOT cause anchor reassignment,
resampling, item exclusion, eligibility changes, weighting changes, or
split-membership changes. What is forbidden is a measurement-**dependent**
selection, not realized equality.

The forbidden dependency is:

```
A/B measurement outcome
    -> anchor choice or sample inclusion
```

The allowed ordering is:

```
protocol-independent anchor mechanism
    -> frozen D_i
    -> A/B measurement
    -> possible post-hoc agreement or disagreement
```

#### 4.1.2 Forbidden anchor mechanisms

The following are explicitly prohibited:

```
D_i := winner_A
D_i := winner_B
```

- running A and then choosing its winner as `D_i`;
- running B and then choosing its winner as `D_i`;
- inspecting A/B scores and then choosing whichever candidate is convenient;
- resampling `D_i` until `D_i != winner_A`;
- resampling `D_i` until `D_i != winner_B`;
- excluding items where `D_i == winner_A`;
- excluding items where `D_i == winner_B`;
- keeping only items where `D_i` differs from both winners.

Agreement conditioning remains prohibited as well (section 5).

#### 4.1.3 Canonical examples

| Case | Sequence | Verdict | Reason |
| --- | --- | --- | --- |
| 1 | anchor frozen first: `shipping`; A winner later `shipping`; B winner later `returns` | VALID | anchor generation did not depend on A/B outcomes |
| 2 | anchor frozen first: `shipping`; A winner later `shipping`; B winner later `shipping` | VALID | agreement is post-hoc coincidence |
| 3 | run A first; A winner `shipping`; then set anchor := `shipping` | INVALID | anchor selection depends on A |
| 4 | anchor frozen first: `shipping`; A winner later `shipping`; then drop the item because anchor == A winner | INVALID | sample inclusion depends on A outcome |
| 5 | anchor frozen first: `shipping`; A winner `returns`; B winner `billing` | VALID | A and B must still score `shipping` if their measurement semantics support a valid score for that anchor |

### 4.2 What must be shared between A and B

For a single paired item, the following must be identical between the two
measurements:

```
same item
same anchor semantic candidate / value
same ground truth
same Y
```

and the two scores are:

```
S_A = A's score for anchor D
S_B = B's score for anchor D
```

They must never be rewritten as "A's winner score" and "B's winner score".

### 4.3 The role of ground truth

Ground truth is used only to compute `Y_i = 1[D_i is correct]`. Ground truth
must not be used to select an anchor that is guaranteed correct, because that
would force `Y_i == 1` and make the experiment uninformative.

### 4.4 Anchor provenance to be recorded

Future R1 must at minimum record, per item:

- the anchor source / protocol identity;
- the anchor semantic value;
- the item identity;
- the ground-truth identity and semantics.

This specification does not design a production or public anchor artifact
schema, and it does not assign a repository fingerprint version to any anchor
artifact. R0 freezes the research requirement only.

---

## 5. Agreement conditioning is prohibited

The frozen-decision condition must never be implemented as:

```
retain only items where CAT winner == OVR winner
```

or:

```
retain only items where anchor == CAT winner == OVR winner
```

Doing so introduces agreement conditioning and selection bias. The entire point
of the frozen-decision condition is that even when A and B would choose
different winners, both are still asked to score the same frozen anchor `D_i`.

A post-hoc equality between the frozen anchor and a protocol's winner
(`D_i == winner_CAT`, `D_i == winner_OVR`, or `D_i == winner_CAT ==
winner_OVR`) is not agreement conditioning and is not by itself an exclusion
reason. Excluding or resampling items because of such equality is forbidden
(section 4.1.2). The concern is measurement dependence, not realized equality.

If a measurement protocol cannot produce a legitimately defined score for an
anchor candidate, a future harness must record an explicit missing or
ineligible reason. It must not quietly substitute the protocol's own winner
score, and it must not drop the item merely because the two protocols disagree.
The concrete missingness policy is deferred to R1/R2 design, but the principle
is frozen here. For the primary estimand these rules operate within the
explicitly defined eligible paired population (section 6).

---

## 6. Paired fitting and paired evaluation discipline

For the primary Frozen-decision estimand, the calibrators `g_A` and `g_B` are
fitted separately, but they MUST be fitted on scores from the same paired
training item IDs, within the explicitly defined eligible paired population:

```
training item i
    |- fixed anchor D_i
    |- same Y_i
    |- S_i^A -> fit g_A
    |- S_i^B -> fit g_B
```

so `g_A` is fitted on `{(S_i^A, Y_i)}` and `g_B` on `{(S_i^B, Y_i)}` for the
same item IDs. They are separately fitted calibrators over different
measurement scores, not calibrators trained on independently sampled item sets.
They must not be fitted on unrelated random samples:

```
g_A trained on random sample X
g_B trained on unrelated random sample Z
```

The purpose is to isolate measurement shift as far as possible, instead of
mixing calibration training-sample variation into the measured transport gap.

For the same reason, evaluation MUST use the same paired held-out target item
IDs:

```
same held-out item IDs
same anchor D_i
same Y_i
paired S_i^A / S_i^B
```

The paired same-ID requirement (for both fitting and evaluation) is scoped to
the explicitly defined eligible paired Frozen-decision population. The
eligibility and missingness rules for that population must be fixed before
confirmatory analysis and must not depend on post-hoc agreement between A and B
or on the observed transport outcome.

Split membership must be disjoint at the observation / item level:

```
train ∩ audit = ∅
train ∩ test = ∅
audit ∩ test = ∅
```

A future R1 must verify this mechanically, not infer it from split names.

The same item IDs are shared across A and B within each split, while the splits
themselves are disjoint. The correct structure is:

```
TRAIN: item IDs T   { A scores, B scores }
AUDIT: item IDs U   { A scores, B scores }
TEST:  item IDs V   { A scores, B scores }

T ∩ U = ∅    T ∩ V = ∅    U ∩ V = ∅
```

Pairing across measurements and disjointness across splits are not in tension:
pairing is about which items receive both A and B scores, disjointness is about
which items belong to train, audit, and test.

---

## 7. Transport estimands: oracle object vs empirical estimator

The specification must keep the theoretical object and the empirical estimate
separate.

### 7.1 Oracle conceptual object

For the target measurement `B`, define the true conditional correctness
function:

```
q_B(s) = P(Y = 1 | S_B = s)
```

The theoretical transport regret of a source calibrator `g_A` is:

```
R_B(g_A) - R_B(q_B)
```

where `R_B` is the risk under the target measurement `B` and the declared loss,
and `g_A` is the calibrator frozen after fitting on source measurement `A`.

However, `q_B` is generally unobservable. An experiment must never present a
fitted `g_B` as if it were the oracle `q_B`.

### 7.2 Empirical transport excess risk

The real experiment compares two finite-data estimators on untouched held-out
target items:

```
Delta_hat[A -> B] = R_hat_B(g_A) - R_hat_B(g_B)
```

where:

```
g_A = fitted on measurement A scores, using the paired training items
g_B = fitted separately on measurement B scores, using the SAME paired training item IDs
evaluation = the same untouched held-out B target scores and items
```

It must be stated explicitly that `g_B` is a finite-data estimator and is not
the oracle `q_B`. For this reason the quantity above is named **empirical
transport excess risk** and must not be called "oracle regret".

Within the explicitly defined eligible paired population (section 6), `g_A` and
`g_B` MUST be fitted on scores from the same paired training item IDs, and they
MUST be evaluated on the same paired held-out target item IDs. They are
separately fitted calibrators over different measurement scores, not
calibrators trained or tested on independently sampled item sets.

---

## 8. Primary and secondary metrics

### 8.1 Primary inferential metric: Brier excess risk

The primary inferential transport metric is the Brier excess risk.

The reason is explicit and must be stated in every use. For a single item with
probability `p` in `[0, 1]` and label `y` in `{0, 1}`, the Brier loss is:

```
(p - y)^2
```

which is bounded:

```
0 <= loss <= 1
```

This boundedness makes it substantially easier to establish rigorous inference
later, should the track proceed to a finite-sample transport audit with
confidence bounds and error control.

### 8.2 Secondary metrics

At minimum, future experiments must also report:

- exact LogLoss;
- reliability diagnostics;
- empirical score-support diagnostics (Section 9);
- task / decision accuracy.

Exact LogLoss keeps the existing Probvenance discipline: **no clipping and no
smoothing**. It must therefore be stated that exact LogLoss is an important
secondary metric but that a future analysis may not directly apply a
finite-sample concentration argument that depends on bounded loss, without
additional tail or parameter assumptions. This specification does not invent a
LogLoss certification theorem.

---

## 9. Empirical score-support diagnostics

Score-support overlap is promoted to a first-class diagnostic. The
specification must not pretend that a finite sample knows a true distribution's
support. It uses only empirical language:

```
empirical score-support diagnostics
observed score-range overlap
quantile coverage
```

and must not unconditionally assert that a support is disjoint.

At minimum, future experiments plan to record:

- the source score distribution;
- the target score distribution;
- the fraction of target scores that fall outside the source observed score
  range;
- robust source quantile-range coverage;
- transport loss inside the empirically well-supported region versus outside /
  poorly-supported region.

The purpose is to distinguish at least two mechanisms that can both contribute
to a real transport failure:

```
the conditional calibration relation changed
```

versus:

```
target scores moved into regions poorly represented during source fitting
```

Items must not be filtered to remove out-of-range observations merely to
produce a prettier result, unless a future confirmatory protocol explicitly
pre-registers that rule. Support diagnostics are for interpretation by default,
not for post-hoc cherry-picking.

---

## 10. Directional, conditioned compatibility

The specification must not define an unconditional global statement such as
"CAT and OVR are compatible". Compatibility is instead expressed as a
conditioned, directional relation:

```
A --epsilon--> B
```

whose meaning is conditioned on at least:

```
source measurement identity A
target measurement identity B
model / source
population P
target semantics
loss / metric L
evidence protocol E
tolerance epsilon
```

Conceptually:

```
A -> B | (model, population, Y semantics, L, E, epsilon)
```

The relation is explicitly:

- **directional**: `A -> B` does not imply `B -> A`;
- **not transitive by default**: `A -> B` and `B -> C` do not imply `A -> C`;
- **evidence-dependent**: the tolerance has meaning only relative to a stated
  evidence protocol.

---

## 11. Conditioned transport graph

A future transport graph has:

```
nodes = declared measurement identities
directed edges = empirically supported transport claims
```

But it must never be described as a universal global graph. It is a conditioned
slice:

```
G | (model/source, population, target semantics, metric, evidence protocol, tolerance)
```

The same pair, for example CAT and OVR, may produce different edges under
different models, populations, tasks, targets, or metrics. A single `CAT -> OVR`
edge must never be read as "CAT is globally calibration-compatible with OVR".

---

## 12. Research questions

Four primary research questions are frozen.

### RQ1 - Measurement effect

> Under controlled semantic decision, population, and correctness target, how
> large is the effect of a change in probability-measurement protocol on
> calibration transport?

Primarily answered by the frozen-decision paired experiment.

### RQ2 - Transport structure

> Does calibration compatibility have stable direction and structure, and how
> does it relate to score distribution, empirical support overlap, raw
> calibration, decision accuracy, and measurement metadata?

A specific concern: whether the same or similar task accuracy is sufficient to
support calibration compatibility.

### RQ3 - Evidence requirement

> Under zero target labels, a small independent audit set, and ample labeled
> evidence, what can be concluded about calibration transport in each regime?

This may eventually motivate an evidence-aware calibration transport audit, but
this specification implements nothing.

### RQ4 - Authorization versus empirical compatibility

> How can declared measurement identity remain the default fail-closed
> authorization boundary while independent empirical evidence expresses
> compatibility between different measurement identities?

This is the point where the Probvenance architecture and the research program
genuinely connect.

---

## 13. Working thesis vs confirmatory hypotheses

The working thesis in Section 1 is not the final confirmatory hypothesis. The
specification must keep the two apart:

```
working research thesis != final confirmatory hypothesis
```

Current working expectations may be recorded, for example:

> Cross-measurement transport may exhibit systematic variation across ordered
> measurement pairs, after controlling model, population, and target.

One important candidate research claim is:

> **Similar task accuracy is insufficient evidence for calibration
> compatibility.**

This remains a candidate until experiments support it. It must not be asserted
as already proven.

### 13.1 H3 support-mismatch question

The following proposition is currently **exploratory**, not confirmatory:

> Transport failure cannot be fully explained by empirical score-support
> mismatch.

Only if pilot data supports continued study does the track decide whether to
freeze it formally in a confirmatory benchmark.

---

## 14. R2 pilot (planned, not implemented)

This specification describes the future R2 pilot; it implements nothing.

The minimal pilot is CAT against OVR. OVR itself is not a novelty claim; its
role is that of a controlled intervention on probability-measurement semantics.

- Primary condition: frozen-decision.
- Secondary condition: end-to-end.

The main transport matrix is:

```
                 evaluated on
               CAT         OVR

fit CAT     CAT->CAT     CAT->OVR
fit OVR     OVR->CAT     OVR->OVR
```

For the frozen-decision condition, all four cells use:

```
same paired train items
same paired test items
same anchor D
same Y
```

---

## 15. Pilot and confirmatory evidence separation

This is a hard discipline of the research track.

```
Pilot data        = hypothesis-generating / design-tuning evidence
Confirmatory data = hypothesis-testing evidence
```

Pilot data may be used to:

- estimate the effect scale;
- expose measurement-definition problems;
- discover missingness;
- study support diagnostics;
- choose a later reasonable epsilon;
- adjust the analysis plan;
- decide whether R3/R4 is worth pursuing.

But once pilot data has been seen and the hypothesis, tolerance, or primary
analysis has been changed in response, that same pilot data can no longer serve
as the final confirmatory evidence.

Before a confirmatory benchmark, the following must be frozen:

```
hypotheses
metrics
tolerance (if applicable)
exclusion rules
primary analysis
split protocol
```

A confirmatory benchmark must contain untouched evaluation conditions and data.
The pattern "run a pilot, tune the rules, then claim the same result validates
the final hypothesis" is prohibited.

---

## 16. Research gate

The track passes through one explicit gate:

```
R2 pilot
    |
========================
RESEARCH GATE
========================
```

Before the gate:

```
DO NOT build the R3 benchmark framework
DO NOT build the R4 audit framework
DO NOT modify the selector for cross-identity reuse
DO NOT build Policy
DO NOT build lifecycle
DO NOT write a theorem just to target a venue
```

The gate's purpose is not to prove the final paper statistically. Its purpose
is to judge:

- whether the transport phenomenon exists;
- the effect scale;
- whether there is a repeatable structure;
- whether the dominant failure is merely support or extrapolation;
- whether investing in R3/R4 is warranted.

Decision rules at the gate:

- if cross transport shows stable, material structure, proceed to R3;
- if transfer is surprisingly strong, R3 may still be warranted, to study
  whether compatibility is broader than exact identity;
- if results are noisy, trivial, or explained entirely by an implementation
  artifact, do not force the CAT-OVR story; move to a second measurement axis.

---

## 17. Future R4 terminology: audit, not certification

Any future method is provisionally named:

```
Evidence-aware Calibration Transport Audit
```

Its conceptual interface may be described as:

```
source fitted calibrator
+ target measurement
+ independent audit evidence
        |
ACCEPT / REJECT / INDETERMINATE
```

Until the track actually has:

```
explicit statistical assumptions
a finite-sample guarantee
a coverage statement
error control
formal tolerance semantics
```

no title or core claim may use:

```
certification
certified
guaranteed compatible
```

If a rigorous guarantee is later established, the terminology may be upgraded
then. This specification implements no audit, no theorem, and no upper
confidence bound.

---

## 18. Non-claims and novelty guardrails

The research must not claim:

- that confidence being protocol-dependent was first discovered here;
- that prompt / formulation calibration was first compared here;
- that OVR confidence is a novel method of ours;
- that calibration never transfers;
- that "token / lexical probability is not semantic uncertainty" was first
  observed here;
- that content-addressed provenance is itself novel;
- that `predicted_correctness` is the model's true belief;
- that exact identity is statistically the only correct compatibility
  boundary.

The program's goal in fact permits the eventual finding that an exact
operational identity is more conservative than empirical compatibility.

---

## 19. Hard isolation from the frozen Phase 4C contracts

This boundary is explicit and non-negotiable.

The frozen-decision research semantics (`fixed-decision-correctness` and
`fixed-decision-semantic-probability`) exist only in the research layer. Using
them does not change Phase 4C, which continues to freeze `winner_correctness`
and the `uncalibrated-selected-probability` input identity.

Therefore the research track is forbidden from:

```
putting an anchor score into CalibrationObservation.selected_probability
passing a fixed-decision target off as winner_correctness
changing CalibrationProfile's current target / input eligibility
changing the selector to allow cross-input reuse
adding a research bypass to runtime application
making catalog discovery responsible for compatibility
writing research transport into a production store
```

If a future research layer needs to execute a deliberately unauthorized
transport, it must do so explicitly in the experiment / research layer, not by
loosening a production semantic contract. The core research design is:

```
system layer:   fail closed
research layer: deliberately violate / recombine identities,
                under controlled experimental conditions,
                to measure the consequences
```

The existing selection and application boundaries also stay frozen:

```
Selection   = semantic authorization
Application = runtime execution capability
```

Selection eligibility remains exactly the current contract: an exact
`CalibrationBinding`, the `winner_correctness` target identity and version, and
the `uncalibrated-selected-probability` input identity and version. Method
execution support, quality, recency, training dataset, ground-truth semantics,
and transport evidence must not be smuggled into the current selector. Catalog
discovery remains candidate discovery, never authorization.

---

## 20. Versioning and revision discipline

The identifiers `research-spec-v1`, `fixed-decision-correctness` version 1, and
`fixed-decision-semantic-probability` version 1 are research-layer identifiers.
They are not a bump of any existing production artifact schema.

This specification does not authorize any change to the frozen production
version matrix, including but not limited to:

```
CalibrationObservation fingerprint
CalibrationBinding fingerprint
GroundTruthSemantics fingerprint
CalibrationDataset fingerprint
CalibrationProfile fingerprint, serialization, and store layout
method / solver / objective / input-transform / endpoint-policy / convergence
catalog fingerprint, serialization, and store layout
formulation / family / plan / execution fingerprints
```

Any future research revision that appears to require such a change must stop
and be raised as a blocker with an explicit payload, schema, or layout analysis,
rather than silently bumping a version.

---

## 21. Relationship to the repository today

As of this specification, the frozen Phase 4C calibration layer exists and is
frozen. Nothing in this document is implemented. Specifically, the following do
not exist and must not be started on the basis of this document:

```
OVR scorer
research runtime / fixed-decision runtime API
research artifact classes / experiment-run classes
transport audit / transport graph
compatibility registry / store
cross-identity selector behavior
automatic compatibility inference
Policy / abstention / review routing
latest / best / default / active / production / staging / promotion / rollback
automatic discovery / selection / calibration
store enumeration / catalog synchronization
multi-writer locking / database / signing / new cryptography / new dependencies
any generic framework
```

The intended progression is:

```
Phase 4C final freeze
    |
this research specification v1
    |
R1 paired experiment-integrity harness      (not implemented)
    |
R2 tiny frozen-decision CAT-OVR pilot       (not implemented)
    |
========================
RESEARCH GATE
========================
    |
only if evidence warrants:
R3 transport benchmark                      (not implemented)
R4 evidence-aware transport audit           (not implemented)
```

The success criterion for this specification is not more code. It is that the
research question becomes precise enough that R1 and R2 can test it without
changing the question mid-experiment.
