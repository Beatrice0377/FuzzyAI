# Choice signal experiment: direct categorical scoring

Status: Phase 2B experimental result. Early development. Not a benchmark.

## 0. What was run

| | |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` (text tower through the experiment-side harness adapter) |
| Revision | none pinned (`revision = null` in the records) |
| Dtype | `bfloat16` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB |
| Backend | `qwen35_loader.Qwen35TextBackend` (subclass of `TransformersBackend`) |
| Label scheme | `categorical-labels-v1` |
| Doctrine | `categorical-semantic-judgment-v1` |
| Case set | `choice-signal-v1` |
| Case-set fingerprint | `9ceb36cf8974680224d767206d107fb6bfc44225e25498978ca7dca10938e2a2` |
| Evaluations | 193, each exactly one model forward pass |
| Wall time | 25.5 s |
| Latency | mean 131.1 ms, median 112.0 ms, min 99.2 ms, max 3243.8 ms (first call, model warm-up) |
| Records | `results/choice-signal-qwen35-2b.jsonl` (one header line, then 193 evaluation records) |

The adapter is a harness-side loader for a multimodal checkpoint. It is not
official FuzzyAI runtime support and it is not wired into any production backend.

`Qwen/Qwen3.5-2B` was chosen because Phase 2A.2 found it the most semantically
sensitive of the three models that were tested; it is not a quality claim.
`MiniCPM5-2B` was the documented fallback and was not needed.

No pass threshold was set anywhere in this experiment.

## 1. Resolved scoring representation

The backend resolved every scoring label against the real rendered continuation
before the forward pass. All six permutations of `A`/`B`/`C` resolved to the same
single tokens, so no evaluation was rejected.

```text
A -> token 32
B -> token 33
C -> token 34
```

Each label is exactly one token, and the three ids are pairwise distinct. The
compiler never saw these ids; it produced the same label strings for every
permutation and the backend did the resolving.

## 2. Ordering and fingerprint behaviour (measured, not just asserted)

For case `3w-billing-01` across the six label permutations:

```text
decision fingerprint unique values: 1
plan fingerprint unique values:     6  (of 6)
execution fingerprint unique values: 6  (of 6)
```

This is the runtime confirmation of the intended separation: the same
`ChoiceDecision` keeps one decision identity while the compiled representation
and the executed representation each move. The implementation's unit tests
assert this; here it also holds against a real model.

## 3. Three-way experiment (N=3), all 6 label permutations

15 cases, 6 permutations each, 90 evaluations.

```text
semantic argmax preservation: 90 / 90  (1.0000)
mean total variation:         0.0334
max total variation:          0.3082
max absolute drift:           0.3082
```

Outcome agreement with the fixture's obvious category, identity mapping only:
`15 / 15`.

Per-case identity results:

| case | expected | winner | billing | shipping | technical | candidate_mass |
| --- | --- | --- | --- | --- | --- | --- |
| 3w-billing-01 | billing | billing | 0.998 | 0.002 | 0.000 | 0.9921 |
| 3w-billing-02 | billing | billing | 0.999 | 0.001 | 0.000 | 0.9935 |
| 3w-billing-03 | billing | billing | 0.997 | 0.002 | 0.001 | 0.9911 |
| 3w-billing-04 | billing | billing | 0.996 | 0.004 | 0.000 | 0.9884 |
| 3w-billing-05 | billing | billing | 0.998 | 0.001 | 0.001 | 0.9957 |
| 3w-shipping-01 | shipping | shipping | 0.003 | 0.996 | 0.001 | 0.9955 |
| 3w-shipping-02 | shipping | shipping | 0.002 | 0.998 | 0.001 | 0.9970 |
| 3w-shipping-03 | shipping | shipping | 0.003 | 0.996 | 0.001 | 0.9957 |
| 3w-shipping-04 | shipping | shipping | 0.001 | 0.999 | 0.000 | 0.9964 |
| 3w-shipping-05 | shipping | shipping | 0.045 | 0.622 | 0.333 | 0.9896 |
| 3w-technical-01 | technical | technical | 0.002 | 0.000 | 0.998 | 0.9972 |
| 3w-technical-02 | technical | technical | 0.018 | 0.002 | 0.980 | 0.9964 |
| 3w-technical-03 | technical | technical | 0.059 | 0.019 | 0.922 | 0.9933 |
| 3w-technical-04 | technical | technical | 0.196 | 0.119 | 0.685 | 0.9859 |
| 3w-technical-05 | technical | technical | 0.003 | 0.001 | 0.997 | 0.9962 |

The winner survived every permutation, but the winner was not always obvious in
the identity mapping: `3w-shipping-05` ("My order arrived in a damaged box with a
broken item inside") put 0.333 on `technical` against 0.622 on `shipping`, and
`3w-technical-04` put 0.196 on `billing` against 0.685 on `technical`.

### Negative finding: the distribution shape is representation-sensitive

Argmax preservation is not stability. The worst 3-way drifts:

| case | permutation change | TV | before | after |
| --- | --- | --- | --- | --- |
| 3w-shipping-05 | ABC to BCA | 0.3082 | shipping 0.622, technical 0.333 | shipping 0.930, technical 0.060 |
| 3w-shipping-05 | ABC to ACB | 0.3005 | shipping 0.622, technical 0.333 | shipping 0.923, technical 0.046 |
| 3w-technical-04 | ABC to ACB | 0.2793 | technical 0.685, billing 0.196 | technical 0.406, shipping 0.316, billing 0.279 |

Changing which letter a candidate is bound to moved probability mass by up to
0.31. The label a candidate is bound to is therefore semantically inert in the
winner but not in the distribution. Any consumer that reads a Choice probability
as a magnitude, rather than only as a ranking, is reading a representation
dependent quantity.

## 4. Five-way experiment (N=5)

15 cases, 5 deterministic permutations each (identity, reverse, cyclic shift,
first-two swap, rotate-by-two), 75 evaluations.

```text
semantic argmax preservation: 75 / 75  (1.0000)
mean total variation:         0.0277
max total variation:          0.2937
max absolute drift:           0.2706
```

Outcome agreement with the fixture's obvious category, identity mapping only:
`15 / 15`. Every case's top token was the label at its expected candidate index
(`A` for billing, `B` for shipping, `C` for returns, `D` for technical, `E` for
account), which was not arranged, only observed.

Worst drifts:

| case | permutation change | TV | before | after |
| --- | --- | --- | --- | --- |
| 5w-account-03 | ABCDE to BCDEA | 0.2937 | account 0.618, technical 0.331 | account 0.889, technical 0.064 |
| 5w-account-02 | ABCDE to BACDE | 0.1545 | account 0.703 | account 0.618 |
| 5w-account-03 | ABCDE to EDCBA | 0.1544 | account 0.618, technical 0.331 | account 0.773, technical 0.221 |

The same pattern repeats at N=5, and again it concentrates on cases where the
identity mapping was not already decisive.

## 5. Irrelevant candidate addition

Six 3-way cases, re-run with a fourth candidate `account deletion` added.

```text
winner preserved:        6 / 6
TV range:                0.0006 to 0.0153
candidate_mass 3-way:    mean 0.9929 over these cases
candidate_mass 4-way:    mean 0.9927 over these cases
```

Adding a candidate that is not semantically present in any of the six contexts
changed the distribution by at most 0.015 and moved no winner. This was not
arranged, and it is a much smaller effect than the label permutation above.

## 6. Description paraphrase

Five 3-way cases, re-run with the `billing` description changed from
`Payment, charges, and invoices` to `Problems involving invoices, charges, or
payments`. The candidate name was unchanged.

```text
winner preserved: 5 / 5
TV range:         0.0002 to 0.0115
```

The largest effect was on `3w-technical-02` (TV 0.0115), a technical case, so the
change did not move the distribution more for the candidate whose description
actually changed.

## 7. Taxonomy overlap sensitivity

Candidate set: `billing`, `payment issue`, `shipping`, `technical`.

| case | note | winner | distribution | candidate_mass |
| --- | --- | --- | --- | --- |
| ov-payment-01 | card declined, money still left the account | payment issue | payment issue 0.689, billing 0.254, technical 0.030, shipping 0.027 | 0.9853 |
| ov-payment-02 | switching payment method | payment issue | payment issue 0.810, billing 0.160, technical 0.019, shipping 0.012 | 0.9896 |
| ov-payment-03 | control, invoice request | billing | billing 0.998 | 0.9961 |

This is a failure characterisation, not a model failure. Between two candidates
that overlap, the model split its mass across both (payment issue and billing
together took 0.943 and 0.970), while the control case with an unambiguous
category stayed sharp at 0.998. A restricted Choice probability is relative to a
taxonomy and its granularity; it is not an intrinsic per-candidate probability.

## 8. Out-of-set behaviour

Candidate set: `billing`, `shipping`, `technical`. All three contexts belong to
none of them.

| case | true topic | winner | distribution | candidate_mass | top token probability |
| --- | --- | --- | --- | --- | --- |
| oos-01 | account deletion | technical | technical 0.893, billing 0.083, shipping 0.024 | 0.9893 | 0.8836 |
| oos-02 | job application | technical | technical 0.711, billing 0.180, shipping 0.109 | 0.9853 | 0.7007 |
| oos-03 | sponsorship enquiry | technical | technical 0.677, billing 0.220, shipping 0.104 | 0.9871 | 0.6679 |

This is the most important negative result of the round. With a candidate set
that does not contain the true topic, the model still returned an in-set answer
with high confidence, and `candidate_mass` stayed at approximately 0.99. Nothing
in the restricted distribution or in the candidate-space mass signalled that the
answer was semantically out of scope.

Direct categorical Choice therefore has no open-set guarantee. `candidate_mass`
measures how much probability the candidate tokens hold, not whether the
candidate set is right. Out-of-set detection must come from the caller or from a
later policy layer, and this experiment is not open-set detection.

## 9. Candidate mass across N

| candidate set | records | mean | min | max |
| --- | --- | --- | --- | --- |
| three_way (N=3) | 104 | 0.9896 | 0.9371 | 0.9972 |
| three_way_plus_irrelevant (N=4) | 6 | 0.9939 | 0.9905 | 0.9961 |
| three_way_billing_paraphrase (N=3) | 5 | 0.9953 | 0.9939 | 0.9965 |
| five_way (N=5) | 75 | 0.9936 | 0.9840 | 0.9972 |
| taxonomy_overlap (N=4) | 3 | 0.9903 | 0.9853 | 0.9961 |

These values are not directly comparable across N. The mass denominator is the
full vocabulary, the candidate tokens differ per set, and both the mass and the
restricted distribution move with the label binding (section 3). What the table
shows is only that for this model on these synthetic contexts the candidate
tokens absorbed almost all of the next-token probability. Normalised entropy over
the restricted distribution ranged from 0.0048 to 0.9886 with a mean of 0.1334,
so the restricted distributions were mostly sharp.

## 10. Determinism notes

Every evaluation used exactly one forward pass and never called `model.generate()`.
The run is deterministic in the sense that the same plan and the same rendered
input produce the same labels, values, and fingerprints. Latency is not
deterministic and the first call included model warm-up (3243.8 ms against a
112.0 ms median).

## 11. What this does and does not establish

Established, for this model, this doctrine, this label scheme, and these
synthetic fixtures:

- the candidate and scoring-label identities are separable end to end, and the
  decision, plan, and execution fingerprints record different things;
- the semantic winner was stable under every label permutation at N=3 and N=5;
- adding an absent candidate and paraphrasing a description moved the winner in
  no case, and moved the distribution very little.

Not established:

- that a Choice probability is an accurate or calibrated quantity (it is not
  calibrated, and `predicted_correctness` remains `None`);
- that a Choice probability magnitude is stable under representation change (it
  is not, up to 0.31 TV here);
- that the candidate set is correct or complete, or that out-of-set inputs can
  be detected (they cannot, section 8);
- that any of this generalises to other models, other doctrines, other label
  schemes, real data, larger N, or cloud providers;
- that the winner is correct. The fixture expectations are the obvious reading
  of hand-written contexts, not ground truth from a dataset.
