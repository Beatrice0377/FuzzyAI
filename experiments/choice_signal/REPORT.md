# Choice signal experiment: direct categorical scoring

Status: Phase 2B.1 experimental result. Early development. Not a benchmark.

Every table below is generated from the raw JSONL records in `results/` through
`metrics.py` and `cross_model.py`, not hand-merged.

## 0. What was run

Two model families, same frozen case set, same doctrine, same label scheme,
same permutation sets, same metrics.

| | Qwen3.5-2B | MiniCPM5-2B |
| --- | --- | --- |
| Model | `Qwen/Qwen3.5-2B` (text tower, experiment-side adapter) | `openbmb/MiniCPM5-2B` |
| Revision | none pinned (`revision = null`) | none pinned (`revision = null`) |
| Dtype | `bfloat16` | `bfloat16` |
| Rendering config | `{"enable_thinking": false}` | `{"enable_thinking": false}` |
| GPU | RTX 5060 Laptop, 8151 MiB | RTX 5060 Laptop, 8151 MiB |
| Peak VRAM | 4829 MiB | 6047 MiB |
| Backend | `qwen35_loader.Qwen35TextBackend` | `fuzzyai.backends.transformers.TransformersBackend` |
| Label scheme | `categorical-labels-v1` | `categorical-labels-v1` |
| Doctrine | `categorical-semantic-judgment-v1` | `categorical-semantic-judgment-v1` |
| Case set | `choice-signal-v1` | `choice-signal-v1` |
| Case-set fingerprint | `9ceb36cf8974680224d767206d107fb6bfc44225e25498978ca7dca10938e2a2` | same |
| Evaluations | 193, one forward pass each | 193, one forward pass each |
| Wall time | 22.9 s | 15.8 s |
| Latency | mean 131.6 ms, median 113.5 ms | mean 81.0 ms, median 68.5 ms |
| Records | `results/choice-signal-qwen35-2b.jsonl` | `results/choice-signal-minicpm5-2b.jsonl` |

The Qwen adapter is a harness-side loader for a multimodal checkpoint. It is not
official FuzzyAI runtime support and it is not wired into any production backend.

No pass threshold was set anywhere in this experiment.

Phase 2B.1 changed three things relative to Phase 2B. First, the Choice
diagnostic field `candidate_mass` was renamed to `scoring_label_mass` (section 1).
Second, the scoring representation is now recorded in the artifact as
`rendering_config`, because a rendering flag can silently move the model out of
the scoring position (section 3). Third, both models were re-run under the same
explicit rendering config. The Qwen re-run reproduced the Phase 2B artifact
across the probability dicts, `scoring_label_mass` values, resolved token ids, and
plan fingerprints: 177 of 177 distinct `(case_id, permutation)` pairs were
identical. The execution fingerprint differed on all 193 records, because this
round records and applies `rendering_config` while the Phase 2B run passed no
rendering flag, which also shows the flag is a no-op for Qwen3.5-2B on this
fixture. The Qwen numbers quoted below are therefore the Phase 2B numbers.

## 1. Terminology: `candidate_mass` is now `scoring_label_mass`

The old name was too wide. The quantity is:

```text
scoring_label_mass = P(next token is one of the declared scoring-label tokens)
log_scoring_label_mass = logsumexp(scoring_label_logits) - logsumexp(vocab_logits)
```

Full-vocabulary normalisation. It answers "how much of the model's raw next-token
probability mass landed on the scoring labels this plan declared", which is a
statement about whether the model is following the scoring protocol.

It does not answer "is the semantic candidate set correct", "is it exhaustive",
"is the true answer inside it", "did the model choose correctly", "is the model
calibrated", or "is the model correct". Section 11 shows out-of-set answers
coinciding with `scoring_label_mass` near 1.0.

The Bool-side name `verbalizer_mass` is unchanged. It already means the same
thing for a two-token scoring representation, and renaming a published Bool field
would be API churn with no semantic gain. `verbalizer_mass` and
`scoring_label_mass` instantiate the same candidate-label-set mass equation under
their respective binary and categorical scoring representations; `scoring_label_mass`
reduces to `verbalizer_mass` when there are exactly two scoring labels.

## 2. Resolved scoring representation

The backend resolved every scoring label against the real rendered continuation
before the forward pass. Every label resolved to exactly one token, and the ids
were pairwise distinct, so no evaluation was rejected on either model.

```text
Qwen3.5-2B    A -> 32   B -> 33   C -> 34   D -> 35   E -> 36
MiniCPM5-2B   A -> 54   B -> 55   C -> 56   D -> 57   E -> 58
```

The compiler never saw these ids. It produced the same label strings for every
permutation and the backend did the resolving.

## 3. A thinking-enabled MiniCPM run that is not a measurement

The first MiniCPM5-2B run passed no rendering flag, so the backend default
applied (effectively `{}`). Those records predate the `rendering_config` artifact
field and therefore do not record it. That run is
kept as `results/choice-signal-minicpm5-2b-thinking-on.jsonl` because it is
evidence, but it is not a Choice measurement:

```text
top token:            <think>  (193 / 193 records)
top token probability: 1.0000  (rounded; minimum 0.999998)
scoring_label_mass:    4.31e-17 to 7.13e-12  (median 2.65e-14)
```

The model wanted to start a reasoning block, so the scoring labels sat in the
extreme tail. The restricted softmax still produced a normalised distribution
over three near-zero logits, and the resulting numbers look like data. They are
renormalised tail noise.

This is the failure mode that Phase 2A.1 was created for and that INV-19 exists
to expose. `scoring_label_mass` caught it immediately: it is the only number in
the artifact that says the model never entered the scoring position. Without it,
this run would have been reported as "MiniCPM shows large representation
instability", which would have been wrong.

Phase 2A.2 ran MiniCPM5-2B with `{"enable_thinking": false}`
(`experiments/semantic_signal/run.py` passes that flag by default), and recorded
healthy masses. Phase 2B.1 therefore re-runs MiniCPM with the same explicit flag,
which restores the comparable environment. The harness now accepts
`--chat-template-kwargs` and records the result as `rendering_config` in every
record and in the run header.

## 4. Three-way experiment (N=3), all 6 label permutations

15 cases, 6 permutations each, 90 evaluations per model.

| metric | Qwen3.5-2B | MiniCPM5-2B |
| --- | --- | --- |
| semantic argmax preservation | 90 / 90 (1.0000) | 85 / 90 (0.9444) |
| mean total variation | 0.0334 | 0.0514 |
| max total variation | 0.3082 | 0.4672 |
| max absolute drift | 0.3082 | 0.4672 |
| identity agreement with expected category | 15 / 15 | 14 / 15 |
| scoring_label_mass (min / mean / max) | 0.9371 / 0.9891 / 0.9972 | 0.9649 / 0.9966 / 1.0000 |

Qwen3.5-2B identity results:

| case | expected | winner | billing | shipping | technical | scoring_label_mass |
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

MiniCPM5-2B identity results:

| case | expected | winner | billing | shipping | technical | scoring_label_mass |
| --- | --- | --- | --- | --- | --- | --- |
| 3w-billing-01 | billing | billing | 0.999 | 0.001 | 0.000 | 0.9999 |
| 3w-billing-02 | billing | billing | 0.999 | 0.001 | 0.000 | 1.0000 |
| 3w-billing-03 | billing | billing | 0.981 | 0.018 | 0.001 | 0.9994 |
| 3w-billing-04 | billing | billing | 0.937 | 0.060 | 0.003 | 0.9992 |
| 3w-billing-05 | billing | billing | 0.999 | 0.001 | 0.000 | 1.0000 |
| 3w-shipping-01 | shipping | shipping | 0.006 | 0.993 | 0.001 | 0.9999 |
| 3w-shipping-02 | shipping | shipping | 0.001 | 0.997 | 0.001 | 0.9993 |
| 3w-shipping-03 | shipping | shipping | 0.005 | 0.995 | 0.000 | 0.9998 |
| 3w-shipping-04 | shipping | shipping | 0.002 | 0.997 | 0.000 | 0.9999 |
| 3w-shipping-05 | shipping | shipping | 0.058 | 0.802 | 0.139 | 0.9996 |
| 3w-technical-01 | technical | technical | 0.001 | 0.000 | 0.999 | 0.9995 |
| 3w-technical-02 | technical | technical | 0.109 | 0.085 | 0.806 | 0.9963 |
| 3w-technical-03 | technical | technical | 0.287 | 0.106 | 0.607 | 0.9806 |
| 3w-technical-04 | technical | billing | 0.685 | 0.093 | 0.222 | 0.9887 |
| 3w-technical-05 | technical | technical | 0.002 | 0.001 | 0.997 | 0.9998 |

MiniCPM's single identity mismatch is `3w-technical-04`, a context whose
`technical` probability (0.222) is below `billing` (0.685) in the identity
mapping. Qwen answered the same case `technical` at 0.685.

Worst 3-way drifts. Qwen3.5-2B:

| case | permutation change | TV | before | after |
| --- | --- | --- | --- | --- |
| 3w-shipping-05 | ABC to BCA | 0.3082 | shipping 0.622, technical 0.333 | shipping 0.930, technical 0.060 |
| 3w-shipping-05 | ABC to ACB | 0.3005 | shipping 0.622, technical 0.333 | shipping 0.923, technical 0.046 |
| 3w-technical-04 | ABC to ACB | 0.2793 | technical 0.685, billing 0.196 | technical 0.406, shipping 0.316, billing 0.279 |

MiniCPM5-2B:

| case | permutation change | TV | before | after |
| --- | --- | --- | --- | --- |
| 3w-technical-03 | ABC to CAB | 0.4672 | technical 0.607, billing 0.287 | billing 0.629, shipping 0.231, technical 0.140 |
| 3w-technical-02 | ABC to CAB | 0.4061 | technical 0.806, billing 0.109 | billing 0.453, shipping 0.147, technical 0.400 |
| 3w-technical-04 | ABC to BAC | 0.3422 | billing 0.685, technical 0.222 | billing 0.384, shipping 0.435, technical 0.181 |

## 5. Five-way experiment (N=5)

15 cases, 5 deterministic permutations each (identity, reverse, cyclic shift,
first-two swap, rotate-by-two), 75 evaluations per model.

| metric | Qwen3.5-2B | MiniCPM5-2B |
| --- | --- | --- |
| semantic argmax preservation | 75 / 75 (1.0000) | 74 / 75 (0.9867) |
| mean total variation | 0.0277 | 0.0381 |
| max total variation | 0.2937 | 0.5746 |
| max absolute drift | 0.2706 | 0.5746 |
| identity agreement with expected category | 15 / 15 | 13 / 15 |
| scoring_label_mass (min / mean / max) | 0.9840 / 0.9936 / 0.9972 | 0.9841 / 0.9983 / 1.0000 |

MiniCPM5-2B identity results:

| case | expected | winner | billing | shipping | returns | technical | account | scoring_label_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5w-account-01 | account | account | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.9997 |
| 5w-account-02 | account | account | 0.008 | 0.000 | 0.000 | 0.000 | 0.992 | 0.9985 |
| 5w-account-03 | account | account | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.9999 |
| 5w-billing-01 | billing | returns | 0.004 | 0.000 | 0.996 | 0.000 | 0.000 | 0.9999 |
| 5w-billing-02 | billing | billing | 0.997 | 0.003 | 0.000 | 0.000 | 0.000 | 0.9999 |
| 5w-billing-03 | billing | returns | 0.047 | 0.010 | 0.938 | 0.001 | 0.004 | 0.9984 |
| 5w-returns-01 | returns | returns | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.0000 |
| 5w-returns-02 | returns | returns | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.0000 |
| 5w-returns-03 | returns | returns | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.0000 |
| 5w-shipping-01 | shipping | shipping | 0.007 | 0.969 | 0.003 | 0.014 | 0.007 | 0.9987 |
| 5w-shipping-02 | shipping | shipping | 0.003 | 0.996 | 0.000 | 0.001 | 0.000 | 0.9995 |
| 5w-shipping-03 | shipping | shipping | 0.002 | 0.940 | 0.013 | 0.025 | 0.020 | 0.9952 |
| 5w-technical-01 | technical | technical | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.9998 |
| 5w-technical-02 | technical | technical | 0.079 | 0.015 | 0.004 | 0.659 | 0.242 | 0.9859 |
| 5w-technical-03 | technical | technical | 0.000 | 0.000 | 0.000 | 0.991 | 0.009 | 0.9994 |

MiniCPM's two identity mismatches are billing cases answered `returns`
(0.996 and 0.938), against Qwen's 15 / 15.

Worst 5-way drifts. Qwen3.5-2B:

| case | permutation change | TV | before | after |
| --- | --- | --- | --- | --- |
| 5w-account-03 | ABCDE to BCDEA | 0.2937 | account 0.618, technical 0.331 | account 0.889, technical 0.064 |
| 5w-account-02 | ABCDE to BACDE | 0.1545 | account 0.703 | account 0.618 |
| 5w-account-03 | ABCDE to EDCBA | 0.1544 | account 0.618, technical 0.331 | account 0.773, technical 0.221 |

MiniCPM5-2B:

| case | permutation change | TV | before | after |
| --- | --- | --- | --- | --- |
| 5w-technical-02 | ABCDE to CDEAB | 0.5746 | technical 0.659, account 0.242 | account 0.707, billing 0.158, technical 0.084 |
| 5w-billing-01 | ABCDE to EDCBA | 0.3220 | returns 0.996 | returns 0.674, billing 0.248 |
| 5w-shipping-03 | ABCDE to BACDE | 0.3050 | shipping 0.940 | shipping 0.635, returns 0.160 |

## 6. Is representation sensitivity Qwen-specific or cross-model?

Cross-model. Both families show the same two-part pattern:

- the semantic winner is largely stable under label permutation;
- the full uncalibrated categorical distribution is measurably sensitive to which
  letter a candidate is bound to, with occasional large moves.

Qwen3.5-2B: winner 100 percent stable, mean TV 0.0334 / 0.0277, max TV 0.3082 / 0.2937.
MiniCPM5-2B: winner 94.4 / 98.7 percent stable, mean TV 0.0514 / 0.0381, max TV 0.4672 / 0.5746.

This was observed on both tested configurations, so the following is supported
for those two configurations only:

> Representation sensitivity was observed as a cross-model property of direct
> categorical token scoring in the tested configurations.

It is not a claim about all models, or about LLMs generally, or about direct
categorical scoring always.

## 7. Is the semantic winner also stable across models?

Largely, not perfectly. Qwen3.5-2B preserved the winner in every permutation at
both N=3 and N=5. MiniCPM5-2B preserved it in 85 / 90 and 74 / 75, and its
identity answers disagreed with the fixture's obvious category in 1 / 15 and
2 / 15 cases. So "the winner is stable" is a weaker statement on the second
family than the Phase 2B round alone suggested, and it is the full distribution,
not the winner, that degrades first in both.

## 8. Irrelevant candidate addition

Six 3-way cases, re-run with a fourth candidate `account deletion` added.

| metric | Qwen3.5-2B | MiniCPM5-2B |
| --- | --- | --- |
| winner preserved | 6 / 6 | 5 / 6 |
| TV range | 0.0006 to 0.0153 | 0.0012 to 0.9629 |
| scoring_label_mass (3-way / 4-way mean) | 0.9943 / 0.9939 | 0.9965 / 0.9971 |

MiniCPM's single flip is `3w-technical-03`, which moved to `account deletion`
with TV 0.9629. Adding a candidate that is semantically absent from the context
was nearly free for Qwen and nearly free for MiniCPM in five of six cases, but
not free in all of them.

## 9. Description paraphrase

Five 3-way cases, re-run with the `billing` description changed from
`Payment, charges, and invoices` to `Problems involving invoices, charges, or
payments`. The candidate name was unchanged.

| metric | Qwen3.5-2B | MiniCPM5-2B |
| --- | --- | --- |
| winner preserved | 5 / 5 | 5 / 5 |
| TV range | 0.0002 to 0.0115 | 0.0021 to 0.3560 |

Both models kept the winner. On Qwen the largest effect landed on a `technical`
case, so it was not the paraphrased candidate that moved most; on MiniCPM the
largest effect (0.3560) also landed on `3w-technical-02`.

## 10. Taxonomy overlap sensitivity

Candidate set: `billing`, `payment issue`, `shipping`, `technical`.

| case | note | Qwen winner / mass | MiniCPM winner / mass |
| --- | --- | --- | --- |
| ov-payment-01 | card declined, money still left the account | payment issue 0.689 / billing 0.254 / mass 0.9853 | payment issue 0.936 / billing 0.060 / mass 0.9997 |
| ov-payment-02 | switching payment method | payment issue 0.810 / billing 0.159 / mass 0.9896 | billing 0.798 / payment issue 0.202 / mass 0.9997 |
| ov-payment-03 | control, invoice request | billing 0.998 / mass 0.9961 | billing 1.000 / mass 1.0000 |

Both models split mass across the two overlapping candidates in the two
ambiguous cases (together 0.943 and 0.970 on Qwen, 0.996 and 1.000 on MiniCPM),
and both stayed sharp on the unambiguous control. They resolve the overlap
differently in `ov-payment-02`, but both show the same structural behaviour.

This is a failure characterisation of a taxonomy that overlaps, not a model
failure, and it is not a robustness invariant. It shows that a restricted Choice
probability is relative to a taxonomy and its granularity.

## 11. Out-of-set behaviour

Candidate set: `billing`, `shipping`, `technical`. All three contexts belong to
none of them.

| case | true topic | Qwen winner / distribution | mass | MiniCPM winner / distribution | mass |
| --- | --- | --- | --- | --- | --- |
| oos-01 | account deletion | technical 0.893, billing 0.083 | 0.9893 | billing 0.754, shipping 0.131 | 0.9968 |
| oos-02 | job application | technical 0.711, billing 0.180 | 0.9853 | billing 0.699, shipping 0.227 | 0.9960 |
| oos-03 | sponsorship enquiry | technical 0.677, billing 0.220 | 0.9871 | billing 0.678, shipping 0.171 | 0.9957 |

This is the most important negative result of the round, and it reproduces on the
second family. With a candidate set that does not contain the true topic, both
models still returned an in-set answer with high confidence, and
`scoring_label_mass` stayed near 1.0 in every case.

Answering the round's question directly: `scoring_label_mass` showed no semantic
coverage capability on either model. In these probes it indicated protocol
adherence rather than semantic coverage. Out-of-set detection must come from the caller or from a later
policy layer, and this experiment is not open-set detection.

## 12. Scoring label mass across N

| model | candidate set | records | mean | min | max |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B | three_way (N=3) | 104 | 0.9896 | 0.9371 | 0.9972 |
| Qwen3.5-2B | three_way_plus_irrelevant (N=4) | 6 | 0.9939 | 0.9905 | 0.9961 |
| Qwen3.5-2B | three_way_billing_paraphrase (N=3) | 5 | 0.9953 | 0.9939 | 0.9965 |
| Qwen3.5-2B | five_way (N=5) | 75 | 0.9936 | 0.9840 | 0.9972 |
| Qwen3.5-2B | taxonomy_overlap (N=4) | 3 | 0.9903 | 0.9853 | 0.9961 |
| MiniCPM5-2B | three_way (N=3) | 104 | 0.9967 | 0.9649 | 1.0000 |
| MiniCPM5-2B | three_way_plus_irrelevant (N=4) | 6 | 0.9971 | 0.9885 | 0.9999 |
| MiniCPM5-2B | three_way_billing_paraphrase (N=3) | 5 | 0.9988 | 0.9950 | 1.0000 |
| MiniCPM5-2B | five_way (N=5) | 75 | 0.9983 | 0.9841 | 1.0000 |
| MiniCPM5-2B | taxonomy_overlap (N=4) | 3 | 0.9998 | 0.9997 | 1.0000 |

These values are not directly comparable across N. The denominator is the full
vocabulary, the scoring tokens differ per set, and both the mass and the
restricted distribution move with the label binding (section 4). The table only
shows that for both models on these synthetic contexts the scoring tokens
absorbed almost all of the next-token probability.

## 13. Layer boundary: single-run diagnostics vs cross-run metrics

`scoring_label_mass`, `scoring_label_token_probabilities`, the top token, the
restricted semantic distribution, and `certainty` are single-run properties and
live in the runtime: `ScoringDiagnostics` / `ChoiceScoringDiagnostics`,
`ChoiceResult`, and `DecisionTrace`.

Total variation, argmax preservation, rank preservation, and probability drift
across permutations require comparing two executions, so they are evaluation
properties and live in the experiment layer (`metrics.py`). They are deliberately
absent from the runtime trace and from the result objects, and no threshold is
attached to them anywhere.

## 14. Determinism notes

Every evaluation used exactly one forward pass and never called `model.generate()`.
The Qwen3.5-2B re-run in this round reproduced the Phase 2B artifact exactly on
the semantic fields: 0 of 177 probability dicts differed, 0 `scoring_label_mass`
values differed, 0 resolved token id maps differed, and 0 plan fingerprints
differed. This is cross-process determinism evidence across the probability,
mass, token-id, and plan-fingerprint portion of the chain. The execution
fingerprint did differ on all 193 records, because this round now records and
applies `rendering_config` while the Phase 2B run passed no rendering flag; since
the semantic fields were identical, the flag is a no-op for Qwen3.5-2B. This also
confirms that the `candidate_mass` to `scoring_label_mass` rename was
value-preserving.

Latency is not deterministic and the first call in each run included model
warm-up.

## 15. What this establishes and does not

Established, for these two models, this doctrine, this label scheme, and these
synthetic fixtures:

- the semantic candidate identity and the scoring-label identity remain separable
  end to end, and the decision, plan, and execution fingerprints record different
  things;
- the semantic winner was largely stable under label permutation on both
  families, exactly stable on Qwen3.5-2B and stable in 85 / 90 and 74 / 75 on
  MiniCPM5-2B;
- the full uncalibrated categorical distribution was representation-sensitive on
  both families, with maximum total variation 0.47 at N=3 and 0.57 at N=5;
- `scoring_label_mass` identifies whether a model occupies the scoring position,
  and it does not indicate whether the candidate set is right.

Not established:

- that a Choice probability is an accurate or calibrated quantity (it is not, and
  `predicted_correctness` remains `None`);
- that a Choice probability magnitude is stable under representation change (it
  is not);
- that the candidate set is correct or complete, or that out-of-set inputs can be
  detected (they cannot, section 11);
- that any of this generalises to other models, other doctrines, other label
  schemes, real data, larger N, or cloud providers;
- that the winner is correct. The fixture expectations are the obvious reading of
  hand-written contexts, not ground truth from a dataset.
