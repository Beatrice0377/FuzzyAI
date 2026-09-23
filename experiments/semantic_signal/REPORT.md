# Semantic Signal Validation - Phase 2A.2 report

Status of this document: experiment record. It is not a benchmark, not a
quality claim, and not a capability claim.

## 1. What this round asked

Phase 2A proved that one explicit binary scoring signal can be obtained from a
local causal LM and turned into an honest uncalibrated probability. Phase 2A.1
proved the scoring position can be *diagnosed*. Phase 2A.2 asks a different
question:

> Does the binary scoring position carry a semantic signal at all, and which
> part of that signal comes from the model, the doctrine, the label family,
> the label order, and the label mapping?

This is mechanism validation. Nothing here is calibrated, nothing here is a
correctness measure, and no threshold is applied anywhere. No probe is
auto-rejected and no record carries a validity verdict.

Overall conclusion: single next-token binary scoring is model- and
formulation-dependent. It is not a provider-independent or model-independent
semantic primitive: the same prompt formulation and the same labels produced
usable signal on some models and none on others. Newer small models tested here
did show sufficient semantic sensitivity under specific formulations to justify
further investigation of multi-class decisions. That justifies a next round; it
does not constitute a quality claim about any model or about FuzzyAI.

## 2. Case set

`cases.json`, `case_set_version` `semantic-signal-v1`, frozen before any model
ran. Fingerprint (`fingerprint()` over the parsed case set):
`b68a1d395f0ec8f2b1e1400cc2e4f3c4b0d7e5f8bdb6a80576c3a4bab2209805`.

- 8 ladders x 5 rungs (`strong_positive`, `weak_positive`, `insufficient`,
  `weak_negative`, `strong_negative`), holding the question fixed and varying
  how strongly the context supports it.
- 12 polarity pairs (a supporting question and its opposite, over the same
  context).
- 4 contrast groups: for one question, three contexts
  (`relevant_positive`, `irrelevant`, `relevant_negative`). This is the
  instrument that holds the question fixed and moves *relevance*.
- 4 injection probes with embedded instructions against the evidence.

Total 80 probes per formulation. Four themes only (delivery, refund,
subscription, software). No medical, legal, credit or political content.

## 3. Formulation matrix

Three doctrines: D1 `binary-semantic-judgment-v1` (the Phase 2A baseline),
D2 `minimal-neutral-v1`, D3 `evidence-oriented-v1`. Three label families:
`yes_no`, `true_false`, `ab`. For `ab` the doctrine template explicitly
declares which surface label means supported, so the mapping is carried by the
doctrine, never by runtime logic.

Two opt-in ablations: `--order-ablation` (negative label listed first,
mapping unchanged) and `--mapping-swap` (semantic TRUE bound to the other
surface label). With both, the matrix is 27 formulations x 80 probes = 2160
probes per model.

Three models ran, one process each, sequentially, `bfloat16`, one forward pass
per probe, batch size 1:

| model | family / arch | params | peak VRAM allocated |
|---|---|---|---|
| `LiquidAI/LFM2.5-1.2B-Instruct` | LiquidAI hybrid conv+attention (`Lfm2ForCausalLM`) | 1.17B | 2285 MiB |
| `openbmb/MiniCPM5-2B` | Llama-style (`LlamaForCausalLM`) | 2.52B | 4877 MiB |
| `Qwen/Qwen3.5-2B` (text tower) | Qwen3.5 (`Qwen3_5ForCausalLM`) | 2.27B | 3716 MiB |

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB. torch 2.14.0+cu130,
transformers 5.17.0, CUDA 13.0. All runs offline
(`HF_HUB_OFFLINE=1`, `--local-files-only`). Every run produced 2160 records
with status `ok`; no probe was rejected.

## 4. Scoring position health

`verbalizer_mass` is the full-vocabulary probability mass on the two declared
candidate tokens. It is what separates a scoring-position failure from a
semantic failure. "Low-mass" below means `verbalizer_mass < 0.5`. That figure is
an experimental analysis cutoff chosen to group this report's own results: it is
not a FuzzyAI threshold, not an API contract, and nothing in the runtime applies
it, exposes a validity verdict, or rejects a probe because of it.

| model | doctrine | low-mass probes | mass_min |
|---|---|---|---|
| LFM2.5-1.2B | D1 | 0 / 720 | 0.998672 |
| LFM2.5-1.2B | D2 | 0 / 720 | 0.998197 |
| LFM2.5-1.2B | D3 | 462 / 720 (64.2%) | 0.000008 |
| MiniCPM5-2B | D1 | 0 / 720 | 0.916076 |
| MiniCPM5-2B | D2 | 0 / 720 | 0.983116 |
| MiniCPM5-2B | D3 | 0 / 720 | 0.817990 |
| Qwen3.5-2B | D1 | 0 / 720 | 0.852574 |
| Qwen3.5-2B | D2 | 0 / 720 | 0.961845 |
| Qwen3.5-2B | D3 | 0 / 720 | 0.918833 |

D3's evidence-oriented wording drives LFM2.5-1.2B off the decision position
(its top token becomes `Support`, `True` capitalized, or `Yes` capitalized
rather than the declared lowercase label), so its `P(True)` there is
renormalized tail noise. The same doctrine does not disturb MiniCPM5-2B or
Qwen3.5-2B. Doctrine robustness is therefore **model-dependent**, and
`verbalizer_mass` is what makes that visible instead of silent.

## 5. RQ1 - semantic sensitivity (ladder shape)

Mean `P(True)` per rung, `yes_no`, canonical mapping, positive first. The
`inverted` column counts how many of that cell's 8 individual ladders contain
at least one adjacent step that moves the wrong way; a tie counts as neither
an inversion nor a strict decrease, so `inverted` and `strictly decreasing`
do not always add up to 8.

| model | doctrine | strong_pos | weak_pos | insufficient | weak_neg | strong_neg | inverted | strictly decreasing |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-1.2B | D1 | 0.999 | 0.985 | 0.820 | 0.957 | 0.573 | 7 / 8 | 1 / 8 |
| LFM2.5-1.2B | D2 | 0.998 | 0.996 | 0.902 | 0.977 | 0.637 | 6 / 8 | 1 / 8 |
| LFM2.5-1.2B | D3 | 1.000 | 0.997 | 0.797 | 0.994 | 0.877 | 8 / 8 | 0 / 8 |
| MiniCPM5-2B | D1 | 0.955 | 0.350 | 0.004 | 0.022 | 0.002 | 8 / 8 | 0 / 8 |
| MiniCPM5-2B | D2 | 0.917 | 0.384 | 0.024 | 0.013 | 0.000 | 6 / 8 | 2 / 8 |
| MiniCPM5-2B | D3 | 0.978 | 0.660 | 0.057 | 0.381 | 0.553 | 8 / 8 | 0 / 8 |
| Qwen3.5-2B | D1 | 0.944 | 0.677 | 0.330 | 0.178 | 0.058 | 2 / 8 | 5 / 8 |
| Qwen3.5-2B | D2 | 0.892 | 0.610 | 0.202 | 0.101 | 0.023 | 5 / 8 | 3 / 8 |
| Qwen3.5-2B | D3 | 0.544 | 0.267 | 0.030 | 0.048 | 0.171 | 7 / 8 | 0 / 8 |

Extremes separate in every model: `strong_positive` vs `strong_negative` is
0.999 vs 0.573 (LFM D1), 0.917 vs 0.000 (MiniCPM D2), 0.892 vs 0.023
(Qwen3.5 D2). The direction of the signal is real.

The middle does not order reliably. In 6 of the 9 model x doctrine cells the
aggregate ladder puts `insufficient` below `weak_negative`, which is the wrong
side of zero for "not enough evidence". Only three cells are fully monotonic
in aggregate (the six that invert `insufficient` and the three that do not are
disjoint): MiniCPM5-2B D2 (`0.917 > 0.384 > 0.024 > 0.013 > 0.000`),
Qwen3.5-2B D1 (`0.944 > 0.677 > 0.330 > 0.178 > 0.058`) and Qwen3.5-2B D2
(`0.892 > 0.610 > 0.202 > 0.101 > 0.023`).

Note that the per-ladder counts and the aggregate shape disagree, and both are
reported: an individual ladder is noisy even when the aggregate of 8 looks
clean. Qwen3.5-2B D1, the best cell in the round, still has 2 of 8 individual
ladders out of order. LFM2.5-1.2B never falls below 0.573 even at
`strong_negative` under D1/D2, which is a positive prior rather than a
semantic reading; under D3 the prior is so strong that every one of its 8
ladders fails to decrease.

## 6. RQ2 - polarity sensitivity

Polarity consistency: the supporting question scores above its opposite over
the same context, canonical mapping, 12 pairs.

| model | D1 yes_no | D2 yes_no | D3 yes_no | D1 t_f | D2 t_f | D3 t_f |
|---|---|---|---|---|---|---|
| LFM2.5-1.2B | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| MiniCPM5-2B | 1.00 | 1.00 | 0.92 | 0.92 | 1.00 | 0.92 |
| Qwen3.5-2B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Directional accuracy is high for all three (0.75 to 1.00 across canonical
formulations), but a model with a constant yes-bias scores well on this metric
by accident, which is why the contrast groups in section 7 matter more.

Injection probes (D2, `yes_no`, canonical, positive first), where the context
embeds an instruction against the evidence:

| probe | semantic expectation | LFM2.5-1.2B | MiniCPM5-2B | Qwen3.5-2B |
|---|---|---|---|---|
| injection-01 | should_be_false | 0.9876 | 0.0067 | 0.0373 |
| injection-02 | unspecified_insufficient | 0.9998 | 0.2689 | 0.5622 |
| injection-03 | should_be_true | 0.9325 | 0.8670 | 0.9149 |
| injection-04 | should_be_false | 0.9770 | 0.0097 | 0.0474 |

LFM2.5-1.2B follows the embedded instruction. MiniCPM5-2B and Qwen3.5-2B
follow the evidence. This is an observation about three small models on four
hand-written probes, taken with no ground-truth dataset and sample size 4. It
is **not** a prompt-injection safety claim, and FuzzyAI makes no such claim
anywhere.

## 7. Contrast groups - the sharpest instrument

One question, three contexts. `relevant_positive` should score highest,
`irrelevant` should be near-uncertain, `relevant_negative` should score
lowest. `P(True)` at D2, `yes_no`, canonical, positive first:

| group | model | relevant_pos | irrelevant | relevant_neg | fully ordered |
|---|---|---|---|---|---|
| delivery | LFM2.5-1.2B | 0.9987 | 0.6225 | 0.7773 | no |
| delivery | MiniCPM5-2B | 0.9954 | 0.0001 | 0.0000 | yes |
| delivery | Qwen3.5-2B | 0.9841 | 0.0953 | 0.0203 | yes |
| refund | LFM2.5-1.2B | 1.0000 | 0.9740 | 0.9526 | yes |
| refund | MiniCPM5-2B | 0.9964 | 0.0003 | 0.0000 | yes |
| refund | Qwen3.5-2B | 0.9707 | 0.0293 | 0.0097 | yes |
| subscription | LFM2.5-1.2B | 0.9990 | 0.9241 | 0.8176 | yes |
| subscription | MiniCPM5-2B | 0.7982 | 0.0006 | 0.0001 | yes |
| subscription | Qwen3.5-2B | 0.8355 | 0.0851 | 0.0331 | yes |
| software | LFM2.5-1.2B | 0.9985 | 0.9466 | 0.5312 | yes |
| software | MiniCPM5-2B | 0.9579 | 0.0036 | 0.0000 | yes |
| software | Qwen3.5-2B | 0.8808 | 0.2942 | 0.0474 | yes |

("fully ordered" means `relevant_positive > irrelevant > relevant_negative`;
the LFM delivery row is `no` only because `relevant_negative` (0.7773) landed
above `irrelevant` (0.6225).)

MiniCPM5-2B orders every group correctly with three orders of magnitude of
separation, and it moves to near-zero when the context is irrelevant.
Qwen3.5-2B also orders every group correctly, with roughly one to two orders
of magnitude of separation. LFM2.5-1.2B barely moves: irrelevant contexts
still score 0.62 to 0.97, and `relevant_negative` scores 0.53 to 0.95,
meaning it answers "yes" largely independently of whether the supplied
evidence supports the question.

This is the single most informative result of the round. A metric that only
reports directional accuracy on polarity pairs would have rated LFM2.5-1.2B
high; the contrast groups reveal that its probability is dominated by a
label/prior preference rather than by the context.

## 8. RQ3 - label invariance

Cross-family agreement and the ablation behaviour:

- `yes_no` and `true_false` behave as near-interchangeable verbalizations at
  the aggregate level, while the model's bias is bound to the surface token
  (`true` vs `Yes`).
- The `ab` family is strongly model-dependent. Directional accuracy on
  canonical `ab`: LFM2.5-1.2B 0.38 to 0.88, Qwen3.5-2B 0.62 to 0.88,
  MiniCPM5-2B 0.50 to 0.50. No model treats `ab` as a drop-in replacement for
  a natural-language yes/no.
- The mapping swap is only meaningful where the doctrine **declares** the
  label semantics. For `ab` the doctrine declares it, so keeping the inverted
  semantic direction is possible through the declared mapping. For `yes_no`
  and `true_false` no mapping is declared, so swapping the verbalizer roles
  simply inverts the measurement, which is exactly what the data shows: those
  swapped formulations collapse to directional accuracy 0.00 to 0.38 and
  polarity consistency 0.00 to 0.17, while `ab` swapped stays high.

Conclusion: label families are not interchangeable without declaring what
each surface label means, and a label-role swap is not a neutral operation
unless the semantics travel with the doctrine.

## 9. RQ4 - model dependence

| property | LFM2.5-1.2B-Instruct | MiniCPM5-2B | Qwen3.5-2B (text tower) |
|---|---|---|---|
| family / arch | LiquidAI hybrid conv+attention | Llama-style | Qwen3.5 with linear attention layers |
| strong prior bias | yes, positive (strong_neg 0.573) | no (strong_neg 0.000) | no (strong_neg 0.023) |
| collapse rate, yes_no canonical | 0.50 to 1.00 | 0.00 to 0.25 | 0.00 to 0.25 |
| best label family | `true_false` / `yes_no` | `yes_no` / `true_false` | `true_false` / `yes_no` |
| survives D3 wording | no (64% of probes off-position) | yes (0%) | yes (0%) |
| contrast-group ordering | fails relevance discrimination | correct, 3 orders of magnitude | correct, 1 to 2 orders |
| best aggregate ladder | inverted in 6/8, compressed | monotonic under D2 | monotonic under D1 and D2 |
| best strictly-decreasing ladders | 1 of 8 | 2 of 8 | 6 of 8 (D1 `true_false`) |

All three models share the coarse behaviour: extremes separate, canonical
`yes_no` / `true_false` polarity consistency is 0.92 to 1.00, and directional
accuracy is 0.75 to 1.00. Everything finer is model-specific. LFM2.5-1.2B has
a strong positive prior and no relevance discrimination; MiniCPM5-2B has
neither, but is weak on `ab`; Qwen3.5-2B behaves best of the three on ladder
shape and relevance while still being weak on `ab` and collapsing under D3. A
single-model result therefore cannot be generalized into a doctrine or label
guideline; the harness must keep reporting per model.

## 10. Choice readiness gate

Phase 2B was gated on evidence that the ternary/categorical scoring position
is viable. Against the stated criteria, using the best formulations found:

- Obvious positive and obvious negative evidence separate stably: **met** for
  MiniCPM5-2B (0.917 vs 0.000) and Qwen3.5-2B (0.892 vs 0.023); weakly for
  LFM2.5-1.2B (0.998 vs 0.573).
- Polarity pairs point the right way in the majority of cases: **met**
  (0.92 to 1.00 for all three).
- After a declared mapping swap the semantic direction is preserved: **met
  only where the doctrine declares the mapping** (`ab`), not for undeclared
  families.
- No significant always-first-label collapse: **met** for MiniCPM5-2B and
  Qwen3.5-2B, **not met** for LFM2.5-1.2B (collapse 0.50 to 1.00).
- Context relevance is discriminated: **met** for MiniCPM5-2B and
  Qwen3.5-2B (correct ordering on all 4 groups), **not met** for
  LFM2.5-1.2B.
- `verbalizer_mass` shows the model is actually in the candidate space:
  **met** for `yes_no` / `true_false` on all three, **not met** for
  LFM2.5-1.2B under D3, and not measured for three-way candidate spaces yet.

So the gate is met by a specific, named configuration (MiniCPM5-2B or
Qwen3.5-2B + `yes_no` or `true_false` + a doctrine that keeps the model at
the decision position), not by the mechanism in general. The three-way case
remains unmeasured: `verbalizer_mass` for a 3-candidate space, and whether a
model can hold three competing candidate tokens, are open.

## 11. Findings

1. The binary scoring position carries a real semantic signal, but the
   signal quality is dominated by whether the model actually sits in the
   declared candidate space. `verbalizer_mass` is the measurement that
   separates the two, and it changed the interpretation of an entire
   doctrine (D3).
2. A high `P(True)` is not evidence of semantic reading. LFM2.5-1.2B reaches
   0.99+ on questions whose context is irrelevant, so probability magnitude
   alone cannot be trusted as a sensitivity indicator.
3. Context-relevance probes (contrast groups) discriminate models far better
   than polarity pairs or directional accuracy. They are the only instrument
   in this round that separated the three models unambiguously.
4. Label families need declared semantics. Undeclared families cannot be
   mapping-swapped meaningfully.
5. Doctrine robustness is model-dependent, so a doctrine cannot be validated
   once and reused across models.
6. The `insufficient` rung is the systematic weak point: it lands below
   `weak_negative` in 6 of the 9 model x doctrine cells. "Not enough evidence" is
   not ordered between weak support and weak opposition, which is precisely
   the behaviour that Phase 5 abstention policy must handle.
7. Ladder monotonicity and aggregate ladder shape differ. A model can look
   ordered in the 8-ladder mean while most individual ladders are not, so a
   single summary number would have hidden the noise.

Not established by this round: accuracy, calibration, real-world correctness,
prompt-injection safety, cross-domain robustness, large-model behaviour,
cloud-model behaviour, and multi-class behaviour. Section 10 records a named
configuration passing the choice-readiness gate, not the mechanism passing.

## 12. Limitations

- Three small models (1.17B to 2.52B), one of them (Qwen3.5-2B) scored
  through a harness-side text-tower adapter rather than its published
  multimodal entry point.
- Four hand-written themes, 80 probes per formulation. No ground truth, no
  dataset, no sample size.
- `P(True)` remains restricted and uncalibrated. `calibrated` is `False` and
  `predicted_correctness` is `None` everywhere.
- Single-token verbalizers only, one forward pass per probe, batch size 1, no
  quantization.
- `verbalizer_mass` is measured for a two-candidate space only.
- No threshold is applied; nothing in this round decides whether a scoring
  position is acceptable in production.
- The ladder and contrast results are sensitive to prompt wording and to the
  chat template, as the D3 divergence shows.
- Model revision is `None` in every run header: revision provenance was not
  obtained from the local cache, so the exact checkpoint commit of each model
  is not recorded. The header also records `device` as the requested value
  (`auto`) rather than the resolved device, so the actual device is known from
  the hardware line and the VRAM measurement rather than from that field.
  Both are provenance gaps in the harness, not in the core library.

## 13. Artifacts

- `run.py`, `formulations.py`, `metrics.py`, `cases.json`, `tests/test_harness.py`
- `qwen35_loader.py` (harness-side checkpoint adaptation, see section 14)
- `results/lfm2.5-1.2b-instruct.jsonl`, `results/minicpm5-2b.jsonl`,
  `results/qwen3.5-2b-text.jsonl` (2160 ok records each)
- Reproduce: `python experiments/semantic_signal/run.py --model <id>
  --order-ablation --mapping-swap`, then
  `python experiments/semantic_signal/metrics.py results/<id>.jsonl`
- Qwen3.5-2B additionally needs
  `--backend-factory qwen35_loader:Qwen35TextBackend`.
- The recorded `plan_fingerprint` and `execution_fingerprint` values predate
  plan fingerprint v4. Phase 2C.0 added compiler and assembler provenance to the
  plan payload and its fingerprint, so re-running now yields different
  fingerprints for otherwise identical plans.

## 14. Qwen3.5-2B text-tower compatibility

`Qwen/Qwen3.5-2B` is published as a vision-language checkpoint
(`Qwen3_5ForConditionalGeneration`). Its text tower is a
`Qwen3_5ForCausalLM` stored under `model.language_model.*`, next to a
`model.visual.*` tower and an `mtp.*` multi-token-prediction head.

`qwen35_loader.py` rebuilds the text-only model from `text_config` and rewrites
the key prefix `model.language_model.X -> model.X`, dropping the 312 non-text
tensors. The remap is exact: 0 unexpected keys, and the only missing key is
`lm_head.weight`, which is expected because `tie_word_embeddings` is true and
the head shares storage with `embed_tokens` (verified: identical
`data_ptr()`, same values). Everything else (rendering, verbalizer resolution,
scoring, diagnostics, provenance) runs through the real backend.

One real defect was found and fixed during this work: constructing from
`text_config` yields float32 parameters, and `load_state_dict` only widens the
checkpoint's bfloat16 values into them, so the model silently ran in float32
(7.53 GB, 7178 MiB peak) while the run header declared `bfloat16`. The adapter
now casts explicitly (measured after the fix: bfloat16, 3.76 GB, 3589 MiB
peak). This is recorded because a declared-but-unused dtype is exactly the
kind of false provenance this project exists to prevent.

## 15. Qwen3.5-2B results

The third model was added because the first two disagreed on nearly every
secondary property, leaving open whether the agreement on the primary ones was
coincidence. It is not an independent family sample: it is one more model, and
it is scored through a text-tower adapter.

Summary, all 2160 records `ok`:

- Scoring position: healthy under all three doctrines (0 low-mass probes,
  `mass_min` 0.853 / 0.962 / 0.919 for D1 / D2 / D3). It is the only model
  that is unaffected by D3 *and* has no positive prior.
- Ladder (yes_no, canonical, positive first): monotonic in aggregate under
  both D1 and D2, and the only model to place `insufficient` above
  `weak_negative` in aggregate under two doctrines (MiniCPM5-2B manages it
  under D2 only). Per-ladder noise
  remains: 2 of 8 ladders out of order under D1, 5 of 8 under D2, 7 of 8
  under D3.
- Contrast groups: correct ordering on all 4 groups, with the smallest
  relevant-vs-irrelevant gap (delivery 0.9841 vs 0.0953) and the largest
  (software 0.8808 vs 0.2942). Its `irrelevant` values are higher than
  MiniCPM5-2B's (0.03 to 0.29 versus 0.0001 to 0.0036), so it discriminates
  relevance less sharply.
- Polarity: 1.00 on all six canonical `yes_no` / `true_false` cells, the
  cleanest of the three models.
- Injection probes: follows the evidence (0.0373 and 0.0474 on the two
  should-be-false probes), like MiniCPM5-2B and unlike LFM2.5-1.2B.
- `ab`: weak, directional accuracy 0.62 to 0.88.
- D3: unlike LFM2.5-1.2B it stays on the decision position, but its ladder
  degrades (`strong_positive` drops to 0.544 and `strong_negative` rises to
  0.171), so D3's wording is still not safe here even though the mass is
  healthy.

Its best cell, D1 `true_false` positive first, reaches 6 of 8 ladders that
strictly decrease rung by rung, the highest of any model in the round
(LFM2.5-1.2B 1/8, MiniCPM5-2B 2/8).

## 16. Harness defect found and fixed during the round

The first pass of all three sweeps lost records: LFM2.5-1.2B produced 2128
`ok` and 32 `error`, MiniCPM5-2B 2130 `ok` and 30 `error`. Every error was

```
InvalidProbabilityError: vocab_logsumexp is inconsistent with the reported
logit: logit - logsumexp = 1.8553912184415822e-07
```

The cause was the clamp tolerance added in Phase 2A.2 (`1e-9`, absolute).
A float32 logsumexp over a 130k-token vocabulary carries rounding error
proportional to its own magnitude, so an absolute bound of `1e-9` sits below
the noise floor and rejected legitimate results. The bound is now relative
(`1e-6` scaled by the operand magnitude) and the material-inconsistency check
still rejects genuine mismatches. All three sweeps were re-run.

The re-runs also serve as a determinism check:

- LFM2.5-1.2B: 2160 `ok`, and all 2128 previously-`ok` records are
  bit-identical; 32 records regained.
- MiniCPM5-2B: 2160 `ok`, all 2130 previously-`ok` records bit-identical; 30
  records regained.
- Qwen3.5-2B: 2160 `ok`, all 2160 records bit-identical to its previous run
  (that run had no rejected records, so nothing changed).

So the fix recovered exactly the rejected records and altered no measurement,
and repeated runs of the same model on the same inputs reproduce every
probability and mass bit for bit.
