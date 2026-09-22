# Claims Register

Purpose: this document exists to stop an AI project from gradually turning
hypotheses into stated facts as the README evolves. Every capability,
performance, quality, calibration, or provider-support claim made about this
project must carry one of the four evidence statuses defined below. A claim
without a status (and, for `[V]`, without a pointer to evidence) does not
belong in any document in this repository.

## How to read this document

- `[V] VERIFIED`: verified by tests in this repository, by a reproducible
  experiment, or by explicit evidence that currently exists. A verified claim
  must point at tests, a code invariant, or a reproducible command.
- `[E] EXPERIMENTAL`: an experimental result observed under specific
  conditions (model, dataset, version, configuration). State the conditions.
  Never present a single experimental result as a general truth.
- `[H] HYPOTHESIS`: plausible and worth verifying, but not established.
  Wording must stay hedged ("may", "hypothesis", "to be evaluated").
- `[R] ROADMAP`: not implemented. A roadmap item is never a capability claim.

This register is referenced by the Architecture Principles of the
[design constitution](./design-constitution.md), and roadmap entries link to
[the roadmap](./roadmap.md) rather than duplicating it here.

## Verified claims

This section contains NO model-quality, performance, or accuracy claims. The
Bool path runs against a real local Hugging Face backend and has been executed
on real hardware; that single session is recorded under "Experimental records"
below and demonstrates nothing about model quality. Everything else is a
statement about the deterministic core, the Bool vertical slice, and their
tests, reproducible with `uv run pytest`. The `transformers` backend tests skip
automatically unless the optional extra is installed
(`uv sync --extra transformers`); they use injected fakes, never download
models, and never need a GPU.

### Deterministic core (Phase 1)

- `[V]` Decision fingerprints are deterministic under the documented
  canonicalization rules. Evidence: `tests/test_fingerprint.py`
  (`test_deterministic`, `test_stable_across_calls`), INV-07 and INV-12 in
  `docs/design-constitution.md`.
- `[V]` Context dict insertion order does not affect fingerprints. Evidence:
  `test_fingerprint_dict_order_independent_for_context` in
  `tests/test_decisions.py`, `test_dict_keys_sorted` in
  `tests/test_fingerprint.py`, INV-10.
- `[V]` Choice ordering affects `ChoiceDecision` semantics and its
  fingerprint. Evidence: `test_choice_order_preserved` and
  `test_fingerprint_order_sensitive_for_choices` in `tests/test_decisions.py`,
  INV-13.
- `[V]` Uncalibrated Phase 1 results cannot expose `predicted_correctness`.
  Evidence: `test_defaults_uncalibrated` and
  `test_predicted_correctness_requires_calibration` in `tests/test_results.py`,
  INV-04.
- `[V]` `NaN` and `Infinity` (in any sign, at any nesting depth) are rejected
  by fingerprinting. Evidence: `test_non_finite_floats_rejected` and
  `test_nan_in_nested_structure_rejected` in `tests/test_fingerprint.py`,
  INV-09.
- `[V]` `ChoiceResult.value` uses a deterministic first-in-order tie-break.
  Evidence: `test_tie_breaks_to_first_in_order` in `tests/test_results.py`,
  INV-03.
- `[V]` `BackendCapabilities` is explicit data and is never probed with
  `hasattr`. Evidence: `tests/test_capabilities.py`, INV-14.
- `[V]` The `Backend` protocol is unaware of decisions and results. Evidence:
  `test_backend_is_decision_agnostic` in `tests/test_plans.py`, INV-16.

### Bool vertical slice (Phase 2A)

- `[V]` `assemble_bool_probability` computes a numerically stable two-way
  softmax over exactly two labeled logits, equal to `sigmoid(l_true - l_false)`,
  with no overflow at extreme magnitudes. Evidence:
  `tests/test_assembler.py::TestSoftmaxMatrix::test_equal_logits_is_half`,
  `::test_ln3_gap_is_three_to_one`, `::test_large_equal_logits_no_overflow`,
  `::test_minus_thousand_true_gap`, `::test_plus_thousand_false_gap`,
  `::test_label_order_true_false_maps_by_label`, INV-01.
- `[V]` Assembled Bool results stay uncalibrated even at extreme logits:
  `calibrated = False` and `predicted_correctness = None`. Evidence:
  `tests/test_assembler.py::TestUncalibratedDefaults::test_extreme_logits_stay_uncalibrated`,
  INV-04.
- `[V]` A verbalizer that does not resolve to exactly one distinct scoring
  token raises `VerbalizerError`: no silent fallback, no truncation, no
  multi-token logit summing. Evidence:
  `tests/test_verbalizers.py::test_multi_token_positive_raises`,
  `::test_multi_token_negative_raises`, `::test_identical_token_for_both_raises`,
  `::test_zero_added_tokens_raises`, and
  `tests/test_backends_transformers.py::test_execute_rejects_multi_token_verbalizer_before_forward`
  (the error is raised before the model runs any forward pass).
- `[V]` `TransformersBackend` declares exactly one capability,
  `binary_token_logits=True`, and every other capability field False.
  Evidence: `tests/test_backends_transformers.py::test_capabilities_are_honest`,
  INV-14.
- `[V]` The backend reads only the final-position logits row. Evidence:
  `tests/test_backends_transformers.py::test_execute_reads_last_position_only`;
  the fake logits object in that test asserts the only permitted read is
  `logits[0, -1, :]`.
- `[V]` The compiler selects a strategy only from declared capabilities;
  insufficient capability raises `UnsupportedCapabilityError` before
  execution. Evidence:
  `tests/test_compiler.py::TestBoolCompilerCompile::test_missing_capability_rejected`,
  `tests/test_runtime.py::TestRuntimeRejections::test_missing_capability_rejected_before_execution`,
  INV-15.
- `[V]` `ChoiceDecision` has no runtime path: compiling or evaluating one
  raises `UnsupportedDecisionError` before execution. Evidence:
  `tests/test_compiler.py::TestBoolCompilerCompile::test_choice_decision_rejected`,
  `tests/test_runtime.py::TestRuntimeRejections::test_choice_decision_rejected_before_execution`.
- `[V]` `DecisionTrace.trace_id` is never derived from any fingerprint.
  Evidence:
  `tests/test_trace.py::TestTraceIdProvenance::test_trace_id_is_not_any_fingerprint`.
- `[V]` The `FuzzyAI` facade composes compile, execute, assemble, and trace in
  a fixed order and rejects evidence whose plan fingerprint does not match the
  plan it came from. Evidence:
  `tests/test_runtime.py::TestEndToEnd::test_evaluate_with_trace_full_pipeline`,
  `::test_trace_lineage_matches_decision_plan_and_evidence`,
  `tests/test_runtime.py::TestRuntimeRejections::test_foreign_plan_fingerprint_rejected`.
- `[V]` Rendered decision context never enters the system prompt; it is placed
  into the user prompt as evidence only. This is a structural placement rule,
  not a prompt-injection-safety claim. Evidence:
  `tests/test_doctrine.py::TestScoringDoctrineRender::test_malicious_context_never_enters_system_prompt`.
- `[V]` The trace's `input_fingerprint` describes the text actually rendered
  and fed to the model rather than the plan fields, and does not depend on
  whether the full rendered text was captured. Two different renderings of the
  same plan produce different fingerprints. Evidence:
  `TestInputFingerprintProvenance` in `tests/test_trace.py`
  (`test_uses_rendered_input_when_evidence_carries_it`,
  `test_falls_back_to_plan_fields_without_rendered_input`,
  `test_capture_flag_does_not_change_fingerprint`,
  `test_different_renderings_give_different_fingerprints`).
- `[V]` The runtime rejects evidence that carries no plan fingerprint, so
  lineage cannot be skipped. Evidence:
  `tests/test_runtime.py::TestRuntimeRejections::test_evidence_without_lineage_is_rejected`.
- `[V]` The transformers backend performs read-only scoring and never calls
  `model.generate()`. Evidence:
  `tests/test_backends_transformers.py::test_execute_never_calls_generate`
  (the fake model records the call and raises, so a regression fails loudly).

### Scoring validity and execution provenance (Phase 2A.1)

- `[V]` Restricted-probability diagnostics are computed from full-vocabulary
  normalization, not from the two candidate logits alone. Evidence:
  `tests/test_diagnostics.py` (35 tests: verbalizer-mass math, full-vocab
  probability, `diagnose_bool_evidence` field validation) and
  `tests/test_backends_transformers.py::test_metadata_carries_full_vocabulary_statistics`.
- `[V]` A `DecisionTrace` always carries `ScoringDiagnostics`; it cannot be
  constructed without them. Evidence:
  `tests/test_runtime.py::TestEndToEnd::test_trace_carries_scoring_diagnostics`.
- `[V]` The execution fingerprint is deterministic for the same configuration
  and changes when the rendered input, the rendering config, the model
  revision, or either resolved token id changes. Evidence:
  `tests/test_trace.py::TestExecutionFingerprint`
  (`test_same_config_same_fingerprint`,
  `test_different_rendered_input_different_fingerprint`,
  `test_different_rendering_config_different_fingerprint`,
  `test_different_model_revision_different_fingerprint`,
  `test_different_positive_token_id_different_fingerprint`,
  `test_different_negative_token_id_different_fingerprint`).
- `[V]` Trace id is per-execution identity, distinct from every fingerprint:
  the same execution fingerprint can carry different trace ids. Evidence:
  `tests/test_trace.py::TestExecutionFingerprint::test_same_execution_fingerprint_different_trace_id`
  and
  `tests/test_trace.py::TestExecutionFingerprint::test_distinct_from_plan_decision_and_input_fingerprints`.
- `[V]` `capture_rendered_input` does not change the input or execution
  fingerprint. Evidence:
  `tests/test_trace.py::TestExecutionFingerprint::test_capture_flag_does_not_change_either_fingerprint`.
- `[V]` One Bool evaluation performs exactly one model forward pass (no extra
  pass for diagnostics). Evidence:
  `tests/test_backends_transformers.py::test_single_forward_per_evaluation`.
- `[V]` Decoding the top token text is best-effort and never fails an
  inference. Evidence:
  `tests/test_backends_transformers.py::test_top_token_text_decode_failure_is_not_fatal`.

### Diagnostics clamp tolerance (Phase 2A.2)

- `[V]` `full_vocab_probability` and `verbalizer_mass` treat a small positive
  `logit - vocab_logsumexp` overshoot as floating-point rounding and clamp it
  to `1.0` under a *relative* bound (`_RELATIVE_OVERSHOOT_TOLERANCE = 1e-6`
  scaled by the operand magnitude), while a materially inconsistent normalizer
  still raises `InvalidProbabilityError`. Evidence:
  `tests/test_diagnostics.py::TestFullVocabProbability::test_rounding_level_overshoot_is_clamped`,
  `::test_measured_float32_overshoot_is_clamped` (pins the exact measured
  float32 overshoot `1.8553912184415822e-07` that the previous absolute `1e-9`
  bound wrongly rejected), and
  `::test_materially_inconsistent_normalizer_is_rejected`. The claim covers
  the deterministic clamp logic in `src/fuzzyai/diagnostics.py` only; it says
  nothing about any model.

### Direct categorical Choice (Phase 2B)

- `[V]` The N-way softmax over the declared candidate logits uses the
  max-subtraction form, so equal logits give a uniform distribution and very
  large or very negative offsets neither overflow nor collapse. Evidence:
  `tests/test_choice_assembler.py::TestNWaySoftmaxMath`
  (`test_equal_logits_are_uniform_thirds`, `test_ln2_gap_is_two_to_one_to_one`,
  `test_very_large_equal_logits_stay_uniform`, `test_one_dominant_logit_absorbs_mass`,
  `test_five_way_uniform`).
- `[V]` `ChoiceResult.probabilities` are keyed by semantic candidate names and
  never by scoring labels, under both the natural and a permuted label binding.
  Evidence: `tests/test_choice_assembler.py::TestSemanticKeying`
  (`test_probabilities_keyed_by_semantic_names_never_labels`,
  `test_semantic_mapping_follows_candidate_order_not_label_order`).
- `[V]` Choice ties resolve to the first candidate in semantic `ChoiceDecision`
  order, never to a scoring label, under both a natural and a permuted mapping.
  Evidence: `tests/test_choice_assembler.py::TestTieBreakFollowsSemanticOrder`.
- `[V]` `RawEvidence` labels must equal the plan's declared `targets` exactly and
  in order; permuted labels with otherwise legal values raise instead of being
  reordered. Evidence:
  `tests/test_choice_assembler.py::TestOrderedEvidenceEnforcement::test_permuted_labels_rejected_even_with_legal_pairs`,
  `tests/test_choice_trace.py::TestDiagnoseChoiceEvidenceGuards::test_permuted_labels_rejected`,
  and `tests/test_choice_runtime.py::TestChoiceEndToEnd::test_evidence_labels_are_execution_labels_not_semantics`.
- `[V]` `candidate_mass` is a full-vocabulary quantity: three candidate logits in
  a uniform four-token vocabulary give `0.75` and in a uniform five-token
  vocabulary give `0.6`, and mass falls as the vocabulary grows with the
  candidate set held fixed. Evidence:
  `tests/test_choice_assembler.py::TestCandidateMassMath`.
- `[V]` The compiler never touches a tokenizer, and the backend resolves and
  validates scoring labels in the real rendered continuation, rejecting a
  multi-token label or colliding label ids before any forward pass. Evidence:
  `tests/test_backends_choice.py::test_multi_token_label_raises_before_forward`
  and `tests/test_backends_choice.py::test_colliding_labels_raise`.
- `[V]` One Choice evaluation performs exactly one model forward pass and never
  calls `model.generate()`. Evidence:
  `tests/test_choice_runtime.py::TestChoiceEndToEnd::test_exactly_one_forward_pass`
  and `tests/test_backends_choice.py::test_metadata_carries_full_vocabulary_statistics`.
- `[V]` Changing the candidate-to-label assignment leaves the decision
  fingerprint unchanged and changes the plan fingerprint, and the prompt keeps
  semantic candidate order under permutation. Evidence:
  `tests/test_choice_compiler.py::TestMappingIdentity::test_permuted_mapping_keeps_decision_fingerprint_and_changes_plan_fingerprint`
  and
  `tests/test_choice_compiler.py::TestMappingIdentity::test_prompt_keeps_semantic_candidate_order_under_permutation`.
- `[V]` A categorical `DecisionTrace` requires the resolved scoring token ids,
  carries `probability_true = None`, and takes its execution fingerprint from
  the resolved token ids. Evidence:
  `tests/test_choice_trace.py::TestCategoricalTraceFields`,
  `tests/test_choice_trace.py::TestExecutionFingerprint::test_different_resolved_token_ids_change_execution_fingerprint`,
  and `tests/test_choice_trace.py::TestCategoricalTraceRejections::test_missing_resolved_ids_rejected`.
- `[V]` A `ChoiceDecision` evaluated against a backend that does not declare
  `categorical_token_logits` is rejected before execution with
  `UnsupportedCapabilityError` and no fallback strategy. Evidence:
  `tests/test_choice_runtime.py::TestChoiceDispatchRejections::test_missing_capability_rejected_before_execution`
  and `tests/test_choice_compiler.py::TestCompileRejections::test_missing_capability_rejected`.

## Experimental records

Single-session observations, each reported with the conditions under which it
was obtained. They are not quality, performance, or support claims, and none of
them generalises to other models, revisions, prompts, or tasks.

- `[E]` Rendering mode changes the decision distribution of the same
  `InferencePlan`. Model `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, dtype `float32`, on an NVIDIA
  GeForce RTX 5060 Laptop GPU (8123 MiB), run offline against the local cache
  through `examples/bool_local.py`. With the tokenizer's default chat-template
  mode the assistant turn starts in thinking mode, the model's next token is
  its reasoning opener at probability 0.9997, and both resolved verbalizer
  tokens (`yes` = 9693, `no` = 2152) sit at probability ~0, so the two-way
  softmax renormalizes tail noise: `P(True) = 0.3479` for "Did the customer's
  package fail to arrive?". With `chat_template_kwargs={"enable_thinking":
  False}` the verbalizers become the actual decision candidates and the same
  plan gives `P(True) = 0.9951`. Both values are uncalibrated. Neither is
  evidence of correctness; the difference records how much the probability
  depended on an unrecorded rendering choice.
- `[E]` One session of three Bool decisions on the same setup, thinking
  disabled: `P(True) = 0.9951` (123 input tokens, first call 1880.3 ms
  including model warm-up), `0.9926` (115 tokens, 51.5 ms), `0.9964` (116
  tokens, 51.9 ms). `calibrated` was `False` and `predicted_correctness` was
  `None` in all three. Each question is one a human would answer "yes", but
  this is not a correctness measurement: no ground truth was collected and
  three samples cannot support a quality claim.
- `[E]` Restricted probability versus verbalizer mass at a decision point.
  Model `Qwen/Qwen3-0.6B`, full precision (float32), `NVIDIA GeForce RTX 5060
  Laptop GPU` with 8123 MiB capacity, offline (`HF_HUB_OFFLINE=1`), scoring
  verbalizers `yes` / `no`. The restricted probability alone could not reveal
  that the model was not at a decision point, while `verbalizer_mass` could.
  Same plan, same question, only the chat-template rendering mode differed:

  | rendering | P(True) | verbalizer_mass | top token | top token probability |
  |---|---|---|---|---|
  | thinking disabled | 0.9988 | 0.954228 | 'yes' (id 9693) | 0.953119 |
  | thinking enabled | 0.5116 | 0.000000 | '<think>' (id 151667) | 0.999699 |

  This is a single-case observation about scoring position, not a quality
  or accuracy measurement. These probabilities are uncalibrated and were not
  evaluated against ground truth.
- `[E]` Six-case exploratory session (thinking disabled) on the same model and
  hardware. Recorded as observations only. All six cases resolved to top token
  'yes'; this is NOT an accuracy claim (one 0.6B model, six hand-written
  examples, no ground truth, no calibration):

  | case | P(True) | verbalizer_mass | top token prob | entropy | margin | input tokens | latency ms |
  |---|---|---|---|---|---|---|---|
  | obvious true | 0.9988 | 0.954228 | 0.953119 | 0.0130 | 0.9977 | 106 | 17781.4 (includes warm-up) |
  | obvious false | 0.9928 | 0.858544 | 0.852394 | 0.0613 | 0.9857 | 112 | 185.5 |
  | ambiguous | 0.9956 | 0.965834 | 0.961592 | 0.0407 | 0.9912 | 111 | 62.4 |
  | insufficient evidence | 0.9931 | 0.730371 | 0.725351 | 0.0593 | 0.9863 | 108 | 53.7 |
  | contradictory | 0.9949 | 0.900218 | 0.895594 | 0.0464 | 0.9897 | 121 | 52.6 |
  | prompt-injection-like | 0.9999 | 0.999018 | 0.998910 | 0.0016 | 0.9998 | 115 | 54.0 |

  In every case `calibrated=False`, `predicted_correctness=None`, and
  `method=binary_token_logits`. No conclusion is drawn about the model being
  right or wrong on any case.
- `[E]` Phase 2A.2 semantic signal validation, three-model sweep. An
  observation of signal behaviour under fixed conditions, NOT a quality,
  accuracy, reliability, or capability claim, and not a benchmark; the full
  record is `experiments/semantic_signal/REPORT.md`. Conditions: models
  `LiquidAI/LFM2.5-1.2B-Instruct` (1.17B), `openbmb/MiniCPM5-2B` (2.52B), and
  `Qwen/Qwen3.5-2B` (2.27B, scored through a harness-side text-tower adapter
  rather than its published multimodal entry point); model revision `None` in
  every run header (revision provenance could not be obtained from the local
  cache); hardware NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB), torch
  2.14.0+cu130, transformers 5.17.0, CUDA 13.0, run offline
  (`HF_HUB_OFFLINE=1`); dtype `bfloat16`, batch size 1, one forward pass per
  probe; 27 formulations (3 doctrines x 3 label families, with the
  order-ablation and mapping-swap variants) x 80 probes = 2160 probes per
  model, and each run finished with 2160 `ok` records. Case set
  `semantic-signal-v1` (fingerprint
  `b68a1d395f0ec8f2b1e1400cc2e4f3c4b0d7e5f8bdb6a80576c3a4bab2209805`): 8
  evidence ladders x 5 rungs, 12 polarity pairs, 4 contrast groups, 4
  injection probes, four hand-written themes (delivery, refund, subscription,
  software); no ground truth and no dataset. `calibrated` is `False` and
  `predicted_correctness` is `None` in every record; no threshold was applied
  and no probe was auto-rejected. `verbalizer_mass` is the full-vocabulary mass
  on the two declared candidate tokens; "low-mass" below means
  `verbalizer_mass < 0.5`, which is an experimental analysis cutoff used only
  to group these results, not a FuzzyAI threshold or API contract. Observed
  under exactly these conditions, generalising to none of them:

  - Scoring-position health: `LFM2.5-1.2B-Instruct` leaves the decision
    position under doctrine D3 `evidence-oriented-v1` for 462 / 720 probes
    (64.2%), `mass_min` 0.000008: its top token becomes a capitalized or
    unrelated variant (`Support`, `True`, `Yes`) rather than the declared
    lowercase label, so its `P(True)` there is renormalized tail noise.
    `MiniCPM5-2B` and `Qwen3.5-2B` show zero low-mass probes under all three
    doctrines (`mass_min` 0.916076 / 0.983116 / 0.817990 and 0.852574 /
    0.961845 / 0.918833 for D1 / D2 / D3). Whether a doctrine's wording keeps
    a model at the decision position is model-dependent, and
    `verbalizer_mass` is what makes such a failure visible instead of silent.
  - Ladder shape: the extremes separate in every model (`strong_positive` vs
    `strong_negative` mean `P(True)`: 0.999 vs 0.573 for LFM D1, 0.917 vs
    0.000 for MiniCPM D2, 0.892 vs 0.023 for Qwen D2), but the middle rungs
    do not order reliably: the `insufficient` rung lands below
    `weak_negative` in the aggregate of 6 of the 9 model x doctrine cells,
    and only 3 cells are fully monotonic in aggregate (MiniCPM D2, Qwen D1,
    Qwen D2). Even the best cell keeps per-ladder noise (Qwen D1: 2 of 8
    individual ladders out of order).
  - Contrast groups (one question, three contexts: `relevant_positive`,
    `irrelevant`, `relevant_negative`; 4 groups x 3 contexts per model,
    doctrine D2 `yes_no` canonical): MiniCPM5-2B and Qwen3.5-2B order all 4
    groups correctly; LFM2.5-1.2B does not discriminate relevance
    (irrelevant contexts still score 0.62 to 0.97, `relevant_negative` 0.53
    to 0.95). The report calls this the sharpest instrument of the round: a
    high `P(True)` is not by itself evidence that the model read the
    context.
  - Polarity pairs (12 pairs): canonical `yes_no` / `true_false` polarity
    consistency is 0.92 to 1.00 for all three models, with directional
    accuracy 0.75 to 1.00 across canonical formulations. The report notes a
    constant yes-bias could score well on this metric by accident, which is
    why the contrast groups carry more weight.
  - Label families: `yes_no` and `true_false` behave as near-interchangeable
    verbalizations at the aggregate level; `ab` is weak and model-dependent
    (directional accuracy 0.38 to 0.88 for LFM, 0.62 to 0.88 for Qwen, 0.50
    to 0.50 for MiniCPM). A mapping swap preserves the semantic direction
    only where the doctrine declares the label semantics (`ab`); for
    undeclared families it simply inverts the measurement (directional
    accuracy 0.00 to 0.38, polarity consistency 0.00 to 0.17).
  - Injection probes: on 4 hand-written probes (doctrine D2 `yes_no`
    canonical; sample size 4, no ground truth), LFM2.5-1.2B followed the
    embedded instruction against the evidence (0.9876, 0.9998, 0.9325,
    0.9770), while MiniCPM5-2B (0.0067, 0.2689, 0.8670, 0.0097) and
    Qwen3.5-2B (0.0373, 0.5622, 0.9149, 0.0474) followed the evidence. This
    is an observation about three small models on four probes. It is NOT a
    prompt-injection safety claim, and FuzzyAI makes no such claim anywhere.
  - Choice-readiness gate (report section 10), assessed per model: against
    the stated criteria the gate is met by a specific named configuration
    (MiniCPM5-2B or Qwen3.5-2B, `yes_no` or `true_false`, with a doctrine
    that keeps the model at the decision position), not by the mechanism in
    general; LFM2.5-1.2B fails the no-collapse and relevance-discrimination
    criteria. The three-way candidate space itself remains unmeasured.
- `[E]` Phase 2A.2 harness defect and fix, recorded as an observed
  engineering lesson. Conditions: same hardware, software, and sweeps as the
  entry above. The first pass of the sweeps lost 62 legitimate probe records
  (32 for LFM2.5-1.2B, 30 for MiniCPM5-2B) with
  `InvalidProbabilityError: vocab_logsumexp is inconsistent with the reported
  logit: logit - logsumexp = 1.8553912184415822e-07`. Cause: the clamp bound
  was absolute (`1e-9`), while a float32 logsumexp over a 130k-token
  vocabulary carries rounding error proportional to its own magnitude, so the
  bound sat below the noise floor. Fix: a relative tolerance (`1e-6` scaled
  by the operand magnitude) in `src/fuzzyai/diagnostics.py`, keeping the
  material-inconsistency rejection (tests cited under Verified claims). All
  three sweeps were re-run: 2160 `ok` records each, and every previously-`ok`
  record is bit-identical (2128 for LFM, 2130 for MiniCPM, all 2160 for
  Qwen), so the fix recovered exactly the rejected records and changed no
  measurement; the re-run doubles as a determinism check. Lesson: an
  absolute tolerance is the wrong shape for rounding error that scales with
  operand magnitude.

- `[E]` Direct categorical Choice distributions are representation-sensitive in
  shape while the winner stays stable. Model `Qwen/Qwen3.5-2B`, revision
  unrecorded (local cache), dtype `bfloat16`, on an NVIDIA GeForce RTX 5060
  Laptop GPU (8 GB), for a three-candidate billing/shipping/technical fixture
  under doctrine `categorical-semantic-judgment-v1` and label scheme
  `categorical-labels-v1`, all six label permutations, 90 evaluations. Semantic
  argmax preserved 90/90, mean total-variation distance 0.0334, maximum total
  variation 0.3082. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`,
  `experiments/choice_signal/REPORT.md`. Observed under exactly these
  conditions, generalising to none of them.
- `[E]` Adding an irrelevant candidate leaves the winner unchanged with small
  drift. Same model, dtype, GPU, doctrine and label scheme, 6 cases, adding an
  unnecessary `account deletion` candidate: winner preserved 6/6, total
  variation 0.0006 to 0.0153. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`.
- `[E]` Paraphrasing a candidate description leaves the winner unchanged. Same
  model, dtype, GPU, doctrine and label scheme, 5 cases, the billing description
  reworded with the candidate name unchanged: winner preserved 5/5, total
  variation 0.0002 to 0.0115. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`.
- `[E]` A direct categorical answer can be confidently in-set when the candidate
  set omits the true topic, and `candidate_mass` does not detect it. Same model,
  dtype, GPU, doctrine and label scheme, three out-of-set probes (account
  deletion, job application, sponsorship enquiry) against a
  billing/shipping/technical set: the model still chose an in-set candidate at
  0.677 to 0.893 while `candidate_mass` stayed near 0.99. This is why no
  open-set guarantee is claimed for Choice. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`.
- `[E]` Overlapping candidate descriptions split probability between the
  overlapping candidates. Same model, dtype, GPU, doctrine and label scheme, 3
  probes against a billing/`payment issue`/shipping/technical set: the two
  overlapping candidates held 0.943 and 0.970 of the restricted mass on the
  ambiguous probes while an unambiguous control stayed at 0.998. This
  characterises a bad taxonomy, not a model defect. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`.

## Current hypotheses

- `[H]` An explicit scoring doctrine may improve cross-model semantic
  stability of the same decision. The doctrine itself now exists as
  `BINARY_SEMANTIC_JUDGMENT_V1`; confirming or refuting the stability claim
  requires evaluation work that does not exist yet.
- `[H]` Small local models combined with task-specific calibration may be
  sufficient for useful narrow semantic decisions. Confirming or refuting
  this requires calibration and evaluation work that does not exist yet.
- `[H]` Cache-preserving compilation may materially reduce local inference
  cost when many decisions share a long context. Confirming or refuting this
  requires benchmark work on real backends that does not exist yet.

Hypotheses are recorded here so they can be tested, and must not be quoted as
features.

## Roadmap claims

Planned and unimplemented directions are tracked in
[the roadmap](./roadmap.md) and must not be restated here as capabilities.
They may carry `[R]` status in this register only once they have a named entry
in that roadmap.
