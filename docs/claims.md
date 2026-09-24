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
- `[V]` `BoolCompiler` and `ChoiceCompiler` are mutually exclusive: `BoolCompiler`
  rejects a `ChoiceDecision` and `ChoiceCompiler` rejects a `BoolDecision`, both
  with `UnsupportedDecisionError` before any execution. Evidence:
  `tests/test_compiler.py::TestBoolCompilerCompile::test_choice_decision_rejected`,
  `tests/test_choice_compiler.py::TestCompileRejections::test_bool_decision_rejected`.
- `[V]` `DecisionTrace.trace_id` is never derived from any fingerprint.
  Evidence:
  `tests/test_trace.py::TestTraceIdProvenance::test_trace_id_is_not_any_fingerprint`.
- `[V]` The `Probvenance` facade composes compile, execute, assemble, and trace in
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
  the deterministic clamp logic in `src/probvenance/diagnostics.py` only; it says
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
- `[V]` `scoring_label_mass` is a full-vocabulary quantity: three candidate logits in
  a uniform four-token vocabulary give `0.75` and in a uniform five-token
  vocabulary give `0.6`, and mass falls as the vocabulary grows with the
  candidate set held fixed. Evidence:
  `tests/test_choice_assembler.py::TestScoringLabelMassMath`.
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
- `[V]` The Choice diagnostics expose a scoring-label mass, not a semantic
  coverage measure. The public name is `scoring_label_mass`, it is documented and
  tested as `P(next token is one of the declared scoring-label tokens)` under
  full-vocabulary normalization, and the pre-2B.1 public names `candidate_mass`
  and `candidate_token_probabilities` no longer exist in any form: there is no
  alias and no deprecation shim. Evidence:
  `tests/test_public_api.py::test_renamed_scoring_label_names_present`,
  `tests/test_public_api.py::test_old_candidate_mass_names_gone`,
  `tests/test_choice_assembler.py::TestDiagnosticsReturned::test_renamed_fields_present_old_candidate_fields_gone`.

### Exact continuation and provenance (Phase 2C.0)
- `[V]` Single-token continuation requires exact preservation of the rendered
  prefix tokenization: a scoring label counts only when tokenizing the prefix
  plus the label reproduces the prefix token ids followed by exactly one
  additional token, so a net increase of one token is not sufficient. Evidence:
  `tests/test_verbalizers.py::test_exact_continuation_rejects_retokenized_prefix_when_net_delta_is_one`,
  `tests/test_backends_choice.py::test_retokenized_prefix_label_raises_before_forward`,
  `tests/test_backends_transformers.py::test_retokenized_prefix_verbalizer_raises_before_forward`.
- `[V]` Compiler and probability-assembler provenance are committed by the
  current Plan fingerprint schema and exposed on `DecisionTrace`:
  `compiler_id`, `compiler_version`, `assembler_id`, and `assembler_version`
  are explicit on `InferencePlan`, included in the plan fingerprint payload,
  and surfaced on the trace and its `to_dict()`. Current plan fingerprint
  schema version = 6. Evidence: `tests/test_plans.py::TestPlanProvenance`. The
  decision fingerprint is unaffected by construction rather than by test:
  `src/probvenance/decisions.py` never reads provenance, and both decision
  payloads remain v1 with no provenance keys.

### Execution-verified assembler provenance (Phase 2C.1)
- `[V]` The runtime executes only the probability assembler whose strategy,
  assembler identity, and assembler version exactly match the plan declaration.
  An unknown id, a known id paired with the wrong strategy, an unsupported
  version, and any plan whose declaration matches no implementation are all
  rejected before a probability result or trace is produced. This is NOT a claim
  that arbitrary third-party assemblers are supported: the set of
  implementations is closed, with no registry, plugin loading, or fallback.
  Evidence: `tests/test_assembler_dispatch.py`.
- `[V]` A decision trace records the plan-fingerprint schema version its
  `plan_fingerprint` was computed under (`plan_fingerprint_version`), taken from
  the plan's own schema version rather than hardcoded in the trace module, so a
  stored hash self-describes its version. Evidence:
  `tests/test_trace.py::TestPlanFingerprintVersion`,
  `tests/test_plans.py::TestPlanFingerprintVersion`.

### Probability formulation identity (Phase 2D)

- `[V]` Probvenance deterministically derives an exact probability formulation
  fingerprint that excludes instance evidence while committing the semantic
  outcome space, the scoring representation, the compiler identity and version,
  the doctrine identity AND version, the strategy, and the probability assembler
  identity and version. Both built-in Bool and Choice doctrines commit an
  explicit version: the version is a declared plan field and is never parsed out
  of the doctrine id string. Evidence:
  `tests/test_probability_identity.py::TestPayloadShape`,
  `tests/test_probability_identity.py::TestEvidenceExclusion`,
  `tests/test_probability_identity.py::TestExactFormulationBasics::test_doctrine_block_records_the_real_identity_and_version`,
  `tests/test_compiler.py::TestBoolCompilerInit::test_doctrine_version_comes_from_metadata_not_the_id`.
- `[V]` Formulation-family fingerprints intentionally collapse the
  representation-specific differences defined by the frozen family contract
  (candidate names, candidate descriptions, and the candidate-to-label
  assignment) while preserving mechanism-level dimensions including arity.
  Evidence:
  `tests/test_probability_identity.py::TestChoiceRepresentation`,
  `tests/test_probability_identity.py::TestBoolRepresentation`.

Same formulation identity concerns, and same formulation-family membership, are
NOT claims that the probabilities are accurate, interchangeable, poolable, or
calibratable together; see INV-23 and INV-24.

### Calibration data foundation (Phase 4A)

- `[V]` Calibration fitting datasets reject mixed exact bindings and
  non-fit-eligible observations. A `CalibrationDataset` refuses observations
  whose status is `taxonomy_miss` or `unresolved` or whose ground-truth
  provenance is not adjudicated, and refuses observations whose
  `CalibrationBinding` canonical payload differs from the dataset binding,
  even when the two traces share an identical formulation-family
  fingerprint; the rejection names the offending status or binding
  fingerprint and the excluded count, and nothing is silently filtered.
  Evidence: `tests/test_calibration.py::TestCalibrationDataset`
  (`test_taxonomy_miss_inclusion_rejected`,
  `test_unresolved_inclusion_rejected`,
  `test_unadjudicated_inclusion_rejected`, `test_mixed_binding_rejected`,
  `test_mixed_formulation_rejected_even_when_family_matches`,
  `test_rejection_names_count_and_reason`, `test_empty_dataset_rejected`).
- `[V]` A `CalibrationObservation`'s status and correctness are
  deterministically derived from the semantic result and the ground-truth
  record, not from caller-supplied confidence labels. The construction path
  accepts no `correct` argument (passing one raises `TypeError`), a resolved
  Bool ground truth must be a real `bool` (strings and numbers are never
  coerced), a resolved Choice ground truth outside the declared candidate
  set derives `taxonomy_miss` with `correct = None` and is retained rather
  than dropped, and `fit_eligible` consults only the derived status and the
  adjudication flag, never any scoring diagnostic or threshold. Evidence:
  `tests/test_calibration.py::TestObservationStatusAndCorrectness`
  (`test_bool_correct`, `test_bool_wrong`,
  `test_bool_ground_truth_must_be_real_bool`,
  `test_choice_correct_and_wrong_by_semantic_name`,
  `test_taxonomy_miss_retained_not_dropped`, `test_unresolved_truth`,
  `test_unadjudicated_truth_not_fit_eligible`,
  `test_caller_cannot_supply_correct`,
  `test_no_scoring_thresholds_in_fit_eligibility`).
- `[V]` Calibration observations are constructed only through supported
  construction paths that require matching runtime linkage identities between
  the result and the trace. `CalibrationObservation.from_evaluation` is the
  only supported construction path: it requires a non-null result linkage id
  matching the supplied trace linkage id and rejects mismatched pairs before
  calibration provenance is derived, it rejects a result whose trace id is
  missing, it rejects direct field construction, and it rejects
  `dataclasses.replace` reconstruction with or without changed fields. The
  binding is derived from the supplied trace and the probabilities and
  selection are derived from the supplied result. This is linkage
  CONSISTENCY, not content attestation: the check protects against accidental
  pairing of objects carrying different runtime linkage identities, and it
  does not prove that a caller-constructed result's probabilities were
  emitted by the supplied trace. Lower-level Python escape hatches such as
  `object.__new__`, `copy`, and `pickle` are not supported construction paths
  and are not defended against.
  Evidence:
  `tests/test_calibration.py::TestObservationProvenanceCoherence`
  (`test_mismatched_pair_rejected`,
  `test_mismatched_evaluation_wrapper_rejected`,
  `test_identical_probabilities_still_rejected`,
  `test_result_without_linkage_rejected`,
  `test_direct_constructor_rejected`,
  `test_replace_binding_rejected`,
  `test_replace_with_no_changes_rejected`,
  `test_replace_derived_field_rejected`,
  `test_construction_token_is_not_an_instance_attribute`).
- `[V]` A `CalibrationDataset` rejects fit-eligible observations whose
  derived `GroundTruthSemanticsIdentity` differs from the dataset's, even
  when the `CalibrationBinding` canonical payload matches: observations
  whose ground truth was established under a different labeling rule, a
  different ambiguity policy, or a different ground-truth taxonomy measure
  a different statistical target and are refused with a
  semantics-specific rejection message that is distinct from the binding
  mismatch message. Evidence: `tests/test_calibration.py`
  (`TestDatasetPoolingSemantics::test_same_binding_different_labeling_rule_rejected`,
  `TestDatasetPoolingSemantics::test_same_binding_different_ambiguity_policy_rejected`,
  `TestDatasetPoolingSemantics::test_same_binding_different_ground_truth_taxonomy_rejected`,
  `TestDatasetPoolingSemantics::test_rejection_messages_distinguish_binding_from_semantics`,
  `TestDatasetPoolingSemantics::test_direct_construction_enforces_same_checks`).
- `[V]` Different label sources may coexist in one `CalibrationDataset`
  when the ground-truth semantics identity matches: `label_source` is
  excluded from the semantics pooling key, so observations from different
  annotators with identical labeling semantics are accepted together while
  their observation fingerprints remain distinct because `label_source`
  stays in the observation identity. Evidence:
  `tests/test_calibration.py`
  (`TestDatasetPoolingSemantics::test_same_binding_same_semantics_different_label_source_accepted`,
  `TestObservationLineageUnderSemanticsSplit::test_different_label_source_different_observation_fingerprint`,
  `TestGroundTruthSemanticsIdentity::test_different_label_source_only_same_semantics`,
  `TestDatasetPoolingSemantics::test_label_source_pooling_and_fingerprint_distinction_combined`).

These claims are about the deterministic data model only. They say nothing
about the quality of any calibration produced from such data; no calibration
has been fitted, and no calibration-quality claim is made anywhere.

### Pre-calibration evaluation foundation (Phase 4A evaluation)

 - `[V]` `CalibrationEvaluationCohort` is the declared evaluation source
   cohort: the full set of observations supplied as one declared evaluation
   split BEFORE metric eligibility is applied. Admission is all-or-nothing
   with accumulated, distinct rejection messages: non-empty tuple container,
   every element a real `CalibrationObservation` (duck-typed look-alikes and
   non-tuple containers rejected), one `CalibrationBinding` canonical payload
   and one `GroundTruthSemanticsIdentity` enforced by canonical-JSON string
   equality (`label_source` differences alone are accepted; `labeling_rule`
   differences are rejected even when the binding matches), and valid split
   metadata (an `EvaluationSplitRole` `split_role`, a non-empty `split_id`).
   A cohort MAY contain fit-eligible resolved rows, taxonomy-miss rows,
   unresolved rows, and resolved-but-unadjudicated rows; those rows are
   retained, never rejected merely because they cannot enter a
   winner-correctness metric. The cohort fingerprint is version 1, row-order
   independent, multiplicity preserving, and commits the split metadata, the
   binding and ground-truth semantics identities with their payload versions,
   and the sorted fingerprints of ALL cohort rows (excluded rows are real
   rows in the identity, not just counters). Evidence:
   `tests/test_calibration_evaluation.py`
   (`TestEvaluationCohortAdmission`, `TestEvaluationCohortIdentity`).
 - `[V]` The cohort partition is deterministic and mutually exclusive with
   fixed precedence (`TAXONOMY_MISS` -> `taxonomy_miss`; `UNRESOLVED` ->
   `unresolved`; `RESOLVED` with `provenance.adjudicated is not True` ->
   `unadjudicated_resolved`; otherwise `eligible`), so
   `source_count == eligible_count + taxonomy_miss_count + unresolved_count +
   unadjudicated_resolved_count` with no overlapping counts and no
   double-counting (an unresolved row is counted exactly once, as unresolved,
   even when its provenance is also unadjudicated). A cohort with zero
   eligible rows is still a valid provenance artifact, but the metric-dataset
   projection fails clearly because a Brier score or a log loss cannot
   average zero evaluated rows. Evidence: `tests/test_calibration_evaluation.py`
   (`TestEvaluationCohortPartition`).
 - `[V]` `CalibrationEvaluationDataset` is the metric-eligible projection of
   exactly one declared `CalibrationEvaluationCohort`, and the only supported
   construction path is `CalibrationEvaluationDataset.from_cohort(cohort)`:
   direct construction is rejected with a construction token, so a caller
   cannot present an arbitrary already-filtered tuple as an evaluation
   dataset with no cohort provenance. The projection derives the eligible
   rows, the binding, the ground-truth semantics, the split metadata, the
   source cohort identity, and the exclusion accounting from the cohort.
   The dataset fingerprint is version 2 (bumped from version 1), row-order
   independent, multiplicity preserving, and commits the source cohort
   fingerprint and payload version, the split metadata, the binding and
   ground-truth semantics identities with their payload versions, the sorted
   eligible observation fingerprints, `source_count`, `eligible_count`, and
   the exclusion accounting. Two cohorts with identical scored rows but
   different taxonomy-miss, unresolved, or unadjudicated exclusions never
   collapse to the same evaluation dataset fingerprint. Evidence:
   `tests/test_calibration_evaluation.py`
   (`TestEvaluationDatasetProjection`, `TestEvaluationDatasetIdentity`,
   `TestCohortProvenanceNoConflation`).
 - `[V]` Metric exclusion is provenance: the decisive regression is two
   cohorts, A with 2 eligible rows and 0 taxonomy misses and B with the same
   2 eligible rows plus 3 taxonomy-miss rows under the same split metadata,
   binding, and ground-truth semantics. A and B have equal eligible
   observation fingerprints, equal metric `count`, and equal Brier and
   log-loss numeric values, but different cohort fingerprints, different
   evaluation dataset fingerprints, different Brier artifact fingerprints,
   different log-loss artifact fingerprints, different `source_count`, and
   different `taxonomy_miss_count`. Both `BrierEvaluationResult` and
   `LogLossEvaluationResult` (artifact fingerprint schema versions 3 and 3;
   Phase 4B moved them to 2 and the Phase 4C.0 target-identity migration moved
   them to 3) commit
   `source_cohort_fingerprint` and its payload version, `source_count`, and
   the exclusion accounting, while `count` remains the number of observations
   actually scored; the metric formulas and the metric versions (Brier
   version 1, log-loss version 1) and the input-score identity (version 1)
   are unchanged. This records the cohort supplied to the evaluation harness;
   it does not prove a caller did not discard rows before cohort
   construction. Evidence: `tests/test_calibration_evaluation.py`
   (`TestCohortProvenanceNoConflation`, `TestVersionGuards`,
   `TestLogLossVersionGuards`).
 - `[V]` The three datasets are distinct contracts and must not be conflated.
   A `CalibrationDataset` (the calibration FITTING dataset) REJECTS
   non-fit-eligible observations: it refuses a taxonomy miss, an unresolved
   ground truth, or an unadjudicated ground truth. A
   `CalibrationEvaluationCohort` (the declared evaluation SOURCE cohort) is
   the opposite contract: it RETAINS every declared source row, including
   taxonomy-miss, unresolved, and resolved-but-unadjudicated rows, and
   ACCOUNTS for them in the four-way partition and in the cohort fingerprint.
   A `CalibrationEvaluationDataset` is the deterministic METRIC-ELIGIBLE
   PROJECTION of exactly one cohort
   (`CalibrationEvaluationDataset.from_cohort(cohort)`): excluded rows are
   excluded from the projected evaluation rows, NOT rejected as source data
   and never silently dropped, and the projection keeps the source cohort
   identity and the exclusion accounting so the exclusion itself remains
   provenance. Structural projection rules are all-or-nothing over
   fit-eligible observations with accumulated, distinct rejection messages;
   binding and ground-truth semantics are enforced by canonical-JSON string
   equality of their identity payloads (`label_source` differences alone are
   accepted; `labeling_rule` differences are rejected even when the binding
   matches); invalid split metadata (a non-`EvaluationSplitRole` `split_role`,
   an empty `split_id`) is rejected. The dataset fingerprint is version 2,
   row-order independent, multiplicity preserving, and commits the declared
   `split_role` and `split_id` plus the source cohort identity and exclusion
   accounting. Evidence: `tests/test_calibration.py::TestCalibrationDataset`
   and `tests/test_calibration_evaluation.py`
   (`TestEvaluationDatasetProjection`, `TestEvaluationDatasetIdentity`).
 - `[V]` Brier evaluation computes `Brier = mean((p_i - y_i)^2)` with
   `math.fsum` accumulation, where `p_i` is the uncalibrated selected semantic
   probability extracted from the recorded `selected_value` by semantic name
   (the winner is never recomputed, tie-breaking is never re-run, and scoring
   labels or token ids are never used) and `y_i` is the derived
   winner-correctness label of the observation. Evidence:
   `tests/test_calibration_evaluation.py`
   (`TestBrierMatrix`, `TestSelectedProbabilityExtraction`).
 - `[V]` `BrierEvaluationResult` commits metric identity and version
    (`brier`, version 1), the shared versioned calibration target identity
   (`winner_correctness`, version 1),
    input-score identity and version
    (`uncalibrated-selected-probability`, version 1), the evaluation dataset
    fingerprint and payload version, the source cohort fingerprint and
    payload version, the source count and exclusion accounting, the constant
    `configuration == {}`, the evaluated `count`, and `value`; the
    fingerprint is the hash of `canonical_payload()`, and
    `evaluate_uncalibrated_winner_brier(dataset)` is the only supported
    construction path (direct construction and `dataclasses.replace(result)`
    raise `InvalidDecisionError`; `dataclasses.replace(result, value=...)` is
    rejected by `dataclasses` itself with a `ValueError`). Two evaluation
    datasets that accidentally produce the same Brier numeric value yield
    different result fingerprints. Evidence: `tests/test_calibration_evaluation.py`
    (`TestBrierEvaluationResult`, `TestVersionGuards`).
 - `[V]` Exact log-loss evaluation computes
   `log_loss = mean(-ln(p_i) if y_i = 1 else -ln(1 - p_i))` with `math.fsum`
   accumulation of the finite terms, where `p_i` and `y_i` are the same
   uncalibrated selected semantic probability and derived winner-correctness
   label used by Brier. The boundary policy is exact: there is no clipping, no
   epsilon, and no smoothing; a correct deterministic endpoint scores exactly
   `0`, an impossible observed endpoint scores positive infinity, and any
   infinite term makes the aggregate positive infinity. A representable
   probability arbitrarily close to 0 or 1 stays finite, and no supported path
   produces NaN or a negative value. Positive infinity is a legitimate metric
   value and is encoded structurally in the canonical payload
   (`{"kind": "positive_infinity", "number": null}`) so the result remains
   fingerprintable while canonical JSON stays free of non-finite numbers.
   Evidence: `tests/test_calibration_evaluation.py` (`TestLogLossFinite`,
   `TestLogLossExactEndpoints`, `TestLogLossNearBoundary`,
   `TestLogLossInfinityCanonicalization`).
  - `[V]` `LogLossEvaluationResult` commits metric identity and version
    (`log-loss`, version 1), the shared versioned calibration target identity
   (`winner_correctness`, version 1),
    input-score identity and version (`uncalibrated-selected-probability`,
    version 1), the evaluation dataset fingerprint and payload version, the
    source cohort fingerprint and payload version, the source count and
    exclusion accounting, the fixed configuration (`boundary_policy = exact`,
    `log_base = e`), the evaluated `count`, and the structurally encoded
    `value`;
   `evaluate_uncalibrated_winner_log_loss(dataset)` is the only supported
   construction path. The same `CalibrationEvaluationDataset` therefore yields
   equal evaluation-dataset fingerprint and version, equal input-score identity
   and version, and an equal target across Brier and log loss, while the metric
   identities and the artifact fingerprints differ. Evidence:
   `tests/test_calibration_evaluation.py` (`TestLogLossEvaluationResult`,
   `TestLogLossCrossMetricProvenance`, `TestLogLossVersionGuards`).
 - `[V]` The winner-correctness companion diagnostics artifact
   `WinnerCorrectnessDiagnosticsResult` derives exactly three quantities from
   the exact metric-eligible evaluation population, with no caller-supplied
   derived values: the empirical winner-correctness rate
   (`correct_count / count`, which on this binary target is also the ordinary
   decision accuracy, committed as ONE numeric field with no duplicate
   accuracy or base-rate identity), the mean selected probability
   (`math.fsum(p_i) / count` over the same uncalibrated selected semantic
   probabilities used by Brier and log loss, never labelled confidence or
   predicted correctness), and the empirical constant Brier reference
   (`math.fsum((q - y_i)^2) / count` with `q` the empirical correctness rate,
   equal to `q * (1 - q)` within floating-point evaluation). Excluded cohort
   rows never enter any denominator: two eligible correct rows plus eight
   taxonomy misses give `count = 2`, rate `1.0`, `source_count = 10`, and
   `taxonomy_miss_count = 8`, never a rate of `0.2`. Evidence:
   `tests/test_calibration_evaluation.py` (`TestWinnerDiagnosticsAccuracy`,
   `TestWinnerDiagnosticsMeanSelectedProbability`,
   `TestWinnerDiagnosticsConstantReference`,
   `TestWinnerDiagnosticsExclusionProvenance`).
 - `[V]` `WinnerCorrectnessDiagnosticsResult` (fingerprint version 1) commits
   the same dataset, cohort, exclusion, and input-score provenance as Brier
   and exact log loss: the evaluation dataset fingerprint and payload version,
   the source cohort fingerprint and payload version, `source_count`, the
   evaluated `count`, the three exclusion counts, the target
   (`winner_correctness`), and the input-score identity and version
   (`uncalibrated-selected-probability`, version 1), plus `correct_count`,
   `incorrect_count`, and each diagnostic's identity and version. On one
   dataset the three evaluators agree on all of that provenance while their
   artifact fingerprints all differ. The only supported construction path is
   `evaluate_uncalibrated_winner_diagnostics(dataset)` (direct construction
   and `dataclasses.replace(result)` raise `InvalidDecisionError`;
   `dataclasses.replace(result, empirical_correctness_rate=...)` is rejected
   by `dataclasses` itself with a `ValueError`). Identical numeric diagnostics
   from cohorts with different exclusions never collapse to the same
   diagnostics fingerprint. Evidence: `tests/test_calibration_evaluation.py`
    (`TestWinnerDiagnosticsConstruction`,
    `TestWinnerDiagnosticsCrossArtifactAlignment`, `TestWinnerDiagnosticsIdentity`,
    `TestWinnerDiagnosticsVersionGuards`).
  - `[V]` The equal-width reliability binning deterministically partitions the
    exact metric-eligible population into `bin_count` regions with the frozen
    interval contract (`bin 0` owns `[0/B, 1/B)`, bin `i` owns
    `[i/B, (i+1)/B)`, the last bin owns `[(B-1)/B, 1]` with an inclusive upper
    bound), so `p = 0` falls in the first bin, `p = 1` falls in the last bin,
    and an exact interior boundary `p = i/B` belongs to bin `i`, never to bin
    `i-1`. Boundary ownership is decided by comparing the stored probability
    value against the mathematical rational boundaries `i/B` via exact
    `fractions.Fraction.from_float` comparison, never by binary floating
    multiplication; `math.nextafter` probes just below and just above each
    tested boundary land in the lower and upper bin respectively, and a
    rational boundary with no exact float representation (such as `1/3`) is
    decided by that exact rational comparison. `bin_count` must be a real
    `int` with `bin_count >= 1` (`True`, `False`, `0`, negatives, floats,
    strings, and `None` are rejected with `InvalidDecisionError`), and there
    is no hidden maximum. Evidence: `tests/test_calibration_evaluation.py`
    (`TestWinnerReliabilityCore`, `TestWinnerReliabilityBoundaries`,
    `TestWinnerReliabilityConstruction`).
  - `[V]` Reliability artifacts retain empty bins and commit dataset, cohort,
    exclusion, input-score, binning, and per-bin membership provenance:
    `len(bins) == bin_count` always holds, an empty bin carries `count = 0`,
    `correct_count = 0`, both statistics as `null` (never NaN, never a fake
    `0.0`, so an empty interval is distinguishable from an observed rate of
    zero), and no members; each non-empty bin carries the mean selected
    probability (`math.fsum` over deterministically sorted member
    probabilities, so dataset row order cannot change a bin mean), the
    empirical correctness rate, and the sorted, multiplicity-preserving
    member observation fingerprints (a duplicated row appears twice, never
    deduplicated). `WinnerReliabilityResult` (fingerprint version 1,
    reliability id `winner-reliability-curve` version 1, binning id
    `equal-width` version 1) commits the evaluation dataset fingerprint and
    payload version, the source cohort fingerprint and payload version,
    `source_count`, the evaluated `count`, the three exclusion counts, the
    target (`winner_correctness`), the input-score identity and version
    (`uncalibrated-selected-probability`, version 1), and the binning
    configuration (`bin_count`); the only supported construction path is
    `evaluate_uncalibrated_winner_reliability(dataset, bin_count=...)`
    (direct construction and `dataclasses.replace(result)` raise
    `InvalidDecisionError`; `dataclasses.replace(result, bins=...)` is
    rejected by `dataclasses` itself with a `ValueError`). On one dataset the
    four evaluators (Brier, log loss, diagnostics, reliability) agree on all
    shared provenance while their artifact fingerprints all differ; different
    `bin_count` values on the same dataset produce different canonical
    payloads and fingerprints; identical bin numerics from cohorts with
    different exclusions never collapse to the same reliability fingerprint;
    and the per-bin totals align with the diagnostics artifact (`sum(bin.count)
    == count`, `sum(bin.correct_count) == correct_count`, the count-weighted
    bin mean matches the diagnostics mean selected probability, and the
    correctness totals match the empirical correctness rate). No per-bin gap,
    calibration-error, or ECE quantity exists in the artifact, and the
    summary is a pre-calibration description of raw-score regions, not
    calibration evidence. Evidence: `tests/test_calibration_evaluation.py`
    (`TestWinnerReliabilityEmptyBins`, `TestWinnerReliabilityGlobalAlignment`,
    `TestWinnerReliabilityCrossArtifactProvenance`,
    `TestWinnerReliabilityIdentity`, `TestWinnerReliabilityNoConflation`,
    `TestWinnerReliabilityVersionGuards`).
  - `[V]` The equal-width binned absolute-gap result is derived from one
    exact `WinnerReliabilityResult` and computes the documented
    sample-weighted absolute difference between each non-empty bin's mean
    selected probability and empirical winner-correctness rate:
    `evaluate_winner_binned_absolute_gap(reliability)` (aggregate id
    `winner-correctness-equal-width-binned-absolute-gap` version 1, result
    fingerprint version 1) never re-bins observations, never re-extracts
    probabilities, never re-assigns bin indices, and never re-implements
    boundary logic; it takes no `bin_count` argument because the source
    reliability partition is accepted as truth, and it accepts only a real
    `WinnerReliabilityResult` (`None`, a mapping, a duck-typed look-alike, a
    `WinnerCorrectnessDiagnosticsResult`, and a `CalibrationEvaluationDataset`
    all fail closed with `InvalidDecisionError`). Empty bins contribute zero
    mass: they are skipped, their `null` statistics are never treated as `0`,
    no fake observed gap is constructed, and an empty bin is not an error.
    The value is `math.fsum` over `(count_b / count) * abs(mean_b - rate_b)`
    for non-empty bins in fixed bin-index order, guaranteed finite and within
    `[0, 1]`; a non-finite statistic, a negative weighted term, or an
    out-of-range aggregate fails closed with `InvalidDecisionError`. The
    artifact commits the source reliability fingerprint and payload schema
    version, the reliability and binning identities with versions and the bin
    configuration, the evaluation dataset and source cohort fingerprints with
    their payload versions, `source_count`, the evaluated `count`, the three
    exclusion counts, the target, the input-score identity, and the value; the
    only supported construction path is
    `evaluate_winner_binned_absolute_gap(reliability)` (direct construction
    and `dataclasses.replace(result)` raise `InvalidDecisionError`;
    `dataclasses.replace(result, value=...)` is rejected by `dataclasses`
    itself with a `ValueError`). Identity regressions hold: the same
    reliability gives a deterministic identical fingerprint; the same numeric
    value from different reliability artifacts gives different aggregate
    fingerprints (numeric equality is not provenance identity); `B = 5` versus
    `B = 10` on the same dataset give different reliability and aggregate
    fingerprints; and identical bin numerics from cohorts with different
    exclusions may match numerically while differing in artifact identity with
    `source_count` and the exclusion counts preserved. Evidence:
    `tests/test_calibration_evaluation.py` (`TestWinnerBinnedAbsoluteGapFormula`,
    `TestWinnerBinnedAbsoluteGapIdentity`,
    `TestWinnerBinnedAbsoluteGapConstruction`,
    `TestWinnerBinnedAbsoluteGapNoConflation`,
    `TestWinnerBinnedAbsoluteGapVersionGuards`). This is NOT a claim that the
    raw selected semantic probability is calibrated, and NOT a claim that the
    number establishes model calibration quality: the formula equals the
    conventional equal-width ECE estimator form, but the canonical semantics
    are the binned absolute-gap diagnostic, the absolute value carries no
    direction, and the ordinary calibration-error interpretation is not
    granted.

 - `[V]` All winner-correctness evaluation artifacts commit ONE shared,
   explicit, versioned calibration target identity
   (`WINNER_CORRECTNESS_TARGET_ID = "winner_correctness"`, version 1). The
   target is no longer owned by Brier: `BrierEvaluationResult`,
   `LogLossEvaluationResult`, `WinnerCorrectnessDiagnosticsResult`,
   `WinnerReliabilityResult`, and `WinnerBinnedAbsoluteGapResult` all report
   the same `target_id` and the same `target_version` when computed over one
   evaluation population, and the target is committed exactly once per
   artifact payload as a nested `{"target_id", "target_version"}` object with
   no separate top-level identity key and no leftover bare unversioned target
   string. Because the target identity became an explicit versioned identity,
   the five artifact payload schema versions moved to Brier 3, log loss 3,
   diagnostics 2, reliability 2, and binned absolute gap 2, while the metric,
   reliability, binning, input-score, and aggregate semantic versions and
   every cohort and dataset identity stayed unchanged: only the artifact
   identity schema moved, not any mathematics. Evidence:
   `tests/test_calibration_evaluation.py::TestSharedCalibrationTargetIdentity`
   and the five version-guard classes.

These claims are deterministic implementation claims about the evaluation
foundation only. They are NOT claims that any model is calibrated, that any
calibration improves anything, or that any pre-calibration baseline is
calibration-quality evidence: no calibrator exists, no fitting has happened,
and no fitted `CalibrationProfile` has ever been produced. The profile
identity/artifact foundation exists in `src/probvenance/calibration.py`, but
nothing in the runtime or the evaluation layer creates one.

### Calibration profile identity foundation (Phase 4C.1)

- [V] `CalibrationProfile` identity commits the exact `CalibrationBinding`
  identity, the exact `GroundTruthSemanticsIdentity`, the winner-correctness
  target ID and version (`winner_correctness` v1), the selected-probability
  input-score ID and version (`uncalibrated-selected-probability` v1), the
  method ID and version, the method configuration, the fitted parameters, and
  the training `CalibrationDataset` fingerprint and schema version. The binding
  and the ground-truth semantics are composed by fingerprint plus schema
  version only; their constituent fields are deliberately not duplicated into
  the profile payload. Evidence:
  `tests/test_calibration_profile.py::TestProfileIdentityDurability`,
  `...::TestCanonicalPayloadShape`.

- [V] An explicit concrete taxonomy contradiction is rejected when an internal
  fitted profile artifact is constructed: a concrete binding taxonomy ID that
  differs from a concrete ground-truth taxonomy ID fails closed, and the same
  concrete taxonomy ID carrying two concrete but different versions fails
  closed with a distinguishable message. An unknown (`None`) taxonomy on either
  side is allowed and is never invented as equal. Evidence:
  `tests/test_calibration_profile.py::TestTaxonomyPrecondition`.

- [V] Profile exact binding matching rejects a supplied mismatched binding and
  provides no formulation-family, model-only, task-only, taxonomy-only, or
  nearest fallback. Evidence:
  `tests/test_calibration_profile.py::TestRequireBindingMatch`.

- [V] The calibrator input-score identity
  (`UNCALIBRATED_SELECTED_PROBABILITY_ID` / `UNCALIBRATED_SELECTED_PROBABILITY_VERSION`)
  is owned by the calibration foundation rather than the evaluation layer; the
  evaluation layer imports and re-exports it under the same names, moving its
  ownership changed no evaluation payload and no evaluation fingerprint, and
  `probvenance.calibration` never imports `probvenance.calibration_evaluation`.
  Evidence:
  `tests/test_calibration_profile.py::TestPhase4BFingerprintsUnchanged`,
  `...::TestImportDirection`.

These are structural identity claims. A valid `CalibrationProfile` proves only
that its provenance and identity are structurally coherent. It does NOT prove
that the calibrator improves Brier or log loss, that it generalizes beyond its
training data, that the training data is representative, or that a
training/evaluation split is independent. No fitting algorithm exists, no
supported public fitter produces a profile, and no result carries a
`predicted_correctness`.

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
  to group these results, not a Probvenance threshold or API contract. Observed
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
    prompt-injection safety claim, and Probvenance makes no such claim anywhere.
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
  by the operand magnitude) in `src/probvenance/diagnostics.py`, keeping the
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
  set omits the true topic, and `scoring_label_mass` does not detect it. Same model,
  dtype, GPU, doctrine and label scheme, three out-of-set probes (account
  deletion, job application, sponsorship enquiry) against a
  billing/shipping/technical set: the model still chose an in-set candidate at
  0.677 to 0.893 while `scoring_label_mass` stayed near 0.99. This is why no
  open-set guarantee is claimed for Choice. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`.
- `[E]` Overlapping candidate descriptions split probability between the
  overlapping candidates. Same model, dtype, GPU, doctrine and label scheme, 3
  probes against a billing/`payment issue`/shipping/technical set: the two
  overlapping candidates held 0.943 and 0.970 of the restricted mass on the
  ambiguous probes while an unambiguous control stayed at 0.998. This
  characterises a bad taxonomy, not a model defect. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`.
- `[E]` Representation sensitivity of the direct categorical distribution
  replicates on a second model family. Models `Qwen/Qwen3.5-2B` and
  `openbmb/MiniCPM5-2B`, both with revision unrecorded (local cache), dtype
  `bfloat16`, rendering config `{"enable_thinking": false}`, on an NVIDIA GeForce
  RTX 5060 Laptop GPU (8 GB), same fixture, doctrine
  `categorical-semantic-judgment-v1`, label scheme `categorical-labels-v1`. At N=3
  over all six label permutations: winner preserved 90/90 (mean total variation
  0.0334, max 0.3082) for Qwen3.5-2B and 85/90 (mean 0.0514, max 0.4672) for
  MiniCPM5-2B. At N=5 over five permutations: 75/75 (mean 0.0277, max 0.2937) and
  74/75 (mean 0.0381, max 0.5746). So the winner is largely stable on both while
  the full uncalibrated distribution is representation-sensitive on both.
  Evidence: `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`,
  `experiments/choice_signal/results/choice-signal-minicpm5-2b.jsonl`,
  `experiments/choice_signal/REPORT.md`. Observed under exactly these conditions,
  generalising to none of them.
- `[E]` Out-of-set confidence with an undetectably high `scoring_label_mass`
  replicates on the second family. `openbmb/MiniCPM5-2B`, same conditions, three
  out-of-set probes (account deletion, job application, sponsorship enquiry)
  against a billing/shipping/technical set: the model chose an in-set candidate
  at 0.678 to 0.754 while `scoring_label_mass` stayed between 0.9957 and 0.9968.
  This matches the Qwen3.5-2B observation and supports the reading that
  `scoring_label_mass` diagnoses protocol adherence, not semantic coverage.
  Evidence: `experiments/choice_signal/results/choice-signal-minicpm5-2b.jsonl`.
- `[E]` `scoring_label_mass` detects a model that never occupied the scoring
  position. `openbmb/MiniCPM5-2B` with no rendering flag passed (the backend
  default, effectively `{}`) put probability 1.0000 (rounded; minimum 0.999998)
  on `<think>` in 193/193 records and drove
  `scoring_label_mass` down to between 4.31e-17 and 7.13e-12 (median 2.65e-14),
  while the restricted three-way softmax still returned normalised-looking
  numbers. With `{"enable_thinking": false}` the same fixture returned masses
  between 0.9649 and 1.0000. The first artifact is retained as evidence that
  without the mass diagnostic this would have been misreported as extreme
  representation instability. Evidence:
  `experiments/choice_signal/results/choice-signal-minicpm5-2b-thinking-on.jsonl`.
- `[E]` The 2B.1 rename was value-preserving. The `Qwen/Qwen3.5-2B` artifact was
  re-run under the renamed code and compared against the migrated pre-rename
  copy (`compare_pre_rename` in the analysis script, which reads the committed
  artifact via `git show`): across 177 distinct `(case_id, permutation)` pairs, 0
  probability dicts, 0 `scoring_label_mass` values, 0 resolved token id maps and 0
  plan fingerprints differed. The execution fingerprint did differ on all 193
  records, because `rendering_config` is now recorded and applied; that is a
  provenance change, not a value change. Evidence:
  `experiments/choice_signal/results/choice-signal-qwen35-2b.jsonl`,
  `experiments/choice_signal/cross_model.py`.

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
