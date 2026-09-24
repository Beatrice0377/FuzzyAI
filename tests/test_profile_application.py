"""Tests for offline profile application and post-calibration evaluation.

Covers the Phase 4C.3 flow: one exact ``CalibrationProfile`` is applied offline
to one compatible ``CalibrationEvaluationDataset`` to produce an immutable
``ProfileAppliedEvaluationDataset``, and the five post-calibration evaluators
consume that artifact.

The decisive property under test is that pre-calibration and post-calibration
evaluation share the same mathematics and the same evaluation population and
differ ONLY in the committed input-score identity and the recorded profile
provenance. Nothing here mutates a runtime result, and no test loads a model.
"""

import ast
import inspect
import math
import textwrap
from dataclasses import replace
from typing import Any

import pytest
from test_calibration import (
    bool_observation,
    choice_observation,
    make_bool_decision,
    make_choice_runtime,
    resolved_truth,
    unresolved_truth,
)
from test_calibration_fitter import bool_observation_at, fit

from probvenance import InvalidDecisionError
from probvenance import calibration as calibration_module
from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID,
    L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION,
    PREDICTED_WINNER_CORRECTNESS_ID,
    PREDICTED_WINNER_CORRECTNESS_VERSION,
    UNCALIBRATED_SELECTED_PROBABILITY_ID,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    WINNER_CORRECTNESS_TARGET_ID,
    WINNER_CORRECTNESS_TARGET_VERSION,
    CalibrationDataset,
    CalibrationProfile,
    fit_l2_logistic_selected_probability,
    predicted_winner_correctness,
)
from probvenance.calibration_evaluation import (
    _PROFILE_APPLIED_EVALUATION_ROW_CONSTRUCTION_TOKEN,
    BRIER_METRIC_ID,
    BRIER_METRIC_VERSION,
    CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
    EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID,
    EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION,
    EMPIRICAL_CORRECTNESS_RATE_ID,
    EMPIRICAL_CORRECTNESS_RATE_VERSION,
    EQUAL_WIDTH_BINNING_ID,
    EQUAL_WIDTH_BINNING_VERSION,
    LOG_LOSS_METRIC_ID,
    LOG_LOSS_METRIC_VERSION,
    MEAN_PREDICTED_CORRECTNESS_ID,
    MEAN_PREDICTED_CORRECTNESS_VERSION,
    MEAN_SELECTED_PROBABILITY_ID,
    POST_CALIBRATION_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_BRIER_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_DIAGNOSTICS_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_LOG_LOSS_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_RELIABILITY_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_WINNER_BINNED_ABSOLUTE_GAP_ID,
    POST_CALIBRATION_WINNER_BINNED_ABSOLUTE_GAP_VERSION,
    POST_CALIBRATION_WINNER_RELIABILITY_ID,
    POST_CALIBRATION_WINNER_RELIABILITY_VERSION,
    PROFILE_APPLIED_EVALUATION_DATASET_FINGERPRINT_VERSION,
    WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION,
    WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION,
    WINNER_RELIABILITY_CURVE_ID,
    WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION,
    CalibrationEvaluationCohort,
    CalibrationEvaluationDataset,
    EvaluationSplitRole,
    ProfileAppliedEvaluationDataset,
    ProfileAppliedEvaluationRow,
    apply_profile_to_evaluation_dataset,
    evaluate_post_calibration_winner_binned_absolute_gap,
    evaluate_post_calibration_winner_brier,
    evaluate_post_calibration_winner_diagnostics,
    evaluate_post_calibration_winner_log_loss,
    evaluate_post_calibration_winner_reliability,
    evaluate_uncalibrated_winner_brier,
    evaluate_uncalibrated_winner_diagnostics,
    evaluate_uncalibrated_winner_log_loss,
    evaluate_uncalibrated_winner_reliability,
    evaluate_winner_binned_absolute_gap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def body_of(function: Any) -> str:
    """Function source with its docstring removed.

    Scope checks assert a specific call does NOT appear in an implementation.
    Prose in a docstring must not make such a check pass or fail, so the
    docstring is stripped through the AST before inspecting.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = node.body[1:]
    return ast.unparse(node)


def evaluation_dataset(
    rows: list[tuple[float, bool]],
    *,
    split_id: str = "eval-1",
    split_role: EvaluationSplitRole = EvaluationSplitRole.VALIDATION,
    extra_excluded: int = 0,
) -> CalibrationEvaluationDataset:
    """One declared evaluation dataset from ``(selected probability, correct)`` rows.

    ``extra_excluded`` appends unresolved observations that the cohort RETAINS
    while the metric-eligible projection EXCLUDES them, which is how the
    provenance attacks add source rows without changing the scored rows.
    """
    observations = [bool_observation_at(p, correct) for p, correct in rows]
    observations.extend(bool_observation(unresolved_truth()) for _ in range(extra_excluded))
    cohort = CalibrationEvaluationCohort(tuple(observations), split_role, split_id)
    return CalibrationEvaluationDataset.from_cohort(cohort)


def training_profile(
    rows: list[tuple[float, bool]], *, l2_strength: float = 0.05
) -> CalibrationProfile:
    return fit(rows, l2_strength)


def training_dataset_for(rows: list[tuple[float, bool]]) -> CalibrationDataset:
    return CalibrationDataset.create([bool_observation_at(p, c) for p, c in rows])


def internal_profile(
    dataset: CalibrationDataset,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
    l2_strength: float = 1.0,
    method_id: str = L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID,
    method_version: int = L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION,
    fitted_parameters: dict[str, Any] | None = None,
) -> CalibrationProfile:
    """A profile built through the internal producer for boundary probes."""
    return CalibrationProfile._from_fitted_state(
        dataset,
        method_id=method_id,
        method_version=method_version,
        method_configuration=calibration_module._l2_logistic_method_configuration(l2_strength),
        fitted_parameters=(
            fitted_parameters
            if fitted_parameters is not None
            else {"slope": slope, "intercept": intercept}
        ),
    )


def forge_profile(profile: CalibrationProfile, **overrides: Any) -> CalibrationProfile:
    """Rebuild a profile with field values replaced, bypassing validation.

    Used only to probe that the APPLICATION layer fails closed on a
    structurally forged profile the constructor would already have rejected.
    """
    forged = object.__new__(CalibrationProfile)
    for field in CalibrationProfile.__slots__:
        object.__setattr__(forged, field, overrides.get(field, getattr(profile, field)))
    for name, value in overrides.items():
        assert getattr(forged, name) is value or getattr(forged, name) == value
    return forged


def binding_mismatched_dataset() -> CalibrationEvaluationDataset:
    """One evaluation dataset whose binding differs from a Bool profile's."""
    cohort = CalibrationEvaluationCohort(
        (choice_observation(),), EvaluationSplitRole.VALIDATION, "eval-choice"
    )
    return CalibrationEvaluationDataset.from_cohort(cohort)


# ---------------------------------------------------------------------------
# Predicted-correctness identity (PART 1 to PART 3, PART 24)
# ---------------------------------------------------------------------------


class TestPredictedCorrectnessIdentity:
    def test_identity_is_versioned_and_owned_by_the_foundation(self) -> None:
        assert PREDICTED_WINNER_CORRECTNESS_ID == "predicted-winner-correctness"
        assert PREDICTED_WINNER_CORRECTNESS_VERSION == 1
        assert calibration_module.PREDICTED_WINNER_CORRECTNESS_ID == PREDICTED_WINNER_CORRECTNESS_ID

    def test_post_identity_differs_from_the_uncalibrated_identity(self) -> None:
        assert PREDICTED_WINNER_CORRECTNESS_ID != UNCALIBRATED_SELECTED_PROBABILITY_ID
        assert UNCALIBRATED_SELECTED_PROBABILITY_VERSION == 1

    def test_scoring_matches_the_declared_mapping(self) -> None:
        profile = training_profile([(0.6, True), (0.6, False), (0.9, True), (0.9, True)])
        observation = bool_observation_at(0.9, True)
        slope, intercept = calibration_module._require_l2_logistic_fitted_parameters(
            profile.fitted_parameters
        )
        expected = calibration_module._stable_sigmoid(
            slope * calibration_module._selected_probability(observation) + intercept
        )
        assert predicted_winner_correctness(profile, observation) == expected

    def test_scoring_does_not_refit(self) -> None:
        body = body_of(predicted_winner_correctness)
        assert "_solve_l2_logistic" not in body
        assert "fit_l2_logistic_selected_probability" not in body
        assert "l2_strength" not in body


# ---------------------------------------------------------------------------
# Application artifact (PART 5 to PART 10, PART 20 to PART 23)
# ---------------------------------------------------------------------------


class TestApplicationArtifact:
    def test_application_produces_one_row_per_observation(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        assert isinstance(applied, ProfileAppliedEvaluationDataset)
        assert applied.count == 3
        assert {row.observation_fingerprint for row in applied.rows} == {
            observation.fingerprint for observation in dataset.observations
        }

    def test_direct_construction_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            ProfileAppliedEvaluationDataset(  # type: ignore[call-arg]
                (), EvaluationSplitRole.VALIDATION, "x"
            )

    def test_replace_cannot_forge_an_artifact(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        with pytest.raises(InvalidDecisionError):
            replace(applied, split_id="forged")

    def test_row_order_is_irrelevant_and_multiplicity_preserved(self) -> None:
        rows = [(0.6, True), (0.9, False), (0.8, True), (0.6, True)]
        profile = training_profile([(0.7, True), (0.7, False)])
        forward = apply_profile_to_evaluation_dataset(evaluation_dataset(rows), profile)
        backward = apply_profile_to_evaluation_dataset(
            evaluation_dataset(list(reversed(rows))), profile
        )
        assert forward.fingerprint == backward.fingerprint
        assert forward.count == 4

    def test_application_identity_distinguishes_profiles(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        training = training_dataset_for([(0.7, True), (0.7, False)])
        first = internal_profile(training, slope=1.0, intercept=0.0, l2_strength=1.0)
        second = internal_profile(training, slope=1.0, intercept=0.0, l2_strength=2.0)
        assert first.fingerprint != second.fingerprint
        first_applied = apply_profile_to_evaluation_dataset(dataset, first)
        second_applied = apply_profile_to_evaluation_dataset(dataset, second)
        assert first_applied.rows == second_applied.rows
        assert first_applied.fingerprint != second_applied.fingerprint

    def test_application_identity_distinguishes_excluded_source_rows(self) -> None:
        profile = training_profile([(0.7, True)])
        plain = evaluation_dataset([(0.6, True), (0.9, False)])
        extended = evaluation_dataset([(0.6, True), (0.9, False)], extra_excluded=2)
        # Excluded source rows never enter the metric-eligible projection, so
        # the scored observations are identical while the identities differ.
        assert plain.observations == extended.observations
        assert plain.fingerprint != extended.fingerprint
        assert plain.source_cohort_fingerprint != extended.source_cohort_fingerprint
        plain_applied = apply_profile_to_evaluation_dataset(plain, profile)
        extended_applied = apply_profile_to_evaluation_dataset(extended, profile)
        assert plain_applied.count == extended_applied.count == 2
        assert [r.predicted_correctness for r in plain_applied.rows] == [
            r.predicted_correctness for r in extended_applied.rows
        ]
        assert plain_applied.fingerprint != extended_applied.fingerprint

    def test_artifact_commits_profile_and_population_provenance(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        profile = training_profile([(0.7, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, profile)
        assert applied.profile_fingerprint == profile.fingerprint
        assert applied.profile_fingerprint_version == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        assert applied.evaluation_dataset_fingerprint == dataset.fingerprint
        assert applied.source_cohort_fingerprint == dataset.source_cohort_fingerprint
        assert (
            applied.source_cohort_fingerprint_version
            == CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION
        )
        assert applied.target_id == WINNER_CORRECTNESS_TARGET_ID
        assert applied.target_version == WINNER_CORRECTNESS_TARGET_VERSION
        assert applied.input_score_id == UNCALIBRATED_SELECTED_PROBABILITY_ID
        assert applied.output_score_id == PREDICTED_WINNER_CORRECTNESS_ID
        assert applied.output_score_version == PREDICTED_WINNER_CORRECTNESS_VERSION
        assert applied.source_count == 2

    def test_artifact_does_not_duplicate_profile_internals(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        payload = applied.canonical_payload()
        assert set(payload["profile"]) == {
            "profile_fingerprint",
            "profile_fingerprint_version",
        }
        assert "fitted_parameters" not in str(payload)
        assert "method_id" not in str(payload)

    def test_selected_value_not_argmax_is_still_used(self) -> None:
        """Application consumes the recorded selection, never a recomputed winner."""
        body = body_of(apply_profile_to_evaluation_dataset)
        assert "argmax" not in body
        assert "predicted_winner_correctness(" in body


# ---------------------------------------------------------------------------
# Compatibility checks (PART 11 to PART 18, PART 47 to PART 49)
# ---------------------------------------------------------------------------


class TestCompatibilityChecks:
    def test_binding_mismatch_fails_before_scoring(self) -> None:
        profile = training_profile([(0.7, True)])
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(binding_mismatched_dataset(), profile)

    def test_ground_truth_semantics_mismatch_fails_closed(self) -> None:
        # Training under a different labeling rule keeps the binding identical
        # while changing the ground-truth semantics the profile targets.
        training = CalibrationDataset.create(
            [bool_observation(resolved_truth(True, labeling_rule="other-labeling-rule"))]
        )
        profile = fit_l2_logistic_selected_probability(training, l2_strength=0.05)
        dataset = evaluation_dataset([(0.6, True)])
        assert (
            profile.ground_truth_semantics.fingerprint != dataset.ground_truth_semantics.fingerprint
        )
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(dataset, profile)

    def test_unsupported_method_fails_closed(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        training = training_dataset_for([(0.7, True)])
        profile = internal_profile(training, method_id="synthetic-unknown-method", method_version=1)
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(dataset, profile)

    def test_missing_fitted_parameter_fails_closed(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        training = training_dataset_for([(0.7, True)])
        profile = internal_profile(training, fitted_parameters={"slope": 1.0})
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(dataset, profile)

    def test_non_finite_fitted_state_fails_closed_at_application(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        training = training_dataset_for([(0.7, True)])
        valid = internal_profile(training, slope=1.0, intercept=0.0)
        forged = forge_profile(valid, fitted_parameters={"slope": math.inf, "intercept": 0.0})
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(dataset, forged)

    def test_wrong_argument_types_are_rejected(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        profile = training_profile([(0.7, True)])
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset("nope", profile)  # type: ignore[arg-type]
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(dataset, "nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Logistic application (PART 16 to PART 20)
# ---------------------------------------------------------------------------


class TestLogisticApplication:
    def test_endpoints_are_representable_and_never_clipped(self) -> None:
        dataset = evaluation_dataset([(0.9, True)])
        training = training_dataset_for([(0.9, True)])
        high = internal_profile(training, slope=1000.0, intercept=0.0)
        low = internal_profile(training, slope=-1000.0, intercept=0.0)
        assert predicted_winner_correctness(high, dataset.observations[0]) == 1.0
        assert predicted_winner_correctness(low, dataset.observations[0]) == 0.0

    def test_mapping_is_monotone_in_the_selected_probability(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, True)])
        training = training_dataset_for([(0.6, True), (0.9, True)])
        profile = internal_profile(training, slope=2.0, intercept=-1.0)
        scores = [
            predicted_winner_correctness(profile, observation)
            for observation in dataset.observations
        ]
        assert scores[0] < scores[1]

    def test_no_clipping_or_epsilon_appears_in_the_application_path(self) -> None:
        body = body_of(predicted_winner_correctness)
        assert "clip(" not in body
        assert "min(max(" not in body
        assert "1e-" not in body

    def test_application_does_not_touch_the_optimizer(self) -> None:
        body = body_of(apply_profile_to_evaluation_dataset)
        assert "_solve_l2_logistic" not in body
        assert "l2_strength" not in body


# ---------------------------------------------------------------------------
# Post-calibration metrics (PART 25 to PART 43)
# ---------------------------------------------------------------------------


class TestPostCalibrationBrier:
    def test_formula_matches_a_recomputation(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        profile = training_profile([(0.7, True), (0.7, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, profile)
        result = evaluate_post_calibration_winner_brier(applied)
        expected = (
            math.fsum(
                (row.predicted_correctness - (1.0 if row.correct else 0.0)) ** 2
                for row in applied.rows
            )
            / applied.count
        )
        assert result.value == expected
        assert result.metric_id == BRIER_METRIC_ID
        assert result.metric_version == BRIER_METRIC_VERSION
        assert result.input_score_id == PREDICTED_WINNER_CORRECTNESS_ID
        assert result.input_score_version == PREDICTED_WINNER_CORRECTNESS_VERSION
        assert result.target_id == WINNER_CORRECTNESS_TARGET_ID

    def test_provenance_commits_profile_and_population(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        profile = training_profile([(0.7, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, profile)
        result = evaluate_post_calibration_winner_brier(applied)
        assert result.application_fingerprint == applied.fingerprint
        assert result.profile_fingerprint == profile.fingerprint
        assert result.evaluation_dataset_fingerprint == dataset.fingerprint
        assert result.source_cohort_fingerprint == applied.source_cohort_fingerprint
        assert result.source_count == 2

    def test_result_schema_is_new_version(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        payload = evaluate_post_calibration_winner_brier(applied).canonical_payload()
        assert payload["v"] == POST_CALIBRATION_BRIER_RESULT_FINGERPRINT_VERSION


class TestPostCalibrationLogLoss:
    def test_formula_matches_a_recomputation(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        profile = training_profile([(0.7, True), (0.7, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, profile)
        result = evaluate_post_calibration_winner_log_loss(applied)
        terms = [
            math.log(row.predicted_correctness)
            if row.correct
            else math.log(1.0 - row.predicted_correctness)
            for row in applied.rows
        ]
        expected = math.fsum(-term for term in terms) / applied.count
        assert result.value == expected
        assert result.input_score_id == PREDICTED_WINNER_CORRECTNESS_ID
        assert result.metric_id == LOG_LOSS_METRIC_ID
        assert result.metric_version == LOG_LOSS_METRIC_VERSION

    def test_exact_endpoints_are_not_clipped(self) -> None:
        dataset = evaluation_dataset([(0.9, True), (0.9, False)])
        training = training_dataset_for([(0.9, True)])
        confident = internal_profile(training, slope=1000.0, intercept=0.0)
        applied = apply_profile_to_evaluation_dataset(dataset, confident)
        assert all(row.predicted_correctness == 1.0 for row in applied.rows)
        result = evaluate_post_calibration_winner_log_loss(applied)
        assert result.value == math.inf

    def test_infinity_is_canonicalized_not_raw_json(self) -> None:
        dataset = evaluation_dataset([(0.9, True), (0.9, False)])
        training = training_dataset_for([(0.9, True)])
        confident = internal_profile(training, slope=1000.0, intercept=0.0)
        applied = apply_profile_to_evaluation_dataset(dataset, confident)
        result = evaluate_post_calibration_winner_log_loss(applied)
        payload = result.canonical_payload()
        assert payload["value"]["kind"] == "positive_infinity"
        assert payload["value"]["number"] is None
        assert "Infinity" not in str(payload)
        assert len(result.fingerprint) == 64

    def test_finite_result_schema_is_new_version(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        payload = evaluate_post_calibration_winner_log_loss(applied).canonical_payload()
        assert payload["v"] == POST_CALIBRATION_LOG_LOSS_RESULT_FINGERPRINT_VERSION


class TestPostCalibrationDiagnostics:
    def test_mean_predicted_correctness_has_its_own_identity(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        result = evaluate_post_calibration_winner_diagnostics(applied)
        expected = math.fsum(row.predicted_correctness for row in applied.rows) / applied.count
        assert result.mean_predicted_correctness == expected
        payload = result.canonical_payload()
        assert payload["mean_predicted_correctness"]["id"] == MEAN_PREDICTED_CORRECTNESS_ID
        assert (
            payload["mean_predicted_correctness"]["version"] == MEAN_PREDICTED_CORRECTNESS_VERSION
        )
        assert MEAN_PREDICTED_CORRECTNESS_ID != MEAN_SELECTED_PROBABILITY_ID

    def test_empirical_correctness_rate_is_the_same_target_statistic(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        post = evaluate_post_calibration_winner_diagnostics(applied)
        pre = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert post.empirical_correctness_rate == pre.empirical_correctness_rate
        assert EMPIRICAL_CORRECTNESS_RATE_ID == "empirical-winner-correctness-rate"
        assert EMPIRICAL_CORRECTNESS_RATE_VERSION == 1

    def test_constant_brier_reference_semantics_unchanged(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        post = evaluate_post_calibration_winner_diagnostics(applied)
        pre = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert post.empirical_constant_brier_reference == pre.empirical_constant_brier_reference
        assert EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID
        assert EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION == 1

    def test_diagnostics_schema_is_new_version(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        payload = evaluate_post_calibration_winner_diagnostics(applied).canonical_payload()
        assert payload["v"] == POST_CALIBRATION_DIAGNOSTICS_RESULT_FINGERPRINT_VERSION


class TestPostCalibrationReliability:
    def test_bin_summary_uses_the_post_field_name(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        result = evaluate_post_calibration_winner_reliability(applied, bin_count=4)
        summary = result.bins[0]
        assert hasattr(summary, "mean_predicted_correctness")
        assert not hasattr(summary, "mean_selected_probability")
        assert result.reliability_id == POST_CALIBRATION_WINNER_RELIABILITY_ID
        assert result.reliability_version == POST_CALIBRATION_WINNER_RELIABILITY_VERSION
        assert result.reliability_id != WINNER_RELIABILITY_CURVE_ID

    def test_binning_policy_is_reused(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        post = evaluate_post_calibration_winner_reliability(applied, bin_count=5)
        pre = evaluate_uncalibrated_winner_reliability(dataset, bin_count=5)
        assert post.binning_id == pre.binning_id == EQUAL_WIDTH_BINNING_ID
        assert post.binning_version == pre.binning_version == EQUAL_WIDTH_BINNING_VERSION
        assert post.bin_count == 5

    def test_empty_bins_are_retained(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        result = evaluate_post_calibration_winner_reliability(applied, bin_count=10)
        assert len(result.bins) == 10
        empty = [b for b in result.bins if b.count == 0]
        assert empty
        for summary in empty:
            assert summary.correct_count == 0
            assert summary.mean_predicted_correctness is None
            assert summary.empirical_correctness_rate is None

    def test_membership_is_sorted_multiplicity_preserving(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        result = evaluate_post_calibration_winner_reliability(applied, bin_count=5)
        assert sum(summary.count for summary in result.bins) == 3
        for summary in result.bins:
            assert list(summary.observation_fingerprints) == sorted(
                summary.observation_fingerprints
            )
            assert len(summary.observation_fingerprints) == summary.count

    def test_reliability_schema_is_new_version(self) -> None:
        dataset = evaluation_dataset([(0.6, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        payload = evaluate_post_calibration_winner_reliability(
            applied, bin_count=4
        ).canonical_payload()
        assert payload["v"] == POST_CALIBRATION_RELIABILITY_RESULT_FINGERPRINT_VERSION
        assert WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION == 2


class TestPostCalibrationBinnedGap:
    def test_gap_derives_only_from_reliability(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        reliability = evaluate_post_calibration_winner_reliability(applied, bin_count=4)
        result = evaluate_post_calibration_winner_binned_absolute_gap(reliability)
        expected = math.fsum(
            summary.count
            / reliability.count
            * abs(summary.mean_predicted_correctness - summary.empirical_correctness_rate)
            for summary in reliability.bins
            if summary.count > 0
        )
        assert abs(result.value - expected) < 1e-15
        assert result.reliability_fingerprint == reliability.fingerprint

    def test_gap_is_weighted_not_an_unweighted_mean(self) -> None:
        dataset = evaluation_dataset([(0.6, True)] * 5 + [(0.9, False)] * 95)
        training = training_dataset_for([(0.7, True)])
        # A steep map pushes the two score levels into different bins, so the
        # bin counts differ and a weighted aggregate must not equal the plain
        # mean of bin gaps.
        steep = internal_profile(training, slope=10.0, intercept=-5.0)
        applied = apply_profile_to_evaluation_dataset(dataset, steep)
        reliability = evaluate_post_calibration_winner_reliability(applied, bin_count=10)
        non_empty = [b for b in reliability.bins if b.count > 0]
        assert len({b.count for b in non_empty}) > 1
        result = evaluate_post_calibration_winner_binned_absolute_gap(reliability)
        unweighted = math.fsum(
            abs(summary.mean_predicted_correctness - summary.empirical_correctness_rate)
            for summary in non_empty
        ) / len(non_empty)
        assert result.value != unweighted

    def test_gap_does_not_reapply_the_profile(self) -> None:
        body = body_of(evaluate_post_calibration_winner_binned_absolute_gap)
        assert "apply_profile" not in body
        assert "predicted_winner_correctness" not in body
        assert "_equal_width_bin_index" not in body

    def test_gap_artifact_identity(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        reliability = evaluate_post_calibration_winner_reliability(applied, bin_count=4)
        result = evaluate_post_calibration_winner_binned_absolute_gap(reliability)
        assert result.aggregate_id == POST_CALIBRATION_WINNER_BINNED_ABSOLUTE_GAP_ID
        assert result.aggregate_version == POST_CALIBRATION_WINNER_BINNED_ABSOLUTE_GAP_VERSION
        assert (
            result.canonical_payload()["v"]
            == POST_CALIBRATION_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION
        )


# ---------------------------------------------------------------------------
# Pre/post coherence (PART 43 to PART 46)
# ---------------------------------------------------------------------------


class TestPrePostCoherence:
    def test_pre_and_post_agree_on_population_and_differ_on_score_identity(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False), (0.8, True)])
        profile = training_profile([(0.7, True), (0.7, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, profile)
        pre_reliability = evaluate_uncalibrated_winner_reliability(dataset, bin_count=4)
        post_reliability = evaluate_post_calibration_winner_reliability(applied, bin_count=4)
        pairs = [
            (
                evaluate_uncalibrated_winner_brier(dataset),
                evaluate_post_calibration_winner_brier(applied),
            ),
            (
                evaluate_uncalibrated_winner_log_loss(dataset),
                evaluate_post_calibration_winner_log_loss(applied),
            ),
            (
                evaluate_uncalibrated_winner_diagnostics(dataset),
                evaluate_post_calibration_winner_diagnostics(applied),
            ),
            (pre_reliability, post_reliability),
            (
                evaluate_winner_binned_absolute_gap(pre_reliability),
                evaluate_post_calibration_winner_binned_absolute_gap(post_reliability),
            ),
        ]
        for pre, post in pairs:
            assert pre.evaluation_dataset_fingerprint == post.evaluation_dataset_fingerprint
            assert pre.source_cohort_fingerprint == post.source_cohort_fingerprint
            assert pre.source_count == post.source_count
            assert pre.count == post.count
            assert pre.target_id == post.target_id
            assert pre.target_version == post.target_version
            assert pre.input_score_id == UNCALIBRATED_SELECTED_PROBABILITY_ID
            assert post.input_score_id == PREDICTED_WINNER_CORRECTNESS_ID
            assert pre.fingerprint != post.fingerprint

    def test_exclusion_attack_changes_post_artifact_fingerprints(self) -> None:
        profile = training_profile([(0.7, True)])
        rows = [(0.6, True), (0.9, False), (0.8, True)]
        plain = evaluation_dataset(rows)
        extended = evaluation_dataset(rows, extra_excluded=3)
        plain_applied = apply_profile_to_evaluation_dataset(plain, profile)
        extended_applied = apply_profile_to_evaluation_dataset(extended, profile)
        assert plain_applied.fingerprint != extended_applied.fingerprint

        pairs = [
            (
                evaluate_post_calibration_winner_brier(plain_applied),
                evaluate_post_calibration_winner_brier(extended_applied),
            ),
            (
                evaluate_post_calibration_winner_log_loss(plain_applied),
                evaluate_post_calibration_winner_log_loss(extended_applied),
            ),
            (
                evaluate_post_calibration_winner_diagnostics(plain_applied),
                evaluate_post_calibration_winner_diagnostics(extended_applied),
            ),
            (
                evaluate_post_calibration_winner_reliability(plain_applied, bin_count=4),
                evaluate_post_calibration_winner_reliability(extended_applied, bin_count=4),
            ),
        ]
        for plain_artifact, extended_artifact in pairs:
            assert plain_artifact.count == extended_artifact.count
            assert plain_artifact.source_count != extended_artifact.source_count
            assert plain_artifact.fingerprint != extended_artifact.fingerprint

    def test_different_profiles_differ_in_post_artifact_fingerprints(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        training = training_dataset_for([(0.7, True), (0.7, False)])
        first = internal_profile(training, slope=1.0, intercept=0.0, l2_strength=1.0)
        second = internal_profile(training, slope=1.0, intercept=0.0, l2_strength=2.0)
        first_applied = apply_profile_to_evaluation_dataset(dataset, first)
        second_applied = apply_profile_to_evaluation_dataset(dataset, second)
        assert first_applied.fingerprint != second_applied.fingerprint
        first_brier = evaluate_post_calibration_winner_brier(first_applied)
        second_brier = evaluate_post_calibration_winner_brier(second_applied)
        assert first_brier.value == second_brier.value
        assert first_brier.fingerprint != second_brier.fingerprint


# ---------------------------------------------------------------------------
# Synthetic effect examples (PART 52 to PART 53)
# ---------------------------------------------------------------------------


class TestSyntheticEffects:
    def test_one_case_where_calibration_improves_both_metrics(self) -> None:
        # The raw scores understate correctness at both levels; the fitted
        # logistic raises them toward the empirical rate.
        rows = [(0.6, True)] * 40 + [(0.6, False)] * 10 + [(0.9, True)] * 49 + [(0.9, False)]
        dataset = evaluation_dataset(rows)
        applied = apply_profile_to_evaluation_dataset(
            dataset, training_profile(rows, l2_strength=0.01)
        )
        pre_brier = evaluate_uncalibrated_winner_brier(dataset)
        post_brier = evaluate_post_calibration_winner_brier(applied)
        pre_log = evaluate_uncalibrated_winner_log_loss(dataset)
        post_log = evaluate_post_calibration_winner_log_loss(applied)
        assert post_brier.value < pre_brier.value
        assert post_log.value < pre_log.value

    def test_one_case_where_calibration_worsens_brier(self) -> None:
        # Fit where a low score is always correct, then evaluate where the same
        # low score is always wrong. The map pushes the score up, so the
        # post-calibration Brier is worse.
        profile = training_profile([(0.6, True)] * 10, l2_strength=0.01)
        dataset = evaluation_dataset([(0.6, False)] * 10)
        applied = apply_profile_to_evaluation_dataset(dataset, profile)
        pre = evaluate_uncalibrated_winner_brier(dataset)
        post = evaluate_post_calibration_winner_brier(applied)
        assert post.value > pre.value

    def test_no_automatic_improvement_verdict_exists(self) -> None:
        module_source = inspect.getsource(
            __import__("probvenance.calibration_evaluation", fromlist=["x"])
        )
        assert "calibration_success" not in module_source
        assert "improved" not in module_source


class TestPostMetricLabelBinding:
    """Post metrics must bind labels to observation identity, never to position."""

    ROWS = [(0.6, True)] * 6 + [(0.9, False)] * 6

    def _applied(self) -> tuple[CalibrationEvaluationDataset, Any]:
        profile = training_profile([(0.6, True)] * 3 + [(0.9, False)] * 3)
        dataset = evaluation_dataset(self.ROWS)
        return dataset, apply_profile_to_evaluation_dataset(dataset, profile)

    def test_metric_values_are_row_order_independent(self) -> None:
        _dataset, applied = self._applied()
        baseline = evaluate_post_calibration_winner_brier(applied).value
        for permuted in (
            list(reversed(self.ROWS)),
            self.ROWS[5:] + self.ROWS[:5],
            self.ROWS[3:] + self.ROWS[:3],
        ):
            other = evaluation_dataset(permuted)
            other_applied = apply_profile_to_evaluation_dataset(
                other, training_profile([(0.6, True)] * 3 + [(0.9, False)] * 3)
            )
            assert evaluate_post_calibration_winner_brier(other_applied).value == baseline

    def test_every_post_metric_is_row_order_independent(self) -> None:
        dataset, applied = self._applied()
        permuted = evaluation_dataset(list(reversed(self.ROWS)))
        permuted_applied = apply_profile_to_evaluation_dataset(
            permuted, training_profile([(0.6, True)] * 3 + [(0.9, False)] * 3)
        )
        for builder in (
            lambda s, a: evaluate_post_calibration_winner_brier(a),
            lambda s, a: evaluate_post_calibration_winner_log_loss(a),
            lambda s, a: evaluate_post_calibration_winner_diagnostics(a),
            lambda s, a: evaluate_post_calibration_winner_reliability(a, bin_count=4),
        ):
            assert (
                builder(dataset, applied).canonical_payload()
                == builder(permuted, permuted_applied).canonical_payload()
            )

    def test_wrong_source_cannot_be_supplied_by_api_shape(self) -> None:
        # The original MF1/MF1b defect was that a metric paired applied rows with
        # a separately supplied source by position. The one-input API removes the
        # call shape entirely, so a wrong source can no longer be passed at all.
        for function in (
            evaluate_post_calibration_winner_brier,
            evaluate_post_calibration_winner_log_loss,
            evaluate_post_calibration_winner_diagnostics,
        ):
            assert list(inspect.signature(function).parameters) == ["applied"]
        assert list(inspect.signature(evaluate_post_calibration_winner_reliability).parameters) == [
            "applied",
            "bin_count",
        ]
        assert list(
            inspect.signature(evaluate_post_calibration_winner_binned_absolute_gap).parameters
        ) == ["reliability"]

    def test_applied_rows_commit_derived_labels_never_caller_labels(self) -> None:
        dataset, applied = self._applied()
        labels_by_fingerprint = {
            observation.fingerprint: observation.correct for observation in dataset.observations
        }
        for row in applied.rows:
            assert row.correct is labels_by_fingerprint[row.observation_fingerprint]
            assert isinstance(row.correct, bool)


class TestApplicationRowLabelIntegrity:
    def _applied(self) -> tuple[CalibrationEvaluationDataset, Any]:
        profile = training_profile([(0.6, True), (0.9, False)])
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        return dataset, apply_profile_to_evaluation_dataset(dataset, profile)

    def test_a_row_requires_a_real_bool_label(self) -> None:
        for forged in (1, 0, None, "true", 1.0):
            with pytest.raises(InvalidDecisionError):
                ProfileAppliedEvaluationRow(
                    "a" * 64,
                    forged,
                    0.5,
                    _construction_token=_PROFILE_APPLIED_EVALUATION_ROW_CONSTRUCTION_TOKEN,
                )

    def test_a_row_rejects_direct_construction_and_replace(self) -> None:
        with pytest.raises(InvalidDecisionError):
            ProfileAppliedEvaluationRow("a" * 64, True, 0.5)
        _, applied = self._applied()
        with pytest.raises(InvalidDecisionError):
            replace(applied.rows[0], correct=not applied.rows[0].correct)

    def test_application_artifact_rejects_direct_construction(self) -> None:
        _, applied = self._applied()
        with pytest.raises(InvalidDecisionError):
            ProfileAppliedEvaluationDataset((applied.rows[0],), EvaluationSplitRole.VALIDATION, "e")
        with pytest.raises(InvalidDecisionError):
            replace(applied, split_id="other")

    def test_forged_dataset_without_a_bool_label_fails_closed(self) -> None:
        profile = training_profile([(0.6, True)] * 3 + [(0.9, False)] * 3)
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        forged_observations = list(dataset.observations)
        object.__setattr__(forged_observations[0], "correct", None)
        forged = object.__new__(CalibrationEvaluationDataset)
        object.__setattr__(forged, "observations", tuple(forged_observations))
        object.__setattr__(forged, "split_role", dataset.split_role)
        object.__setattr__(forged, "split_id", dataset.split_id)
        object.__setattr__(forged, "source_cohort_fingerprint", dataset.source_cohort_fingerprint)
        object.__setattr__(forged, "source_count", dataset.source_count)
        object.__setattr__(forged, "taxonomy_miss_count", dataset.taxonomy_miss_count)
        object.__setattr__(forged, "unresolved_count", dataset.unresolved_count)
        object.__setattr__(
            forged, "unadjudicated_resolved_count", dataset.unadjudicated_resolved_count
        )
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(forged, profile)


# ---------------------------------------------------------------------------
# Reproducibility and mutation isolation (PART 57 to PART 59)
# ---------------------------------------------------------------------------


class TestReproducibilityAndIsolation:
    def test_repeat_application_is_identical(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        profile = training_profile([(0.7, True)])
        first = apply_profile_to_evaluation_dataset(dataset, profile)
        second = apply_profile_to_evaluation_dataset(dataset, profile)
        assert first.canonical_payload() == second.canonical_payload()
        assert first.fingerprint == second.fingerprint

    def test_payload_mutation_does_not_affect_state(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        applied = apply_profile_to_evaluation_dataset(dataset, training_profile([(0.7, True)]))
        payload = applied.canonical_payload()
        payload["split_id"] = "mutated"
        payload["profile"]["profile_fingerprint"] = "mutated"
        payload["rows"].clear()
        assert applied.split_id != "mutated"
        assert applied.profile_fingerprint != "mutated"
        assert len(applied.rows) == 2
        assert applied.canonical_payload()["split_id"] != "mutated"

    def test_application_does_not_mutate_inputs(self) -> None:
        dataset = evaluation_dataset([(0.6, True), (0.9, False)])
        profile = training_profile([(0.7, True)])
        dataset_fingerprint = dataset.fingerprint
        profile_fingerprint = profile.fingerprint
        observations = dataset.observations
        apply_profile_to_evaluation_dataset(dataset, profile)
        assert dataset.fingerprint == dataset_fingerprint
        assert profile.fingerprint == profile_fingerprint
        assert dataset.observations == observations


# ---------------------------------------------------------------------------
# Frozen identities (PART 60 to PART 63)
# ---------------------------------------------------------------------------


class TestFrozenIdentities:
    def test_pre_calibration_result_schemas_unchanged(self) -> None:
        assert WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION == 2
        assert WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION == 2
        assert WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION == 2
        assert CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION == 1

    def test_application_schema_bumped_and_post_result_schemas_stayed(self) -> None:
        # Adding the derived `correct` label to each canonical application row is
        # a real application payload schema change, so its version moved 1 -> 2.
        # The post result payloads did NOT change shape: they merely carry the new
        # upstream application fingerprint/version value, so their own versions
        # stay 1.
        assert PROFILE_APPLIED_EVALUATION_DATASET_FINGERPRINT_VERSION == 2
        assert POST_CALIBRATION_BRIER_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_LOG_LOSS_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_DIAGNOSTICS_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_RELIABILITY_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION == 1

    def test_calibration_profile_schema_frozen(self) -> None:
        assert CALIBRATION_PROFILE_FINGERPRINT_VERSION == 1

    def test_metric_semantic_versions_unchanged(self) -> None:
        assert BRIER_METRIC_ID == "brier"
        assert BRIER_METRIC_VERSION == 1
        assert LOG_LOSS_METRIC_VERSION == 1
        assert EQUAL_WIDTH_BINNING_VERSION == 1


# ---------------------------------------------------------------------------
# Scope boundaries (PART 50, PART 51, PART 70, PART 71)
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    def test_no_profile_registry_or_lookup_exists(self) -> None:
        module_source = inspect.getsource(
            __import__("probvenance.calibration_evaluation", fromlist=["x"])
        )
        assert "class ProfileRegistry" not in module_source
        assert "def find_profile" not in module_source
        assert "def nearest_profile" not in module_source

    def test_runtime_results_remain_uncalibrated(self) -> None:
        runtime, _ = make_choice_runtime(None)
        result = runtime.evaluate_with_trace(make_bool_decision()).result
        assert result.predicted_correctness is None
        assert result.calibrated is False

    def test_no_plotting_or_heavy_dependency_imported(self) -> None:
        module_source = inspect.getsource(
            __import__("probvenance.calibration_evaluation", fromlist=["x"])
        )
        for forbidden in (
            "import matplotlib",
            "from matplotlib",
            "import numpy",
            "from numpy",
            "import scipy",
            "from scipy",
        ):
            assert forbidden not in module_source

    def test_no_equal_mass_or_quantile_binning_algorithm(self) -> None:
        module_source = inspect.getsource(
            __import__("probvenance.calibration_evaluation", fromlist=["x"])
        )
        for forbidden in ("quantile(", "equal_mass_binning", "adaptive_binning"):
            assert forbidden not in module_source
