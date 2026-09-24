"""Runtime-linked calibration application (Phase 4C.4).

Covers the explicit transformation of an uncalibrated runtime ``Evaluation``
into a calibrated one through an exact, caller-supplied ``CalibrationProfile``.
"""

from dataclasses import replace
from typing import Any

import pytest
from test_calibration import (
    make_bool_decision,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
)
from test_profile_application import forge_profile

from probvenance import InvalidDecisionError, InvalidProbabilityError
from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationDataset,
    CalibrationObservation,
    fit_l2_logistic_selected_probability,
    predicted_winner_correctness,
)
from probvenance.calibration import (
    apply_profile_to_runtime_evaluation as apply_profile,
)
from probvenance.results import BoolResult, ChoiceResult
from probvenance.runtime import Evaluation


def _bool_evaluation() -> Evaluation:
    runtime, _ = make_choice_runtime()
    return runtime.evaluate_with_trace(make_bool_decision())


def _choice_evaluation() -> Evaluation:
    runtime, _ = make_choice_runtime()
    return runtime.evaluate_with_trace(make_choice_decision())


def _bool_observation(evaluation: Evaluation) -> CalibrationObservation:
    return CalibrationObservation.from_evaluation(evaluation, resolved_truth(True))


def _choice_observation(evaluation: Evaluation) -> CalibrationObservation:
    return CalibrationObservation.from_evaluation(evaluation, resolved_truth("shipping"))


def _profile_for(observation: CalibrationObservation, l2_strength: float = 1.0):
    dataset = CalibrationDataset.create([observation])
    return fit_l2_logistic_selected_probability(dataset, l2_strength=l2_strength)


def _bool_evaluation_and_profile() -> tuple[Evaluation, CalibrationObservation, Any]:
    evaluation = _bool_evaluation()
    observation = _bool_observation(evaluation)
    return evaluation, observation, _profile_for(observation)


def _choice_evaluation_and_profile() -> tuple[Evaluation, CalibrationObservation, Any]:
    evaluation = _choice_evaluation()
    observation = _choice_observation(evaluation)
    return evaluation, observation, _profile_for(observation)


# ---------------------------------------------------------------------------
# Offline/runtime numeric equivalence (PART 25, PART 26)
# ---------------------------------------------------------------------------


class TestOfflineRuntimeEquivalence:
    def test_bool_offline_and_runtime_scores_agree(self) -> None:
        evaluation, observation, profile = _bool_evaluation_and_profile()
        offline = predicted_winner_correctness(profile, observation)
        applied = apply_profile(evaluation, profile)
        assert applied.result.predicted_correctness == offline

    def test_choice_offline_and_runtime_scores_agree(self) -> None:
        evaluation, observation, profile = _choice_evaluation_and_profile()
        offline = predicted_winner_correctness(profile, observation)
        applied = apply_profile(evaluation, profile)
        assert applied.result.predicted_correctness == offline

    def test_runtime_application_reads_recorded_selection(self) -> None:
        evaluation, _, profile = _choice_evaluation_and_profile()
        applied = apply_profile(evaluation, profile)
        assert isinstance(evaluation.result, ChoiceResult)
        assert isinstance(applied.result, ChoiceResult)
        assert applied.result.value == evaluation.result.value
        assert applied.result.probabilities == evaluation.result.probabilities


# ---------------------------------------------------------------------------
# Result and trace provenance (PART 4, PART 15, PART 18, PART 24)
# ---------------------------------------------------------------------------


class TestCalibratedResultAndTrace:
    def test_result_carries_exact_profile_identity(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        result = apply_profile(evaluation, profile).result
        assert result.calibrated is True
        assert result.calibration_profile_fingerprint == profile.fingerprint
        assert (
            result.calibration_profile_fingerprint_version
            == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        )

    def test_trace_mirrors_profile_provenance(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        trace = apply_profile(evaluation, profile).trace
        assert trace.calibration_profile_fingerprint == profile.fingerprint
        assert (
            trace.calibration_profile_fingerprint_version == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        )

    def test_result_and_trace_provenance_agree(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        applied = apply_profile(evaluation, profile)
        assert (
            applied.result.calibration_profile_fingerprint
            == applied.trace.calibration_profile_fingerprint
        )
        assert (
            applied.result.calibration_profile_fingerprint_version
            == applied.trace.calibration_profile_fingerprint_version
        )

    def test_same_concrete_result_type(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        assert isinstance(apply_profile(evaluation, profile).result, BoolResult)

    def test_original_evaluation_is_unchanged(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        apply_profile(evaluation, profile)
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None
        assert evaluation.result.calibration_profile_fingerprint is None
        assert evaluation.trace.calibration_profile_fingerprint is None

    def test_semantic_distribution_is_unchanged(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        applied = apply_profile(evaluation, profile).result
        assert isinstance(evaluation.result, BoolResult)
        assert isinstance(applied, BoolResult)
        assert applied.probability_true == evaluation.result.probability_true
        assert applied.certainty == evaluation.result.certainty
        assert applied.method == evaluation.result.method
        assert applied.trace_id == evaluation.result.trace_id

    def test_trace_id_is_preserved(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        assert apply_profile(evaluation, profile).trace.trace_id == evaluation.trace.trace_id


# ---------------------------------------------------------------------------
# Execution identity (PART 20, PART 21, PART 22)
# ---------------------------------------------------------------------------


class TestExecutionIdentityUnchanged:
    def test_execution_fingerprint_is_unchanged(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        assert (
            apply_profile(evaluation, profile).trace.execution_fingerprint
            == evaluation.trace.execution_fingerprint
        )

    def test_plan_and_decision_fingerprints_are_unchanged(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        applied = apply_profile(evaluation, profile).trace
        assert applied.plan_fingerprint == evaluation.trace.plan_fingerprint
        assert applied.decision_fingerprint == evaluation.trace.decision_fingerprint

    def test_to_dict_includes_calibration_provenance(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        payload = apply_profile(evaluation, profile).trace.to_dict()
        assert payload["calibration_profile_fingerprint"] == profile.fingerprint
        assert (
            payload["calibration_profile_fingerprint_version"]
            == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        )

    def test_uncalibrated_trace_serializes_explicit_null(self) -> None:
        payload = _bool_evaluation().trace.to_dict()
        assert payload["calibration_profile_fingerprint"] is None
        assert payload["calibration_profile_fingerprint_version"] is None


# ---------------------------------------------------------------------------
# Default runtime behaviour (PART 35)
# ---------------------------------------------------------------------------


class TestUncalibratedByDefault:
    def test_evaluate_with_trace_stays_uncalibrated(self) -> None:
        evaluation = _bool_evaluation()
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None
        assert evaluation.result.calibration_profile_fingerprint is None
        assert evaluation.trace.calibration_profile_fingerprint is None


# ---------------------------------------------------------------------------
# Already-calibrated input (PART 14)
# ---------------------------------------------------------------------------


class TestAlreadyCalibratedRejected:
    def test_calibrated_result_rejected(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        already = apply_profile(evaluation, profile)
        with pytest.raises(InvalidDecisionError, match="already calibrated"):
            apply_profile(already, profile)

    def test_trace_provenance_without_result_rejected(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        forged = Evaluation(
            result=evaluation.result,
            trace=replace(
                evaluation.trace,
                calibration_profile_fingerprint=profile.fingerprint,
                calibration_profile_fingerprint_version=1,
            ),
        )
        with pytest.raises(InvalidDecisionError, match="uncalibrated trace"):
            apply_profile(forged, profile)


# ---------------------------------------------------------------------------
# Result/trace linkage (PART 13)
# ---------------------------------------------------------------------------


class TestLinkageValidation:
    def test_missing_trace_id_rejected(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        forged = Evaluation(
            result=replace(evaluation.result, trace_id=None),
            trace=evaluation.trace,
        )
        with pytest.raises(InvalidDecisionError, match="trace id"):
            apply_profile(forged, profile)

    def test_mismatched_trace_id_rejected(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        forged = Evaluation(
            result=replace(evaluation.result, trace_id="a-different-trace"),
            trace=evaluation.trace,
        )
        with pytest.raises(InvalidDecisionError, match="trace id"):
            apply_profile(forged, profile)


# ---------------------------------------------------------------------------
# Exact binding match (PART 27, PART 28)
# ---------------------------------------------------------------------------


class TestBindingMatch:
    def test_choice_profile_on_bool_evaluation_rejected(self) -> None:
        evaluation = _bool_evaluation()
        _, _, choice_profile = _choice_evaluation_and_profile()
        with pytest.raises(InvalidDecisionError, match="binding does not match"):
            apply_profile(evaluation, choice_profile)

    def test_caller_declaration_mismatch_rejected(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        with pytest.raises(InvalidDecisionError, match="binding does not match"):
            apply_profile(evaluation, profile, task_id="declared-task")

    def test_unknown_declarations_match_exactly(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        applied = apply_profile(evaluation, profile)
        assert applied.result.calibrated is True

    def test_extra_declaration_on_both_sides_matches(self) -> None:
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_bool_decision())
        observation = CalibrationObservation.from_evaluation(evaluation, resolved_truth(True))
        dataset = CalibrationDataset.create([observation])
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        applied = apply_profile(evaluation, profile, task_id=None, domain_id=None)
        assert applied.result.calibrated is True


# ---------------------------------------------------------------------------
# Method and fitted state fail closed (PART 29, PART 30)
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_unknown_method_rejected(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        forged = forge_profile(profile, method_id="an-unsupported-method")
        with pytest.raises(InvalidDecisionError, match="not supported"):
            apply_profile(evaluation, forged)

    @pytest.mark.parametrize(
        "fitted",
        [
            {"slope": 1.0},
            {"slope": 1.0, "intercept": 0.0, "extra": 1.0},
            {"slope": float("nan"), "intercept": 0.0},
            {"slope": float("inf"), "intercept": 0.0},
            {"slope": "1.0", "intercept": 0.0},
        ],
    )
    def test_malformed_fitted_state_rejected(self, fitted: dict[str, Any]) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        forged = forge_profile(profile, fitted_parameters=fitted)
        with pytest.raises(InvalidDecisionError):
            apply_profile(evaluation, forged)


# ---------------------------------------------------------------------------
# Endpoints (PART 37)
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_endpoint_one_allowed(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        steep = forge_profile(profile, fitted_parameters={"slope": 1000.0, "intercept": 0.0})
        applied = apply_profile(evaluation, steep)
        assert applied.result.predicted_correctness == 1.0

    def test_endpoint_zero_allowed(self) -> None:
        evaluation, _, profile = _bool_evaluation_and_profile()
        steep = forge_profile(profile, fitted_parameters={"slope": -1000.0, "intercept": 0.0})
        applied = apply_profile(evaluation, steep)
        assert applied.result.predicted_correctness == 0.0


# ---------------------------------------------------------------------------
# Manual calibrated result state machine (PART 5, PART 6, PART 36)
# ---------------------------------------------------------------------------


def _manual_result(**overrides: Any) -> BoolResult:
    values: dict[str, Any] = {
        "certainty": _bool_evaluation().result.certainty,
        "method": "manual",
        "probability_true": 0.6,
        "predicted_correctness": 0.8,
        "calibrated": True,
        "calibration_profile_fingerprint": "a" * 64,
        "calibration_profile_fingerprint_version": 1,
    }
    values.update(overrides)
    return BoolResult(**values)


class TestManualResultStateMachine:
    def test_coherent_manual_calibrated_result_allowed(self) -> None:
        result = _manual_result()
        assert result.calibrated is True
        assert result.predicted_correctness == 0.8

    @pytest.mark.parametrize(
        "overrides",
        [
            {"predicted_correctness": None},
            {"calibration_profile_fingerprint": None},
            {"calibrated": False},
            {
                "calibrated": False,
                "predicted_correctness": None,
                "calibration_profile_fingerprint": None,
            },
            {"calibration_profile_fingerprint_version": None},
            {"calibration_profile_fingerprint_version": True},
            {"calibration_profile_fingerprint_version": 0},
            {"calibration_profile_fingerprint": ""},
            {"predicted_correctness": float("nan")},
        ],
    )
    def test_forbidden_states_rejected(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(InvalidProbabilityError):
            _manual_result(**overrides)

    def test_calibrated_false_with_provenance_rejected(self) -> None:
        with pytest.raises(InvalidProbabilityError):
            _manual_result(
                calibrated=False,
                predicted_correctness=None,
                calibration_profile_fingerprint="a" * 64,
                calibration_profile_fingerprint_version=1,
            )

    def test_uncalibrated_default_has_no_provenance(self) -> None:
        result = BoolResult(
            certainty=_bool_evaluation().result.certainty,
            method="manual",
            probability_true=0.6,
        )
        assert result.calibrated is False
        assert result.predicted_correctness is None
        assert result.calibration_profile_fingerprint is None
        assert result.calibration_profile_fingerprint_version is None


class TestTraceProvenanceStateMachine:
    """The trace carries the same calibration-provenance rule as the result."""

    def _trace(self) -> Any:
        return _bool_evaluation().trace

    def test_trace_rejects_half_state(self) -> None:
        trace = self._trace()
        with pytest.raises(InvalidProbabilityError):
            replace(trace, calibration_profile_fingerprint="a" * 64)
        with pytest.raises(InvalidProbabilityError):
            replace(trace, calibration_profile_fingerprint_version=1)

    @pytest.mark.parametrize(
        ("fingerprint", "version"),
        [
            (12345, "not-int"),
            (12345, None),
            ("", 1),
            ("a" * 64, True),
            ("a" * 64, 0),
            ("a" * 64, -1),
        ],
    )
    def test_trace_rejects_wrong_typed_or_incoherent_provenance(
        self, fingerprint: Any, version: Any
    ) -> None:
        with pytest.raises(InvalidProbabilityError):
            replace(
                self._trace(),
                calibration_profile_fingerprint=fingerprint,
                calibration_profile_fingerprint_version=version,
            )

    def test_trace_accepts_coherent_provenance(self) -> None:
        trace = self._trace()
        calibrated = replace(
            trace,
            calibration_profile_fingerprint="a" * 64,
            calibration_profile_fingerprint_version=1,
        )
        assert calibrated.calibration_profile_fingerprint == "a" * 64
        assert calibrated.calibration_profile_fingerprint_version == 1


class TestResultTraceFamilyCoherence:
    """Runtime application mirrors the offline result/trace family check."""

    def test_runtime_application_rejects_family_mismatch(self) -> None:
        bool_evaluation = _bool_evaluation()
        choice_evaluation = _choice_evaluation()
        profile = _profile_for(_bool_observation(bool_evaluation))
        mismatched = Evaluation(
            result=replace(bool_evaluation.result, trace_id=choice_evaluation.trace.trace_id),
            trace=choice_evaluation.trace,
        )
        with pytest.raises(InvalidDecisionError, match="requires a ChoiceResult"):
            apply_profile(mismatched, profile)

    def test_runtime_application_accepts_coherent_pair(self) -> None:
        evaluation = _bool_evaluation()
        profile = _profile_for(_bool_observation(evaluation))
        calibrated = apply_profile(evaluation, profile)
        assert calibrated.result.calibrated is True


class TestChoiceProbabilitySnapshot:
    """The ChoiceResult snapshot is read-only and never aliases its input."""

    def test_probabilities_are_read_only(self) -> None:
        result = _choice_evaluation().result
        assert isinstance(result, ChoiceResult)
        probabilities: Any = result.probabilities
        with pytest.raises(TypeError):
            probabilities["billing"] = 0.99

    def test_snapshot_does_not_alias_the_input_mapping(self) -> None:
        certainty = _choice_evaluation().result.certainty
        source = {"billing": 0.5, "shipping": 0.3, "returns": 0.2}
        result = ChoiceResult(
            certainty=certainty,
            method="manual",
            probabilities=source,
        )
        source["billing"] = 0.99
        assert result.probabilities["billing"] == 0.5
