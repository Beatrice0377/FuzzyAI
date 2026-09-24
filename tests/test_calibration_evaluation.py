"""Tests for the pre-calibration evaluation foundation.

Covers :mod:`probvenance.calibration_evaluation`: the declared evaluation
source cohort contract, the metric-eligible dataset projection, the Brier
winner-correctness metric, and the result artifact.

The fixture helpers below build REAL ``CalibrationObservation`` objects through
the supported ``from_evaluation`` path (a hand-built result paired with a real
evaluation trace, the established pattern in ``test_calibration.py``). The
evaluator itself is never faked.

One fixture deserves explanation: ``recorded_selection_observation`` records a
``selected_value`` that is not the argmax of the recorded probabilities. The
runtime would never select a non-argmax outcome, but the observation contract
records the selection carried by the supplied runtime-linked result, and the
evaluation layer must use the record without recomputing the winner. The
Brier matrix entries with ``p = 0.0`` require exactly such a record (the
argmax of a normalized distribution is never ``0.0``), and the
hand-calculated matrix uses one for its ``p = 0.2`` row over the standard
three-candidate decision.

All choice observations in one dataset share the standard three-candidate
decision because the binding includes the formulation (candidate names and
order) and the fake backend's honest ``vocab_logsumexp`` budget only covers
three candidates.
"""

import math
from dataclasses import fields, replace
from fractions import Fraction

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    bool_observation,
    choice_observation,
    make_bool_decision,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
    unresolved_truth,
)

from probvenance import (
    BoolResult,
    Certainty,
    ChoiceDecision,
    ChoiceResult,
    InvalidDecisionError,
    Probvenance,
)
from probvenance.calibration import (
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    CALIBRATION_DATASET_FINGERPRINT_VERSION,
    CALIBRATION_OBSERVATION_FINGERPRINT_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    RENDERING_SEMANTICS_VERSION,
    CalibrationObservation,
    CalibrationObservationStatus,
)
from probvenance.calibration_evaluation import (
    BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION,
    BRIER_METRIC_ID,
    BRIER_METRIC_VERSION,
    CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
    CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
    EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID,
    EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION,
    EMPIRICAL_CORRECTNESS_RATE_ID,
    EMPIRICAL_CORRECTNESS_RATE_VERSION,
    EQUAL_WIDTH_BINNING_ID,
    EQUAL_WIDTH_BINNING_VERSION,
    LOG_LOSS_BOUNDARY_POLICY,
    LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION,
    LOG_LOSS_LOG_BASE,
    LOG_LOSS_METRIC_ID,
    LOG_LOSS_METRIC_VERSION,
    LOG_LOSS_TARGET,
    MEAN_SELECTED_PROBABILITY_ID,
    MEAN_SELECTED_PROBABILITY_VERSION,
    UNCALIBRATED_SELECTED_PROBABILITY_ID,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION,
    WINNER_RELIABILITY_CURVE_ID,
    WINNER_RELIABILITY_CURVE_VERSION,
    WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION,
    BrierEvaluationResult,
    CalibrationEvaluationCohort,
    CalibrationEvaluationDataset,
    EvaluationSplitRole,
    LogLossEvaluationResult,
    ReliabilityBinSummary,
    WinnerCorrectnessDiagnosticsResult,
    WinnerReliabilityResult,
    _binary_log_loss_term,
    evaluate_uncalibrated_winner_brier,
    evaluate_uncalibrated_winner_diagnostics,
    evaluate_uncalibrated_winner_log_loss,
    evaluate_uncalibrated_winner_reliability,
)
from probvenance.fingerprint import fingerprint

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def evaluation_cohort(
    observations,
    *,
    split_role=EvaluationSplitRole.TEST,
    split_id="eval-split-1",
):
    return CalibrationEvaluationCohort(
        observations=tuple(observations), split_role=split_role, split_id=split_id
    )


def evaluation_dataset(
    observations,
    *,
    split_role=EvaluationSplitRole.TEST,
    split_id="eval-split-1",
):
    """Build a dataset through the supported cohort projection path."""
    return CalibrationEvaluationDataset.from_cohort(
        evaluation_cohort(observations, split_role=split_role, split_id=split_id)
    )


def bool_observation_with_probability_true(probability_true, truth):
    """A real Bool observation with a controlled recorded probability."""
    runtime, _ = make_choice_runtime()
    evaluation = runtime.evaluate_with_trace(make_bool_decision())
    result = BoolResult(
        certainty=Certainty.from_probabilities([1.0 - probability_true, probability_true]),
        method="token_logits",
        probability_true=probability_true,
        trace_id=evaluation.trace.trace_id,
    )
    return CalibrationObservation.from_evaluation((result, evaluation.trace), truth)


def three_way_probabilities(**by_name):
    """Recorded probabilities for the standard three-candidate decision."""
    names = ("billing", "shipping", "returns")
    return {name: by_name.get(name, 0.0) for name in names}


def choice_observation_with_probabilities(probabilities, truth):
    """A real Choice observation with controlled recorded probabilities."""
    runtime, _ = make_choice_runtime()
    evaluation = runtime.evaluate_with_trace(make_choice_decision())
    result = ChoiceResult(
        certainty=Certainty.from_probabilities(list(probabilities.values())),
        method="categorical_token_logits",
        trace_id=evaluation.trace.trace_id,
        probabilities=probabilities,
    )
    return CalibrationObservation.from_evaluation((result, evaluation.trace), truth)


def recorded_selection_observation(probabilities, recorded_value, truth):
    """A real Choice observation whose recorded selection is not the argmax.

    The recorded ``selected_value`` is the selection carried by the supplied
    runtime-linked result: the observation below is built through the
    supported ``from_evaluation`` path, and the recorded selection is honored
    exactly by the evaluation layer (the winner is never recomputed).
    """
    runtime, _ = make_choice_runtime()
    evaluation = runtime.evaluate_with_trace(make_choice_decision())

    class _RecordedSelectionResult(ChoiceResult):
        @property
        def value(self):
            return recorded_value

    result = _RecordedSelectionResult(
        certainty=Certainty.from_probabilities(list(probabilities.values())),
        method="categorical_token_logits",
        trace_id=evaluation.trace.trace_id,
        probabilities=probabilities,
    )
    return CalibrationObservation.from_evaluation((result, evaluation.trace), truth)


# ---------------------------------------------------------------------------
# Brier matrix
# ---------------------------------------------------------------------------


class TestBrierMatrix:
    def test_perfect_predictions_score_zero(self):
        confident = choice_observation_with_probabilities(
            three_way_probabilities(billing=1.0), resolved_truth("billing")
        )
        zero_confidence_wrong = recorded_selection_observation(
            three_way_probabilities(billing=0.5, shipping=0.5),
            "returns",
            resolved_truth("shipping"),
        )
        assert confident.selected_value == "billing"
        assert confident.probabilities[0] == ("billing", pytest.approx(1.0))
        assert confident.correct is True
        assert zero_confidence_wrong.selected_value == "returns"
        assert zero_confidence_wrong.correct is False
        result = evaluate_uncalibrated_winner_brier(
            evaluation_dataset([confident, zero_confidence_wrong])
        )
        assert result.value == pytest.approx(0.0)

    def test_maximally_wrong_predictions_score_one(self):
        confident_wrong = choice_observation_with_probabilities(
            three_way_probabilities(billing=1.0), resolved_truth("shipping")
        )
        zero_confidence_correct = recorded_selection_observation(
            three_way_probabilities(billing=0.5, shipping=0.5),
            "returns",
            resolved_truth("returns"),
        )
        assert confident_wrong.correct is False
        assert zero_confidence_correct.selected_value == "returns"
        assert zero_confidence_correct.correct is True
        result = evaluate_uncalibrated_winner_brier(
            evaluation_dataset([confident_wrong, zero_confidence_correct])
        )
        assert result.value == pytest.approx(1.0)

    def test_neutral_predictions_score_one_quarter(self):
        y_one = bool_observation_with_probability_true(0.5, resolved_truth(True))
        y_zero = bool_observation_with_probability_true(0.5, resolved_truth(False))
        assert y_one.selected_value is False  # the deterministic tie selects False
        assert y_one.correct is False
        assert y_zero.correct is True
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([y_one, y_zero]))
        assert result.value == pytest.approx(0.25)

    def test_hand_calculated_three_observation_value(self):
        # p = [0.8, 0.6, 0.2], y = [1, 0, 0]
        # Brier = ((0.8 - 1)^2 + (0.6 - 0)^2 + (0.2 - 0)^2) / 3 = 0.44 / 3
        confident_right = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        confident_wrong = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
            resolved_truth("billing"),
        )
        # The recorded selection is "billing" at p = 0.2 even though the argmax
        # of the recorded distribution is "shipping"; the record is honored.
        low_confidence_wrong = recorded_selection_observation(
            three_way_probabilities(billing=0.2, shipping=0.4, returns=0.4),
            "billing",
            resolved_truth("shipping"),
        )
        assert confident_right.correct is True
        assert confident_right.probabilities[0] == ("billing", pytest.approx(0.8))
        assert confident_wrong.correct is False
        assert low_confidence_wrong.selected_value == "billing"
        assert low_confidence_wrong.probabilities[0] == ("billing", pytest.approx(0.2))
        assert low_confidence_wrong.correct is False
        result = evaluate_uncalibrated_winner_brier(
            evaluation_dataset([confident_right, confident_wrong, low_confidence_wrong])
        )
        assert result.value == pytest.approx(0.44 / 3)


# ---------------------------------------------------------------------------
# Selected-probability extraction
# ---------------------------------------------------------------------------


class TestSelectedProbabilityExtraction:
    def test_bool_true_selection_uses_probability_true(self):
        # P(True) = 0.8, selected True, correct -> (0.8 - 1)^2 = 0.04.
        # An implementation that always uses P(False) fails this test.
        observation = bool_observation_with_probability_true(0.8, resolved_truth(True))
        assert observation.selected_value is True
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([observation]))
        assert result.value == pytest.approx((1.0 - 0.8) ** 2)

    def test_bool_false_selection_uses_probability_false(self):
        # P(True) = 0.2 so P(False) = 0.8, selected False, truth True -> wrong.
        # (0.8 - 0)^2 = 0.64. An implementation that always uses P(True)
        # would score (0.2)^2 = 0.04 and fail this test.
        observation = bool_observation_with_probability_true(0.2, resolved_truth(True))
        assert observation.selected_value is False
        assert observation.correct is False
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([observation]))
        assert result.value == pytest.approx((1.0 - 0.2) ** 2)
        assert result.value != pytest.approx(0.2**2)

    def test_bool_tie_honors_deterministic_false_selection(self):
        # P(True) = 0.5 -> the runtime records selected = False with p = 0.5.
        observation = bool_observation_with_probability_true(0.5, resolved_truth(True))
        assert observation.selected_value is False
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([observation]))
        assert result.value == pytest.approx((1.0 - 0.5) ** 2)

    def test_choice_uses_recorded_semantic_probability(self):
        # Standard three-candidate decision: label B lands on "shipping" with
        # p = 0.6. Correct and wrong variants.
        correct = choice_observation(resolved_truth("shipping"))
        wrong = choice_observation(resolved_truth("billing"))
        assert correct.selected_value == "shipping"
        assert correct.probabilities[1] == ("shipping", pytest.approx(0.6))
        assert evaluate_uncalibrated_winner_brier(evaluation_dataset([correct])).value == (
            pytest.approx((1.0 - 0.6) ** 2)
        )
        assert evaluate_uncalibrated_winner_brier(evaluation_dataset([wrong])).value == (
            pytest.approx(0.6**2)
        )

    def test_choice_permutation_uses_semantic_names_not_scoring_labels(self):
        # Reordering the candidates moves the scoring labels: in the permuted
        # decision "returns" takes label A and "shipping" takes label C, so the
        # runtime distribution becomes returns 0.2 / billing 0.6 / shipping
        # 0.2. The winner (argmax) is now "billing" at p = 0.6, and "shipping"
        # sits at p = 0.2. Any implementation that looks probabilities up by
        # scoring label, or that always reads the label-B slot, produces the
        # wrong numbers below.
        permuted = ChoiceDecision(
            "Which department should handle this request?",
            context="The customer asks about a refund for a damaged parcel.",
            choices={
                "returns": "Refunds, exchanges, warranty",
                "billing": "Payment, charges, invoices",
                "shipping": "Delivery, couriers, parcels",
            },
        )
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(permuted)
        result = ChoiceResult(
            certainty=Certainty.from_probabilities([0.2, 0.6, 0.2]),
            method="categorical_token_logits",
            trace_id=evaluation.trace.trace_id,
            probabilities={"returns": 0.2, "billing": 0.6, "shipping": 0.2},
        )
        winner = CalibrationObservation.from_evaluation(
            (result, evaluation.trace), resolved_truth("billing")
        )
        assert winner.selected_value == "billing"
        assert winner.probabilities == (
            ("returns", pytest.approx(0.2)),
            ("billing", pytest.approx(0.6)),
            ("shipping", pytest.approx(0.2)),
        )
        assert evaluate_uncalibrated_winner_brier(
            evaluation_dataset([winner])
        ).value == pytest.approx((1.0 - 0.6) ** 2)

    def test_choice_never_recomputes_the_winner(self):
        # A recorded selection that is not the argmax must be used as
        # recorded: "returns" carries p = 0.0 and is wrong, so the contribution
        # is (0.0 - 0.0)^2 = 0.0 even though the argmax is "billing"/"shipping"
        # at p = 0.5. Recomputing the winner would score 0.25.
        observation = recorded_selection_observation(
            three_way_probabilities(billing=0.5, shipping=0.5),
            "returns",
            resolved_truth("shipping"),
        )
        assert observation.selected_value == "returns"
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([observation]))
        assert result.value == pytest.approx(0.0)
        assert result.value != pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Evaluation cohort admission
# ---------------------------------------------------------------------------


class TestEvaluationCohortAdmission:
    def test_empty_observations_rejected(self):
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            evaluation_cohort([])

    def test_observations_must_be_a_tuple(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationEvaluationCohort(
                observations=[observation],  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_none_observations_rejected_with_invalid_decision_error(self):
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationEvaluationCohort(
                observations=None,  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_generator_observations_rejected_with_invalid_decision_error(self):
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationEvaluationCohort(
                observations=(item for item in [choice_observation()]),  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_non_observation_element_rejected_with_invalid_decision_error(self):
        with pytest.raises(InvalidDecisionError, match="expected a CalibrationObservation"):
            CalibrationEvaluationCohort(
                observations=("not an observation",),  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_duck_typed_look_alike_not_admitted(self):
        class LookAlike:
            decision_family = "choice"
            outcome_order = ("billing", "shipping", "returns")
            probabilities = (("billing", 1.0), ("shipping", 0.0), ("returns", 0.0))
            selected_value = "billing"
            correct = True
            status = "resolved"
            fit_eligible = True
            fingerprint = "fabricated"

        with pytest.raises(InvalidDecisionError, match="expected a CalibrationObservation"):
            CalibrationEvaluationCohort(
                observations=(LookAlike(),),  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_unresolved_rows_retained_in_cohort(self):
        observation = choice_observation(unresolved_truth())
        cohort = evaluation_cohort([observation])
        assert cohort.source_count == 1
        assert cohort.unresolved_count == 1
        assert cohort.eligible_count == 0

    def test_taxonomy_miss_rows_retained_in_cohort(self):
        # "account" is not among the billing/shipping/returns candidates.
        observation = choice_observation(resolved_truth("account"))
        assert observation.status.name == "TAXONOMY_MISS"
        assert observation.correct is None
        cohort = evaluation_cohort([observation])
        assert cohort.source_count == 1
        assert cohort.taxonomy_miss_count == 1
        assert cohort.eligible_count == 0

    def test_unadjudicated_rows_retained_in_cohort(self):
        observation = choice_observation(resolved_truth("shipping", adjudicated=False))
        cohort = evaluation_cohort([observation])
        assert cohort.source_count == 1
        assert cohort.unadjudicated_resolved_count == 1
        assert cohort.eligible_count == 0

    def test_mixed_binding_rejected(self):
        same = choice_observation()
        other_model = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
            evaluation_cohort([same, other_model])

    def test_mixed_ground_truth_semantics_rejected(self):
        same = choice_observation()
        other_rule = choice_observation(
            resolved_truth("shipping", labeling_rule="a different labeling rule")
        )
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics"):
            evaluation_cohort([same, other_rule])

    def test_different_label_source_only_is_accepted(self):
        human = choice_observation()
        reviewer = choice_observation(resolved_truth("shipping", label_source="reviewer"))
        cohort = evaluation_cohort([human, reviewer])
        assert len(cohort.observations) == 2

    def test_split_role_must_be_an_evaluation_split_role(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="split_role must be"):
            CalibrationEvaluationCohort(
                observations=(observation,),
                split_role="validation",  # type: ignore[arg-type]
                split_id="split",
            )
        with pytest.raises(InvalidDecisionError, match="split_role must be"):
            CalibrationEvaluationCohort(
                observations=(observation,),
                split_role="train",  # type: ignore[arg-type]
                split_id="split",
            )

    def test_split_id_must_be_non_empty(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="split_id must be a non-empty"):
            CalibrationEvaluationCohort(
                observations=(observation,),
                split_role=EvaluationSplitRole.TEST,
                split_id="",
            )

    def test_rejection_messages_distinguish_reason_kinds(self):
        # One message that mixes a non-observation element, a binding
        # mismatch, a semantics mismatch, and invalid split metadata; the
        # message must distinguish all four kinds and count the observations
        # and reasons.
        unresolved = choice_observation(unresolved_truth())
        other_model = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        other_rule = choice_observation(
            resolved_truth("shipping", labeling_rule="a different labeling rule")
        )
        with pytest.raises(InvalidDecisionError) as excinfo:
            CalibrationEvaluationCohort(
                observations=(unresolved, other_model, other_rule),
                split_role="validation",  # type: ignore[arg-type]
                split_id="",
            )
        message = str(excinfo.value)
        assert "3 observation(s)" in message
        assert "4 reason(s)" in message
        assert "binding canonical payload" in message
        assert "ground-truth semantics" in message
        assert "split_role must be an EvaluationSplitRole" in message
        assert "split_id must be a non-empty" in message


# ---------------------------------------------------------------------------
# Cohort identity
# ---------------------------------------------------------------------------


class TestEvaluationCohortIdentity:
    def test_row_order_does_not_change_fingerprint(self):
        o1 = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        o2 = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
            resolved_truth("shipping"),
        )
        first = evaluation_cohort([o1, o2])
        second = evaluation_cohort([o2, o1])
        assert first.fingerprint == second.fingerprint

    def test_multiplicity_is_preserved(self):
        o1 = choice_observation()
        o2 = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        smaller = evaluation_cohort([o1, o2])
        larger = evaluation_cohort([o1, o1, o2])
        assert smaller.fingerprint != larger.fingerprint
        assert larger.source_count == 3

    def test_split_role_changes_fingerprint(self):
        observation = choice_observation()
        as_validation = evaluation_cohort([observation], split_role=EvaluationSplitRole.VALIDATION)
        as_test = evaluation_cohort([observation], split_role=EvaluationSplitRole.TEST)
        assert as_validation.fingerprint != as_test.fingerprint

    def test_split_id_changes_fingerprint(self):
        observation = choice_observation()
        first = evaluation_cohort([observation], split_id="holdout-a")
        second = evaluation_cohort([observation], split_id="holdout-b")
        assert first.fingerprint != second.fingerprint

    def test_fingerprint_version_is_one(self):
        assert CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION == 1

    def test_fingerprint_payload_commits_identity_versions_and_all_rows(self):
        eligible = choice_observation()
        taxonomy_miss = choice_observation(resolved_truth("account"))
        cohort = evaluation_cohort(
            [eligible, taxonomy_miss],
            split_role=EvaluationSplitRole.VALIDATION,
            split_id="holdout",
        )
        expected: dict = {
            "v": CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
            "split_role": "validation",
            "split_id": "holdout",
            "binding_fingerprint": cohort.binding.fingerprint,
            "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "ground_truth_semantics_fingerprint": (cohort.ground_truth_semantics.fingerprint),
            "ground_truth_semantics_fingerprint_version": (
                GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
            ),
            "observation_fingerprints": sorted(
                [observation.fingerprint for observation in cohort.observations]
            ),
        }
        assert cohort.fingerprint == fingerprint(expected)


# ---------------------------------------------------------------------------
# Cohort partition and exclusion accounting
# ---------------------------------------------------------------------------


class TestEvaluationCohortPartition:
    def test_partition_precedence_counts_each_row_once(self):
        # An unresolved ground truth is also unadjudicated; the documented
        # precedence counts it exactly once, as unresolved.
        unresolved = choice_observation(unresolved_truth())
        assert unresolved.status is CalibrationObservationStatus.UNRESOLVED
        cohort = evaluation_cohort([unresolved])
        assert cohort.source_count == 1
        assert cohort.unresolved_count == 1
        assert cohort.unadjudicated_resolved_count == 0
        assert cohort.eligible_count == 0
        assert cohort.taxonomy_miss_count == 0
        assert (
            cohort.source_count
            == cohort.eligible_count
            + cohort.taxonomy_miss_count
            + cohort.unresolved_count
            + cohort.unadjudicated_resolved_count
        )

    def test_partition_covers_all_four_buckets(self):
        eligible = choice_observation()
        taxonomy_miss = choice_observation(resolved_truth("account"))
        unresolved = choice_observation(unresolved_truth())
        unadjudicated = choice_observation(resolved_truth("shipping", adjudicated=False))
        cohort = evaluation_cohort([eligible, taxonomy_miss, unresolved, unadjudicated])
        assert cohort.source_count == 4
        assert cohort.eligible_count == 1
        assert cohort.taxonomy_miss_count == 1
        assert cohort.unresolved_count == 1
        assert cohort.unadjudicated_resolved_count == 1
        assert (
            cohort.source_count
            == cohort.eligible_count
            + cohort.taxonomy_miss_count
            + cohort.unresolved_count
            + cohort.unadjudicated_resolved_count
        )
        assert cohort.eligible_observations == (eligible,)

    def test_accounting_identity_holds_on_several_cohorts(self):
        cohorts = [
            evaluation_cohort([choice_observation()]),
            evaluation_cohort(
                [
                    choice_observation(),
                    choice_observation(resolved_truth("account")),
                    choice_observation(unresolved_truth()),
                    choice_observation(resolved_truth("shipping", adjudicated=False)),
                ]
            ),
            evaluation_cohort(
                [
                    choice_observation(resolved_truth("account")),
                    choice_observation(resolved_truth("account")),
                ]
            ),
        ]
        for cohort in cohorts:
            assert (
                cohort.source_count
                == cohort.eligible_count
                + cohort.taxonomy_miss_count
                + cohort.unresolved_count
                + cohort.unadjudicated_resolved_count
            )

    def test_zero_eligible_cohort_is_valid_but_projection_fails(self):
        cohort = evaluation_cohort([choice_observation(resolved_truth("account"))])
        assert cohort.source_count == 1
        assert cohort.taxonomy_miss_count == 1
        assert cohort.eligible_count == 0
        assert cohort.fingerprint  # the cohort itself is a valid artifact
        with pytest.raises(InvalidDecisionError, match="zero metric-eligible rows"):
            CalibrationEvaluationDataset.from_cohort(cohort)

    def test_all_taxonomy_miss_cohort_is_a_legitimate_provenance_artifact(self):
        cohort = evaluation_cohort(
            [
                choice_observation(resolved_truth("account")),
                choice_observation(resolved_truth("fraud")),
            ]
        )
        assert cohort.eligible_count == 0
        assert cohort.taxonomy_miss_count == 2
        assert len(cohort.fingerprint) == 64


# ---------------------------------------------------------------------------
# Evaluation dataset projection
# ---------------------------------------------------------------------------


class TestEvaluationDatasetProjection:
    def test_direct_construction_rejected(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="from_cohort"):
            CalibrationEvaluationDataset(
                observations=(observation,),
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )
        with pytest.raises(InvalidDecisionError, match="from_cohort"):
            CalibrationEvaluationDataset()

    def test_from_cohort_requires_a_cohort(self):
        with pytest.raises(InvalidDecisionError, match="CalibrationEvaluationCohort"):
            CalibrationEvaluationDataset.from_cohort("not a cohort")  # type: ignore[arg-type]

    def test_projection_derives_eligible_rows_and_accounting(self):
        eligible = choice_observation()
        taxonomy_miss = choice_observation(resolved_truth("account"))
        unresolved = choice_observation(unresolved_truth())
        unadjudicated = choice_observation(resolved_truth("shipping", adjudicated=False))
        cohort = evaluation_cohort([eligible, taxonomy_miss, unresolved, unadjudicated])
        dataset = CalibrationEvaluationDataset.from_cohort(cohort)
        assert dataset.observations == (eligible,)
        assert dataset.split_role == cohort.split_role
        assert dataset.split_id == cohort.split_id
        assert dataset.source_cohort_fingerprint == cohort.fingerprint
        assert dataset.source_count == 4
        assert dataset.taxonomy_miss_count == 1
        assert dataset.unresolved_count == 1
        assert dataset.unadjudicated_resolved_count == 1
        assert len(dataset.observations) == 1

    def test_projection_rejects_manual_subsets_without_cohort_provenance(self):
        # A caller cannot present an arbitrary already-filtered tuple as an
        # evaluation dataset: the supported path derives the projection from
        # a declared cohort, so no silent manual-subset identity exists.
        eligible = choice_observation()
        taxonomy_miss = choice_observation(resolved_truth("account"))
        with pytest.raises(InvalidDecisionError, match="from_cohort"):
            CalibrationEvaluationDataset(
                observations=(eligible,),
                split_role=EvaluationSplitRole.TEST,
                split_id="eval-split-1",
            )
        # The cohort path keeps the excluded row in the accounting.
        dataset = evaluation_dataset([eligible, taxonomy_miss])
        assert dataset.source_count == 2
        assert dataset.taxonomy_miss_count == 1
        assert len(dataset.observations) == 1

    def test_dataset_binding_and_semantics_derived_from_cohort(self):
        cohort = evaluation_cohort([choice_observation()])
        dataset = CalibrationEvaluationDataset.from_cohort(cohort)
        assert dataset.binding == cohort.binding
        assert dataset.ground_truth_semantics == cohort.ground_truth_semantics


# ---------------------------------------------------------------------------
# The decisive A-vs-B regression: metric exclusion is provenance
# ---------------------------------------------------------------------------


class TestCohortProvenanceNoConflation:
    def _eligible_pair(self):
        first = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        second = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
            resolved_truth("shipping"),
        )
        return first, second

    def _taxonomy_miss_rows(self, count):
        # Distinct taxonomy-miss rows so multiplicity is unambiguous.
        return [choice_observation(resolved_truth(f"account-{index}")) for index in range(count)]

    def test_identical_scored_rows_with_different_exclusions_do_not_collapse(self):
        eligible_a, eligible_b = self._eligible_pair()
        cohort_a = evaluation_cohort([eligible_a, eligible_b])
        cohort_b = evaluation_cohort([eligible_a, eligible_b, *self._taxonomy_miss_rows(3)])
        assert cohort_a.eligible_observations == cohort_b.eligible_observations
        assert cohort_a.eligible_count == cohort_b.eligible_count == 2
        assert cohort_a.source_count == 2
        assert cohort_b.source_count == 5
        assert cohort_a.taxonomy_miss_count == 0
        assert cohort_b.taxonomy_miss_count == 3
        assert cohort_a.fingerprint != cohort_b.fingerprint

        dataset_a = CalibrationEvaluationDataset.from_cohort(cohort_a)
        dataset_b = CalibrationEvaluationDataset.from_cohort(cohort_b)
        assert dataset_a.observations == dataset_b.observations
        assert dataset_a.fingerprint != dataset_b.fingerprint

        brier_a = evaluate_uncalibrated_winner_brier(dataset_a)
        brier_b = evaluate_uncalibrated_winner_brier(dataset_b)
        assert brier_a.value == brier_b.value
        assert brier_a.count == brier_b.count == 2
        assert brier_a.fingerprint != brier_b.fingerprint

        log_loss_a = evaluate_uncalibrated_winner_log_loss(dataset_a)
        log_loss_b = evaluate_uncalibrated_winner_log_loss(dataset_b)
        assert log_loss_a.value == log_loss_b.value
        assert log_loss_a.count == log_loss_b.count == 2
        assert log_loss_a.fingerprint != log_loss_b.fingerprint

    def test_metric_artifacts_commit_source_cohort_provenance(self):
        eligible_a, eligible_b = self._eligible_pair()
        cohort_b = evaluation_cohort([eligible_a, eligible_b, *self._taxonomy_miss_rows(3)])
        dataset_b = CalibrationEvaluationDataset.from_cohort(cohort_b)
        for evaluate in (
            evaluate_uncalibrated_winner_brier,
            evaluate_uncalibrated_winner_log_loss,
        ):
            result = evaluate(dataset_b)
            assert result.source_cohort_fingerprint == cohort_b.fingerprint
            assert result.source_cohort_fingerprint_version == (
                CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION
            )
            assert result.source_count == 5
            assert result.taxonomy_miss_count == 3
            assert result.unresolved_count == 0
            assert result.unadjudicated_resolved_count == 0
            assert result.count == 2

    def test_both_metrics_expose_identical_cohort_accounting(self):
        cohort = evaluation_cohort(
            [
                choice_observation(),
                choice_observation(resolved_truth("account")),
                choice_observation(unresolved_truth()),
                choice_observation(resolved_truth("shipping", adjudicated=False)),
            ]
        )
        dataset = CalibrationEvaluationDataset.from_cohort(cohort)
        brier = evaluate_uncalibrated_winner_brier(dataset)
        log_loss = evaluate_uncalibrated_winner_log_loss(dataset)
        assert brier.source_cohort_fingerprint == log_loss.source_cohort_fingerprint
        assert brier.source_cohort_fingerprint_version == log_loss.source_cohort_fingerprint_version
        assert brier.source_count == log_loss.source_count == 4
        assert brier.taxonomy_miss_count == log_loss.taxonomy_miss_count == 1
        assert brier.unresolved_count == log_loss.unresolved_count == 1
        assert brier.unadjudicated_resolved_count == log_loss.unadjudicated_resolved_count == 1
        assert brier.count == log_loss.count == 1
        assert brier.evaluation_dataset_fingerprint == log_loss.evaluation_dataset_fingerprint


# ---------------------------------------------------------------------------
# Evaluation dataset identity
# ---------------------------------------------------------------------------


class TestEvaluationDatasetIdentity:
    def test_row_order_does_not_change_fingerprint(self):
        o1 = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        o2 = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
            resolved_truth("shipping"),
        )
        first = evaluation_dataset([o1, o2])
        second = evaluation_dataset([o2, o1])
        assert first.fingerprint == second.fingerprint

    def test_multiplicity_is_preserved(self):
        o1 = choice_observation()
        o2 = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        smaller = evaluation_dataset([o1, o2])
        larger = evaluation_dataset([o1, o1, o2])
        assert smaller.fingerprint != larger.fingerprint

    def test_split_role_changes_fingerprint(self):
        observation = choice_observation()
        as_validation = evaluation_dataset([observation], split_role=EvaluationSplitRole.VALIDATION)
        as_test = evaluation_dataset([observation], split_role=EvaluationSplitRole.TEST)
        assert as_validation.fingerprint != as_test.fingerprint

    def test_split_id_changes_fingerprint(self):
        observation = choice_observation()
        first = evaluation_dataset([observation], split_id="holdout-a")
        second = evaluation_dataset([observation], split_id="holdout-b")
        assert first.fingerprint != second.fingerprint

    def test_fingerprint_version_is_two(self):
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 2

    def test_fingerprint_payload_commits_identity_versions_and_cohort_provenance(self):
        observation = choice_observation()
        cohort = evaluation_cohort(
            [observation], split_role=EvaluationSplitRole.VALIDATION, split_id="holdout"
        )
        dataset = CalibrationEvaluationDataset.from_cohort(cohort)
        expected: dict = {
            "v": CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
            "source_cohort_fingerprint": cohort.fingerprint,
            "source_cohort_fingerprint_version": (
                CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION
            ),
            "split_role": "validation",
            "split_id": "holdout",
            "binding_fingerprint": dataset.binding.fingerprint,
            "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "ground_truth_semantics_fingerprint": (dataset.ground_truth_semantics.fingerprint),
            "ground_truth_semantics_fingerprint_version": (
                GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
            ),
            "observation_fingerprints": sorted(
                [observation.fingerprint for observation in dataset.observations]
            ),
            "source_count": 1,
            "eligible_count": 1,
            "taxonomy_miss_count": 0,
            "unresolved_count": 0,
            "unadjudicated_resolved_count": 0,
        }
        assert dataset.fingerprint == fingerprint(expected)


# ---------------------------------------------------------------------------
# Brier result artifact
# ---------------------------------------------------------------------------


class TestBrierEvaluationResult:
    def test_value_and_fingerprint_are_deterministic(self):
        first = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        second = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        assert first.value == second.value
        assert first.fingerprint == second.fingerprint

    def test_identity_fields_are_actually_set(self):
        # D1 regression guard: with the broken hand-written __init__ these
        # attributes were never assigned at all.
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        assert result.metric_id == "brier"
        assert result.metric_version == 1
        assert result.target == "winner_correctness"
        assert result.input_score_id == "uncalibrated-selected-probability"
        assert result.input_score_version == 1
        assert result.count == 1
        assert isinstance(result.value, float)

    def test_canonical_payload_commits_identity_and_value(self):
        dataset = evaluation_dataset([choice_observation()])
        result = evaluate_uncalibrated_winner_brier(dataset)
        payload = result.canonical_payload()
        assert payload == {
            "v": BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION,
            "metric_id": "brier",
            "metric_version": 1,
            "target": "winner_correctness",
            "input_score_id": "uncalibrated-selected-probability",
            "input_score_version": 1,
            "evaluation_dataset_fingerprint": dataset.fingerprint,
            "evaluation_dataset_fingerprint_version": (
                CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION
            ),
            "source_cohort_fingerprint": dataset.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": (
                CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION
            ),
            "source_count": 1,
            "taxonomy_miss_count": 0,
            "unresolved_count": 0,
            "unadjudicated_resolved_count": 0,
            "configuration": {},
            "count": 1,
            "value": result.value,
        }
        assert result.fingerprint == fingerprint(payload)

    def test_fingerprint_is_stable_across_payload_mutation(self):
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        before = result.fingerprint
        result.canonical_payload()["value"] = 0.123
        assert result.fingerprint == before

    def test_direct_construction_rejected(self):
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_brier"):
            BrierEvaluationResult()
        dataset = evaluation_dataset([choice_observation()])
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_brier"):
            BrierEvaluationResult(dataset)

    def test_wrong_construction_token_rejected(self):
        dataset = evaluation_dataset([choice_observation()])
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_brier"):
            BrierEvaluationResult(dataset, _construction_token=object())

    def test_dataclasses_replace_without_fields_rejected(self):
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_brier"):
            replace(result)

    def test_dataclasses_replace_with_value_rejected_by_dataclasses(self):
        # replace(result, value=...) fails inside dataclasses itself (value is
        # declared init=False) and never reaches __init__; the rejection is a
        # ValueError, not an InvalidDecisionError.
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        with pytest.raises(ValueError, match="init=False"):
            replace(result, value=0.01)

    def test_evaluator_rejects_non_dataset_argument(self):
        with pytest.raises(InvalidDecisionError, match="CalibrationEvaluationDataset"):
            evaluate_uncalibrated_winner_brier("not a dataset")  # type: ignore[arg-type]

    def test_forging_the_value_is_rejected(self):
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        with pytest.raises(AttributeError):
            result.value = 0.5  # type: ignore[misc]
        assert result.value != 0.5

    def test_identical_value_from_different_datasets_yields_different_fingerprints(self):
        # Both datasets produce Brier = 0.25, but they commit different
        # observation rows, so the result fingerprints must differ.
        neutral_correct = bool_observation_with_probability_true(0.5, resolved_truth(False))
        neutral_wrong = bool_observation_with_probability_true(0.5, resolved_truth(True))
        first = evaluate_uncalibrated_winner_brier(evaluation_dataset([neutral_correct]))
        second = evaluate_uncalibrated_winner_brier(evaluation_dataset([neutral_wrong]))
        assert first.value == pytest.approx(0.25)
        assert second.value == pytest.approx(0.25)
        assert (
            first.canonical_payload()["evaluation_dataset_fingerprint"]
            != (second.canonical_payload()["evaluation_dataset_fingerprint"])
        )
        assert first.fingerprint != second.fingerprint

    def test_result_commits_dataset_identity_through_split_id(self):
        observation = choice_observation()
        first = evaluate_uncalibrated_winner_brier(
            evaluation_dataset([observation], split_id="holdout-a")
        )
        second = evaluate_uncalibrated_winner_brier(
            evaluation_dataset([observation], split_id="holdout-b")
        )
        assert first.value == second.value
        assert first.fingerprint != second.fingerprint


# ---------------------------------------------------------------------------
# No-conflation guards
# ---------------------------------------------------------------------------


class TestNoConflation:
    def test_result_has_no_calibration_or_prediction_fields(self):
        field_names = {field.name for field in fields(BrierEvaluationResult)}
        assert "configuration" not in field_names
        assert "calibrated" not in field_names
        assert "predicted_correctness" not in field_names

    def test_result_instances_carry_no_calibration_attributes(self):
        result = evaluate_uncalibrated_winner_brier(evaluation_dataset([choice_observation()]))
        assert not hasattr(result, "calibrated")
        assert not hasattr(result, "predicted_correctness")
        assert not hasattr(result, "configuration")

    def test_no_calibration_profile_exists(self):
        import probvenance.calibration as calibration_module

        assert not hasattr(calibration_module, "CalibrationProfile")

    def test_evaluation_does_not_mutate_observations_or_dataset(self):
        o1 = bool_observation()
        o2 = bool_observation(resolved_truth(False))
        dataset = evaluation_dataset([o1, o2])
        before = (
            dataset.fingerprint,
            dataset.observations,
            o1.fingerprint,
            o1.probabilities,
            o1.selected_value,
            o1.correct,
            o2.fingerprint,
            o2.probabilities,
            o2.selected_value,
            o2.correct,
        )
        result = evaluate_uncalibrated_winner_brier(dataset)
        after = (
            dataset.fingerprint,
            dataset.observations,
            o1.fingerprint,
            o1.probabilities,
            o1.selected_value,
            o1.correct,
            o2.fingerprint,
            o2.probabilities,
            o2.selected_value,
            o2.correct,
        )
        assert before == after
        assert result.count == 2
        assert Probvenance is not None  # runtime import surface untouched


# ---------------------------------------------------------------------------
# Version guards
# ---------------------------------------------------------------------------


class TestVersionGuards:
    def test_new_versions_are_two(self):
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 2
        assert BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION == 2
        assert UNCALIBRATED_SELECTED_PROBABILITY_VERSION == 1
        assert BRIER_METRIC_VERSION == 1

    def test_pre_existing_versions_unchanged(self):
        assert CALIBRATION_BINDING_FINGERPRINT_VERSION == 1
        assert GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION == 1
        assert CALIBRATION_OBSERVATION_FINGERPRINT_VERSION == 1
        assert CALIBRATION_DATASET_FINGERPRINT_VERSION == 2
        assert RENDERING_SEMANTICS_VERSION == 1

    def test_new_identities_are_string_constants(self):
        assert UNCALIBRATED_SELECTED_PROBABILITY_ID == "uncalibrated-selected-probability"
        assert BRIER_METRIC_ID == "brier"
        assert BRIER_METRIC_VERSION == 1
        assert math.isfinite(float(BRIER_METRIC_VERSION))


# ---------------------------------------------------------------------------
# Log-loss: finite values
# ---------------------------------------------------------------------------


class TestLogLossFinite:
    def test_neutral_probability_scores_ln_two(self):
        # P(True) = 0.5 selects False (the deterministic tie), so the selected
        # probability is 0.5 on both sides.
        correct_side = bool_observation_with_probability_true(0.5, resolved_truth(False))
        wrong_side = bool_observation_with_probability_true(0.5, resolved_truth(True))
        assert correct_side.selected_value is False
        assert correct_side.correct is True
        assert wrong_side.correct is False
        assert evaluate_uncalibrated_winner_log_loss(
            evaluation_dataset([correct_side])
        ).value == pytest.approx(math.log(2.0))
        assert evaluate_uncalibrated_winner_log_loss(
            evaluation_dataset([wrong_side])
        ).value == pytest.approx(math.log(2.0))

    def test_correct_finite_scores_negative_log_p(self):
        observation = bool_observation_with_probability_true(0.8, resolved_truth(True))
        assert observation.selected_value is True
        assert observation.correct is True
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert result.value == pytest.approx(-math.log(0.8))

    def test_wrong_finite_scores_negative_log_one_minus_p(self):
        # An implementation that always used P(True) would give -ln(0.8) here.
        observation = bool_observation_with_probability_true(0.8, resolved_truth(False))
        assert observation.selected_value is True
        assert observation.correct is False
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert result.value == pytest.approx(-math.log(0.2))

    def test_hand_calculated_three_observation_value(self):
        # p = [0.8, 0.6, 0.2], y = [1, 0, 0]
        # loss = (-ln(0.8) - ln(0.4) - ln(0.8)) / 3
        confident_right = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        confident_wrong = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
            resolved_truth("billing"),
        )
        low_confidence_wrong = recorded_selection_observation(
            three_way_probabilities(billing=0.2, shipping=0.4, returns=0.4),
            "billing",
            resolved_truth("shipping"),
        )
        assert confident_right.correct is True
        assert confident_wrong.correct is False
        assert low_confidence_wrong.correct is False
        result = evaluate_uncalibrated_winner_log_loss(
            evaluation_dataset([confident_right, confident_wrong, low_confidence_wrong])
        )
        expected = (-math.log(0.8) - math.log(0.4) - math.log(0.8)) / 3
        assert result.value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Log-loss: exact endpoints
# ---------------------------------------------------------------------------


class TestLogLossExactEndpoints:
    def test_correct_deterministic_endpoints_score_zero(self):
        assert _binary_log_loss_term(1.0, True) == 0.0
        assert _binary_log_loss_term(0.0, False) == 0.0

    def test_impossible_observed_endpoints_score_positive_infinity(self):
        assert _binary_log_loss_term(0.0, True) == math.inf
        assert _binary_log_loss_term(1.0, False) == math.inf

    def test_integration_confident_correct_scores_zero(self):
        observation = choice_observation_with_probabilities(
            three_way_probabilities(billing=1.0), resolved_truth("billing")
        )
        assert observation.selected_value == "billing"
        assert observation.correct is True
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert result.value == 0.0

    def test_integration_confident_wrong_scores_positive_infinity(self):
        observation = choice_observation_with_probabilities(
            three_way_probabilities(billing=1.0), resolved_truth("shipping")
        )
        assert observation.correct is False
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert result.value == math.inf

    def test_integration_zero_probability_wrong_selection_scores_zero(self):
        # The recorded selection is "billing" at p = 0.0 and the truth is
        # "returns": a correct deterministic endpoint scores exactly 0.
        observation = recorded_selection_observation(
            three_way_probabilities(shipping=1.0), "billing", resolved_truth("returns")
        )
        assert observation.selected_value == "billing"
        assert observation.correct is False
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert result.value == 0.0

    def test_integration_zero_probability_correct_selection_scores_infinity(self):
        observation = recorded_selection_observation(
            three_way_probabilities(shipping=1.0), "billing", resolved_truth("billing")
        )
        assert observation.correct is True
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert result.value == math.inf


# ---------------------------------------------------------------------------
# Log-loss: near-boundary behaviour
# ---------------------------------------------------------------------------


class TestLogLossNearBoundary:
    def test_near_zero_probability_stays_finite(self):
        tiny = math.nextafter(0.0, 1.0)
        term = _binary_log_loss_term(tiny, True)
        assert math.isfinite(term)
        assert term == pytest.approx(-math.log(tiny))

    def test_near_one_probability_stays_finite(self):
        almost_one = math.nextafter(1.0, 0.0)
        term = _binary_log_loss_term(almost_one, False)
        assert math.isfinite(term)
        assert term == pytest.approx(-math.log1p(-almost_one))

    def test_integration_near_one_wrong_selection_stays_finite(self):
        # Only the exact impossible event scores infinity: a representable
        # probability arbitrarily close to 1 must stay finite.
        almost_one = math.nextafter(1.0, 0.0)
        observation = bool_observation_with_probability_true(almost_one, resolved_truth(False))
        assert observation.selected_value is True
        assert observation.correct is False
        result = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))
        assert math.isfinite(result.value)
        assert result.value > 0.0


# ---------------------------------------------------------------------------
# Log-loss: infinity canonicalization
# ---------------------------------------------------------------------------


class TestLogLossInfinityCanonicalization:
    def _infinity_result(self):
        observation = choice_observation_with_probabilities(
            three_way_probabilities(billing=1.0), resolved_truth("shipping")
        )
        return evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))

    def _finite_result(self):
        observation = choice_observation_with_probabilities(
            three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
            resolved_truth("billing"),
        )
        return evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation]))

    def test_positive_infinity_is_the_python_metric_value(self):
        result = self._infinity_result()
        assert result.value == math.inf
        assert math.isinf(result.value)
        assert result.value > 0.0

    def test_infinity_result_is_fingerprintable(self):
        result = self._infinity_result()
        encoded = result.canonical_payload()["value"]
        assert isinstance(encoded, dict)
        assert encoded == {"kind": "positive_infinity", "number": None}
        # fingerprint() canonicalises the payload, so it would raise
        # FingerprintError if a non-finite float had leaked into the payload.
        assert isinstance(result.fingerprint, str)
        assert len(result.fingerprint) == 64

    def test_finite_payload_is_structurally_encoded(self):
        encoded = self._finite_result().canonical_payload()["value"]
        assert isinstance(encoded, dict)
        assert encoded["kind"] == "finite"
        number = encoded["number"]
        assert isinstance(number, float)
        assert math.isfinite(number)

    def test_no_non_finite_number_enters_the_canonical_payload(self):
        result = self._infinity_result()
        payload = result.canonical_payload()
        encoded = payload["value"]
        assert isinstance(encoded, dict)
        assert encoded["number"] is None
        # Canonicalising the whole payload is the real guard: it rejects any
        # non-finite JSON number anywhere in the payload.
        assert fingerprint(payload) == result.fingerprint

    def test_finite_and_infinite_values_have_different_identities(self):
        finite = self._finite_result()
        infinite = self._infinity_result()
        assert finite.canonical_payload()["value"] != infinite.canonical_payload()["value"]
        assert finite.fingerprint != infinite.fingerprint

    def test_identical_value_from_different_datasets_yields_different_fingerprints(self):
        def build(split_id):
            observation = choice_observation_with_probabilities(
                three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
                resolved_truth("billing"),
            )
            return evaluate_uncalibrated_winner_log_loss(
                evaluation_dataset([observation], split_id=split_id)
            )

        first = build("eval-a")
        second = build("eval-b")
        assert first.value == pytest.approx(second.value)
        assert first.evaluation_dataset_fingerprint != second.evaluation_dataset_fingerprint
        assert first.fingerprint != second.fingerprint


# ---------------------------------------------------------------------------
# Log-loss: result artifact
# ---------------------------------------------------------------------------


class TestLogLossEvaluationResult:
    def _result(self):
        return evaluate_uncalibrated_winner_log_loss(evaluation_dataset([choice_observation()]))

    def test_identity_fields_are_actually_set(self):
        result = self._result()
        assert result.metric_id == "log-loss"
        assert result.metric_version == 1
        assert result.target == "winner_correctness"
        assert result.input_score_id == "uncalibrated-selected-probability"
        assert result.input_score_version == 1
        assert result.count == 1
        assert isinstance(result.value, float)

    def test_configuration_is_fixed_and_committed(self):
        result = self._result()
        assert result.canonical_payload()["configuration"] == {
            "boundary_policy": "exact",
            "log_base": "e",
        }

    def test_no_caller_controlled_configuration_field_exists(self):
        assert "configuration" not in {f.name for f in fields(LogLossEvaluationResult)}

    def test_value_and_fingerprint_are_deterministic(self):
        assert self._result().value == self._result().value
        assert self._result().fingerprint == self._result().fingerprint

    def test_fingerprint_hashes_the_canonical_payload(self):
        result = self._result()
        assert result.fingerprint == fingerprint(result.canonical_payload())

    def test_direct_construction_rejected(self):
        with pytest.raises(InvalidDecisionError):
            LogLossEvaluationResult()

    def test_dataclasses_replace_without_fields_rejected(self):
        with pytest.raises(InvalidDecisionError):
            replace(self._result())

    def test_dataclasses_replace_with_value_rejected_by_dataclasses(self):
        with pytest.raises(ValueError):
            replace(self._result(), value=0.01)

    def test_evaluator_rejects_non_dataset_argument(self):
        with pytest.raises(InvalidDecisionError, match="CalibrationEvaluationDataset"):
            evaluate_uncalibrated_winner_log_loss("not a dataset")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Log-loss: cross-metric provenance
# ---------------------------------------------------------------------------


class TestLogLossCrossMetricProvenance:
    def test_same_dataset_shares_provenance_but_not_metric_identity(self):
        dataset = evaluation_dataset([choice_observation()])
        brier = evaluate_uncalibrated_winner_brier(dataset)
        log_loss = evaluate_uncalibrated_winner_log_loss(dataset)
        assert log_loss.evaluation_dataset_fingerprint == brier.evaluation_dataset_fingerprint
        assert (
            log_loss.evaluation_dataset_fingerprint_version
            == brier.evaluation_dataset_fingerprint_version
        )
        assert log_loss.input_score_id == brier.input_score_id
        assert log_loss.input_score_version == brier.input_score_version
        assert log_loss.target == brier.target
        assert log_loss.count == brier.count
        assert log_loss.metric_id != brier.metric_id
        assert log_loss.fingerprint != brier.fingerprint


# ---------------------------------------------------------------------------
# Log-loss: no semantic conflation
# ---------------------------------------------------------------------------


class TestLogLossNoConflation:
    def test_result_has_no_calibration_or_prediction_fields(self):
        names = {f.name for f in fields(LogLossEvaluationResult)}
        assert "calibrated" not in names
        assert "predicted_correctness" not in names

    def test_value_is_never_nan(self):
        finite = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([choice_observation()]))
        assert not math.isnan(finite.value)
        infinite = evaluate_uncalibrated_winner_log_loss(
            evaluation_dataset(
                [
                    choice_observation_with_probabilities(
                        three_way_probabilities(billing=1.0), resolved_truth("shipping")
                    )
                ]
            )
        )
        assert not math.isnan(infinite.value)

    def test_value_is_never_negative(self):
        observations = (
            choice_observation(),
            bool_observation_with_probability_true(0.8, resolved_truth(True)),
            bool_observation_with_probability_true(0.8, resolved_truth(False)),
        )
        for observation in observations:
            value = evaluate_uncalibrated_winner_log_loss(evaluation_dataset([observation])).value
            assert value >= 0.0
            assert value != -math.inf

    def test_evaluation_does_not_mutate_observations_or_dataset(self):
        observation = choice_observation()
        dataset = evaluation_dataset([observation])
        before_observation = observation.fingerprint
        before_dataset = dataset.fingerprint
        evaluate_uncalibrated_winner_log_loss(dataset)
        assert observation.fingerprint == before_observation
        assert dataset.fingerprint == before_dataset


# ---------------------------------------------------------------------------
# Log-loss: version guards
# ---------------------------------------------------------------------------


class TestLogLossVersionGuards:
    def test_new_log_loss_versions_are_two(self):
        assert LOG_LOSS_METRIC_VERSION == 1
        assert LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION == 2

    def test_log_loss_identity_constants(self):
        assert LOG_LOSS_METRIC_ID == "log-loss"
        assert LOG_LOSS_TARGET == "winner_correctness"
        assert LOG_LOSS_BOUNDARY_POLICY == "exact"
        assert LOG_LOSS_LOG_BASE == "e"

    def test_pre_existing_versions_unchanged(self):
        assert BRIER_METRIC_VERSION == 1
        assert CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION == 1
        assert UNCALIBRATED_SELECTED_PROBABILITY_VERSION == 1


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: accuracy and rate
# ---------------------------------------------------------------------------


def _diagnostics_taxonomy_miss_rows(count):
    # Distinct taxonomy-miss rows so multiplicity is unambiguous.
    return [choice_observation(resolved_truth(f"account-{index}")) for index in range(count)]


def _diagnostics_correct_observation():
    # Selected "shipping" at p = 0.6, truth "shipping" -> correct.
    return choice_observation()


def _diagnostics_wrong_observation():
    # Selected "shipping" at p = 0.6, truth "billing" -> wrong.
    return choice_observation(resolved_truth("billing"))


class TestWinnerDiagnosticsAccuracy:
    def test_labels_one_one_zero_zero_give_rate_one_half(self):
        # y = [1, 1, 0, 0]: two correct, two incorrect.
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
                _diagnostics_wrong_observation(),
            ]
        )
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert result.correct_count == 2
        assert result.incorrect_count == 2
        assert result.empirical_correctness_rate == pytest.approx(0.5)

    def test_all_correct_gives_rate_one(self):
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([_diagnostics_correct_observation()])
        )
        assert result.correct_count == 1
        assert result.incorrect_count == 0
        assert result.empirical_correctness_rate == pytest.approx(1.0)

    def test_all_wrong_gives_rate_zero(self):
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([_diagnostics_wrong_observation()])
        )
        assert result.correct_count == 0
        assert result.incorrect_count == 1
        assert result.empirical_correctness_rate == pytest.approx(0.0)

    def test_count_accounting_invariant(self):
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
                _diagnostics_correct_observation(),
            ]
        )
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert result.correct_count + result.incorrect_count == result.count == 3

    def test_exactly_one_numeric_field_for_the_rate(self):
        # On this binary winner-correctness target the empirical correctness
        # rate IS the ordinary decision accuracy: mean(Y_correct). The frozen
        # semantic decision is that this is ONE quantity, so the artifact
        # carries exactly one numeric field for it and no duplicate
        # accuracy/base-rate identity.
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([_diagnostics_correct_observation()])
        )
        names = {f.name for f in fields(WinnerCorrectnessDiagnosticsResult)}
        assert "empirical_correctness_rate" in names
        for forbidden in ("accuracy", "base_rate", "correctness_base_rate"):
            assert forbidden not in names
        payload = result.canonical_payload()
        assert "accuracy" not in payload
        assert "base_rate" not in payload
        diagnostics = payload["diagnostics"]
        assert isinstance(diagnostics, dict)
        assert set(diagnostics) == {
            "empirical_correctness_rate",
            "mean_selected_probability",
            "empirical_constant_brier_reference",
        }


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: mean selected probability
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsMeanSelectedProbability:
    def test_hand_calculated_mean_of_three_selected_probabilities(self):
        # Selected probabilities 0.8, 0.6, 0.4 -> mean 0.6.
        observations = (
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
                resolved_truth("billing"),
            ),
            choice_observation(),
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.3, shipping=0.4, returns=0.3),
                resolved_truth("shipping"),
            ),
        )
        result = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset(list(observations)))
        assert result.mean_selected_probability == pytest.approx((0.8 + 0.6 + 0.4) / 3)
        assert result.mean_selected_probability == pytest.approx(0.6)

    def test_bool_true_selection_uses_probability_true(self):
        observation = bool_observation_with_probability_true(0.8, resolved_truth(True))
        assert observation.selected_value is True
        result = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset([observation]))
        assert result.mean_selected_probability == pytest.approx(0.8)

    def test_bool_false_selection_uses_probability_false(self):
        # P(True) = 0.2 so P(False) = 0.8, selected False. The mean must use
        # P(False) = 0.8, not P(True) = 0.2.
        observation = bool_observation_with_probability_true(0.2, resolved_truth(True))
        assert observation.selected_value is False
        result = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset([observation]))
        assert result.mean_selected_probability == pytest.approx(0.8)
        assert result.mean_selected_probability != pytest.approx(0.2)

    def test_choice_uses_semantic_lookup(self):
        # Default choice observation: selected "shipping" at p = 0.6.
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([choice_observation()])
        )
        assert result.mean_selected_probability == pytest.approx(0.6)

    def test_recorded_non_argmax_selection_is_used_not_the_scoring_label(self):
        # The recorded selection is "returns" whose semantic probability is
        # 0.0. The mean must use 0.0, never the scoring-label slot and never
        # the argmax probability 0.6.
        observation = recorded_selection_observation(
            three_way_probabilities(billing=0.6, shipping=0.4, returns=0.0),
            "returns",
            resolved_truth("returns"),
        )
        assert observation.selected_value == "returns"
        result = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset([observation]))
        assert result.mean_selected_probability == pytest.approx(0.0)

    def test_mean_is_aggregate_raw_score_behaviour_not_confidence(self):
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([choice_observation()])
        )
        names = {f.name for f in fields(WinnerCorrectnessDiagnosticsResult)}
        assert "confidence" not in names
        assert result.mean_selected_probability == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: constant Brier reference
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsConstantReference:
    def test_labels_one_one_zero_zero_give_reference_one_quarter(self):
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
                _diagnostics_wrong_observation(),
            ]
        )
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert result.empirical_correctness_rate == pytest.approx(0.5)
        assert result.empirical_constant_brier_reference == pytest.approx(0.25)

    def test_three_correct_one_wrong_gives_reference_three_sixteenths(self):
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_correct_observation(),
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
            ]
        )
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert result.empirical_correctness_rate == pytest.approx(0.75)
        assert result.empirical_constant_brier_reference == pytest.approx(0.1875)

    def test_all_correct_endpoint_gives_reference_zero(self):
        # q = 1 -> q * (1 - q) = 0. This is the deliberate hindsight
        # prevalence endpoint, not a bug.
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([_diagnostics_correct_observation()])
        )
        assert result.empirical_correctness_rate == pytest.approx(1.0)
        assert result.empirical_constant_brier_reference == pytest.approx(0.0)

    def test_all_wrong_endpoint_gives_reference_zero(self):
        # q = 0 -> q * (1 - q) = 0. Same deliberate endpoint.
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([_diagnostics_wrong_observation()])
        )
        assert result.empirical_correctness_rate == pytest.approx(0.0)
        assert result.empirical_constant_brier_reference == pytest.approx(0.0)

    def test_reference_equals_rate_times_one_minus_rate(self):
        observations = [
            _diagnostics_correct_observation(),
            _diagnostics_wrong_observation(),
            _diagnostics_correct_observation(),
        ]
        result = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset(observations))
        q = result.empirical_correctness_rate
        assert result.empirical_constant_brier_reference == pytest.approx(q * (1.0 - q))

    def test_reference_stays_within_theoretical_range(self):
        observations = [
            _diagnostics_correct_observation(),
            _diagnostics_wrong_observation(),
            _diagnostics_correct_observation(),
            _diagnostics_wrong_observation(),
        ]
        result = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset(observations))
        assert 0.0 <= result.empirical_constant_brier_reference <= 0.25


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: exclusion provenance
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsExclusionProvenance:
    def test_excluded_rows_never_enter_rate_mean_or_reference(self):
        # 2 eligible correct rows + 8 taxonomy misses: the rate must be 1.0
        # over count = 2, never 0.2 over source_count = 10.
        eligible = [
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
                resolved_truth("billing"),
            ),
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
                resolved_truth("shipping"),
            ),
        ]
        dataset = evaluation_dataset(eligible + _diagnostics_taxonomy_miss_rows(8))
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert result.source_count == 10
        assert result.count == 2
        assert result.taxonomy_miss_count == 8
        assert result.correct_count == 2
        assert result.incorrect_count == 0
        assert result.empirical_correctness_rate == pytest.approx(1.0)
        assert result.mean_selected_probability == pytest.approx((0.8 + 0.6) / 2)
        assert result.empirical_constant_brier_reference == pytest.approx(0.0)

    def test_excluded_rows_never_enter_a_mixed_population_denominator(self):
        # Same two eligible rows (one correct, one wrong) plus taxonomy
        # misses: rate stays 0.5 over count = 2, never diluted.
        eligible = [
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
                resolved_truth("billing"),
            ),
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
                resolved_truth("billing"),
            ),
        ]
        dataset = evaluation_dataset(eligible + _diagnostics_taxonomy_miss_rows(8))
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert result.count == 2
        assert result.correct_count == 1
        assert result.incorrect_count == 1
        assert result.empirical_correctness_rate == pytest.approx(0.5)
        assert result.empirical_constant_brier_reference == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: identity
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsIdentity:
    def test_row_order_independence(self):
        observations = [
            _diagnostics_correct_observation(),
            _diagnostics_wrong_observation(),
            _diagnostics_correct_observation(),
        ]
        forward = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset(observations))
        reversed_result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset(list(reversed(observations)))
        )
        assert forward.fingerprint == reversed_result.fingerprint
        assert forward.empirical_correctness_rate == reversed_result.empirical_correctness_rate

    def test_multiplicity_is_preserved(self):
        observation = _diagnostics_correct_observation()
        single = evaluate_uncalibrated_winner_diagnostics(evaluation_dataset([observation]))
        doubled = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([observation, observation])
        )
        # The duplicated row enters the population twice.
        assert doubled.count == 2
        assert doubled.correct_count == 2
        assert doubled.empirical_correctness_rate == single.empirical_correctness_rate
        assert doubled.mean_selected_probability == single.mean_selected_probability
        assert doubled.empirical_constant_brier_reference == (
            single.empirical_constant_brier_reference
        )
        # ...but the artifacts are distinct because the populations differ.
        assert doubled.fingerprint != single.fingerprint

    def test_same_dataset_gives_same_diagnostics_fingerprint(self):
        dataset = evaluation_dataset(
            [_diagnostics_correct_observation(), _diagnostics_wrong_observation()]
        )
        first = evaluate_uncalibrated_winner_diagnostics(dataset)
        second = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert first.fingerprint == second.fingerprint

    def test_same_numerics_with_extra_taxonomy_misses_do_not_collapse(self):
        # A-vs-B: identical eligible rows (one correct, one wrong) so every
        # numeric diagnostic matches, but cohort B carries three extra
        # taxonomy misses. The diagnostics numbers are equal while the
        # diagnostics artifact fingerprints differ.
        eligible = [
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
                resolved_truth("billing"),
            ),
            choice_observation_with_probabilities(
                three_way_probabilities(billing=0.1, shipping=0.6, returns=0.3),
                resolved_truth("billing"),
            ),
        ]
        cohort_a = evaluation_cohort(eligible)
        cohort_b = evaluation_cohort(eligible + _diagnostics_taxonomy_miss_rows(3))
        diagnostics_a = evaluate_uncalibrated_winner_diagnostics(
            CalibrationEvaluationDataset.from_cohort(cohort_a)
        )
        diagnostics_b = evaluate_uncalibrated_winner_diagnostics(
            CalibrationEvaluationDataset.from_cohort(cohort_b)
        )
        assert diagnostics_a.empirical_correctness_rate == (
            diagnostics_b.empirical_correctness_rate
        )
        assert diagnostics_a.mean_selected_probability == (diagnostics_b.mean_selected_probability)
        assert diagnostics_a.empirical_constant_brier_reference == (
            diagnostics_b.empirical_constant_brier_reference
        )
        assert diagnostics_a.correct_count == diagnostics_b.correct_count
        assert diagnostics_a.count == diagnostics_b.count
        assert diagnostics_a.source_count != diagnostics_b.source_count
        assert diagnostics_a.taxonomy_miss_count != diagnostics_b.taxonomy_miss_count
        assert diagnostics_a.fingerprint != diagnostics_b.fingerprint

    def test_diagnostics_are_deterministic(self):
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
                choice_observation_with_probabilities(
                    three_way_probabilities(billing=0.8, shipping=0.1, returns=0.1),
                    resolved_truth("billing"),
                ),
            ]
        )
        first = evaluate_uncalibrated_winner_diagnostics(dataset)
        second = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert first.canonical_payload() == second.canonical_payload()
        assert first.fingerprint == second.fingerprint


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: construction guard
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsConstruction:
    def test_direct_construction_is_rejected(self):
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_diagnostics"):
            WinnerCorrectnessDiagnosticsResult()

    def test_replace_without_fields_is_rejected(self):
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([choice_observation()])
        )
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_diagnostics"):
            replace(result)

    def test_replace_with_derived_value_is_rejected(self):
        # dataclasses.replace raises its own ValueError for init=False fields
        # before the token-guarded __init__ ever runs.
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([choice_observation()])
        )
        with pytest.raises(ValueError, match="init=False"):
            replace(result, empirical_correctness_rate=0.9)

    def test_evaluator_rejects_non_dataset(self):
        with pytest.raises(InvalidDecisionError):
            evaluate_uncalibrated_winner_diagnostics("not-a-dataset")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: cross-artifact alignment
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsCrossArtifactAlignment:
    def test_three_evaluators_share_provenance_but_not_artifact_identity(self):
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
                _diagnostics_taxonomy_miss_rows(1)[0],
            ]
        )
        brier = evaluate_uncalibrated_winner_brier(dataset)
        log_loss = evaluate_uncalibrated_winner_log_loss(dataset)
        diagnostics = evaluate_uncalibrated_winner_diagnostics(dataset)

        assert diagnostics.evaluation_dataset_fingerprint == (brier.evaluation_dataset_fingerprint)
        assert diagnostics.evaluation_dataset_fingerprint == (
            log_loss.evaluation_dataset_fingerprint
        )
        assert diagnostics.evaluation_dataset_fingerprint_version == (
            brier.evaluation_dataset_fingerprint_version
        )
        assert diagnostics.source_cohort_fingerprint == brier.source_cohort_fingerprint
        assert diagnostics.source_cohort_fingerprint == log_loss.source_cohort_fingerprint
        assert diagnostics.source_cohort_fingerprint_version == (
            brier.source_cohort_fingerprint_version
        )
        assert diagnostics.source_count == brier.source_count == log_loss.source_count
        assert diagnostics.count == brier.count == log_loss.count
        assert diagnostics.taxonomy_miss_count == (
            brier.taxonomy_miss_count == log_loss.taxonomy_miss_count
        )
        assert diagnostics.unresolved_count == brier.unresolved_count
        assert diagnostics.unresolved_count == log_loss.unresolved_count
        assert diagnostics.unadjudicated_resolved_count == brier.unadjudicated_resolved_count
        assert diagnostics.unadjudicated_resolved_count == log_loss.unadjudicated_resolved_count
        assert diagnostics.target == brier.target == log_loss.target
        assert diagnostics.input_score_id == brier.input_score_id == log_loss.input_score_id
        assert diagnostics.input_score_version == (
            brier.input_score_version == log_loss.input_score_version
        )
        assert len({brier.fingerprint, log_loss.fingerprint, diagnostics.fingerprint}) == 3


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: no semantic conflation
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsNoConflation:
    def test_result_has_no_calibration_prediction_or_confidence_fields(self):
        names = {f.name for f in fields(WinnerCorrectnessDiagnosticsResult)}
        assert "calibrated" not in names
        assert "predicted_correctness" not in names
        assert "confidence" not in names

    def test_no_uncommitted_metric_identity_attribute(self):
        # Regression guard (cross-review): a stray metric_id/metric_version once
        # existed on this artifact while being excluded from canonical_payload(),
        # duplicating the rate identity without binding it into the fingerprint.
        # The container has no single metric identity; each quantity is
        # identified inside payload["diagnostics"] instead.
        assert not hasattr(WinnerCorrectnessDiagnosticsResult, "metric_id")
        assert not hasattr(WinnerCorrectnessDiagnosticsResult, "metric_version")
        result = evaluate_uncalibrated_winner_diagnostics(
            evaluation_dataset([_diagnostics_correct_observation()])
        )
        payload = result.canonical_payload()
        assert "metric_id" not in payload
        assert "metric_version" not in payload
        diagnostics = payload["diagnostics"]
        assert isinstance(diagnostics, dict)
        expected = {
            "empirical_correctness_rate": EMPIRICAL_CORRECTNESS_RATE_ID,
            "mean_selected_probability": MEAN_SELECTED_PROBABILITY_ID,
            "empirical_constant_brier_reference": EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID,
        }
        for key, identity in expected.items():
            entry = diagnostics[key]
            assert isinstance(entry, dict)
            assert entry["id"] == identity

    def test_payload_is_finite_fingerprintable_and_stable(self):
        dataset = evaluation_dataset(
            [
                _diagnostics_correct_observation(),
                _diagnostics_wrong_observation(),
            ]
        )
        result = evaluate_uncalibrated_winner_diagnostics(dataset)
        payload = result.canonical_payload()

        def _assert_finite(value):
            if isinstance(value, float):
                assert math.isfinite(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    _assert_finite(nested)
            elif isinstance(value, list):
                for nested in value:
                    _assert_finite(nested)

        _assert_finite(payload)
        assert result.fingerprint == fingerprint(payload)
        assert len(result.fingerprint) == 64
        # A second evaluation of the same dataset reproduces the payload.
        again = evaluate_uncalibrated_winner_diagnostics(dataset)
        assert again.canonical_payload() == payload

    def test_evaluation_does_not_mutate_observations_or_dataset(self):
        observation = choice_observation()
        dataset = evaluation_dataset([observation])
        before_observation = observation.fingerprint
        before_dataset = dataset.fingerprint
        evaluate_uncalibrated_winner_diagnostics(dataset)
        assert observation.fingerprint == before_observation
        assert dataset.fingerprint == before_dataset

    def test_rate_is_never_written_as_a_prediction(self):
        # The empirical correctness rate is a population-level rate, never a
        # per-example predicted probability for a new observation. The
        # evaluator is read-only: every recorded observation field is
        # untouched afterwards.
        observation = choice_observation()
        dataset = evaluation_dataset([observation])
        before = (
            observation.selected_value,
            observation.probabilities,
            observation.correct,
            observation.status,
            observation.fingerprint,
        )
        evaluate_uncalibrated_winner_diagnostics(dataset)
        after = (
            observation.selected_value,
            observation.probabilities,
            observation.correct,
            observation.status,
            observation.fingerprint,
        )
        assert after == before


# ---------------------------------------------------------------------------
# Winner-correctness companion diagnostics: version guards
# ---------------------------------------------------------------------------


class TestWinnerDiagnosticsVersionGuards:
    def test_new_diagnostics_fingerprint_version_is_one(self):
        assert WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION == 1

    def test_diagnostics_identity_constants(self):
        assert EMPIRICAL_CORRECTNESS_RATE_ID == "empirical-winner-correctness-rate"
        assert EMPIRICAL_CORRECTNESS_RATE_VERSION == 1
        assert MEAN_SELECTED_PROBABILITY_ID == "mean-uncalibrated-selected-probability"
        assert MEAN_SELECTED_PROBABILITY_VERSION == 1
        assert (
            EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID
            == "empirical-correctness-rate-constant-brier-reference"
        )
        assert EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION == 1

    def test_pre_existing_versions_unchanged(self):
        assert CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION == 1
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 2
        assert BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION == 2
        assert LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION == 2
        assert BRIER_METRIC_VERSION == 1
        assert LOG_LOSS_METRIC_VERSION == 1
        assert UNCALIBRATED_SELECTED_PROBABILITY_VERSION == 1


# ---------------------------------------------------------------------------
# Reliability summary: shared fixtures
# ---------------------------------------------------------------------------


def _reliability_observation(probability, correct):
    """One recorded-selection observation with selected 'billing' at p.

    The recorded selection is 'billing' even when it is not the argmax; the
    record is honored exactly, so the reliability bin sees exactly ``p``.
    """
    tail = (1.0 - probability) / 2
    return recorded_selection_observation(
        three_way_probabilities(billing=probability, shipping=tail, returns=tail),
        "billing",
        resolved_truth("billing") if correct else resolved_truth("shipping"),
    )


def _reliability_bin_index(probability, bin_count):
    result = evaluate_uncalibrated_winner_reliability(
        evaluation_dataset([_reliability_observation(probability, True)]),
        bin_count=bin_count,
    )
    counts = [bin.count for bin in result.bins]
    assert sum(counts) == 1
    return counts.index(1)


# ---------------------------------------------------------------------------
# Reliability summary: core hand calculation
# ---------------------------------------------------------------------------


class TestWinnerReliabilityCore:
    def test_hand_calculated_two_bin_summary(self):
        # p/y: 0.10/0, 0.40/1, 0.60/1, 0.90/0 with bin_count = 2.
        # bin0 [0.0, 0.5): p 0.10 (y=0), 0.40 (y=1) -> count 2, correct 1,
        # mean 0.25, rate 0.5. bin1 [0.5, 1.0]: p 0.60 (y=1), 0.90 (y=0)
        # -> count 2, correct 1, mean 0.75, rate 0.5.
        observations = [
            _reliability_observation(0.10, False),
            _reliability_observation(0.40, True),
            _reliability_observation(0.60, True),
            _reliability_observation(0.90, False),
        ]
        result = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset(observations), bin_count=2
        )
        assert len(result.bins) == 2
        bin0, bin1 = result.bins
        assert (bin0.index, bin0.count, bin0.correct_count) == (0, 2, 1)
        assert bin0.mean_selected_probability == pytest.approx(0.25)
        assert bin0.empirical_correctness_rate == pytest.approx(0.5)
        assert (bin1.index, bin1.count, bin1.correct_count) == (1, 2, 1)
        assert bin1.mean_selected_probability == pytest.approx(0.75)
        assert bin1.empirical_correctness_rate == pytest.approx(0.5)
        assert sum(bin.count for bin in result.bins) == 4
        assert sum(bin.correct_count for bin in result.bins) == 2

    def test_bin_count_is_committed_in_the_identity(self):
        dataset = evaluation_dataset([_reliability_observation(0.6, True)])
        result = evaluate_uncalibrated_winner_reliability(dataset, bin_count=3)
        assert result.bin_count == 3
        assert result.binning_id == "equal-width"
        assert result.binning_version == 1
        assert result.reliability_id == "winner-reliability-curve"
        assert result.reliability_version == 1


# ---------------------------------------------------------------------------
# Reliability summary: boundary ownership
# ---------------------------------------------------------------------------


class TestWinnerReliabilityBoundaries:
    @pytest.mark.parametrize(
        ("probability", "expected_bin"),
        [
            (0.0, 0),
            (0.1, 0),
            (math.nextafter(0.5, 0.0), 0),
            (0.5, 1),
            (math.nextafter(0.5, 1.0), 1),
            (0.9, 1),
            (1.0, 1),
        ],
    )
    def test_two_bin_endpoint_and_boundary_ownership(self, probability, expected_bin):
        assert _reliability_bin_index(probability, 2) == expected_bin

    @pytest.mark.parametrize(
        ("probability", "expected_bin"),
        [
            (0.0, 0),
            (math.nextafter(0.25, 0.0), 0),
            (0.25, 1),
            (math.nextafter(0.25, 1.0), 1),
            (0.5, 2),
            (0.75, 3),
            (1.0, 3),
        ],
    )
    def test_four_bin_interior_boundaries(self, probability, expected_bin):
        # 0.25, 0.5, and 0.75 are exactly float-representable interior
        # boundaries for bin_count = 4; each exact boundary belongs to the
        # upper bin, nextafter-below belongs to the lower bin.
        assert _reliability_bin_index(probability, 4) == expected_bin

    def test_non_representable_rational_boundary_is_decided_by_the_exact_rational(self):
        # 1/3 and 2/3 have no exact float representation. The contract compares
        # the STORED float's exact rational value against the rational boundary,
        # so both outcomes are pinned here independently of any floor(p / B)
        # formula: each stored double falls strictly below its own boundary.
        one_third = 1 / 3
        two_thirds = 2 / 3
        assert Fraction.from_float(one_third) < Fraction(1, 3)
        assert Fraction.from_float(two_thirds) < Fraction(2, 3)
        assert _reliability_bin_index(one_third, 3) == 0
        assert _reliability_bin_index(two_thirds, 3) == 1
        assert _reliability_bin_index(math.nextafter(one_third, 1.0), 3) == 1


# ---------------------------------------------------------------------------
# Reliability summary: empty bins
# ---------------------------------------------------------------------------


class TestWinnerReliabilityEmptyBins:
    def test_empty_bins_are_retained_with_none_statistics(self):
        observations = [
            _reliability_observation(0.1, True),
            _reliability_observation(0.95, False),
        ]
        result = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset(observations), bin_count=4
        )
        assert len(result.bins) == 4
        for index in (1, 2):
            empty = result.bins[index]
            assert (empty.index, empty.count, empty.correct_count) == (index, 0, 0)
            assert empty.mean_selected_probability is None
            assert empty.empirical_correctness_rate is None
            assert empty.observation_fingerprints == ()
        assert result.bins[0].count == 1
        assert result.bins[3].count == 1

    def test_empty_bins_serialize_as_null_not_nan_not_zero(self):
        observations = [
            _reliability_observation(0.1, True),
            _reliability_observation(0.95, False),
        ]
        result = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset(observations), bin_count=4
        )
        payload = result.canonical_payload()
        for index in (1, 2):
            entry = payload["bins"][index]
            assert entry["mean_selected_probability"] is None
            assert entry["empirical_correctness_rate"] is None
            assert entry["observation_fingerprints"] == []

    def test_empty_bin_is_not_confused_with_an_observed_zero_rate(self):
        # bin0 with one wrong observation has rate 0.0 and a real mean; an
        # empty bin has None. The two must never collapse.
        wrong_only = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset([_reliability_observation(0.1, False)]), bin_count=2
        )
        assert wrong_only.bins[0].empirical_correctness_rate == 0.0
        assert wrong_only.bins[0].mean_selected_probability == pytest.approx(0.1)
        assert wrong_only.bins[1].empirical_correctness_rate is None
        assert wrong_only.bins[1].mean_selected_probability is None


# ---------------------------------------------------------------------------
# Reliability summary: global diagnostics alignment
# ---------------------------------------------------------------------------


class TestWinnerReliabilityGlobalAlignment:
    def test_bin_totals_match_the_diagnostics_artifact(self):
        observations = [
            _reliability_observation(0.05, True),
            _reliability_observation(0.30, False),
            _reliability_observation(0.55, True),
            _reliability_observation(0.80, False),
            _reliability_observation(0.95, True),
        ]
        dataset = evaluation_dataset(observations)
        diagnostics = evaluate_uncalibrated_winner_diagnostics(dataset)
        reliability = evaluate_uncalibrated_winner_reliability(dataset, bin_count=3)
        assert sum(bin.count for bin in reliability.bins) == diagnostics.count == 5
        assert sum(bin.correct_count for bin in reliability.bins) == diagnostics.correct_count
        assert sum(bin.correct_count for bin in reliability.bins) / reliability.count == (
            pytest.approx(diagnostics.empirical_correctness_rate)
        )
        weighted_mean = (
            math.fsum(
                bin.mean_selected_probability * bin.count
                for bin in reliability.bins
                if bin.count > 0
            )
            / reliability.count
        )
        assert weighted_mean == pytest.approx(diagnostics.mean_selected_probability)


# ---------------------------------------------------------------------------
# Reliability summary: cross-artifact provenance alignment
# ---------------------------------------------------------------------------


class TestWinnerReliabilityCrossArtifactProvenance:
    def test_four_evaluators_share_provenance_but_not_artifact_identity(self):
        dataset = evaluation_dataset(
            [
                _reliability_observation(0.2, True),
                _reliability_observation(0.7, False),
                _diagnostics_taxonomy_miss_rows(1)[0],
            ]
        )
        brier = evaluate_uncalibrated_winner_brier(dataset)
        log_loss = evaluate_uncalibrated_winner_log_loss(dataset)
        diagnostics = evaluate_uncalibrated_winner_diagnostics(dataset)
        reliability = evaluate_uncalibrated_winner_reliability(dataset, bin_count=4)

        assert reliability.evaluation_dataset_fingerprint == (brier.evaluation_dataset_fingerprint)
        assert reliability.evaluation_dataset_fingerprint == (
            log_loss.evaluation_dataset_fingerprint
        )
        assert reliability.evaluation_dataset_fingerprint == (
            diagnostics.evaluation_dataset_fingerprint
        )
        assert reliability.evaluation_dataset_fingerprint_version == (
            brier.evaluation_dataset_fingerprint_version
        )
        assert reliability.source_cohort_fingerprint == brier.source_cohort_fingerprint
        assert reliability.source_cohort_fingerprint == diagnostics.source_cohort_fingerprint
        assert reliability.source_cohort_fingerprint_version == (
            brier.source_cohort_fingerprint_version
        )
        assert reliability.source_count == brier.source_count == log_loss.source_count
        assert reliability.count == brier.count == log_loss.count == diagnostics.count
        assert reliability.taxonomy_miss_count == brier.taxonomy_miss_count
        assert reliability.unresolved_count == log_loss.unresolved_count
        assert reliability.unadjudicated_resolved_count == (
            diagnostics.unadjudicated_resolved_count
        )
        assert reliability.target == brier.target == log_loss.target == diagnostics.target
        assert reliability.input_score_id == brier.input_score_id
        assert reliability.input_score_version == diagnostics.input_score_version
        assert (
            len(
                {
                    brier.fingerprint,
                    log_loss.fingerprint,
                    diagnostics.fingerprint,
                    reliability.fingerprint,
                }
            )
            == 4
        )


# ---------------------------------------------------------------------------
# Reliability summary: identity, provenance, multiplicity, permutation
# ---------------------------------------------------------------------------


class TestWinnerReliabilityIdentity:
    def test_different_bin_count_is_a_different_artifact(self):
        dataset = evaluation_dataset(
            [
                _reliability_observation(0.1, True),
                _reliability_observation(0.6, False),
                _reliability_observation(0.9, True),
            ]
        )
        five = evaluate_uncalibrated_winner_reliability(dataset, bin_count=5)
        ten = evaluate_uncalibrated_winner_reliability(dataset, bin_count=10)
        assert five.canonical_payload() != ten.canonical_payload()
        assert five.fingerprint != ten.fingerprint

    def test_row_permutation_gives_identical_bins_and_fingerprint(self):
        observations = [
            _reliability_observation(0.15, True),
            _reliability_observation(0.85, False),
            _reliability_observation(0.45, True),
            _reliability_observation(0.55, False),
        ]
        forward = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset(observations), bin_count=4
        )
        reversed_result = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset(list(reversed(observations))), bin_count=4
        )
        assert forward.bins == reversed_result.bins
        assert forward.fingerprint == reversed_result.fingerprint

    def test_multiplicity_is_preserved_never_deduplicated(self):
        observation = _reliability_observation(0.6, True)
        single = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset([observation]), bin_count=2
        )
        doubled = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset([observation, observation]), bin_count=2
        )
        assert single.bins[1].count == 1
        assert doubled.bins[1].count == 2
        assert doubled.bins[1].correct_count == 2
        assert doubled.bins[1].observation_fingerprints == (
            observation.fingerprint,
            observation.fingerprint,
        )
        assert doubled.bins[1].mean_selected_probability == (
            single.bins[1].mean_selected_probability
        )
        assert doubled.bins[1].empirical_correctness_rate == (
            single.bins[1].empirical_correctness_rate
        )
        assert doubled.fingerprint != single.fingerprint

    def test_same_numerics_with_extra_taxonomy_misses_do_not_collapse(self):
        eligible = [
            _reliability_observation(0.3, True),
            _reliability_observation(0.7, False),
        ]
        cohort_a = evaluation_cohort(eligible)
        cohort_b = evaluation_cohort(eligible + _diagnostics_taxonomy_miss_rows(3))
        reliability_a = evaluate_uncalibrated_winner_reliability(
            CalibrationEvaluationDataset.from_cohort(cohort_a), bin_count=2
        )
        reliability_b = evaluate_uncalibrated_winner_reliability(
            CalibrationEvaluationDataset.from_cohort(cohort_b), bin_count=2
        )
        for bin_a, bin_b in zip(reliability_a.bins, reliability_b.bins, strict=True):
            assert (bin_a.count, bin_a.correct_count) == (bin_b.count, bin_b.correct_count)
            assert bin_a.mean_selected_probability == bin_b.mean_selected_probability
            assert bin_a.empirical_correctness_rate == bin_b.empirical_correctness_rate
        assert reliability_a.source_cohort_fingerprint != reliability_b.source_cohort_fingerprint
        assert reliability_a.evaluation_dataset_fingerprint != (
            reliability_b.evaluation_dataset_fingerprint
        )
        assert reliability_a.fingerprint != reliability_b.fingerprint

    def test_same_dataset_gives_same_fingerprint(self):
        dataset = evaluation_dataset(
            [_reliability_observation(0.2, True), _reliability_observation(0.8, False)]
        )
        first = evaluate_uncalibrated_winner_reliability(dataset, bin_count=3)
        second = evaluate_uncalibrated_winner_reliability(dataset, bin_count=3)
        assert first.canonical_payload() == second.canonical_payload()
        assert first.fingerprint == second.fingerprint


# ---------------------------------------------------------------------------
# Reliability summary: construction guard and bin_count validation
# ---------------------------------------------------------------------------


class TestWinnerReliabilityConstruction:
    def test_direct_construction_is_rejected(self):
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_reliability"):
            WinnerReliabilityResult()

    def test_replace_without_fields_is_rejected(self):
        result = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset([_reliability_observation(0.5, True)]), bin_count=2
        )
        with pytest.raises(InvalidDecisionError, match="evaluate_uncalibrated_winner_reliability"):
            replace(result)

    def test_replace_with_bins_is_rejected_by_dataclasses(self):
        # bins is declared init=False, so dataclasses.replace raises its own
        # ValueError before the token-guarded __init__ ever runs.
        result = evaluate_uncalibrated_winner_reliability(
            evaluation_dataset([_reliability_observation(0.5, True)]), bin_count=2
        )
        with pytest.raises(ValueError, match="init=False"):
            replace(result, bins=())

    def test_evaluator_rejects_non_dataset(self):
        with pytest.raises(InvalidDecisionError, match="CalibrationEvaluationDataset"):
            evaluate_uncalibrated_winner_reliability("not-a-dataset", bin_count=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_bin_count", [True, False, 0, -1, 2.0, "2", None])
    def test_bin_count_validation_rejects_non_real_ints(self, bad_bin_count):
        dataset = evaluation_dataset([_reliability_observation(0.5, True)])
        with pytest.raises(InvalidDecisionError, match="bin_count"):
            evaluate_uncalibrated_winner_reliability(dataset, bin_count=bad_bin_count)

    def test_no_hidden_maximum_bin_count(self):
        dataset = evaluation_dataset([_reliability_observation(0.5, True)])
        result = evaluate_uncalibrated_winner_reliability(dataset, bin_count=1000)
        assert len(result.bins) == 1000
        assert sum(bin.count for bin in result.bins) == 1


# ---------------------------------------------------------------------------
# Reliability summary: no semantic conflation
# ---------------------------------------------------------------------------


class TestWinnerReliabilityNoConflation:
    def test_result_has_no_calibration_prediction_confidence_or_gap_fields(self):
        names = {f.name for f in fields(WinnerReliabilityResult)}
        for forbidden in (
            "calibrated",
            "predicted_correctness",
            "confidence",
            "gap",
            "absolute_gap",
            "calibration_error",
            "overconfidence",
            "underconfidence",
        ):
            assert forbidden not in names
        bin_names = {f.name for f in fields(ReliabilityBinSummary)}
        for forbidden in ("gap", "absolute_gap", "calibration_error", "confidence"):
            assert forbidden not in bin_names

    def test_no_ece_symbol_exists_in_the_module(self):
        import probvenance.calibration_evaluation as module

        for symbol in ("ece", "expected_calibration_error", "weighted_gap"):
            assert not hasattr(module, symbol)
        source_names = {name for name in dir(module)}
        assert not any("ece" in name.lower() for name in source_names)

    def test_no_equal_mass_implementation_exists(self):
        import probvenance.calibration_evaluation as module

        assert not any("equal_mass" in name for name in dir(module))
        assert not any("quantile" in name.lower() for name in dir(module))

    def test_evaluation_does_not_mutate_observations_or_dataset(self):
        observation = _reliability_observation(0.4, True)
        dataset = evaluation_dataset([observation])
        before_observation = observation.fingerprint
        before_dataset = dataset.fingerprint
        evaluate_uncalibrated_winner_reliability(dataset, bin_count=3)
        assert observation.fingerprint == before_observation
        assert dataset.fingerprint == before_dataset

    def test_payload_is_finite_fingerprintable_and_stable(self):
        dataset = evaluation_dataset(
            [
                _reliability_observation(0.1, True),
                _reliability_observation(0.9, False),
            ]
        )
        result = evaluate_uncalibrated_winner_reliability(dataset, bin_count=4)
        payload = result.canonical_payload()

        def _assert_finite(value):
            if isinstance(value, float):
                assert math.isfinite(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    _assert_finite(nested)
            elif isinstance(value, list):
                for nested in value:
                    _assert_finite(nested)

        _assert_finite(payload)
        assert result.fingerprint == fingerprint(payload)
        assert len(result.fingerprint) == 64


# ---------------------------------------------------------------------------
# Reliability summary: version guards
# ---------------------------------------------------------------------------


class TestWinnerReliabilityVersionGuards:
    def test_new_reliability_versions_are_one(self):
        assert EQUAL_WIDTH_BINNING_ID == "equal-width"
        assert EQUAL_WIDTH_BINNING_VERSION == 1
        assert WINNER_RELIABILITY_CURVE_ID == "winner-reliability-curve"
        assert WINNER_RELIABILITY_CURVE_VERSION == 1
        assert WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION == 1

    def test_pre_existing_versions_unchanged(self):
        assert CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION == 1
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 2
        assert BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION == 2
        assert LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION == 2
        assert WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION == 1
        assert BRIER_METRIC_VERSION == 1
        assert LOG_LOSS_METRIC_VERSION == 1
        assert EMPIRICAL_CORRECTNESS_RATE_VERSION == 1
        assert MEAN_SELECTED_PROBABILITY_VERSION == 1
        assert EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION == 1
        assert UNCALIBRATED_SELECTED_PROBABILITY_VERSION == 1
