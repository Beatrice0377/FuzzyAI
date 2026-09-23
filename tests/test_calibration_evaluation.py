"""Tests for the pre-calibration evaluation foundation.

Covers :mod:`probvenance.calibration_evaluation`: the evaluation dataset
contract, the Brier winner-correctness metric, and the result artifact.

The fixture helpers below build REAL ``CalibrationObservation`` objects through
the supported ``from_evaluation`` path (a hand-built result paired with a real
evaluation trace, the established pattern in ``test_calibration.py``). The
evaluator itself is never faked.

One fixture deserves explanation: ``recorded_selection_observation`` records a
``selected_value`` that is not the argmax of the recorded probabilities. The
runtime would never select a non-argmax outcome, but the observation contract
treats ``selected_value`` as a recorded execution fact, and the evaluation
layer must use the record without recomputing the winner. The Brier matrix
entries with ``p = 0.0`` require exactly such a record (the argmax of a
normalized distribution is never ``0.0``), and the hand-calculated matrix uses
one for its ``p = 0.2`` row over the standard three-candidate decision.

All choice observations in one dataset share the standard three-candidate
decision because the binding includes the formulation (candidate names and
order) and the fake backend's honest ``vocab_logsumexp`` budget only covers
three candidates.
"""

import math
from dataclasses import fields, replace

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
)
from probvenance.calibration_evaluation import (
    BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION,
    BRIER_METRIC_ID,
    BRIER_METRIC_VERSION,
    CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
    UNCALIBRATED_SELECTED_PROBABILITY_ID,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    BrierEvaluationResult,
    CalibrationEvaluationDataset,
    EvaluationSplitRole,
    evaluate_uncalibrated_winner_brier,
)
from probvenance.fingerprint import fingerprint

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def evaluation_dataset(
    observations,
    *,
    split_role=EvaluationSplitRole.TEST,
    split_id="eval-split-1",
):
    return CalibrationEvaluationDataset(
        observations=tuple(observations), split_role=split_role, split_id=split_id
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

    The recorded ``selected_value`` is an execution fact: the observation below
    is built through the supported ``from_evaluation`` path, and the recorded
    selection is honored exactly by the evaluation layer (the winner is never
    recomputed).
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
# Evaluation dataset admission
# ---------------------------------------------------------------------------


class TestEvaluationDatasetAdmission:
    def test_empty_observations_rejected(self):
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            evaluation_dataset([])

    def test_observations_must_be_a_tuple(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationEvaluationDataset(
                observations=[observation],  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_none_observations_rejected_with_invalid_decision_error(self):
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationEvaluationDataset(
                observations=None,  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_generator_observations_rejected_with_invalid_decision_error(self):
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationEvaluationDataset(
                observations=(item for item in [choice_observation()]),  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_non_observation_element_rejected_with_invalid_decision_error(self):
        with pytest.raises(InvalidDecisionError, match="expected a CalibrationObservation"):
            CalibrationEvaluationDataset(
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
            CalibrationEvaluationDataset(
                observations=(LookAlike(),),  # type: ignore[arg-type]
                split_role=EvaluationSplitRole.TEST,
                split_id="split",
            )

    def test_unresolved_ground_truth_rejected(self):
        observation = choice_observation(unresolved_truth())
        with pytest.raises(InvalidDecisionError, match="not fit-eligible"):
            evaluation_dataset([observation])

    def test_taxonomy_miss_rejected_not_encoded_as_correct_false(self):
        # "account" is not among the billing/shipping/returns candidates.
        observation = choice_observation(resolved_truth("account"))
        assert observation.status.name == "TAXONOMY_MISS"
        assert observation.correct is None
        with pytest.raises(InvalidDecisionError, match="not fit-eligible"):
            evaluation_dataset([observation])

    def test_unadjudicated_ground_truth_rejected(self):
        observation = choice_observation(resolved_truth("shipping", adjudicated=False))
        with pytest.raises(InvalidDecisionError, match="not fit-eligible"):
            evaluation_dataset([observation])

    def test_mixed_binding_rejected(self):
        same = choice_observation()
        other_model = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
            evaluation_dataset([same, other_model])

    def test_mixed_ground_truth_semantics_rejected(self):
        same = choice_observation()
        other_rule = choice_observation(
            resolved_truth("shipping", labeling_rule="a different labeling rule")
        )
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics"):
            evaluation_dataset([same, other_rule])

    def test_different_label_source_only_is_accepted(self):
        human = choice_observation()
        reviewer = choice_observation(resolved_truth("shipping", label_source="reviewer"))
        dataset = evaluation_dataset([human, reviewer])
        assert len(dataset.observations) == 2

    def test_split_role_must_be_an_evaluation_split_role(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="split_role must be"):
            CalibrationEvaluationDataset(
                observations=(observation,),
                split_role="validation",  # type: ignore[arg-type]
                split_id="split",
            )
        with pytest.raises(InvalidDecisionError, match="split_role must be"):
            CalibrationEvaluationDataset(
                observations=(observation,),
                split_role="train",  # type: ignore[arg-type]
                split_id="split",
            )

    def test_split_id_must_be_non_empty(self):
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="split_id must be a non-empty"):
            CalibrationEvaluationDataset(
                observations=(observation,),
                split_role=EvaluationSplitRole.TEST,
                split_id="",
            )

    def test_rejection_messages_distinguish_reason_kinds(self):
        # One message that mixes a non-fit-eligible row, a binding mismatch, a
        # semantics mismatch, and invalid split metadata; the message must
        # distinguish all four kinds and count the observations and reasons.
        unresolved = choice_observation(unresolved_truth())
        other_model = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        other_rule = choice_observation(
            resolved_truth("shipping", labeling_rule="a different labeling rule")
        )
        with pytest.raises(InvalidDecisionError) as excinfo:
            CalibrationEvaluationDataset(
                observations=(unresolved, other_model, other_rule),
                split_role="validation",  # type: ignore[arg-type]
                split_id="",
            )
        message = str(excinfo.value)
        assert "3 observation(s)" in message
        assert "5 reason(s)" in message
        assert "not fit-eligible" in message
        assert "binding canonical payload" in message
        assert "ground-truth semantics" in message
        assert "split_role must be an EvaluationSplitRole" in message
        assert "split_id must be a non-empty" in message


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

    def test_fingerprint_version_is_one(self):
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 1

    def test_fingerprint_payload_commits_identity_versions(self):
        observation = choice_observation()
        dataset = evaluation_dataset(
            [observation], split_role=EvaluationSplitRole.VALIDATION, split_id="holdout"
        )
        expected: dict = {
            "v": CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
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
    def test_new_versions_are_one(self):
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 1
        assert BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION == 1
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
