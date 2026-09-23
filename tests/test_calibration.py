"""Tests for the calibration data foundation (Phase 4A, Parts D through M).

Observations are built through the REAL runtime with small deterministic
fake backends, so the constructor is exercised against real provenance.
"""

import inspect
import math
from typing import Any

import pytest

from probvenance import (
    BackendCapabilities,
    BoolDecision,
    ChoiceDecision,
    EvidenceKind,
    InvalidDecisionError,
    Probvenance,
    RawEvidence,
)
from probvenance.calibration import (
    CALIBRATION_DATASET_FINGERPRINT_VERSION,
    CALIBRATION_OBSERVATION_FINGERPRINT_VERSION,
    RENDERING_SEMANTICS_VERSION,
    CalibrationBinding,
    CalibrationDataset,
    CalibrationObservation,
    CalibrationObservationStatus,
    GroundTruthProvenance,
    GroundTruthRecord,
    GroundTruthResolutionStatus,
    canonical_rendering_semantics,
)
from probvenance.errors import ProbvenanceError
from probvenance.fingerprint import JSONValue

# ---------------------------------------------------------------------------
# Fake backends (deterministic, honest metadata, no model downloads)
# ---------------------------------------------------------------------------

LOGIT_BILLING = 0.0
LOGIT_SHIPPING = math.log(3.0)
LOGIT_RETURNS = 0.0
VOCAB_LOGSUMEXP = math.log(6.0)


def _honest_choice_metadata(plan: Any) -> dict[str, Any]:
    resolved = [[label, 100 + index] for index, label in enumerate(plan.targets)]
    return {
        "vocab_logsumexp": VOCAB_LOGSUMEXP,
        "top_token_id": 101,
        "top_token_logit": LOGIT_SHIPPING,
        "resolved_target_token_ids": resolved,
        "model": "fake-model",
        "model_revision": "rev-1",
        "input_token_count": 42,
    }


class FakeCategoricalBackend:
    """Deterministic backend handling both strategies with honest metadata."""

    def __init__(
        self,
        *,
        model: str | None = "fake-model",
        model_revision: str | None = "rev-1",
        tokenizer: str | None = "fake-tokenizer",
        tokenizer_revision: str | None = "tok-rev-1",
        rendering_config: dict[str, JSONValue] | None = None,
    ) -> None:
        self.capabilities = BackendCapabilities(
            binary_token_logits=True,
            categorical_token_logits=True,
        )
        self.model = model
        self.model_revision = model_revision
        self.tokenizer = tokenizer
        self.tokenizer_revision = tokenizer_revision
        self.rendering_config = rendering_config if rendering_config is not None else {}

    def execute(self, plan: Any) -> RawEvidence:
        if plan.strategy.value == "binary_token_logits":
            return RawEvidence(
                kind=EvidenceKind.LOGITS,
                labels=("false", "true"),
                values=(0.0, math.log(3.0)),
                plan_fingerprint=plan.fingerprint,
                metadata={
                    "vocab_logsumexp": math.log(4.0),
                    "top_token_id": 9642,
                    "top_token_logit": math.log(3.0),
                    "positive_token_id": 9642,
                    "negative_token_id": 3134,
                    "model": self.model,
                    "model_revision": self.model_revision,
                    "tokenizer": self.tokenizer,
                    "tokenizer_revision": self.tokenizer_revision,
                    "rendering_config": self.rendering_config,
                },
            )
        values = tuple(
            LOGIT_BILLING if label == "A" else LOGIT_SHIPPING if label == "B" else LOGIT_RETURNS
            for label in plan.targets
        )
        return RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=plan.targets,
            values=values,
            plan_fingerprint=plan.fingerprint,
            metadata={
                **_honest_choice_metadata(plan),
                "model": self.model,
                "model_revision": self.model_revision,
                "tokenizer": self.tokenizer,
                "tokenizer_revision": self.tokenizer_revision,
                "rendering_config": self.rendering_config,
            },
        )


def make_choice_runtime(
    backend: FakeCategoricalBackend | None = None,
) -> tuple[Probvenance, FakeCategoricalBackend]:
    active = backend if backend is not None else FakeCategoricalBackend()
    return Probvenance(backend=active), active


def make_choice_decision() -> ChoiceDecision:
    return ChoiceDecision(
        "Which department should handle this request?",
        context="The customer asks about a refund for a damaged parcel.",
        choices={
            "billing": "Payment, charges, invoices",
            "shipping": "Delivery, couriers, parcels",
            "returns": "Refunds, exchanges, warranty",
        },
    )


def make_bool_decision() -> BoolDecision:
    return BoolDecision("Was the parcel delivered on the promised day?")


def adjudicated_provenance(**overrides: Any) -> GroundTruthProvenance:
    values: dict[str, Any] = {
        "label_source": "human-annotator",
        "labeling_rule": "route to the department named in the ticket",
        "adjudicated": True,
        "ambiguity_policy": "exclude ambiguous tickets",
    }
    values.update(overrides)
    return GroundTruthProvenance(**values)


def resolved_truth(value: JSONValue, **overrides: Any) -> GroundTruthRecord:
    return GroundTruthRecord(
        value=value,
        resolution_status=GroundTruthResolutionStatus.RESOLVED,
        provenance=adjudicated_provenance(**overrides),
    )


def unresolved_truth() -> GroundTruthRecord:
    return GroundTruthRecord(
        value=None,
        resolution_status=GroundTruthResolutionStatus.UNRESOLVED,
        provenance=adjudicated_provenance(),
    )


def choice_observation(
    truth: GroundTruthRecord | None = None,
    backend: FakeCategoricalBackend | None = None,
) -> CalibrationObservation:
    runtime, _ = make_choice_runtime(backend)
    evaluation = runtime.evaluate_with_trace(make_choice_decision())
    return CalibrationObservation.from_evaluation(
        evaluation, truth if truth is not None else resolved_truth("shipping")
    )


def bool_observation(
    truth: GroundTruthRecord | None = None,
    backend: FakeCategoricalBackend | None = None,
) -> CalibrationObservation:
    runtime, _ = make_choice_runtime(backend)
    evaluation = runtime.evaluate_with_trace(make_bool_decision())
    return CalibrationObservation.from_evaluation(
        evaluation, truth if truth is not None else resolved_truth(True)
    )


# ---------------------------------------------------------------------------
# Ground truth provenance and records (Part F)
# ---------------------------------------------------------------------------


class TestGroundTruthProvenance:
    def test_valid_provenance_round_trips_payload(self) -> None:
        provenance = adjudicated_provenance(taxonomy_id="departments", taxonomy_version=2)
        payload = provenance.canonical_payload()
        assert payload["label_source"] == "human-annotator"
        assert payload["adjudicated"] is True
        assert payload["taxonomy_id"] == "departments"
        assert payload["taxonomy_version"] == 2

    def test_taxonomy_pair_may_be_independently_none(self) -> None:
        # A known taxonomy with an unknown version is a real state.
        provenance = adjudicated_provenance(taxonomy_id="departments")
        assert provenance.taxonomy_id == "departments"
        assert provenance.taxonomy_version is None
        provenance = adjudicated_provenance(taxonomy_version=3)
        assert provenance.taxonomy_id is None
        assert provenance.taxonomy_version == 3

    def test_empty_label_source_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="label_source"):
            adjudicated_provenance(label_source="   ")

    def test_non_bool_adjudicated_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="adjudicated"):
            adjudicated_provenance(adjudicated=1)

    def test_callable_labeling_rule_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="labeling_rule"):
            adjudicated_provenance(labeling_rule=len)

    def test_object_ambiguity_policy_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="ambiguity_policy"):
            adjudicated_provenance(ambiguity_policy=object())

    def test_zero_taxonomy_version_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="taxonomy_version"):
            adjudicated_provenance(taxonomy_version=0)

    def test_bool_taxonomy_version_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="taxonomy_version"):
            adjudicated_provenance(taxonomy_version=True)


class TestGroundTruthRecord:
    def test_resolved_record_carries_value(self) -> None:
        record = resolved_truth("shipping")
        assert record.value == "shipping"
        assert record.resolution_status is GroundTruthResolutionStatus.RESOLVED

    def test_resolved_record_rejects_none_value(self) -> None:
        with pytest.raises(InvalidDecisionError, match="RESOLVED"):
            GroundTruthRecord(
                value=None,
                resolution_status=GroundTruthResolutionStatus.RESOLVED,
                provenance=adjudicated_provenance(),
            )

    def test_unresolved_record_requires_none_value(self) -> None:
        with pytest.raises(InvalidDecisionError, match="UNRESOLVED"):
            GroundTruthRecord(
                value="shipping",
                resolution_status=GroundTruthResolutionStatus.UNRESOLVED,
                provenance=adjudicated_provenance(),
            )

    def test_unresolved_record_with_none_value_is_valid(self) -> None:
        record = unresolved_truth()
        assert record.value is None

    def test_resolution_status_has_no_taxonomy_miss_member(self) -> None:
        members = {status.value for status in GroundTruthResolutionStatus}
        assert "taxonomy_miss" not in members
        assert members == {"resolved", "unresolved"}

    def test_non_json_value_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="JSON-compatible"):
            GroundTruthRecord(
                value={"a": set()},  # type: ignore[dict-item]
                resolution_status=GroundTruthResolutionStatus.RESOLVED,
                provenance=adjudicated_provenance(),
            )


# ---------------------------------------------------------------------------
# Rendering semantics (Part I5)
# ---------------------------------------------------------------------------


class TestCanonicalRenderingSemantics:
    def test_absent_enable_thinking_is_unknown_not_false(self) -> None:
        projection = canonical_rendering_semantics({})
        assert projection["enable_thinking"] is None
        assert projection["enable_thinking"] is not False

    def test_present_values_pass_through(self) -> None:
        assert canonical_rendering_semantics({"enable_thinking": True})["enable_thinking"] is True
        assert canonical_rendering_semantics({"enable_thinking": False})["enable_thinking"] is False

    def test_projection_is_versioned(self) -> None:
        assert canonical_rendering_semantics({})["v"] == RENDERING_SEMANTICS_VERSION
        assert RENDERING_SEMANTICS_VERSION == 1

    def test_none_input_is_all_unknown(self) -> None:
        projection = canonical_rendering_semantics(None)
        assert projection == {"v": 1, "enable_thinking": None}

    def test_non_mapping_input_is_all_unknown(self) -> None:
        assert canonical_rendering_semantics("nope") == {  # type: ignore[arg-type]
            "v": 1,
            "enable_thinking": None,
        }

    def test_centralized_single_definition(self) -> None:
        # The projection is one public function, not scattered .get() calls.
        import probvenance.calibration as calibration_module

        assert calibration_module.canonical_rendering_semantics is (canonical_rendering_semantics)


# ---------------------------------------------------------------------------
# Calibration binding (Part I)
# ---------------------------------------------------------------------------


class TestCalibrationBinding:
    def test_identical_source_produces_identical_bindings(self) -> None:
        first = choice_observation().binding
        second = choice_observation().binding
        assert first == second
        assert first.fingerprint == second.fingerprint

    def test_different_model_produces_different_binding(self) -> None:
        base = choice_observation().binding
        other = choice_observation(backend=FakeCategoricalBackend(model="other-model")).binding
        assert other.model == "other-model"
        assert base.model == "fake-model"
        assert base.fingerprint != other.fingerprint

    def test_thinking_on_and_off_produce_different_bindings(self) -> None:
        off = choice_observation(
            backend=FakeCategoricalBackend(rendering_config={"enable_thinking": False})
        ).binding
        on = choice_observation(
            backend=FakeCategoricalBackend(rendering_config={"enable_thinking": True})
        ).binding
        assert off.rendering_semantics["enable_thinking"] is False
        assert on.rendering_semantics["enable_thinking"] is True
        assert off.fingerprint != on.fingerprint

    def test_model_revision_none_is_explicit_null_not_wildcard(self) -> None:
        binding = choice_observation(backend=FakeCategoricalBackend(model_revision=None)).binding
        payload = binding.canonical_payload()
        assert payload["model_revision"] is None
        assert "model_revision" in payload
        assert payload["model_revision"] != "*"
        assert binding.is_fully_resolved is False

    def test_is_fully_resolved_true_when_all_known(self) -> None:
        observation = choice_observation(
            backend=FakeCategoricalBackend(rendering_config={"enable_thinking": False})
        )
        binding = observation.binding
        assert binding.model is not None
        assert binding.model_revision is not None
        assert binding.is_fully_resolved is True

    def test_no_profile_matching_policy_exists(self) -> None:
        # The binding answers "which population", nothing more: no policy
        # about sharing a profile may exist on it.
        binding = choice_observation().binding
        for forbidden in (
            "may_share_profile",
            "matches",
            "compatible_with",
            "profile_id",
            "is_interchangeable_with",
        ):
            assert not hasattr(binding, forbidden)

    def test_no_formulation_internals_duplicated(self) -> None:
        binding = choice_observation().binding
        payload = binding.canonical_payload()
        for forbidden in (
            "compiler_id",
            "compiler_version",
            "assembler_id",
            "assembler_version",
            "candidate_mapping",
        ):
            assert forbidden not in payload
        assert not any(key.startswith("doctrine_") for key in payload)

    def test_declaration_fields_are_provenance_only(self) -> None:
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        binding = CalibrationBinding.from_trace(
            evaluation.trace,
            task_id="routing",
            domain_id="support",
            taxonomy_id="departments",
            taxonomy_version=1,
        )
        assert binding.task_id == "routing"
        assert binding.domain_id == "support"
        assert binding.taxonomy_id == "departments"
        assert binding.taxonomy_version == 1


# ---------------------------------------------------------------------------
# Observation status and derived correctness (Parts G, H)
# ---------------------------------------------------------------------------


class TestObservationStatusAndCorrectness:
    def test_bool_correct(self) -> None:
        observation = bool_observation(resolved_truth(True))
        assert observation.status is CalibrationObservationStatus.RESOLVED
        assert observation.correct is True
        assert observation.fit_eligible is True
        assert observation.selected_value is True

    def test_bool_wrong(self) -> None:
        observation = bool_observation(resolved_truth(False))
        assert observation.status is CalibrationObservationStatus.RESOLVED
        assert observation.correct is False
        assert observation.fit_eligible is True

    def test_bool_ground_truth_must_be_real_bool(self) -> None:
        with pytest.raises(InvalidDecisionError, match="real bool"):
            bool_observation(resolved_truth("true"))
        with pytest.raises(InvalidDecisionError, match="real bool"):
            bool_observation(resolved_truth(1))

    def test_choice_correct_and_wrong_by_semantic_name(self) -> None:
        correct = choice_observation(resolved_truth("shipping"))
        assert correct.status is CalibrationObservationStatus.RESOLVED
        assert correct.correct is True
        assert correct.fit_eligible is True
        wrong = choice_observation(resolved_truth("billing"))
        assert wrong.status is CalibrationObservationStatus.RESOLVED
        assert wrong.correct is False
        assert wrong.fit_eligible is True

    def test_scoring_labels_never_used_as_ground_truth_outcome(self) -> None:
        # A/B/C are execution labels; the ground truth is a semantic name.
        observation = choice_observation(resolved_truth("shipping"))
        assert observation.selected_value == "shipping"
        assert observation.selected_value not in ("A", "B", "C")
        assert set(observation.outcome_order) == {"billing", "shipping", "returns"}
        for name, _ in observation.probabilities:
            assert name not in ("A", "B", "C")

    def test_taxonomy_miss_retained_not_dropped(self) -> None:
        observation = choice_observation(resolved_truth("account"))
        assert observation.status is CalibrationObservationStatus.TAXONOMY_MISS
        assert observation.correct is None
        assert observation.fit_eligible is False
        # Retained: the observation exists and carries full information.
        assert observation.selected_value == "shipping"
        assert observation.ground_truth.value == "account"

    def test_unresolved_truth(self) -> None:
        observation = choice_observation(unresolved_truth())
        assert observation.status is CalibrationObservationStatus.UNRESOLVED
        assert observation.correct is None
        assert observation.fit_eligible is False

    def test_unadjudicated_truth_not_fit_eligible(self) -> None:
        observation = choice_observation(resolved_truth("shipping", adjudicated=False))
        assert observation.status is CalibrationObservationStatus.RESOLVED
        assert observation.correct is True
        assert observation.fit_eligible is False

    def test_caller_cannot_supply_correct(self) -> None:
        # Neither the direct constructor nor the runtime classmethod accepts
        # a correctness argument: it is always derived.
        signature = inspect.signature(CalibrationObservation.__init__)
        assert "correct" not in signature.parameters
        assert "status" not in signature.parameters
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        forbidden_kwargs: dict[str, Any] = {"correct": False}
        with pytest.raises(TypeError):
            CalibrationObservation.from_evaluation(
                evaluation,
                resolved_truth("shipping"),
                **forbidden_kwargs,
            )

    def test_no_scoring_thresholds_in_fit_eligibility(self) -> None:
        # fit_eligible depends only on status + adjudication; there is no
        # certainty, margin, or mass threshold anywhere in the module.
        import probvenance.calibration as calibration_module

        source = inspect.getsource(calibration_module)
        for forbidden in ("scoring_label_mass", "certainty", "margin"):
            assert forbidden not in source

    def test_no_predicted_correctness_field(self) -> None:
        observation = choice_observation()
        assert not hasattr(observation, "predicted_correctness")
        source = inspect.getsource(
            inspect.getmodule(CalibrationObservation)  # type: ignore[arg-type]
        )
        assert "predicted_correctness" not in source


# ---------------------------------------------------------------------------
# Observation construction and fingerprints (Parts J, K)
# ---------------------------------------------------------------------------


class TestObservationConstruction:
    def test_from_evaluation_pair(self) -> None:
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        observation = CalibrationObservation.from_evaluation(
            (evaluation.result, evaluation.trace), resolved_truth("shipping")
        )
        assert observation.decision_family == "choice"
        assert observation.binding.probability_formulation_fingerprint == (
            evaluation.trace.probability_formulation_fingerprint
        )

    def test_invalid_evaluation_shape_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="Evaluation"):
            CalibrationObservation.from_evaluation("not an evaluation", resolved_truth("x"))  # type: ignore[arg-type]

    def test_bool_family_requires_bool_result(self) -> None:
        runtime, _ = make_choice_runtime()
        bool_evaluation = runtime.evaluate_with_trace(make_bool_decision())
        choice_evaluation = runtime.evaluate_with_trace(make_choice_decision())
        with pytest.raises(InvalidDecisionError, match="BoolResult"):
            CalibrationObservation.from_evaluation(
                (choice_evaluation.result, bool_evaluation.trace), unresolved_truth()
            )

    def test_choice_family_requires_choice_result(self) -> None:
        runtime, _ = make_choice_runtime()
        bool_evaluation = runtime.evaluate_with_trace(make_bool_decision())
        choice_evaluation = runtime.evaluate_with_trace(make_choice_decision())
        with pytest.raises(InvalidDecisionError, match="ChoiceResult"):
            CalibrationObservation.from_evaluation(
                (bool_evaluation.result, choice_evaluation.trace), unresolved_truth()
            )

    def test_probabilities_are_order_preserving_pairs(self) -> None:
        observation = choice_observation()
        assert isinstance(observation.probabilities, tuple)
        assert [name for name, _ in observation.probabilities] == list(observation.outcome_order)

    def test_bool_tie_selects_false(self) -> None:
        from probvenance import BoolResult, Certainty

        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_bool_decision())
        tie_result = BoolResult(
            certainty=Certainty.from_probabilities([0.5, 0.5]),
            method="token_logits",
            probability_true=0.5,
        )
        observation = CalibrationObservation.from_evaluation(
            (tie_result, evaluation.trace), resolved_truth(True)
        )
        assert observation.selected_value is False
        assert observation.correct is False

    def test_execution_fingerprint_is_audit_metadata_only(self) -> None:
        observation = choice_observation()
        assert observation.execution_fingerprint is not None
        payload = observation.canonical_payload()
        assert "execution_fingerprint" not in payload
        assert "trace_id" not in payload


class TestObservationFingerprint:
    def test_deterministic(self) -> None:
        first = choice_observation()
        second = choice_observation()
        assert first.fingerprint == second.fingerprint

    def test_trace_id_excluded(self) -> None:
        runtime, _ = make_choice_runtime()
        first = runtime.evaluate_with_trace(make_choice_decision())
        second = runtime.evaluate_with_trace(make_choice_decision())
        assert first.trace.trace_id != second.trace.trace_id
        obs_first = CalibrationObservation.from_evaluation(first, resolved_truth("shipping"))
        obs_second = CalibrationObservation.from_evaluation(second, resolved_truth("shipping"))
        assert obs_first.fingerprint == obs_second.fingerprint

    def test_ground_truth_value_change_changes_fingerprint(self) -> None:
        shipping = choice_observation(resolved_truth("shipping"))
        billing = choice_observation(resolved_truth("billing"))
        assert shipping.fingerprint != billing.fingerprint

    def test_label_provenance_is_identity_bearing(self) -> None:
        base = choice_observation(resolved_truth("shipping"))
        other_source = choice_observation(
            resolved_truth("shipping", label_source="senior-annotator")
        )
        other_rule = choice_observation(
            resolved_truth("shipping", labeling_rule="a different rule")
        )
        assert base.fingerprint != other_source.fingerprint
        assert base.fingerprint != other_rule.fingerprint

    def test_formulation_difference_changes_fingerprint(self) -> None:
        # A permuted candidate order is a different formulation identity.
        from probvenance import Choice

        runtime, _ = make_choice_runtime()
        permuted = ChoiceDecision(
            "Which department should handle this request?",
            context="The customer asks about a refund for a damaged parcel.",
            choices=[
                Choice(name="returns", description="Refunds, exchanges, warranty"),
                Choice(name="billing", description="Payment, charges, invoices"),
                Choice(name="shipping", description="Delivery, couriers, parcels"),
            ],
        )
        base_evaluation = runtime.evaluate_with_trace(make_choice_decision())
        permuted_evaluation = runtime.evaluate_with_trace(permuted)
        assert (
            base_evaluation.trace.probability_formulation_fingerprint
            != permuted_evaluation.trace.probability_formulation_fingerprint
        )
        base = CalibrationObservation.from_evaluation(base_evaluation, resolved_truth("shipping"))
        other = CalibrationObservation.from_evaluation(
            permuted_evaluation, resolved_truth("shipping")
        )
        assert base.fingerprint != other.fingerprint

    def test_fingerprint_version_is_committed(self) -> None:
        observation = choice_observation()
        assert observation.canonical_payload()["v"] == (CALIBRATION_OBSERVATION_FINGERPRINT_VERSION)
        assert CALIBRATION_OBSERVATION_FINGERPRINT_VERSION == 1


# ---------------------------------------------------------------------------
# Calibration dataset (Parts L, M)
# ---------------------------------------------------------------------------


class TestCalibrationDataset:
    def _eligible_pair(self) -> list[CalibrationObservation]:
        return [choice_observation(), choice_observation()]

    def test_order_independent_fingerprint(self) -> None:
        o1, o2, o3 = (
            choice_observation(),
            choice_observation(),
            choice_observation(),
        )
        forward = CalibrationDataset.create([o1, o2, o3])
        shuffled = CalibrationDataset.create([o3, o1, o2])
        assert forward.fingerprint == shuffled.fingerprint

    def test_multiplicity_preserved(self) -> None:
        o1, o2 = choice_observation(), choice_observation()
        two = CalibrationDataset.create([o1, o2])
        three = CalibrationDataset.create([o1, o1, o2])
        assert two.fingerprint != three.fingerprint

    def test_mixed_binding_rejected(self) -> None:
        same = choice_observation()
        other_model = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        with pytest.raises(InvalidDecisionError, match="binding fingerprint"):
            CalibrationDataset.create([same, other_model])

    def test_mixed_formulation_rejected_even_when_family_matches(self) -> None:
        from probvenance import Choice

        runtime, _ = make_choice_runtime()
        base_evaluation = runtime.evaluate_with_trace(make_choice_decision())
        permuted = ChoiceDecision(
            "Which department should handle this request?",
            context="The customer asks about a refund for a damaged parcel.",
            choices=[
                Choice(name="returns", description="Refunds, exchanges, warranty"),
                Choice(name="billing", description="Payment, charges, invoices"),
                Choice(name="shipping", description="Delivery, couriers, parcels"),
            ],
        )
        permuted_evaluation = runtime.evaluate_with_trace(permuted)
        assert (
            base_evaluation.trace.formulation_family_fingerprint
            == permuted_evaluation.trace.formulation_family_fingerprint
        )
        base = CalibrationObservation.from_evaluation(base_evaluation, resolved_truth("shipping"))
        other = CalibrationObservation.from_evaluation(
            permuted_evaluation, resolved_truth("shipping")
        )
        assert base.binding.fingerprint != other.binding.fingerprint
        with pytest.raises(InvalidDecisionError, match="binding fingerprint"):
            CalibrationDataset.create([base, other])

    def test_taxonomy_miss_inclusion_rejected(self) -> None:
        eligible = choice_observation()
        miss = choice_observation(resolved_truth("account"))
        assert miss.status is CalibrationObservationStatus.TAXONOMY_MISS
        with pytest.raises(InvalidDecisionError, match="taxonomy_miss"):
            CalibrationDataset.create([eligible, miss])

    def test_unresolved_inclusion_rejected(self) -> None:
        eligible = choice_observation()
        unresolved = choice_observation(unresolved_truth())
        with pytest.raises(InvalidDecisionError, match="unresolved"):
            CalibrationDataset.create([eligible, unresolved])

    def test_unadjudicated_inclusion_rejected(self) -> None:
        eligible = choice_observation()
        unadjudicated = choice_observation(resolved_truth("shipping", adjudicated=False))
        with pytest.raises(InvalidDecisionError, match="not fit-eligible"):
            CalibrationDataset.create([eligible, unadjudicated])

    def test_empty_dataset_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationDataset.create([])
        with pytest.raises(InvalidDecisionError, match="at least one observation"):
            CalibrationDataset(binding=choice_observation().binding, observations=())

    def test_rejection_names_count_and_reason(self) -> None:
        eligible = choice_observation()
        miss = choice_observation(resolved_truth("account"))
        with pytest.raises(InvalidDecisionError) as excinfo:
            CalibrationDataset.create([eligible, miss])
        message = str(excinfo.value)
        assert "1 of 2" in message
        assert "taxonomy_miss" in message

    def test_valid_dataset_exposes_binding_observations_fingerprint(self) -> None:
        observations = self._eligible_pair()
        dataset = CalibrationDataset.create(observations)
        assert dataset.binding is observations[0].binding
        assert dataset.observations == tuple(observations)
        assert dataset.fingerprint

    def test_dataset_fingerprint_version_is_committed(self) -> None:
        dataset = CalibrationDataset.create(self._eligible_pair())
        assert CALIBRATION_DATASET_FINGERPRINT_VERSION == 1
        # The version is committed inside the fingerprint payload; the
        # fingerprint itself is deterministic across constructions.
        other = CalibrationDataset.create(self._eligible_pair())
        assert dataset.fingerprint == other.fingerprint

    def test_dataset_is_not_a_set_hash(self) -> None:
        o1, o2 = choice_observation(), choice_observation()
        dataset = CalibrationDataset.create([o1, o2, o1])
        assert dataset.fingerprint != CalibrationDataset.create([o1, o2]).fingerprint


# ---------------------------------------------------------------------------
# Public API freeze (Part D2)
# ---------------------------------------------------------------------------


class TestPublicApiFreeze:
    def test_calibration_names_not_in_public_api(self) -> None:
        import probvenance

        for name in (
            "CalibrationObservation",
            "CalibrationBinding",
            "CalibrationDataset",
            "GroundTruthRecord",
            "GroundTruthProvenance",
        ):
            assert name not in probvenance.__all__

    def test_module_importable_without_init_changes(self) -> None:
        import probvenance.calibration as calibration_module

        assert calibration_module.CalibrationObservation is CalibrationObservation


# ---------------------------------------------------------------------------
# Error taxonomy (Errors section)
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    def test_validation_errors_use_frozen_error_type(self) -> None:
        with pytest.raises(ProbvenanceError):
            adjudicated_provenance(label_source="")
        with pytest.raises(ProbvenanceError):
            CalibrationDataset.create([])
