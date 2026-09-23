"""Tests for the calibration data foundation (Phase 4A, Parts D through M).

Observations are built through the REAL runtime with small deterministic
fake backends, so the constructor is exercised against real provenance.
"""

import inspect
import math
from dataclasses import fields, replace
from typing import Any

import pytest

from probvenance import (
    BackendCapabilities,
    BoolDecision,
    ChoiceDecision,
    ChoiceResult,
    EvidenceKind,
    InvalidDecisionError,
    Probvenance,
    RawEvidence,
)
from probvenance.calibration import (
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    CALIBRATION_DATASET_FINGERPRINT_VERSION,
    CALIBRATION_OBSERVATION_FINGERPRINT_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    RENDERING_SEMANTICS_VERSION,
    CalibrationBinding,
    CalibrationDataset,
    CalibrationObservation,
    CalibrationObservationStatus,
    GroundTruthProvenance,
    GroundTruthRecord,
    GroundTruthResolutionStatus,
    GroundTruthSemanticsIdentity,
    canonical_rendering_semantics,
)
from probvenance.errors import ProbvenanceError
from probvenance.fingerprint import JSONValue, fingerprint
from probvenance.runtime import Evaluation

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


def hand_bound_binding(
    *,
    rendering_semantics: dict[str, Any],
    base: CalibrationBinding | None = None,
) -> CalibrationBinding:
    """Hand-build a binding, optionally mirroring an existing one's fields.

    Used to probe the binding construction rules directly, bypassing
    ``from_trace``: everything except ``rendering_semantics`` is copied
    from ``base`` (the first observation's binding by default).
    """
    source = base if base is not None else choice_observation().binding
    return CalibrationBinding(
        probability_formulation_fingerprint=source.probability_formulation_fingerprint,
        probability_formulation_fingerprint_version=(
            source.probability_formulation_fingerprint_version
        ),
        model=source.model,
        model_revision=source.model_revision,
        tokenizer=source.tokenizer,
        tokenizer_revision=source.tokenizer_revision,
        rendering_semantics=rendering_semantics,
        task_id=source.task_id,
        domain_id=source.domain_id,
        taxonomy_id=source.taxonomy_id,
        taxonomy_version=source.taxonomy_version,
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

    def test_enable_thinking_int_rejected_at_construction(self) -> None:
        # Root-cause guard for the audit counterexample: an int like 1 must
        # not masquerade as True under Python equality, so a hand-built
        # binding carrying enable_thinking=1 is rejected before any
        # fingerprint can be computed from it.
        for bad in (1, 0):
            with pytest.raises(InvalidDecisionError, match="enable_thinking"):
                hand_bound_binding(rendering_semantics={"v": 1, "enable_thinking": bad})

    def test_rendering_semantics_non_bool_non_none_enable_thinking_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="enable_thinking"):
            hand_bound_binding(rendering_semantics={"v": 1, "enable_thinking": "true"})

    def test_rendering_semantics_non_str_keys_rejected(self) -> None:
        bad: Any = {"v": 1, 2: None}
        with pytest.raises(InvalidDecisionError, match="keys must be str"):
            hand_bound_binding(rendering_semantics=bad)

    def test_rendering_semantics_non_json_values_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="JSON-compatible"):
            hand_bound_binding(
                rendering_semantics={"v": 1, "enable_thinking": None, "bad": object()}
            )

    def test_rendering_semantics_valid_values_accepted_unchanged(self) -> None:
        # Validation must NOT renormalize the stored projection:
        # canonical_payload keeps returning exactly what was passed in.
        binding = hand_bound_binding(rendering_semantics={"v": 1, "enable_thinking": True})
        assert binding.canonical_payload()["rendering_semantics"] == {
            "v": 1,
            "enable_thinking": True,
        }


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
        # Align the linkage identities so the coherence gate passes and the
        # family gate under test is what fires.
        choice_result = replace(choice_evaluation.result, trace_id=bool_evaluation.trace.trace_id)
        with pytest.raises(InvalidDecisionError, match="BoolResult"):
            CalibrationObservation.from_evaluation(
                (choice_result, bool_evaluation.trace), unresolved_truth()
            )

    def test_choice_family_requires_choice_result(self) -> None:
        runtime, _ = make_choice_runtime()
        bool_evaluation = runtime.evaluate_with_trace(make_bool_decision())
        choice_evaluation = runtime.evaluate_with_trace(make_choice_decision())
        # Align the linkage identities so the coherence gate passes and the
        # family gate under test is what fires.
        bool_result = replace(bool_evaluation.result, trace_id=choice_evaluation.trace.trace_id)
        with pytest.raises(InvalidDecisionError, match="ChoiceResult"):
            CalibrationObservation.from_evaluation(
                (bool_result, choice_evaluation.trace), unresolved_truth()
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
            trace_id=evaluation.trace.trace_id,
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


class TestObservationProvenanceCoherence:
    def test_coherent_evaluation_accepted(self) -> None:
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        observation = CalibrationObservation.from_evaluation(evaluation, resolved_truth("shipping"))
        assert observation.decision_family == "choice"
        assert observation.selected_value == "shipping"

    def test_mismatched_pair_rejected(self) -> None:
        runtime, _ = make_choice_runtime()
        decision = make_choice_decision()
        first = runtime.evaluate_with_trace(decision)
        second = runtime.evaluate_with_trace(decision)
        assert first.trace.trace_id != second.trace.trace_id
        with pytest.raises(InvalidDecisionError, match="matching runtime linkage"):
            CalibrationObservation.from_evaluation(
                (first.result, second.trace), resolved_truth("shipping")
            )

    def test_mismatched_evaluation_wrapper_rejected(self) -> None:
        runtime, _ = make_choice_runtime()
        decision = make_choice_decision()
        first = runtime.evaluate_with_trace(decision)
        second = runtime.evaluate_with_trace(decision)
        with pytest.raises(InvalidDecisionError, match="matching runtime linkage"):
            CalibrationObservation.from_evaluation(
                Evaluation(first.result, second.trace), resolved_truth("shipping")
            )

    def test_identical_probabilities_still_rejected(self) -> None:
        # Linkage is never inferred from probability values: two executions of
        # the same decision can emit identical distributions, and the
        # cross-pairing must still be rejected because the trace ids differ.
        runtime, _ = make_choice_runtime()
        decision = make_choice_decision()
        first = runtime.evaluate_with_trace(decision)
        second = runtime.evaluate_with_trace(decision)
        assert isinstance(first.result, ChoiceResult)
        assert isinstance(second.result, ChoiceResult)
        assert first.result.probabilities == second.result.probabilities
        assert first.trace.trace_id != second.trace.trace_id
        with pytest.raises(InvalidDecisionError, match="matching runtime linkage"):
            CalibrationObservation.from_evaluation(
                (first.result, second.trace), resolved_truth("shipping")
            )

    def test_result_without_linkage_rejected(self) -> None:
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        unlinked_result = replace(evaluation.result, trace_id=None)
        with pytest.raises(InvalidDecisionError, match="cannot be verified"):
            CalibrationObservation.from_evaluation(
                (unlinked_result, evaluation.trace), resolved_truth("shipping")
            )

    def test_direct_constructor_rejected(self) -> None:
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="from_evaluation"):
            CalibrationObservation(
                decision_family=observation.decision_family,
                outcome_order=observation.outcome_order,
                probabilities=observation.probabilities,
                selected_value=observation.selected_value,
                ground_truth=observation.ground_truth,
                binding=observation.binding,
                decision_fingerprint=observation.decision_fingerprint,
                execution_fingerprint=observation.execution_fingerprint,
            )

    def test_replace_binding_rejected(self) -> None:
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        observation = CalibrationObservation.from_evaluation(evaluation, resolved_truth("shipping"))
        second_backend = FakeCategoricalBackend(model="other-model")
        second_runtime, _ = make_choice_runtime(second_backend)
        second = second_runtime.evaluate_with_trace(make_choice_decision())
        foreign_binding = CalibrationBinding.from_trace(second.trace)
        assert observation.binding != foreign_binding
        # Replacing only the binding is the exact Frankenstein combination the
        # coherence round exists to prevent: probabilities and
        # decision_fingerprint from execution A, binding (and therefore the
        # model/revision/rendering provenance) from execution B.
        with pytest.raises(InvalidDecisionError, match="from_evaluation"):
            replace(observation, binding=foreign_binding)

    def test_replace_with_no_changes_rejected(self) -> None:
        observation = choice_observation()
        # No changed fields at all: the guard lives on the constructor itself,
        # not on any particular field change.
        with pytest.raises(InvalidDecisionError, match="from_evaluation"):
            replace(observation)

    def test_replace_derived_field_rejected(self) -> None:
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="from_evaluation"):
            replace(observation, probabilities=observation.probabilities)

    def test_construction_token_is_not_an_instance_attribute(self) -> None:
        observation = choice_observation()
        assert not hasattr(observation, "_construction_token")
        assert "_construction_token" not in {f.name for f in fields(CalibrationObservation)}

    def test_replay_same_fingerprint_different_trace_id(self) -> None:
        runtime, _ = make_choice_runtime()
        decision = make_choice_decision()
        first = runtime.evaluate_with_trace(decision)
        second = runtime.evaluate_with_trace(decision)
        assert first.trace.trace_id != second.trace.trace_id
        obs_first = CalibrationObservation.from_evaluation(first, resolved_truth("shipping"))
        obs_second = CalibrationObservation.from_evaluation(second, resolved_truth("shipping"))
        assert obs_first.fingerprint == obs_second.fingerprint

    def test_error_names_both_linkage_identities(self) -> None:
        runtime, _ = make_choice_runtime()
        decision = make_choice_decision()
        first = runtime.evaluate_with_trace(decision)
        second = runtime.evaluate_with_trace(decision)
        with pytest.raises(InvalidDecisionError) as exc_info:
            CalibrationObservation.from_evaluation(
                (first.result, second.trace), resolved_truth("shipping")
            )
        message = str(exc_info.value)
        result_linkage = first.result.trace_id
        assert result_linkage is not None
        assert result_linkage in message
        assert second.trace.trace_id in message

    def test_error_type_is_frozen_taxonomy(self) -> None:
        runtime, _ = make_choice_runtime()
        decision = make_choice_decision()
        first = runtime.evaluate_with_trace(decision)
        second = runtime.evaluate_with_trace(decision)
        with pytest.raises(InvalidDecisionError):
            CalibrationObservation.from_evaluation(
                (first.result, second.trace), resolved_truth("shipping")
            )
        import probvenance.errors

        assert "InvalidDecisionError" in dir(probvenance.errors)
        assert not any(
            name.endswith("LinkageError") or name.endswith("CoherenceError")
            for name in dir(probvenance.errors)
        )


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
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
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
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
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
        assert CALIBRATION_DATASET_FINGERPRINT_VERSION == 2
        # The version is committed inside the fingerprint payload; the
        # fingerprint itself is deterministic across constructions.
        other = CalibrationDataset.create(self._eligible_pair())
        assert dataset.fingerprint == other.fingerprint

    def test_dataset_is_not_a_set_hash(self) -> None:
        o1, o2 = choice_observation(), choice_observation()
        dataset = CalibrationDataset.create([o1, o2, o1])
        assert dataset.fingerprint != CalibrationDataset.create([o1, o2]).fingerprint

    def test_dataset_rejects_dict_equal_but_fingerprint_different_binding(self) -> None:
        # MUST-FIX regression: Python dict equality conflates True == 1, so
        # the dataset check compares canonical JSON strings, not objects. A
        # forged "v": True rendering key is dict-equal to the real "v": 1
        # projection yet yields a different binding fingerprint.
        observation = choice_observation()
        forged = hand_bound_binding(
            base=observation.binding,
            rendering_semantics={"v": True, "enable_thinking": None},
        )
        assert observation.binding.canonical_payload() == forged.canonical_payload()
        assert observation.binding.fingerprint != forged.fingerprint
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
            CalibrationDataset(binding=forged, observations=(observation,))

    def test_enable_thinking_int_binding_rejected_before_dataset_layer(self) -> None:
        # The audit counterexample binding cannot even reach the dataset
        # check: binding construction itself rejects enable_thinking=1, so
        # the construction layer rejects first and the dataset check
        # backstops anything a caller might still smuggle past it.
        observation = choice_observation()
        with pytest.raises(InvalidDecisionError, match="enable_thinking"):
            hand_bound_binding(
                base=observation.binding,
                rendering_semantics={"v": 1, "enable_thinking": 1},
            )

    def test_replace_observations_reruns_dataset_checks(self) -> None:
        dataset = CalibrationDataset.create(self._eligible_pair())
        other_rule = choice_observation(
            resolved_truth("shipping", labeling_rule="a different rule")
        )
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics identity"):
            replace(dataset, observations=(dataset.observations[0], other_rule))
        other_model = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
            replace(dataset, observations=(dataset.observations[0], other_model))


# ---------------------------------------------------------------------------
# Ground-truth semantics identity (Phase 4A.2, Part A)
# ---------------------------------------------------------------------------


class TestGroundTruthSemanticsIdentity:
    def test_same_semantics_same_fingerprint(self) -> None:
        first = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        second = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        assert first == second
        assert first.fingerprint == second.fingerprint

    def test_different_labeling_rule_different_fingerprint(self) -> None:
        base = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        other = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(labeling_rule="a different rule")
        )
        assert base != other
        assert base.fingerprint != other.fingerprint

    def test_different_ambiguity_policy_different_fingerprint(self) -> None:
        base = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        other = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(ambiguity_policy="drop ambiguous tickets")
        )
        assert base != other
        assert base.fingerprint != other.fingerprint

    def test_different_taxonomy_id_different_fingerprint(self) -> None:
        base = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(taxonomy_id="departments", taxonomy_version=2)
        )
        other = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(taxonomy_id="queues", taxonomy_version=2)
        )
        assert base != other
        assert base.fingerprint != other.fingerprint

    def test_different_taxonomy_version_different_fingerprint(self) -> None:
        base = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(taxonomy_id="departments", taxonomy_version=2)
        )
        other = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(taxonomy_id="departments", taxonomy_version=3)
        )
        assert base != other
        assert base.fingerprint != other.fingerprint

    def test_different_label_source_only_same_semantics(self) -> None:
        base = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        other = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(label_source="senior-annotator")
        )
        assert base == other
        assert base.fingerprint == other.fingerprint

    def test_from_provenance_is_deterministic(self) -> None:
        provenance = adjudicated_provenance(taxonomy_id="departments", taxonomy_version=2)
        assert GroundTruthSemanticsIdentity.from_provenance(
            provenance
        ) == GroundTruthSemanticsIdentity.from_provenance(provenance)

    def test_canonical_payload_has_no_version_key(self) -> None:
        identity = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        payload = identity.canonical_payload()
        assert set(payload) == {
            "labeling_rule",
            "ambiguity_policy",
            "taxonomy_id",
            "taxonomy_version",
        }
        assert "v" not in payload

    def test_fingerprint_is_version_tagged(self) -> None:
        identity = GroundTruthSemanticsIdentity.from_provenance(adjudicated_provenance())
        assert GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION == 1
        unversioned = fingerprint(identity.canonical_payload())
        assert identity.fingerprint != unversioned

    def test_taxonomy_pair_may_be_independently_none(self) -> None:
        identity = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(taxonomy_id="departments")
        )
        assert identity.taxonomy_id == "departments"
        assert identity.taxonomy_version is None
        identity = GroundTruthSemanticsIdentity.from_provenance(
            adjudicated_provenance(taxonomy_version=3)
        )
        assert identity.taxonomy_id is None
        assert identity.taxonomy_version == 3

    def test_invalid_fields_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="labeling_rule"):
            GroundTruthSemanticsIdentity("   ", "policy", None, None)
        with pytest.raises(InvalidDecisionError, match="ambiguity_policy"):
            GroundTruthSemanticsIdentity("rule", "  ", None, None)
        with pytest.raises(InvalidDecisionError, match="taxonomy_version"):
            GroundTruthSemanticsIdentity("rule", "policy", None, 0)
        with pytest.raises(InvalidDecisionError, match="taxonomy_version"):
            GroundTruthSemanticsIdentity("rule", "policy", None, True)


# ---------------------------------------------------------------------------
# Dataset pooling semantics (Phase 4A.2, Part C)
# ---------------------------------------------------------------------------


class TestDatasetPoolingSemantics:
    def test_same_binding_same_semantics_different_label_source_accepted(self) -> None:
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(resolved_truth("shipping", label_source="senior-annotator"))
        dataset = CalibrationDataset.create([first, second])
        assert dataset.observations == (first, second)

    def test_label_source_pooling_and_fingerprint_distinction_combined(self) -> None:
        # Both halves of the label-source claim together: observations from
        # different annotators are pooled into ONE dataset, while their
        # observation fingerprints stay distinct because label_source stays
        # in the observation identity.
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(resolved_truth("shipping", label_source="senior-annotator"))
        dataset = CalibrationDataset.create([first, second])
        assert dataset.observations == (first, second)
        assert first.fingerprint != second.fingerprint

    def test_same_binding_different_labeling_rule_rejected(self) -> None:
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(resolved_truth("shipping", labeling_rule="a different rule"))
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics identity"):
            CalibrationDataset.create([first, second])

    def test_same_binding_different_ambiguity_policy_rejected(self) -> None:
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(
            resolved_truth("shipping", ambiguity_policy="drop ambiguous tickets")
        )
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics identity"):
            CalibrationDataset.create([first, second])

    def test_same_binding_different_ground_truth_taxonomy_rejected(self) -> None:
        first = choice_observation(
            resolved_truth("shipping", taxonomy_id="departments", taxonomy_version=2)
        )
        second = choice_observation(
            resolved_truth("shipping", taxonomy_id="departments", taxonomy_version=3)
        )
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics identity"):
            CalibrationDataset.create([first, second])

    def test_different_binding_same_semantics_rejected(self) -> None:
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(
            resolved_truth("shipping"), backend=FakeCategoricalBackend(model="other-model")
        )
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
            CalibrationDataset.create([first, second])

    def test_rejection_messages_distinguish_binding_from_semantics(self) -> None:
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(
            resolved_truth("shipping"), backend=FakeCategoricalBackend(model="other-model")
        )
        third = choice_observation(resolved_truth("shipping", labeling_rule="a different rule"))
        with pytest.raises(InvalidDecisionError) as excinfo:
            CalibrationDataset.create([first, second, third])
        message = str(excinfo.value)
        assert "binding canonical payload" in message
        assert "ground-truth semantics identity" in message
        assert "observation 1" in message
        assert "observation 2" in message
        assert "2 of 3" in message

    def test_direct_construction_enforces_same_checks(self) -> None:
        first = choice_observation(resolved_truth("shipping"))
        second = choice_observation(resolved_truth("shipping", labeling_rule="a different rule"))
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics identity"):
            CalibrationDataset(binding=first.binding, observations=(first, second))
        other_model = choice_observation(
            resolved_truth("shipping"), backend=FakeCategoricalBackend(model="other-model")
        )
        with pytest.raises(InvalidDecisionError, match="binding canonical payload"):
            CalibrationDataset(binding=first.binding, observations=(first, other_model))

    def test_ground_truth_semantics_is_derived_not_settable(self) -> None:
        dataset = CalibrationDataset.create([choice_observation(resolved_truth("shipping"))])
        derived = dataset.ground_truth_semantics
        assert isinstance(derived, GroundTruthSemanticsIdentity)
        assert derived == GroundTruthSemanticsIdentity.from_provenance(
            dataset.observations[0].ground_truth.provenance
        )
        assert "ground_truth_semantics" not in {f.name for f in fields(CalibrationDataset)}
        # A frozen slots dataclass raises TypeError for a non-field name
        # (a plain property would raise AttributeError); either way the
        # property is not assignable.
        with pytest.raises((AttributeError, TypeError)):
            dataset.ground_truth_semantics = derived  # type: ignore[misc]

    def test_ground_truth_semantics_docstring_documents_disagreement(self) -> None:
        docstring = inspect.getdoc(CalibrationDataset.ground_truth_semantics)
        assert docstring is not None
        assert "first observation" in docstring


# ---------------------------------------------------------------------------
# Fingerprint version guards (Phase 4A.2, Parts B, D, E)
# ---------------------------------------------------------------------------


class TestFingerprintVersionGuards:
    def _eligible_pair(self) -> list[CalibrationObservation]:
        return [choice_observation(), choice_observation()]

    def test_binding_fingerprint_version_is_one(self) -> None:
        assert CALIBRATION_BINDING_FINGERPRINT_VERSION == 1

    def test_binding_fingerprint_deterministic_for_same_canonical_binding(self) -> None:
        first = choice_observation().binding
        second = choice_observation().binding
        assert first.canonical_payload() == second.canonical_payload()
        assert first.fingerprint == second.fingerprint

    def test_changed_binding_semantic_dimension_changes_fingerprint(self) -> None:
        base = choice_observation().binding
        other = choice_observation(backend=FakeCategoricalBackend(model="other-model")).binding
        assert base.canonical_payload() != other.canonical_payload()
        assert base.fingerprint != other.fingerprint

    def test_binding_fingerprint_is_version_tagged(self) -> None:
        binding = choice_observation().binding
        unversioned = fingerprint(binding.canonical_payload())
        assert binding.fingerprint != unversioned

    def test_dataset_fingerprint_version_is_two(self) -> None:
        assert CALIBRATION_DATASET_FINGERPRINT_VERSION == 2

    def test_dataset_fingerprint_commits_identity_versions(self) -> None:
        dataset = CalibrationDataset.create(self._eligible_pair())
        payload = {
            "v": CALIBRATION_DATASET_FINGERPRINT_VERSION,
            "binding_fingerprint": dataset.binding.fingerprint,
            "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "ground_truth_semantics_fingerprint": (dataset.ground_truth_semantics.fingerprint),
            "ground_truth_semantics_fingerprint_version": (
                GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
            ),
            "observation_fingerprints": sorted(
                (o.fingerprint for o in dataset.observations), key=str
            ),
        }
        assert dataset.fingerprint == fingerprint(payload)

    def test_observation_fingerprint_version_stays_one(self) -> None:
        # Regression guard: the Phase 4A.2 round adds a new identity object
        # (GroundTruthSemanticsIdentity) but does NOT change the observation
        # payload schema, so the observation fingerprint version must stay 1.
        assert CALIBRATION_OBSERVATION_FINGERPRINT_VERSION == 1

    def test_observation_payload_still_embeds_binding_canonical_payload(self) -> None:
        observation = choice_observation()
        payload = observation.canonical_payload()
        assert payload["binding"] == observation.binding.canonical_payload()
        ground_truth_payload = payload["ground_truth"]
        assert isinstance(ground_truth_payload, dict)
        assert "provenance" in ground_truth_payload


# ---------------------------------------------------------------------------
# Observation lineage under the semantics split (Phase 4A.2, Part E)
# ---------------------------------------------------------------------------


class TestObservationLineageUnderSemanticsSplit:
    def test_different_label_source_different_observation_fingerprint(self) -> None:
        base = choice_observation(resolved_truth("shipping"))
        other = choice_observation(resolved_truth("shipping", label_source="senior-annotator"))
        assert base.fingerprint != other.fingerprint

    def test_different_labeling_rule_different_observation_fingerprint(self) -> None:
        base = choice_observation(resolved_truth("shipping"))
        other = choice_observation(resolved_truth("shipping", labeling_rule="a different rule"))
        assert base.fingerprint != other.fingerprint


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
