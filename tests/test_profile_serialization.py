"""Serialization and identity-verified loading of ``CalibrationProfile``.

The central contract: a serialized document materializes enough state to
restore the fitted artifact, and the loader re-derives every nested identity
instead of trusting the hashes the document claims.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    choice_observation,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
)

from probvenance import InvalidDecisionError
from probvenance.calibration import (
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_SERIALIZATION_TYPE,
    CALIBRATION_PROFILE_SERIALIZATION_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    CalibrationDataset,
    CalibrationProfile,
    apply_profile_to_runtime_evaluation,
    fit_l2_logistic_selected_probability,
    load_calibration_profile,
    predicted_winner_correctness,
    serialize_calibration_profile,
)
from probvenance.calibration_evaluation import (
    CalibrationEvaluationCohort,
    CalibrationEvaluationDataset,
    EvaluationSplitRole,
    apply_profile_to_evaluation_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fitted_profile() -> CalibrationProfile:
    return fit_l2_logistic_selected_probability(
        CalibrationDataset.create([choice_observation()]), l2_strength=1.0
    )


def document_of(profile: CalibrationProfile) -> dict[str, Any]:
    return json.loads(serialize_calibration_profile(profile))


def reload_document(document: dict[str, Any]) -> CalibrationProfile:
    return load_calibration_profile(json.dumps(document))


def assert_document_rejected(document: dict[str, Any]) -> None:
    with pytest.raises(InvalidDecisionError):
        reload_document(document)


def tampered(mutate: Any) -> dict[str, Any]:
    document = document_of(fitted_profile())
    mutate(document)
    return document


def evaluation_dataset(observations: tuple[Any, ...]) -> CalibrationEvaluationDataset:
    cohort = CalibrationEvaluationCohort(observations, EvaluationSplitRole.VALIDATION, "eval-1")
    return CalibrationEvaluationDataset.from_cohort(cohort)


def runtime_evaluation() -> Any:
    runtime, _ = make_choice_runtime()
    return runtime.evaluate_with_trace(make_choice_decision())


# ---------------------------------------------------------------------------
# Round trip (PART 37, PART 38, PART 39, PART 40)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_loaded_identity_equals_original(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(serialize_calibration_profile(profile))
        assert loaded.fingerprint == profile.fingerprint
        assert loaded.canonical_payload() == profile.canonical_payload()
        assert loaded.binding.fingerprint == profile.binding.fingerprint
        assert (
            loaded.ground_truth_semantics.fingerprint == profile.ground_truth_semantics.fingerprint
        )

    def test_loaded_materialized_payloads_equal_original(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(serialize_calibration_profile(profile))
        assert loaded.binding.canonical_payload() == profile.binding.canonical_payload()
        assert (
            loaded.ground_truth_semantics.canonical_payload()
            == profile.ground_truth_semantics.canonical_payload()
        )
        assert dict(loaded.method_configuration) == dict(profile.method_configuration)
        assert dict(loaded.fitted_parameters) == dict(profile.fitted_parameters)

    def test_serialization_is_deterministic(self) -> None:
        profile = fitted_profile()
        assert serialize_calibration_profile(profile) == serialize_calibration_profile(profile)

    def test_serialize_load_serialize_is_a_fixed_point(self) -> None:
        profile = fitted_profile()
        first = serialize_calibration_profile(profile)
        second = serialize_calibration_profile(load_calibration_profile(first))
        assert second == first


class TestNoncanonicalNormalization:
    def test_noncanonical_json_loads_and_normalizes(self) -> None:
        profile = fitted_profile()
        canonical = serialize_calibration_profile(profile)
        noncanonical = json.dumps(json.loads(canonical), indent=4, sort_keys=True)
        assert noncanonical != canonical
        loaded = load_calibration_profile(noncanonical)
        assert loaded.fingerprint == profile.fingerprint
        assert serialize_calibration_profile(loaded) == canonical


# ---------------------------------------------------------------------------
# Tamper attacks (PART 44 to PART 48)
# ---------------------------------------------------------------------------


class TestTamperAttacks:
    def test_outer_profile_fingerprint_tamper_rejected(self) -> None:
        assert_document_rejected(tampered(lambda d: d.__setitem__("profile_fingerprint", "0" * 64)))

    def test_binding_materialization_tamper_rejected(self) -> None:
        assert_document_rejected(
            tampered(lambda d: d["materialized_binding"].__setitem__("model_revision", "rev-x"))
        )

    def test_ground_truth_materialization_tamper_rejected(self) -> None:
        assert_document_rejected(
            tampered(
                lambda d: d["materialized_ground_truth_semantics"].__setitem__(
                    "labeling_rule", "an-evil-rule"
                )
            )
        )

    def test_fitted_parameters_tamper_rejected(self) -> None:
        assert_document_rejected(
            tampered(
                lambda d: d["profile_identity"]["fitted_parameters"].__setitem__("slope", 999.0)
            )
        )

    def test_method_configuration_tamper_rejected(self) -> None:
        assert_document_rejected(
            tampered(
                lambda d: d["profile_identity"].__setitem__(
                    "method_configuration", {"l2_strength": 0.0}
                )
            )
        )

    def test_fully_recomputed_document_is_caught_by_expected_pin(self) -> None:
        profile = fitted_profile()
        trusted = profile.fingerprint
        document = document_of(profile)
        document["profile_identity"]["fitted_parameters"]["slope"] = 999.0
        document["profile_identity"]["fitted_parameters"]["intercept"] = 0.0
        document["materialized_binding"]["model"] = "a-different-model"
        # A fully recomputed document is internally consistent, so the loader
        # cannot detect it from embedded hashes alone.
        recomputed = _recompute_identity(document)
        assert load_calibration_profile(recomputed).fingerprint != trusted
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(
                recomputed,
                expected_profile_fingerprint=trusted,
                expected_profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
            )


def _recompute_identity(document: dict[str, Any]) -> str:
    """Rebuild a self-consistent malicious document, as an attacker would.

    Reconstructs the nested objects from the (tampered) materialized payloads
    and recomputes every embedded fingerprint so the document is internally
    coherent. Only an out-of-band expected fingerprint can catch this.
    """
    from probvenance.calibration import _restore_calibration_binding

    binding = _restore_calibration_binding(document["materialized_binding"])
    document["profile_identity"]["binding"]["binding_fingerprint"] = binding.fingerprint
    profile = CalibrationProfile._from_serialized_state(
        binding=binding,
        ground_truth_semantics=document["profile_identity"]["ground_truth_semantics"]
        and _restore_gt(document),
        target_id=document["profile_identity"]["target_id"],
        target_version=document["profile_identity"]["target_version"],
        input_score_id=document["profile_identity"]["input_score_id"],
        input_score_version=document["profile_identity"]["input_score_version"],
        method_id=document["profile_identity"]["method_id"],
        method_version=document["profile_identity"]["method_version"],
        method_configuration=document["profile_identity"]["method_configuration"],
        fitted_parameters=document["profile_identity"]["fitted_parameters"],
        training_dataset_fingerprint=document["profile_identity"]["training_dataset_fingerprint"],
        training_dataset_fingerprint_version=document["profile_identity"][
            "training_dataset_fingerprint_version"
        ],
    )
    document["profile_fingerprint"] = profile.fingerprint
    return json.dumps(document)


def _restore_gt(document: dict[str, Any]) -> Any:
    from probvenance.calibration import _restore_ground_truth_semantics_identity

    semantics = _restore_ground_truth_semantics_identity(
        document["materialized_ground_truth_semantics"]
    )
    document["profile_identity"]["ground_truth_semantics"]["ground_truth_semantics_fingerprint"] = (
        semantics.fingerprint
    )
    return semantics


# ---------------------------------------------------------------------------
# Expected identity pinning (PART 49, PART 50, PART 51)
# ---------------------------------------------------------------------------


class TestExpectedIdentityPinning:
    def test_matching_pin_succeeds(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(
            serialize_calibration_profile(profile),
            expected_profile_fingerprint=profile.fingerprint,
            expected_profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
        )
        assert loaded.fingerprint == profile.fingerprint

    def test_wrong_expected_fingerprint_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(
                serialize_calibration_profile(fitted_profile()),
                expected_profile_fingerprint="0" * 64,
                expected_profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
            )

    def test_wrong_expected_version_rejected(self) -> None:
        profile = fitted_profile()
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(
                serialize_calibration_profile(profile),
                expected_profile_fingerprint=profile.fingerprint,
                expected_profile_fingerprint_version=2,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"expected_profile_fingerprint": "0" * 64},
            {"expected_profile_fingerprint_version": 1},
        ],
    )
    def test_half_specified_expected_identity_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(InvalidDecisionError, match="together or not at all"):
            load_calibration_profile(serialize_calibration_profile(fitted_profile()), **kwargs)

    @pytest.mark.parametrize("version", [True, False, 1.0, "1"])
    def test_non_integer_expected_version_rejected(self, version: Any) -> None:
        profile = fitted_profile()
        with pytest.raises(InvalidDecisionError, match="must be an integer"):
            load_calibration_profile(
                serialize_calibration_profile(profile),
                expected_profile_fingerprint=profile.fingerprint,
                expected_profile_fingerprint_version=version,
            )


# ---------------------------------------------------------------------------
# Strict parser and schema (PART 52 to PART 57)
# ---------------------------------------------------------------------------


class TestStrictParser:
    @pytest.mark.parametrize("version", [0, True, "1", 2, None])
    def test_unsupported_serialization_version_rejected(self, version: Any) -> None:
        assert_document_rejected(
            tampered(lambda d: d.__setitem__("serialization_version", version))
        )

    @pytest.mark.parametrize("artifact_type", ["other", "", None])
    def test_bad_artifact_type_rejected(self, artifact_type: Any) -> None:
        assert_document_rejected(tampered(lambda d: d.__setitem__("artifact_type", artifact_type)))

    def test_missing_artifact_type_rejected(self) -> None:
        assert_document_rejected(tampered(lambda d: d.pop("artifact_type")))

    @pytest.mark.parametrize(
        "level",
        ["envelope", "profile_identity", "materialized_binding", "materialized_gt"],
    )
    def test_unknown_key_rejected(self, level: str) -> None:
        def mutate(d: dict[str, Any]) -> None:
            if level == "envelope":
                d["unexpected"] = 1
            elif level == "profile_identity":
                d["profile_identity"]["unexpected"] = 1
            elif level == "materialized_binding":
                d["materialized_binding"]["unexpected"] = 1
            else:
                d["materialized_ground_truth_semantics"]["unexpected"] = 1

        assert_document_rejected(tampered(mutate))

    @pytest.mark.parametrize(
        "level",
        ["envelope", "profile_identity", "materialized_binding", "materialized_gt"],
    )
    def test_missing_key_rejected(self, level: str) -> None:
        def mutate(d: dict[str, Any]) -> None:
            if level == "envelope":
                d.pop("profile_identity")
            elif level == "profile_identity":
                d["profile_identity"].pop("method_id")
            elif level == "materialized_binding":
                d["materialized_binding"].pop("model")
            else:
                d["materialized_ground_truth_semantics"].pop("labeling_rule")

        assert_document_rejected(tampered(mutate))

    @pytest.mark.parametrize(
        "level",
        ["envelope", "profile_identity", "materialized_binding", "materialized_gt"],
    )
    def test_duplicate_key_rejected(self, level: str) -> None:
        canonical = serialize_calibration_profile(fitted_profile())
        key = {
            "envelope": "artifact_type",
            "profile_identity": "target_id",
            "materialized_binding": "model",
            "materialized_gt": "labeling_rule",
        }[level]
        match = re.search(rf'"{key}":(?:"[^"]*"|[^,}}]+)', canonical)
        assert match is not None
        pair = match.group(0)
        duplicated = canonical.replace(pair, f"{pair},{pair}", 1)
        assert duplicated != canonical
        with pytest.raises(InvalidDecisionError, match="duplicate"):
            load_calibration_profile(duplicated)

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e9999"])
    def test_non_finite_number_rejected(self, literal: str) -> None:
        canonical = serialize_calibration_profile(fitted_profile())
        injected = canonical.replace(
            f'"serialization_version":{CALIBRATION_PROFILE_SERIALIZATION_VERSION}',
            f'"serialization_version":{literal}',
        )
        assert injected != canonical
        with pytest.raises(InvalidDecisionError, match="non-finite"):
            load_calibration_profile(injected)

    def test_deeply_nested_document_rejected(self) -> None:
        deeply_nested = "[" * 2000 + "]" * 2000
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(deeply_nested)

    def test_document_that_parses_but_nests_too_deeply_is_rejected(self) -> None:
        document = document_of(fitted_profile())
        nested: Any = "x"
        for _ in range(495):
            nested = {"n": nested}
        document["materialized_binding"]["rendering_semantics"] = nested
        assert_document_rejected(document)

    @pytest.mark.parametrize("text", ["[]", '"string"', "123", "null"])
    def test_non_object_top_level_rejected(self, text: str) -> None:
        with pytest.raises(InvalidDecisionError, match="must be a JSON object"):
            load_calibration_profile(text)

    def test_invalid_json_rejected_without_leaking_parser_error(self) -> None:
        with pytest.raises(InvalidDecisionError, match="not valid JSON"):
            load_calibration_profile("{not json")

    def test_non_string_input_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(b"{}")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability and construction (PART 60, PART 61)
# ---------------------------------------------------------------------------


class TestImmutabilityAndConstruction:
    def test_loaded_profile_is_deeply_immutable(self) -> None:
        loaded = load_calibration_profile(serialize_calibration_profile(fitted_profile()))
        fingerprint_before = loaded.fingerprint
        with pytest.raises(TypeError):
            loaded.method_configuration["l2_strength"] = 0.0  # type: ignore[index]
        with pytest.raises(TypeError):
            loaded.fitted_parameters["slope"] = 0.0  # type: ignore[index]
        with pytest.raises(TypeError):
            loaded.binding.rendering_semantics["enable_thinking"] = True  # type: ignore[index]
        with pytest.raises(AttributeError):
            loaded.target_id = "other"  # type: ignore[misc]
        assert loaded.fingerprint == fingerprint_before

    def test_direct_construction_still_rejected(self) -> None:
        profile = fitted_profile()
        with pytest.raises(InvalidDecisionError):
            CalibrationProfile(
                binding=profile.binding,
                ground_truth_semantics=profile.ground_truth_semantics,
                target_id=profile.target_id,
                target_version=profile.target_version,
                input_score_id=profile.input_score_id,
                input_score_version=profile.input_score_version,
                method_id=profile.method_id,
                method_version=profile.method_version,
                method_configuration=profile.method_configuration,
                fitted_parameters=profile.fitted_parameters,
                training_dataset_fingerprint=profile.training_dataset_fingerprint,
                training_dataset_fingerprint_version=profile.training_dataset_fingerprint_version,
            )

    def test_dataclasses_replace_still_rejected(self) -> None:
        import dataclasses

        profile = fitted_profile()
        with pytest.raises(InvalidDecisionError):
            dataclasses.replace(profile, method_id="other")


# ---------------------------------------------------------------------------
# Application equivalence and surviving gates (PART 63 to PART 68)
# ---------------------------------------------------------------------------


class TestApplicationEquivalence:
    def test_loaded_profile_scores_offline_identically(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(serialize_calibration_profile(profile))
        dataset = evaluation_dataset((choice_observation(),))
        original = apply_profile_to_evaluation_dataset(dataset, profile)
        restored = apply_profile_to_evaluation_dataset(dataset, loaded)
        assert [row.predicted_correctness for row in original.rows] == [
            row.predicted_correctness for row in restored.rows
        ]
        assert restored.profile_fingerprint == original.profile_fingerprint

    def test_loaded_profile_scores_runtime_linked_identically(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(serialize_calibration_profile(profile))
        evaluation = runtime_evaluation()
        original = apply_profile_to_runtime_evaluation(evaluation, profile)
        restored = apply_profile_to_runtime_evaluation(evaluation, loaded)
        assert original.result.predicted_correctness == restored.result.predicted_correctness
        assert (
            original.result.calibration_profile_fingerprint
            == restored.result.calibration_profile_fingerprint
            == profile.fingerprint
        )
        assert (
            original.trace.execution_fingerprint
            == restored.trace.execution_fingerprint
            == evaluation.trace.execution_fingerprint
        )

    def test_loaded_profile_preserves_scorer_binding_gate(self) -> None:
        loaded = load_calibration_profile(serialize_calibration_profile(fitted_profile()))
        other = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        with pytest.raises(InvalidDecisionError, match="binding does not match"):
            predicted_winner_correctness(loaded, other)

    def test_loaded_profile_preserves_offline_ground_truth_gate(self) -> None:
        loaded = load_calibration_profile(serialize_calibration_profile(fitted_profile()))
        other_semantics = choice_observation(
            resolved_truth("shipping", labeling_rule="a different labeling rule")
        )
        dataset = evaluation_dataset((other_semantics,))
        with pytest.raises(InvalidDecisionError, match="ground-truth semantics"):
            apply_profile_to_evaluation_dataset(dataset, loaded)

    def test_loaded_profile_preserves_runtime_declaration_gate(self) -> None:
        loaded = load_calibration_profile(serialize_calibration_profile(fitted_profile()))
        evaluation = runtime_evaluation()
        assert apply_profile_to_runtime_evaluation(evaluation, loaded).result.calibrated
        with pytest.raises(InvalidDecisionError, match="binding does not match"):
            apply_profile_to_runtime_evaluation(evaluation, loaded, task_id="declared-task")

    def test_runtime_provenance_carries_profile_not_serialization_identity(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(serialize_calibration_profile(profile))
        applied = apply_profile_to_runtime_evaluation(runtime_evaluation(), loaded)
        assert applied.result.calibration_profile_fingerprint == loaded.fingerprint
        assert (
            applied.result.calibration_profile_fingerprint_version
            == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        )
        serialized_keys = set(applied.trace.to_dict())
        assert "serialization_version" not in serialized_keys
        assert "profile_fingerprint" not in serialized_keys


# ---------------------------------------------------------------------------
# Version and scope boundaries (PART 29, PART 69 to PART 71)
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    def test_versions_are_unchanged_by_serialization(self) -> None:
        profile = fitted_profile()
        loaded = load_calibration_profile(serialize_calibration_profile(profile))
        payload: dict[str, Any] = profile.canonical_payload()
        assert payload["v"] == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        assert (
            payload["binding"]["binding_fingerprint_version"]
            == CALIBRATION_BINDING_FINGERPRINT_VERSION
        )
        assert (
            payload["ground_truth_semantics"]["ground_truth_semantics_fingerprint_version"]
            == GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
        )
        assert loaded.canonical_payload() == profile.canonical_payload()

    def test_envelope_is_a_single_versioned_artifact_type(self) -> None:
        document = document_of(fitted_profile())
        assert document["artifact_type"] == CALIBRATION_PROFILE_SERIALIZATION_TYPE
        assert document["serialization_version"] == CALIBRATION_PROFILE_SERIALIZATION_VERSION

    def test_canonical_payload_alone_does_not_round_trip(self) -> None:
        from probvenance.fingerprint import canonical_json

        profile = fitted_profile()
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(canonical_json(profile.canonical_payload()))

    def test_no_filesystem_or_registry_api_exists(self) -> None:
        import probvenance.calibration as calibration_module

        for forbidden in (
            "save_calibration_profile",
            "load_calibration_profile_from_path",
            "ProfileRegistry",
            "register_profile",
            "lookup_profile",
            "select_profile",
        ):
            assert not hasattr(calibration_module, forbidden)
