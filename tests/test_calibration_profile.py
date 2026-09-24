"""Tests for the calibration profile identity artifact (Phase 4C.1).

Profiles are built through the internal ``_from_fitted_state`` producer over
REAL fitting datasets built with the ``test_calibration.py`` helpers: real
observations through the supported ``from_evaluation`` path, pooled with
``CalibrationDataset.create``. The evaluation-layer helpers in
``test_calibration_evaluation.py`` build evaluation datasets, which are not
fitting datasets, and are used here only for the Phase 4B fingerprint
regression guard.
"""

import inspect
import subprocess
import sys
from dataclasses import fields, replace
from typing import Any

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    choice_observation,
    make_bool_decision,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
)
from test_calibration_evaluation import _reliability_observation, evaluation_dataset

from probvenance import InvalidDecisionError
from probvenance.calibration import (
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    CALIBRATION_DATASET_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    UNCALIBRATED_SELECTED_PROBABILITY_ID,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    WINNER_CORRECTNESS_TARGET_ID,
    WINNER_CORRECTNESS_TARGET_VERSION,
    CalibrationBinding,
    CalibrationDataset,
    CalibrationObservation,
    CalibrationProfile,
)
from probvenance.calibration_evaluation import (
    evaluate_uncalibrated_winner_brier,
    evaluate_uncalibrated_winner_diagnostics,
    evaluate_uncalibrated_winner_log_loss,
    evaluate_uncalibrated_winner_reliability,
    evaluate_winner_binned_absolute_gap,
)
from probvenance.fingerprint import JSONValue

# ---------------------------------------------------------------------------
# Fitting dataset and profile helpers
# ---------------------------------------------------------------------------


def fitting_dataset(truths: tuple[str, ...] = ("shipping", "billing")) -> CalibrationDataset:
    """A valid fitting dataset over the standard three-candidate decision."""
    observations = [choice_observation(resolved_truth(truth)) for truth in truths]
    return CalibrationDataset.create(observations)


def make_profile(dataset: CalibrationDataset | None = None, **overrides: Any) -> CalibrationProfile:
    kwargs: dict[str, Any] = {"method_id": "test-method", "method_version": 1}
    kwargs.update(overrides)
    return CalibrationProfile._from_fitted_state(
        dataset if dataset is not None else fitting_dataset(), **kwargs
    )


def taxonomy_mismatch_dataset(
    *,
    binding_taxonomy_id: str,
    binding_taxonomy_version: int,
    truth_taxonomy_id: str,
    truth_taxonomy_version: int,
) -> CalibrationDataset:
    """A fitting dataset whose binding and ground-truth taxonomies are declared."""
    runtime, _ = make_choice_runtime()
    evaluation = runtime.evaluate_with_trace(make_choice_decision())
    observation = CalibrationObservation.from_evaluation(
        evaluation,
        resolved_truth(
            "shipping", taxonomy_id=truth_taxonomy_id, taxonomy_version=truth_taxonomy_version
        ),
        taxonomy_id=binding_taxonomy_id,
        taxonomy_version=binding_taxonomy_version,
    )
    return CalibrationDataset.create([observation])


# ---------------------------------------------------------------------------
# Construction guard (tests 1, 2, 3)
# ---------------------------------------------------------------------------


def direct_construction_kwargs() -> dict[str, Any]:
    """Field values a caller would need to hand-build a profile."""
    return {
        "binding": fitting_dataset().binding,
        "ground_truth_semantics": fitting_dataset().ground_truth_semantics,
        "target_id": WINNER_CORRECTNESS_TARGET_ID,
        "target_version": WINNER_CORRECTNESS_TARGET_VERSION,
        "input_score_id": UNCALIBRATED_SELECTED_PROBABILITY_ID,
        "input_score_version": UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
        "method_id": "test-method",
        "method_version": 1,
        "method_configuration": {},
        "fitted_parameters": {},
        "training_dataset_fingerprint": "fabricated",
        "training_dataset_fingerprint_version": 1,
    }


class TestProfileConstructionGuard:
    def test_direct_construction_rejected(self):
        with pytest.raises(InvalidDecisionError, match="cannot be constructed directly"):
            CalibrationProfile(**direct_construction_kwargs())

    def test_rejection_message_does_not_advertise_the_internal_builder(self):
        with pytest.raises(InvalidDecisionError) as excinfo:
            CalibrationProfile(**direct_construction_kwargs())
        message = str(excinfo.value)
        assert "supported calibration fitter" in message
        assert "fit_l2_logistic_selected_probability" in message
        assert "_from_fitted_state" not in message

    def test_replace_rejected(self):
        profile = make_profile()
        with pytest.raises(InvalidDecisionError, match="cannot be constructed directly"):
            replace(profile)

    def test_non_dataset_training_source_rejected(self):
        with pytest.raises(InvalidDecisionError, match="CalibrationDataset"):
            CalibrationProfile._from_fitted_state(
                "not a dataset",  # type: ignore[arg-type]
                method_id="test-method",
                method_version=1,
            )


# ---------------------------------------------------------------------------
# Derived identity is not caller-overridable (tests 4, 5)
# ---------------------------------------------------------------------------


class TestDerivedIdentityNotOverridable:
    def test_target_identity_is_not_a_parameter(self):
        parameters = inspect.signature(CalibrationProfile._from_fitted_state).parameters
        assert "target_id" not in parameters
        assert "target_version" not in parameters

    def test_input_score_identity_is_not_a_parameter(self):
        parameters = inspect.signature(CalibrationProfile._from_fitted_state).parameters
        assert "input_score_id" not in parameters
        assert "input_score_version" not in parameters

    def test_dataset_derived_fields_are_not_parameters(self):
        # binding, ground_truth_semantics, and the training dataset
        # fingerprint are derived from the dataset, never supplied.
        parameters = inspect.signature(CalibrationProfile._from_fitted_state).parameters
        for forbidden in (
            "binding",
            "ground_truth_semantics",
            "training_dataset_fingerprint",
            "training_dataset_fingerprint_version",
        ):
            assert forbidden not in parameters

    def test_derived_identities_come_from_the_module_constants(self):
        dataset = fitting_dataset()
        profile = make_profile(dataset)
        assert profile.target_id == WINNER_CORRECTNESS_TARGET_ID
        assert profile.target_version == WINNER_CORRECTNESS_TARGET_VERSION
        assert profile.input_score_id == UNCALIBRATED_SELECTED_PROBABILITY_ID
        assert profile.input_score_version == UNCALIBRATED_SELECTED_PROBABILITY_VERSION
        assert profile.binding == dataset.binding
        assert profile.ground_truth_semantics == dataset.ground_truth_semantics
        assert profile.training_dataset_fingerprint == dataset.fingerprint
        assert profile.training_dataset_fingerprint_version == (
            CALIBRATION_DATASET_FINGERPRINT_VERSION
        )


# ---------------------------------------------------------------------------
# Method identity and method-state validation (tests 6, 7, 8, 9)
# ---------------------------------------------------------------------------


class TestMethodStateValidation:
    def test_empty_or_whitespace_method_id_rejected(self):
        for bad in ("", "   "):
            with pytest.raises(InvalidDecisionError, match="method_id"):
                make_profile(method_id=bad)

    @pytest.mark.parametrize("bad_version", [True, 0, -1, 2.0, "2", None])
    def test_invalid_method_version_rejected(self, bad_version):
        with pytest.raises(InvalidDecisionError, match="method_version"):
            make_profile(method_version=bad_version)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_configuration_rejected(self, bad):
        with pytest.raises(InvalidDecisionError, match="method_configuration"):
            make_profile(method_configuration={"temperature": bad})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_fitted_parameters_rejected(self, bad):
        with pytest.raises(InvalidDecisionError, match="fitted_parameters"):
            make_profile(fitted_parameters={"slope": bad})

    def test_none_method_state_treated_as_empty(self):
        profile = make_profile(method_configuration=None, fitted_parameters=None)
        assert profile.canonical_payload()["method_configuration"] == {}
        assert profile.canonical_payload()["fitted_parameters"] == {}

    def test_non_mapping_method_state_rejected(self):
        with pytest.raises(InvalidDecisionError, match="method_configuration"):
            make_profile(method_configuration=[("temperature", 1.0)])  # type: ignore[arg-type]
        with pytest.raises(InvalidDecisionError, match="fitted_parameters"):
            make_profile(fitted_parameters="slope")  # type: ignore[arg-type]

    def test_non_str_method_state_keys_rejected(self):
        bad: Any = {1: 1.0}
        with pytest.raises(InvalidDecisionError, match="keys must be str"):
            make_profile(method_configuration=bad)
        with pytest.raises(InvalidDecisionError, match="keys must be str"):
            make_profile(fitted_parameters=bad)

    def test_tuples_sets_callables_and_objects_rejected(self):
        with pytest.raises(InvalidDecisionError, match="method_configuration"):
            make_profile(method_configuration={"temperature": (1.0, 2.0)})
        with pytest.raises(InvalidDecisionError, match="method_configuration"):
            make_profile(method_configuration={"bad": {1, 2}})
        with pytest.raises(InvalidDecisionError, match="method_configuration"):
            make_profile(method_configuration={"bad": len})
        with pytest.raises(InvalidDecisionError, match="method_configuration"):
            make_profile(method_configuration={"bad": object()})


# ---------------------------------------------------------------------------
# Deep-freeze durability (tests 10, 11, 12)
# ---------------------------------------------------------------------------


class TestMethodStateDeepFreeze:
    def test_mutating_caller_nested_configuration_does_not_change_profile(self):
        configuration: dict[str, JSONValue] = {"temperature": {"initial": 1.0}}
        profile = make_profile(method_configuration=configuration)
        before_payload = profile.canonical_payload()
        before_fingerprint = profile.fingerprint
        configuration["temperature"]["initial"] = 99.0  # type: ignore[index]
        assert profile.canonical_payload() == before_payload
        assert profile.fingerprint == before_fingerprint

    def test_mutating_caller_nested_fitted_list_does_not_change_profile(self):
        parameters: dict[str, JSONValue] = {"coefficients": [1.0, 2.0]}
        profile = make_profile(fitted_parameters=parameters)
        before_payload = profile.canonical_payload()
        before_fingerprint = profile.fingerprint
        parameters["coefficients"].append(3.0)  # type: ignore[attr-defined]
        assert profile.canonical_payload() == before_payload
        assert profile.fingerprint == before_fingerprint

    def test_mutating_returned_payload_does_not_change_profile(self):
        profile = make_profile(
            method_configuration={"temperature": 1.0},
            fitted_parameters={"slope": 0.5},
        )
        before_fingerprint = profile.fingerprint
        payload = profile.canonical_payload()
        payload["method_id"] = "forged"
        binding = payload["binding"]
        assert isinstance(binding, dict)
        binding["binding_fingerprint"] = "forged"
        configuration = payload["method_configuration"]
        assert isinstance(configuration, dict)
        configuration["temperature"] = 9.0
        fitted = payload["fitted_parameters"]
        assert isinstance(fitted, dict)
        fitted["slope"] = 9.0
        assert profile.fingerprint == before_fingerprint
        fresh = profile.canonical_payload()
        assert fresh["method_id"] == "test-method"
        fresh_binding = fresh["binding"]
        assert isinstance(fresh_binding, dict)
        assert fresh_binding["binding_fingerprint"] != "forged"
        assert fresh["method_configuration"] == {"temperature": 1.0}
        assert fresh["fitted_parameters"] == {"slope": 0.5}

    def test_stored_method_state_is_deeply_immutable(self):
        profile = make_profile(
            method_configuration={"temperature": {"initial": 1.0}},
            fitted_parameters={"coefficients": [1.0, 2.0]},
        )
        with pytest.raises(TypeError):
            profile.method_configuration["temperature"]["initial"] = 9.0  # type: ignore[index]
        with pytest.raises(AttributeError):
            profile.fitted_parameters["coefficients"].append(3.0)  # type: ignore[attr-defined]
        assert profile.canonical_payload()["method_configuration"] == {
            "temperature": {"initial": 1.0}
        }
        assert profile.canonical_payload()["fitted_parameters"] == {"coefficients": [1.0, 2.0]}


# ---------------------------------------------------------------------------
# Identity durability (tests 13, 14, 15, 16, 17)
# ---------------------------------------------------------------------------


class TestProfileIdentityDurability:
    def test_different_dataset_identical_method_state_different_fingerprint(self):
        first = make_profile(fitting_dataset(truths=("shipping",)))
        second = make_profile(fitting_dataset(truths=("billing",)))
        dataset_a = fitting_dataset(truths=("shipping",))
        dataset_b = fitting_dataset(truths=("billing",))
        assert dataset_a.fingerprint != dataset_b.fingerprint
        assert first.fingerprint != second.fingerprint

    def test_row_permutation_same_profile_fingerprint(self):
        first_observation = choice_observation(resolved_truth("shipping"))
        second_observation = choice_observation(resolved_truth("billing"))
        forward = CalibrationDataset.create([first_observation, second_observation])
        shuffled = CalibrationDataset.create([second_observation, first_observation])
        assert forward.fingerprint == shuffled.fingerprint
        assert make_profile(forward).fingerprint == make_profile(shuffled).fingerprint

    def test_duplicated_observation_different_profile_fingerprint(self):
        first_observation = choice_observation(resolved_truth("shipping"))
        second_observation = choice_observation(resolved_truth("billing"))
        base = CalibrationDataset.create([first_observation, second_observation])
        duplicated = CalibrationDataset.create(
            [first_observation, first_observation, second_observation]
        )
        assert base.fingerprint != duplicated.fingerprint
        assert make_profile(base).fingerprint != make_profile(duplicated).fingerprint

    def test_different_method_configuration_different_fingerprint(self):
        first = make_profile(method_configuration={"temperature": 1.0})
        second = make_profile(method_configuration={"temperature": 2.0})
        assert first.fingerprint != second.fingerprint

    def test_different_fitted_parameters_different_fingerprint(self):
        first = make_profile(fitted_parameters={"slope": 0.5})
        second = make_profile(fitted_parameters={"slope": 0.6})
        assert first.fingerprint != second.fingerprint

    def test_different_method_identity_different_fingerprint(self):
        first = make_profile(method_id="method-a")
        second = make_profile(method_id="method-b")
        assert first.fingerprint != second.fingerprint
        third = make_profile(method_version=2)
        assert first.fingerprint != third.fingerprint

    def test_configuration_and_fitted_parameters_are_distinct_components(self):
        # Moving the same value between the two components changes the
        # identity: configuration is never encoded inside parameter names.
        first = make_profile(method_configuration={"temperature": 1.0}, fitted_parameters={})
        second = make_profile(method_configuration={}, fitted_parameters={"temperature": 1.0})
        assert first.fingerprint != second.fingerprint

    def test_profile_fingerprint_is_deterministic(self):
        assert make_profile().fingerprint == make_profile().fingerprint


# ---------------------------------------------------------------------------
# require_binding_match (tests 18, 19, 20, 21)
# ---------------------------------------------------------------------------


class TestRequireBindingMatch:
    def test_exact_binding_match_returns_none(self):
        dataset = fitting_dataset()
        profile = make_profile(dataset)
        assert profile.require_binding_match(dataset.binding) is None

    def test_different_formulation_binding_rejected(self):
        from probvenance import Choice, ChoiceDecision

        profile = make_profile()
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
        permuted_evaluation = runtime.evaluate_with_trace(permuted)
        foreign_binding = permuted_evaluation.trace and CalibrationBinding.from_trace(
            permuted_evaluation.trace
        )
        assert foreign_binding is not None
        assert foreign_binding != profile.binding
        with pytest.raises(InvalidDecisionError, match="binding fingerprint"):
            profile.require_binding_match(foreign_binding)

    def test_different_rendering_semantics_binding_rejected(self):
        profile = make_profile()
        other = choice_observation(
            backend=FakeCategoricalBackend(rendering_config={"enable_thinking": False})
        )
        assert other.binding != profile.binding
        with pytest.raises(InvalidDecisionError, match="binding fingerprint"):
            profile.require_binding_match(other.binding)

    def test_same_formulation_family_still_rejected(self):
        # Family similarity authorizes nothing: only the exact binding
        # canonical identity matches.
        from probvenance import Choice, ChoiceDecision

        profile = make_profile()
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
            base_evaluation.trace.formulation_family_fingerprint
            == permuted_evaluation.trace.formulation_family_fingerprint
        )
        foreign_binding = CalibrationBinding.from_trace(permuted_evaluation.trace)
        assert foreign_binding != profile.binding
        with pytest.raises(InvalidDecisionError, match="binding fingerprint"):
            profile.require_binding_match(foreign_binding)

    def test_non_binding_argument_rejected(self):
        profile = make_profile()
        with pytest.raises(InvalidDecisionError, match="CalibrationBinding"):
            profile.require_binding_match("not a binding")  # type: ignore[arg-type]

    def test_no_fallback_matching_api_on_binding(self):
        make_profile()
        for forbidden in ("matches_profile", "compatible_with", "may_share_profile"):
            assert not hasattr(CalibrationBinding, forbidden)


# ---------------------------------------------------------------------------
# Taxonomy precondition (tests 22, 23, 24, 25)
# ---------------------------------------------------------------------------


class TestTaxonomyPrecondition:
    def test_different_taxonomy_ids_fail_closed(self):
        dataset = taxonomy_mismatch_dataset(
            binding_taxonomy_id="support",
            binding_taxonomy_version=3,
            truth_taxonomy_id="legacy-support",
            truth_taxonomy_version=1,
        )
        with pytest.raises(InvalidDecisionError) as excinfo:
            make_profile(dataset)
        message = str(excinfo.value)
        assert "support" in message
        assert "legacy-support" in message
        assert "taxonomy mapping identity" in message

    def test_same_taxonomy_id_version_mismatch_fails_closed(self):
        dataset = taxonomy_mismatch_dataset(
            binding_taxonomy_id="departments",
            binding_taxonomy_version=3,
            truth_taxonomy_id="departments",
            truth_taxonomy_version=2,
        )
        with pytest.raises(InvalidDecisionError) as excinfo:
            make_profile(dataset)
        message = str(excinfo.value)
        assert "departments" in message
        assert "3" in message
        assert "2" in message
        # A version mismatch is distinguishable from an ID mismatch.
        assert "taxonomy_version" in message

    def test_unknown_taxonomy_on_one_side_is_allowed(self):
        # The binding declares no taxonomy while the ground truth does:
        # absence is not a contradiction and no equality is invented.
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        observation = CalibrationObservation.from_evaluation(
            evaluation,
            resolved_truth("shipping", taxonomy_id="departments", taxonomy_version=2),
        )
        dataset = CalibrationDataset.create([observation])
        profile = make_profile(dataset)
        assert profile.binding.taxonomy_id is None
        assert profile.ground_truth_semantics.taxonomy_id == "departments"
        assert profile.fingerprint

    def test_matching_taxonomy_and_version_allowed(self):
        dataset = taxonomy_mismatch_dataset(
            binding_taxonomy_id="departments",
            binding_taxonomy_version=2,
            truth_taxonomy_id="departments",
            truth_taxonomy_version=2,
        )
        profile = make_profile(dataset)
        assert profile.fingerprint

    def test_no_taxonomy_mapping_object_exists(self):
        import probvenance.calibration as calibration_module

        assert not hasattr(calibration_module, "taxonomy_mapping")
        assert not hasattr(calibration_module, "TaxonomyMappingIdentity")


# ---------------------------------------------------------------------------
# Import direction (tests 26, 27, 28)
# ---------------------------------------------------------------------------


class TestImportDirection:
    def test_import_calibration_first_works(self):
        subprocess.run(
            [sys.executable, "-c", "import probvenance.calibration"],
            check=True,
            capture_output=True,
        )

    def test_import_calibration_evaluation_first_works(self):
        subprocess.run(
            [sys.executable, "-c", "import probvenance.calibration_evaluation"],
            check=True,
            capture_output=True,
        )

    def test_calibration_alone_does_not_load_calibration_evaluation(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import probvenance.calibration; "
                "assert 'probvenance.calibration_evaluation' not in sys.modules",
            ],
            check=True,
            capture_output=True,
        )
        assert result.returncode == 0

    def test_calibration_module_source_never_imports_evaluation(self):
        import ast

        import probvenance.calibration as calibration_module

        tree = ast.parse(inspect.getsource(calibration_module))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        assert not any("calibration_evaluation" in name for name in imported)


# ---------------------------------------------------------------------------
# Phase 4B fingerprint regression guard (test 29)
# ---------------------------------------------------------------------------


class TestPhase4BFingerprintsUnchanged:
    def test_five_evaluation_artifact_fingerprints_are_unchanged(self):
        dataset = evaluation_dataset(
            [
                _reliability_observation(0.10, False),
                _reliability_observation(0.40, True),
                _reliability_observation(0.60, True),
                _reliability_observation(0.90, False),
            ]
        )
        reliability = evaluate_uncalibrated_winner_reliability(dataset, bin_count=2)
        assert evaluate_uncalibrated_winner_brier(dataset).fingerprint == (
            "9b1841c3cd3749d8ef8f65bec2b70312803371b9eebcbd59ada5e149823ffd3c"
        )
        assert evaluate_uncalibrated_winner_log_loss(dataset).fingerprint == (
            "db9b7e2435e9a8d74c3fc6c064d2e6bc2f7dedff4c6c42201433f4f6ae1c9db5"
        )
        assert evaluate_uncalibrated_winner_diagnostics(dataset).fingerprint == (
            "4405acc66efbaf97e48fb2783e301a98c67fa143d7351e4e40deb6cbfb055279"
        )
        assert reliability.fingerprint == (
            "0c555f03397268e03e7093556ba228b4288c597be564ea51ee853a044eb006d0"
        )
        assert evaluate_winner_binned_absolute_gap(reliability).fingerprint == (
            "7bb2e5ad24b6f05e4885aa756d93f032cc20994dcb010236b85d24d7303f9368"
        )


# ---------------------------------------------------------------------------
# Runtime stays uncalibrated (test 30)
# ---------------------------------------------------------------------------


class TestRuntimeStaysUncalibrated:
    def test_bool_result_remains_uncalibrated(self):
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_bool_decision())
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None

    def test_choice_result_remains_uncalibrated(self):
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None


# ---------------------------------------------------------------------------
# Canonical payload shape
# ---------------------------------------------------------------------------


class TestCanonicalPayloadShape:
    def test_exact_key_set(self):
        profile = make_profile(
            method_configuration={"temperature": 1.0},
            fitted_parameters={"slope": 0.5},
        )
        payload = profile.canonical_payload()
        assert set(payload) == {
            "v",
            "binding",
            "ground_truth_semantics",
            "target_id",
            "target_version",
            "input_score_id",
            "input_score_version",
            "method_id",
            "method_version",
            "method_configuration",
            "fitted_parameters",
            "training_dataset_fingerprint",
            "training_dataset_fingerprint_version",
        }
        binding = payload["binding"]
        assert isinstance(binding, dict)
        assert set(binding) == {
            "binding_fingerprint",
            "binding_fingerprint_version",
        }
        semantics = payload["ground_truth_semantics"]
        assert isinstance(semantics, dict)
        assert set(semantics) == {
            "ground_truth_semantics_fingerprint",
            "ground_truth_semantics_fingerprint_version",
        }

    def test_payload_commits_versions_and_no_constituent_fields(self):
        profile = make_profile()
        payload = profile.canonical_payload()
        assert payload["v"] == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        assert CALIBRATION_PROFILE_FINGERPRINT_VERSION == 1
        binding = payload["binding"]
        assert isinstance(binding, dict)
        assert binding["binding_fingerprint"] == profile.binding.fingerprint
        assert binding["binding_fingerprint_version"] == CALIBRATION_BINDING_FINGERPRINT_VERSION
        semantics = payload["ground_truth_semantics"]
        assert isinstance(semantics, dict)
        assert semantics["ground_truth_semantics_fingerprint"] == (
            profile.ground_truth_semantics.fingerprint
        )
        assert semantics["ground_truth_semantics_fingerprint_version"] == (
            GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
        )
        # Constituent fields of the binding and the semantics identity are
        # deliberately not duplicated in the profile payload.
        for forbidden in (
            "model",
            "tokenizer",
            "rendering_semantics",
            "labeling_rule",
            "ambiguity_policy",
            "taxonomy_id",
            "taxonomy_version",
        ):
            assert forbidden not in payload

    def test_fingerprint_hashes_the_canonical_payload(self):
        from probvenance.fingerprint import fingerprint

        profile = make_profile()
        assert profile.fingerprint == fingerprint(profile.canonical_payload())
        assert len(profile.fingerprint) == 64

    def test_construction_token_is_not_an_instance_attribute(self):
        profile = make_profile()
        assert not hasattr(profile, "_construction_token")
        assert "_construction_token" not in {f.name for f in fields(CalibrationProfile)}
