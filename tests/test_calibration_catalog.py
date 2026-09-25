"""Contract tests for the explicit calibration profile catalog and discovery.

The catalog is an immutable discovery index, not a selector. These tests pin
that discovery proposes exact references while selection and application keep
their existing authority: no ranking, no store access, no GT-semantics or
method filtering, deterministic order, and explicit fail-closed boundaries.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    choice_observation,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
)
from test_profile_application import forge_profile

from probvenance import InvalidDecisionError
from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationDataset,
    CalibrationObservation,
    CalibrationProfile,
    fit_l2_logistic_selected_probability,
)
from probvenance.calibration_catalog import (
    CalibrationProfileCatalog,
    CalibrationProfileReference,
    discover_calibration_profile_references_for_runtime,
)
from probvenance.calibration_selection import select_calibration_profile_for_runtime
from probvenance.calibration_store import (
    CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION,
    DirectoryCalibrationProfileStore,
)
from probvenance.errors import (
    AmbiguousCalibrationProfileSelectionError,
    CalibrationProfileNotFoundError,
    CalibrationProfileStoreIntegrityError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def runtime_evaluation(decision: Any = None) -> Any:
    runtime, _ = make_choice_runtime()
    return runtime.evaluate_with_trace(decision if decision is not None else make_choice_decision())


def fit(observations: list[Any], l2_strength: float = 1.0) -> CalibrationProfile:
    return fit_l2_logistic_selected_probability(
        CalibrationDataset.create(observations), l2_strength=l2_strength
    )


def eligible_profile() -> CalibrationProfile:
    return fit([choice_observation()])


def other_binding_profile() -> CalibrationProfile:
    return fit([choice_observation(backend=FakeCategoricalBackend(model="other-model"))])


def other_semantics_profile() -> CalibrationProfile:
    return fit([choice_observation(resolved_truth("shipping", labeling_rule="another rule"))])


def other_training_profile() -> CalibrationProfile:
    return fit([choice_observation(), choice_observation(resolved_truth("billing"))])


def other_strength_profile() -> CalibrationProfile:
    return fit([choice_observation()], l2_strength=0.5)


def declared_profile() -> CalibrationProfile:
    observation = CalibrationObservation.from_evaluation(
        runtime_evaluation(),
        resolved_truth("shipping"),
        task_id="support-routing",
        taxonomy_id="support",
        taxonomy_version=3,
    )
    return fit([observation])


def catalog_of(*profiles: CalibrationProfile) -> CalibrationProfileCatalog:
    return CalibrationProfileCatalog.from_profiles(tuple(profiles))


def discover(evaluation: Any, catalog: CalibrationProfileCatalog, **declarations: Any) -> Any:
    return discover_calibration_profile_references_for_runtime(
        evaluation,
        catalog,
        task_id=declarations.get("task_id"),
        domain_id=declarations.get("domain_id"),
        taxonomy_id=declarations.get("taxonomy_id"),
        taxonomy_version=declarations.get("taxonomy_version"),
    )


def module_code_without_docstrings(module: Any) -> str:
    """Return a module's source with every docstring removed.

    Docstrings legitimately name forbidden behaviour, so scope checks must
    inspect executable code only.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, holders) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# Reference type (PART 4, PART 5, PART 6, PART 37)
# ---------------------------------------------------------------------------


class TestReference:
    def test_reference_matches_the_source_profile_identity(self) -> None:
        profile = eligible_profile()
        catalog = catalog_of(profile)
        (reference,) = catalog.references
        assert reference.profile_fingerprint == profile.fingerprint
        assert reference.profile_fingerprint_version == CALIBRATION_PROFILE_FINGERPRINT_VERSION

    def test_reference_carries_only_the_exact_identity(self) -> None:
        assert set(CalibrationProfileReference.__slots__) == {
            "profile_fingerprint",
            "profile_fingerprint_version",
        }

    def test_reference_has_no_second_identity(self) -> None:
        reference = CalibrationProfileReference("a" * 64, 1)
        for absent in ("reference_fingerprint", "reference_version", "serialization_version"):
            assert not hasattr(reference, absent)

    @pytest.mark.parametrize(
        "fingerprint",
        ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 63 + "-", 12345, None, b"a" * 64],
    )
    def test_malformed_fingerprint_is_rejected(self, fingerprint: Any) -> None:
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileReference(fingerprint, 1)

    @pytest.mark.parametrize("version", [True, False, 0, -1, 1.0, "1", None])
    def test_malformed_version_is_rejected(self, version: Any) -> None:
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileReference("a" * 64, version)

    def test_reference_is_immutable(self) -> None:
        reference = CalibrationProfileReference("a" * 64, 1)
        with pytest.raises(FrozenInstanceError):
            reference.profile_fingerprint = "b" * 64  # type: ignore[misc]

    def test_references_are_equal_by_value(self) -> None:
        assert CalibrationProfileReference("a" * 64, 1) == CalibrationProfileReference("a" * 64, 1)
        assert CalibrationProfileReference("a" * 64, 1) != CalibrationProfileReference("b" * 64, 1)


# ---------------------------------------------------------------------------
# Construction (PART 7, PART 12, PART 13, PART 14, PART 44, PART 45, PART 47, PART 58)
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_catalog_is_built_from_an_explicit_tuple(self) -> None:
        profile = eligible_profile()
        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        assert catalog.references == (CalibrationProfileReference(profile.fingerprint, 1),)

    def test_empty_tuple_builds_a_valid_catalog(self) -> None:
        catalog = CalibrationProfileCatalog.from_profiles(())
        assert catalog.references == ()

    @pytest.mark.parametrize("container", [[], (), iter(()), frozenset()])
    def test_non_tuple_container_is_rejected(self, container: Any) -> None:
        if isinstance(container, tuple):
            return
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog.from_profiles(container)  # type: ignore[arg-type]

    def test_tuple_subclass_is_rejected(self) -> None:
        class Profiles(tuple):  # type: ignore[type-arg]
            pass

        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog.from_profiles(Profiles((eligible_profile(),)))

    @pytest.mark.parametrize("member", [None, object(), "profile", 7, ("a", "b")])
    def test_malformed_member_is_rejected_not_skipped(self, member: Any) -> None:
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog.from_profiles((eligible_profile(), member))  # type: ignore[arg-type]

    def test_duplicate_exact_identity_is_rejected(self) -> None:
        profile = eligible_profile()
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog.from_profiles((profile, profile))

    def test_duplicate_identity_from_independent_fits_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog.from_profiles((eligible_profile(), eligible_profile()))

    def test_hand_declared_entries_are_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog(())  # type: ignore[call-arg]

    def test_internal_order_is_deterministic_across_input_permutations(self) -> None:
        first, second, third = (
            eligible_profile(),
            other_binding_profile(),
            other_semantics_profile(),
        )
        expected = catalog_of(first, second, third).references
        for ordering in permutations((first, second, third)):
            assert catalog_of(*ordering).references == expected

    def test_catalog_does_not_retain_full_profiles(self) -> None:
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        entries = catalog._entries
        assert entries
        for entry in entries:
            assert not isinstance(entry, CalibrationProfile)
            assert set(entry.__slots__) == {
                "reference",
                "binding",
                "target_id",
                "target_version",
                "input_score_id",
                "input_score_version",
            }
            for slot in entry.__slots__:
                assert not isinstance(getattr(entry, slot), CalibrationProfile)

    def test_construction_does_not_mutate_source_profiles(self) -> None:
        profile = eligible_profile()
        before = (profile.fingerprint, profile.canonical_payload())
        catalog_of(profile, other_binding_profile())
        assert (profile.fingerprint, profile.canonical_payload()) == before

    def test_catalog_membership_is_not_part_of_profile_identity(self) -> None:
        profile = eligible_profile()
        before = profile.fingerprint
        catalog_of(profile)
        assert profile.fingerprint == before

    def test_catalog_is_immutable(self) -> None:
        catalog = catalog_of(eligible_profile())
        with pytest.raises(FrozenInstanceError):
            catalog._entries = ()  # type: ignore[misc]

    def test_references_property_returns_an_immutable_tuple(self) -> None:
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        references = catalog.references
        assert isinstance(references, tuple)
        assert len(references) == 2


# ---------------------------------------------------------------------------
# Discovery cardinality (PART 17, PART 18, PART 34, PART 35, PART 36)
# ---------------------------------------------------------------------------


class TestDiscoveryCardinality:
    def test_zero_matches_returns_empty_tuple(self) -> None:
        catalog = catalog_of(other_binding_profile())
        assert discover(runtime_evaluation(), catalog) == ()

    def test_one_match_returns_one_reference(self) -> None:
        profile = eligible_profile()
        (reference,) = discover(runtime_evaluation(), catalog_of(profile))
        assert reference.profile_fingerprint == profile.fingerprint

    def test_many_matches_return_all_references(self) -> None:
        first = eligible_profile()
        second = other_semantics_profile()
        third = other_training_profile()
        references = discover(runtime_evaluation(), catalog_of(first, second, third))
        assert len(references) == 3
        assert {reference.profile_fingerprint for reference in references} == {
            first.fingerprint,
            second.fingerprint,
            third.fingerprint,
        }

    def test_empty_catalog_discovers_nothing(self) -> None:
        assert discover(runtime_evaluation(), CalibrationProfileCatalog.from_profiles(())) == ()

    def test_multiple_matches_do_not_raise_ambiguity(self) -> None:
        catalog = catalog_of(eligible_profile(), other_semantics_profile())
        assert len(discover(runtime_evaluation(), catalog)) == 2

    def test_zero_matches_do_not_raise_no_eligible(self) -> None:
        catalog = catalog_of(other_binding_profile())
        assert discover(runtime_evaluation(), catalog) == ()


# ---------------------------------------------------------------------------
# Discovery filters and deliberate non-filters (PART 9, PART 10, PART 11, PART 21)
# ---------------------------------------------------------------------------


class TestDiscoveryEligibility:
    def test_ground_truth_semantics_is_not_a_filter(self) -> None:
        first = eligible_profile()
        second = other_semantics_profile()
        assert first.binding.canonical_payload() == second.binding.canonical_payload()
        assert len(discover(runtime_evaluation(), catalog_of(first, second))) == 2

    def test_method_is_not_a_filter(self) -> None:
        base = eligible_profile()
        unsupported = forge_profile(base, method_id="some-unsupported-method")
        references = discover(runtime_evaluation(), catalog_of(base, unsupported))
        assert len(references) == 2

    def test_method_configuration_is_not_a_filter(self) -> None:
        base = eligible_profile()
        reconfigured = forge_profile(base, method_configuration={"l2_strength": 0.25})
        assert len(discover(runtime_evaluation(), catalog_of(base, reconfigured))) == 2

    def test_training_dataset_is_not_a_filter(self) -> None:
        first = eligible_profile()
        second = other_training_profile()
        assert first.training_dataset_fingerprint != second.training_dataset_fingerprint
        assert len(discover(runtime_evaluation(), catalog_of(first, second))) == 2

    def test_fitted_parameters_are_not_a_filter(self) -> None:
        first = eligible_profile()
        second = other_strength_profile()
        assert first.fitted_parameters != second.fitted_parameters
        assert len(discover(runtime_evaluation(), catalog_of(first, second))) == 2

    def test_different_binding_is_not_discovered(self) -> None:
        eligible = eligible_profile()
        catalog = catalog_of(eligible, other_binding_profile())
        (reference,) = discover(runtime_evaluation(), catalog)
        assert reference.profile_fingerprint == eligible.fingerprint

    @pytest.mark.parametrize(
        "overrides",
        [
            {"target_id": "some-other-target"},
            {"target_version": 2},
            {"input_score_id": "some-other-score"},
            {"input_score_version": 2},
        ],
    )
    def test_wrong_target_or_input_is_not_discovered(self, overrides: dict[str, Any]) -> None:
        forged = forge_profile(eligible_profile(), **overrides)
        assert discover(runtime_evaluation(), catalog_of(forged)) == ()


# ---------------------------------------------------------------------------
# Declarations (PART 26, PART 27)
# ---------------------------------------------------------------------------


class TestDeclarations:
    def test_omitted_declarations_do_not_discover_a_declared_profile(self) -> None:
        assert discover(runtime_evaluation(), catalog_of(declared_profile())) == ()

    def test_matching_declarations_discover_it(self) -> None:
        profile = declared_profile()
        references = discover(
            runtime_evaluation(),
            catalog_of(profile),
            task_id="support-routing",
            taxonomy_id="support",
            taxonomy_version=3,
        )
        assert len(references) == 1

    def test_contradicting_declarations_do_not_discover_it(self) -> None:
        profile = declared_profile()
        assert (
            discover(
                runtime_evaluation(),
                catalog_of(profile),
                task_id="other-routing",
                taxonomy_id="support",
                taxonomy_version=3,
            )
            == ()
        )

    def test_none_is_not_a_wildcard(self) -> None:
        profile = declared_profile()
        assert discover(runtime_evaluation(), catalog_of(profile), task_id=None) == ()

    def test_declarations_are_not_copied_from_the_catalog_entry(self) -> None:
        profile = declared_profile()
        catalog = catalog_of(profile)
        assert discover(runtime_evaluation(), catalog) == ()


# ---------------------------------------------------------------------------
# Runtime input contract (PART 23, PART 24, PART 25, PART 67)
# ---------------------------------------------------------------------------


class TestRuntimeContract:
    def test_already_calibrated_evaluation_is_rejected(self) -> None:
        evaluation = runtime_evaluation()
        from probvenance.calibration import apply_profile_to_runtime_evaluation

        profile = eligible_profile()
        calibrated = apply_profile_to_runtime_evaluation(evaluation, profile)
        with pytest.raises(InvalidDecisionError):
            discover(calibrated, catalog_of(profile))

    def test_mismatched_result_and_trace_are_rejected(self) -> None:
        evaluation = runtime_evaluation()
        other = runtime_evaluation()
        incoherent = type(evaluation)(
            result=replace(evaluation.result, trace_id=other.trace.trace_id),
            trace=evaluation.trace,
        )
        with pytest.raises(InvalidDecisionError):
            discover(incoherent, catalog_of(eligible_profile()))

    def test_non_evaluation_input_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            discover(object(), catalog_of(eligible_profile()))

    def test_non_catalog_input_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            discover(runtime_evaluation(), object())  # type: ignore[arg-type]

    def test_discovery_does_not_mutate_the_evaluation(self) -> None:
        evaluation = runtime_evaluation()
        before = (evaluation.result.calibrated, evaluation.result.predicted_correctness)
        discover(evaluation, catalog_of(eligible_profile()))
        assert (evaluation.result.calibrated, evaluation.result.predicted_correctness) == before


# ---------------------------------------------------------------------------
# Layer separation (PART 19, PART 41, PART 42, PART 43, PART 48, PART 49, PART 53)
# ---------------------------------------------------------------------------


class TestLayerSeparation:
    def test_discovery_returns_references_not_profiles(self) -> None:
        references = discover(runtime_evaluation(), catalog_of(eligible_profile()))
        assert references
        for reference in references:
            assert isinstance(reference, CalibrationProfileReference)
            assert not isinstance(reference, CalibrationProfile)

    def test_catalog_module_imports_no_store_or_filesystem_or_metrics(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in (
            "calibration_store",
            "DirectoryCalibrationProfileStore",
            "calibration_evaluation",
            "os.",
            "environ",
            "getenv",
            "Path",
            "open(",
            "listdir",
            "glob",
            "walk",
            "brier",
            "log_loss",
            "reliability",
        ):
            assert forbidden not in code, forbidden

    def test_catalog_module_does_not_read_the_selected_probability(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in ("probability_true", "probabilities", "predicted_correctness", "value"):
            assert forbidden not in code, forbidden

    def test_catalog_exposes_no_search_or_mutation_api(self) -> None:
        for absent in (
            "find_by_model",
            "find_by_method",
            "find_by_taxonomy",
            "query",
            "search",
            "latest",
            "best",
            "register",
            "add",
            "remove",
            "update",
            "select",
            "apply",
            "load",
        ):
            assert not hasattr(CalibrationProfileCatalog, absent), absent

    def test_store_public_api_is_frozen(self) -> None:
        for absent in ("list", "scan", "references", "catalog", "find", "search"):
            assert not hasattr(DirectoryCalibrationProfileStore, absent), absent

    def test_selector_still_consumes_real_profiles(self) -> None:
        parameters = list(inspect.signature(select_calibration_profile_for_runtime).parameters)
        assert parameters[:2] == ["evaluation", "candidates"]


# ---------------------------------------------------------------------------
# End-to-end choreography (PART 30, PART 31, PART 32, PART 33, PART 38 to PART 56)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_ambiguous_case_proves_discovery_did_not_become_selection(self, tmp_path: Path) -> None:
        evaluation = runtime_evaluation()
        first = eligible_profile()
        second = other_semantics_profile()
        third = other_binding_profile()
        catalog = catalog_of(first, second, third)
        store = DirectoryCalibrationProfileStore(tmp_path / "ambiguous")
        for profile in (first, second, third):
            store.put(profile)
        references = discover(evaluation, catalog)
        assert len(references) == 2
        profiles = tuple(
            store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )
            for reference in references
        )
        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(evaluation, profiles)

    def test_unique_case_composes_all_four_layers(self, tmp_path: Path) -> None:
        evaluation = runtime_evaluation()
        eligible = eligible_profile()
        catalog = catalog_of(eligible, other_binding_profile())
        store = DirectoryCalibrationProfileStore(tmp_path / "unique")
        store.put(eligible)
        (reference,) = discover(evaluation, catalog)
        loaded = store.get(
            profile_fingerprint=reference.profile_fingerprint,
            profile_fingerprint_version=reference.profile_fingerprint_version,
        )
        selected = select_calibration_profile_for_runtime(evaluation, (loaded,))
        assert selected is loaded
        from probvenance.calibration import apply_profile_to_runtime_evaluation

        calibrated = apply_profile_to_runtime_evaluation(evaluation, selected)
        assert calibrated.result.calibrated is True
        assert calibrated.result.predicted_correctness is not None

    def test_missing_store_artifact_is_a_store_not_found(self, tmp_path: Path) -> None:
        evaluation = runtime_evaluation()
        profile = eligible_profile()
        catalog = catalog_of(profile)
        store = DirectoryCalibrationProfileStore(tmp_path / "missing")
        (reference,) = discover(evaluation, catalog)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )

    def test_corrupt_store_artifact_is_a_store_integrity_error(self, tmp_path: Path) -> None:
        evaluation = runtime_evaluation()
        profile = eligible_profile()
        catalog = catalog_of(profile)
        store = DirectoryCalibrationProfileStore(tmp_path / "corrupt")
        store.put(profile)
        (reference,) = discover(evaluation, catalog)
        target = (
            store.root
            / f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}"
            / f"profile-fingerprint-v{CALIBRATION_PROFILE_FINGERPRINT_VERSION}"
            / f"{reference.profile_fingerprint}.json"
        )
        target.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )


# ---------------------------------------------------------------------------
# Scope boundaries (PART 15, PART 16, PART 41, PART 65, PART 66)
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    def test_no_filesystem_catalog_store_exists(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        for absent in (
            "save_calibration_profile_catalog",
            "load_calibration_profile_catalog_from_path",
            "DirectoryCalibrationProfileCatalogStore",
            "CalibrationProfileCatalogStore",
        ):
            assert not hasattr(catalog_module, absent), absent

    def test_serialization_names_exist_but_no_store_or_registry_does(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        for present in (
            "serialize_calibration_profile_catalog",
            "load_calibration_profile_catalog",
            "CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION",
            "CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION",
        ):
            assert hasattr(catalog_module, present), present

    def test_no_registry_or_environment_discovery_in_source(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in ("registry", "environ", "getenv", "expanduser", "XDG"):
            assert forbidden not in code, forbidden

    def test_no_catalog_construction_token_at_module_scope(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        assert not hasattr(catalog_module, "_CATALOG_CONSTRUCTION_TOKEN")
        with pytest.raises(InvalidDecisionError):
            CalibrationProfileCatalog(())  # type: ignore[call-arg]
        assert isinstance(
            CalibrationProfileCatalog._CONSTRUCTION_TOKEN,  # type: ignore[attr-defined]
            object,
        )

    def test_no_new_dependency(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in ("import numpy", "import torch", "import requests", "import httpx"):
            assert forbidden not in code, forbidden

    def test_catalog_names_are_not_exported_at_package_root(self) -> None:
        import probvenance

        for absent in (
            "CalibrationProfileCatalog",
            "CalibrationProfileReference",
            "discover_calibration_profile_references_for_runtime",
        ):
            assert not hasattr(probvenance, absent), absent
