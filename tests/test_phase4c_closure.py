"""Phase 4C closure audit: cross-layer contracts, version matrix, error matrices.

Subsystem behavior is pinned in each subsystem's own test module. This module
holds only the invariants that SPAN layers, because a boundary defect is exactly
what a per-module suite can miss:

* the exact Phase 4C version matrix, so a closure round cannot silently bump a
  schema;
* one strict-JSON attack matrix applied to BOTH artifact loaders;
* one store error matrix applied to BOTH exact content-addressed stores;
* the end-to-end authorization chain, including the stale/forged catalog path
  that proves discovery is never an authorization boundary;
* scope guards that keep registry, lifecycle, and automation vocabulary out.

Nothing here is duplicated from the subsystem suites; each test either crosses a
module boundary or pins a contract that no single subsystem owns.
"""

from __future__ import annotations

import ast
import errno
import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_calibration import choice_observation
from test_calibration_catalog_serialization import (
    catalog_of,
    deep_value,
    eligible_profile,
    other_binding_profile,
    other_training_profile,
    runtime_evaluation,
    unsupported_method_profile,
)
from test_profile_serialization import fitted_profile

from probvenance import InvalidDecisionError
from probvenance import _directory_artifact_store as shared_filesystem
from probvenance._strict_json import _MAX_ARTIFACT_NESTING_DEPTH, require_bounded_nesting
from probvenance.calibration import (
    _L2_LOGISTIC_SOLVER_VERSION,
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    CALIBRATION_DATASET_FINGERPRINT_VERSION,
    CALIBRATION_OBSERVATION_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_SERIALIZATION_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION,
    PREDICTED_WINNER_CORRECTNESS_VERSION,
    RENDERING_SEMANTICS_VERSION,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    WINNER_CORRECTNESS_TARGET_VERSION,
    CalibrationDataset,
    CalibrationProfile,
    _apply_profile_to_selected_probability,
    apply_profile_to_runtime_evaluation,
    fit_l2_logistic_selected_probability,
    load_calibration_profile,
    predicted_winner_correctness,
    serialize_calibration_profile,
)
from probvenance.calibration_catalog import (
    CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION,
    CalibrationProfileCatalog,
    discover_calibration_profile_references_for_runtime,
    load_calibration_profile_catalog,
    serialize_calibration_profile_catalog,
)
from probvenance.calibration_catalog_store import (
    CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION,
    DirectoryCalibrationProfileCatalogStore,
)
from probvenance.calibration_evaluation import (
    BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION,
    CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
    CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
    EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION,
    EMPIRICAL_CORRECTNESS_RATE_VERSION,
    EQUAL_WIDTH_BINNING_VERSION,
    LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION,
    MEAN_PREDICTED_CORRECTNESS_VERSION,
    MEAN_SELECTED_PROBABILITY_VERSION,
    POST_CALIBRATION_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_BRIER_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_DIAGNOSTICS_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_LOG_LOSS_RESULT_FINGERPRINT_VERSION,
    POST_CALIBRATION_RELIABILITY_RESULT_FINGERPRINT_VERSION,
    PROFILE_APPLIED_EVALUATION_DATASET_FINGERPRINT_VERSION,
    WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION,
    WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION,
    WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION,
)
from probvenance.calibration_selection import select_calibration_profile_for_runtime
from probvenance.calibration_store import (
    CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION,
    DirectoryCalibrationProfileStore,
)
from probvenance.errors import (
    AmbiguousCalibrationProfileSelectionError,
    CalibrationProfileCatalogNotFoundError,
    CalibrationProfileCatalogStoreError,
    CalibrationProfileCatalogStoreIntegrityError,
    CalibrationProfileNotFoundError,
    CalibrationProfileStoreError,
    CalibrationProfileStoreIntegrityError,
    NoEligibleCalibrationProfileError,
)
from probvenance.plans import PLAN_FINGERPRINT_VERSION
from probvenance.trace import EXECUTION_FINGERPRINT_VERSION

#: A bound no honest store error message needs to exceed. Corruption messages
#: carry short context (the path and a bounded echo of the offending value), so
#: a megabyte of attacker input must never inflate them.
_MAX_MESSAGE_CHARS = 2000


def other_profile() -> CalibrationProfile:
    """Return a fitted profile with a fingerprint distinct from ``fitted_profile``."""
    return fit_l2_logistic_selected_probability(
        CalibrationDataset.create([choice_observation()]), l2_strength=0.5
    )


def write_text_at(target: Path, text: str) -> None:
    """Write ``text`` at ``target``, creating the managed parents."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def profile_domain(root: Path) -> SimpleNamespace:
    """Uniform view over the exact profile store for the closure matrices."""
    profile = fitted_profile()
    store = DirectoryCalibrationProfileStore(root)
    target = (
        store.root
        / f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}"
        / f"profile-fingerprint-v{CALIBRATION_PROFILE_FINGERPRINT_VERSION}"
        / f"{profile.fingerprint}.json"
    )
    return SimpleNamespace(
        label="profile",
        store=store,
        artifact=profile,
        target=target,
        valid_text=serialize_calibration_profile(profile),
        other_text=serialize_calibration_profile(other_profile()),
        get=lambda: store.get(
            profile_fingerprint=profile.fingerprint,
            profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
        ),
        put=lambda: store.put(profile),
        not_found=CalibrationProfileNotFoundError,
        integrity=CalibrationProfileStoreIntegrityError,
        operational=CalibrationProfileStoreError,
        temporary_prefix=".tmp-profile-",
    )


def catalog_domain(root: Path) -> SimpleNamespace:
    """Uniform view over the exact catalog store for the closure matrices."""
    catalog = catalog_of(eligible_profile())
    store = DirectoryCalibrationProfileCatalogStore(root)
    target = (
        store.root
        / f"catalog-store-v{CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION}"
        / f"catalog-fingerprint-v{CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION}"
        / f"{catalog.fingerprint}.json"
    )
    other_catalog = catalog_of(other_binding_profile())
    return SimpleNamespace(
        label="catalog",
        store=store,
        artifact=catalog,
        target=target,
        valid_text=serialize_calibration_profile_catalog(catalog),
        other_text=serialize_calibration_profile_catalog(other_catalog),
        get=lambda: store.get(
            catalog_fingerprint=catalog.fingerprint,
            catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
        ),
        put=lambda: store.put(catalog),
        not_found=CalibrationProfileCatalogNotFoundError,
        integrity=CalibrationProfileCatalogStoreIntegrityError,
        operational=CalibrationProfileCatalogStoreError,
        temporary_prefix=".tmp-catalog-",
    )


def domain_of(tmp_path: Path, kind: str) -> SimpleNamespace:
    """Build the store domain named by ``kind``."""
    return (
        profile_domain(tmp_path / "store")
        if kind == "profile"
        else catalog_domain(tmp_path / "store")
    )


def imported_modules(module: Any) -> set[str]:
    """Return every module name imported at the top level of ``module``.

    ``from probvenance import submodule`` records the module ``probvenance``
    rather than the submodule, so that form is expanded here; otherwise a
    module could import a forbidden layer without the assertion noticing.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            if node.module == "probvenance":
                names.update(f"probvenance.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


# ---------------------------------------------------------------------------
# Version matrix (PART 41, PART 42)
# ---------------------------------------------------------------------------


class TestPhase4CVersionMatrix:
    """The closure round freezes the matrix; no value may drift by accident."""

    def test_every_version_is_the_frozen_value(self) -> None:
        assert CALIBRATION_OBSERVATION_FINGERPRINT_VERSION == 1
        assert CALIBRATION_BINDING_FINGERPRINT_VERSION == 1
        assert GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION == 1
        assert CALIBRATION_DATASET_FINGERPRINT_VERSION == 2

        assert CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION == 1
        assert CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION == 2

        assert BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION == 3
        assert LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION == 3
        assert WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION == 2
        assert WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION == 2
        assert WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION == 2

        assert CALIBRATION_PROFILE_FINGERPRINT_VERSION == 1
        assert CALIBRATION_PROFILE_SERIALIZATION_VERSION == 1
        assert CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION == 1

        assert L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION == 1
        assert _L2_LOGISTIC_SOLVER_VERSION == 2

        assert PROFILE_APPLIED_EVALUATION_DATASET_FINGERPRINT_VERSION == 2
        assert POST_CALIBRATION_BRIER_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_LOG_LOSS_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_DIAGNOSTICS_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_RELIABILITY_RESULT_FINGERPRINT_VERSION == 1
        assert POST_CALIBRATION_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION == 1

        assert CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION == 1
        assert CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION == 1
        assert CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION == 1

        assert PLAN_FINGERPRINT_VERSION == 6
        assert EXECUTION_FINGERPRINT_VERSION == 2

    def test_named_metric_and_target_versions_are_one(self) -> None:
        assert MEAN_SELECTED_PROBABILITY_VERSION == 1
        assert EMPIRICAL_CORRECTNESS_RATE_VERSION == 1
        assert EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION == 1
        assert EQUAL_WIDTH_BINNING_VERSION == 1
        assert MEAN_PREDICTED_CORRECTNESS_VERSION == 1
        assert WINNER_CORRECTNESS_TARGET_VERSION == 1
        assert UNCALIBRATED_SELECTED_PROBABILITY_VERSION == 1
        assert PREDICTED_WINNER_CORRECTNESS_VERSION == 1
        assert RENDERING_SEMANTICS_VERSION == 1

    def test_store_layouts_are_independent_namespaces(self) -> None:
        # The two stores must never share a layout directory name, because they
        # are separate namespaces that may share a root without interpreting each
        # other's artifacts.
        from probvenance.calibration_catalog_store import (
            _STORE_DIRECTORY as catalog_layout,
        )
        from probvenance.calibration_store import _STORE_DIRECTORY as profile_layout

        assert profile_layout
        assert catalog_layout
        assert profile_layout != catalog_layout
        assert CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION == 1
        assert CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION == 1
        assert CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION == 1
        assert CALIBRATION_PROFILE_FINGERPRINT_VERSION == 1

    def test_method_component_and_formulation_versions_are_pinned(self) -> None:
        # These feed the profile canonical payload, so silent drift would change
        # profile identity without any fingerprint version moving.
        from probvenance.calibration import (
            _L2_LOGISTIC_CONVERGENCE_VERSION,
            _L2_LOGISTIC_ENDPOINT_POLICY_VERSION,
            _L2_LOGISTIC_INPUT_TRANSFORM_VERSION,
            _L2_LOGISTIC_OBJECTIVE_VERSION,
        )
        from probvenance.probability_identity import (
            FORMULATION_FAMILY_FINGERPRINT_VERSION,
            PROBABILITY_FORMULATION_FINGERPRINT_VERSION,
        )

        assert _L2_LOGISTIC_OBJECTIVE_VERSION == 1
        assert _L2_LOGISTIC_INPUT_TRANSFORM_VERSION == 1
        assert _L2_LOGISTIC_ENDPOINT_POLICY_VERSION == 1
        assert _L2_LOGISTIC_CONVERGENCE_VERSION == 1
        assert _L2_LOGISTIC_SOLVER_VERSION == 2
        assert PROBABILITY_FORMULATION_FINGERPRINT_VERSION == 2
        assert FORMULATION_FAMILY_FINGERPRINT_VERSION == 2


# ---------------------------------------------------------------------------
# Dependency graph (PART 5, PART 39)
# ---------------------------------------------------------------------------


class TestDependencyDirection:
    """The layer order is enforced by imports, not by convention."""

    def test_runtime_never_imports_the_calibration_subsystem(self) -> None:
        from probvenance import runtime

        imported = imported_modules(runtime)
        for forbidden in (
            "probvenance.calibration",
            "probvenance.calibration_selection",
            "probvenance.calibration_catalog",
            "probvenance.calibration_store",
            "probvenance.calibration_catalog_store",
            "probvenance.calibration_evaluation",
        ):
            assert forbidden not in imported

    def test_foundation_never_imports_catalog_store_or_selection(self) -> None:
        from probvenance import calibration

        imported = imported_modules(calibration)
        for forbidden in (
            "probvenance.calibration_catalog",
            "probvenance.calibration_store",
            "probvenance.calibration_catalog_store",
            "probvenance.calibration_selection",
        ):
            assert forbidden not in imported

    def test_the_two_stores_are_mutually_independent(self) -> None:
        from probvenance import calibration_catalog_store, calibration_store

        assert "probvenance.calibration_catalog_store" not in imported_modules(calibration_store)
        assert "probvenance.calibration_store" not in imported_modules(calibration_catalog_store)

    def test_shared_filesystem_helper_is_domain_neutral(self) -> None:
        # It may import stdlib modules, but no probvenance domain module: a
        # dependency on a store, profile, catalog, or binding type would make it
        # a second source of truth for the store invariants.
        imported = {
            name for name in imported_modules(shared_filesystem) if name.startswith("probvenance")
        }
        assert imported == set()
        public = {name for name in dir(shared_filesystem) if not name.startswith("_")}
        for name in public:
            lowered = name.lower()
            assert "calibration" not in lowered
            assert "profile" not in lowered
            assert "catalog" not in lowered
            assert "binding" not in lowered


# ---------------------------------------------------------------------------
# Shared strict-JSON attack matrix (PART 34)
# ---------------------------------------------------------------------------

RAW_ATTACKS: list[tuple[str, str]] = [
    ("invalid_json", "{ not json"),
    ("duplicate_key", '{"a": 1, "a": 2}'),
    ("nan", '{"a": NaN}'),
    ("infinity", '{"a": Infinity}'),
    ("negative_infinity", '{"a": -Infinity}'),
    ("overflowing_float", '{"a": 1e9999}'),
    ("huge_integer", '{"a": ' + "1" * 6000 + "}"),
    ("deep_nesting", '{"a": ' + "[" * 500 + "]" * 500 + "}"),
    ("array_not_object", "[1, 2, 3]"),
    ("scalar_not_object", '"hello"'),
]


class TestSharedStrictJsonMatrix:
    """Both loaders must fail the same way on the same hostile text."""

    @pytest.mark.parametrize(("label", "text"), RAW_ATTACKS)
    def test_profile_loader_reports_a_controlled_error(self, label: str, text: str) -> None:
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile(text)

    @pytest.mark.parametrize(("label", "text"), RAW_ATTACKS)
    def test_catalog_loader_reports_a_controlled_error(self, label: str, text: str) -> None:
        with pytest.raises(InvalidDecisionError):
            load_calibration_profile_catalog(text)

    def test_document_root_is_level_one(self) -> None:
        # Convention: the document root is level 1, each containment adds one
        # level, and no node may exceed level 64. ``deep_value(value, n)`` wraps
        # ``value`` in ``n`` nested objects, so n = 63 puts the leaf at level 64
        # (accepted) and n = 64 puts it at level 65 (rejected).
        assert _MAX_ARTIFACT_NESTING_DEPTH == 64
        require_bounded_nesting(deep_value("x", 63), subject="test artifact")
        with pytest.raises(InvalidDecisionError, match="nested more than 64 levels deep"):
            require_bounded_nesting(deep_value("x", 64), subject="test artifact")


# ---------------------------------------------------------------------------
# Store error matrix (PART 35)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["profile", "catalog"])
class TestStoreErrorSymmetry:
    """Both exact stores classify the same situations analogously."""

    def test_absence_is_not_found_and_creates_nothing(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        with pytest.raises(domain.not_found):
            domain.get()
        assert not domain.store.root.exists()

    def test_corrupt_json_is_integrity_not_absence(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        write_text_at(domain.target, "{ not json")
        with pytest.raises(domain.integrity):
            domain.get()

    def test_substituted_identity_at_the_right_path_is_integrity(
        self, tmp_path: Path, kind: str
    ) -> None:
        domain = domain_of(tmp_path, kind)
        write_text_at(domain.target, domain.other_text)
        with pytest.raises(domain.integrity):
            domain.get()

    def test_valid_but_noncanonical_json_is_integrity(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        write_text_at(domain.target, json.dumps(json.loads(domain.valid_text), indent=2))
        with pytest.raises(domain.integrity):
            domain.get()

    def test_invalid_utf8_is_integrity(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        domain.target.parent.mkdir(parents=True, exist_ok=True)
        domain.target.write_bytes(b"\xff\xfe\x00not utf8")
        with pytest.raises(domain.integrity):
            domain.get()

    def test_deep_json_is_integrity_not_recursion_error(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        write_text_at(domain.target, '{"a": ' + "[" * 500 + "]" * 500 + "}")
        with pytest.raises(domain.integrity):
            domain.get()

    def test_huge_integer_is_integrity_not_value_error(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        write_text_at(domain.target, '{"a": ' + "1" * 6000 + "}")
        with pytest.raises(domain.integrity):
            domain.get()

    def test_managed_directory_replaced_by_file_is_integrity(
        self, tmp_path: Path, kind: str
    ) -> None:
        domain = domain_of(tmp_path, kind)
        layout = domain.target.parent.parent
        layout.parent.mkdir(parents=True, exist_ok=True)
        layout.write_text("not a directory", encoding="utf-8")
        with pytest.raises(domain.integrity):
            domain.get()

    def test_root_replaced_by_file_is_operational(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        domain.store.root.parent.mkdir(parents=True, exist_ok=True)
        domain.store.root.write_text("not a directory", encoding="utf-8")
        with pytest.raises(domain.operational) as info:
            domain.get()
        assert type(info.value) is domain.operational

    @pytest.mark.skipif(os.getuid() == 0, reason="root bypasses directory permissions")
    def test_unreadable_managed_directory_is_operational_not_raw_oserror(
        self, tmp_path: Path, kind: str
    ) -> None:
        domain = domain_of(tmp_path, kind)
        version_directory = domain.target.parent
        version_directory.mkdir(parents=True, exist_ok=True)
        os.chmod(version_directory, 0o000)
        try:
            with pytest.raises(domain.operational) as info:
                domain.get()
            assert type(info.value) is domain.operational
            assert not isinstance(info.value, domain.not_found)
            assert not isinstance(info.value, domain.integrity)
        finally:
            os.chmod(version_directory, 0o755)

    def test_huge_attacker_payload_keeps_the_message_bounded(
        self, tmp_path: Path, kind: str
    ) -> None:
        domain = domain_of(tmp_path, kind)
        huge_key = "k" * 1_000_000
        write_text_at(domain.target, '{"' + huge_key + '": 1, "' + huge_key + '": 2}')
        with pytest.raises(domain.integrity) as info:
            domain.get()
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS

    def test_idempotent_put_keeps_the_same_bytes(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        domain.put()
        first = domain.target.read_text(encoding="utf-8")
        domain.put()
        assert domain.target.read_text(encoding="utf-8") == first
        assert domain.get().fingerprint == domain.artifact.fingerprint

    def test_corrupt_existing_target_is_never_overwritten(self, tmp_path: Path, kind: str) -> None:
        domain = domain_of(tmp_path, kind)
        write_text_at(domain.target, "{ not json")
        with pytest.raises(domain.integrity):
            domain.put()
        assert domain.target.read_text(encoding="utf-8") == "{ not json"


class TestPublicationClosesTemporaryResources:
    """A controlled publication failure must leak neither a descriptor nor a file."""

    def test_failed_descriptor_wrap_closes_the_descriptor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = profile_domain(tmp_path / "store")
        captured: list[int] = []
        real_fdopen = os.fdopen

        def failing_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
            captured.append(descriptor)
            raise OSError(errno.EMFILE, "too many open files")

        monkeypatch.setattr(os, "fdopen", failing_fdopen)
        with pytest.raises(CalibrationProfileStoreError):
            domain.put()
        monkeypatch.setattr(os, "fdopen", real_fdopen)

        assert captured, "os.fdopen was never reached, so the case did not run"
        # The descriptor must have been closed, so any further use is EBADF.
        with pytest.raises(OSError):
            os.fstat(captured[0])

    def test_successful_publication_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        domain = catalog_domain(tmp_path / "store")
        domain.put()
        leftovers = [
            path
            for path in domain.target.parent.iterdir()
            if path.name.startswith(domain.temporary_prefix)
        ]
        assert leftovers == []


# ---------------------------------------------------------------------------
# End-to-end authorization chains (PART 48 through PART 56)
# ---------------------------------------------------------------------------


def shared_stores(tmp_path: Path) -> tuple[DirectoryCalibrationProfileStore, Any]:
    """Return a profile store and a catalog store sharing one root directory."""
    root = tmp_path / "shared"
    return DirectoryCalibrationProfileStore(root), DirectoryCalibrationProfileCatalogStore(root)


class TestEndToEndChains:
    """The full chain, including its failure branches, closes authorization."""

    def test_unique_path_applies_exactly_one_profile(self, tmp_path: Path) -> None:
        profile_store, catalog_store = shared_stores(tmp_path)
        profile = eligible_profile()
        profile_store.put(profile)
        loaded_profile = profile_store.get(
            profile_fingerprint=profile.fingerprint,
            profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
        )
        assert loaded_profile.fingerprint == profile.fingerprint

        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        catalog_store.put(catalog)
        loaded_catalog = catalog_store.get(
            catalog_fingerprint=catalog.fingerprint,
            catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
        )
        assert loaded_catalog.fingerprint == catalog.fingerprint

        evaluation = runtime_evaluation()
        references = discover_calibration_profile_references_for_runtime(evaluation, loaded_catalog)
        assert len(references) == 1
        assert references[0].profile_fingerprint == profile.fingerprint

        selected = select_calibration_profile_for_runtime(evaluation, (loaded_profile,))
        assert selected.fingerprint == profile.fingerprint

        calibrated = apply_profile_to_runtime_evaluation(evaluation, selected)
        assert calibrated.result.calibrated is True
        assert calibrated.result.predicted_correctness is not None
        assert calibrated.result.calibration_profile_fingerprint == selected.fingerprint
        assert (
            calibrated.result.calibration_profile_fingerprint_version
            == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        )
        assert calibrated.trace.calibration_profile_fingerprint == selected.fingerprint
        # Explicit calibration is a result-level transformation only.
        assert calibrated.trace.execution_fingerprint == evaluation.trace.execution_fingerprint
        assert (
            calibrated.trace.probability_formulation_fingerprint
            == evaluation.trace.probability_formulation_fingerprint
        )

        # The offline scorer and the runtime path share one numerical kernel, so
        # the same profile must produce the same number for the same selected
        # score: re-applying must be deterministic.
        again = apply_profile_to_runtime_evaluation(evaluation, selected)
        assert again.result.predicted_correctness == calibrated.result.predicted_correctness

    def test_ambiguity_path_survives_the_store_and_fails_at_selection(self, tmp_path: Path) -> None:
        profile_store, catalog_store = shared_stores(tmp_path)
        first = eligible_profile()
        second = other_training_profile()
        assert first.fingerprint != second.fingerprint

        profile_store.put(first)
        profile_store.put(second)
        catalog = CalibrationProfileCatalog.from_profiles((first, second))
        catalog_store.put(catalog)
        loaded_catalog = catalog_store.get(
            catalog_fingerprint=catalog.fingerprint,
            catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
        )

        evaluation = runtime_evaluation()
        references = discover_calibration_profile_references_for_runtime(evaluation, loaded_catalog)
        assert len(references) == 2

        loaded = tuple(
            profile_store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )
            for reference in references
        )
        assert {profile.fingerprint for profile in loaded} == {
            first.fingerprint,
            second.fingerprint,
        }
        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(evaluation, loaded)

    def test_stale_catalog_metadata_cannot_authorize_a_mismatched_profile(
        self, tmp_path: Path
    ) -> None:
        profile_store, catalog_store = shared_stores(tmp_path)
        real_profile = other_binding_profile()
        profile_store.put(real_profile)

        catalog = CalibrationProfileCatalog.from_profiles((real_profile,))
        document = json.loads(serialize_calibration_profile_catalog(catalog))
        # Forge the discovery projection so it claims the runtime's binding while
        # the entries still point at the real (incompatible) profile.
        from probvenance.fingerprint import canonical_json, fingerprint

        document["catalog_identity"]["entries"][0]["binding"] = json.loads(
            canonical_json(eligible_profile().binding.canonical_payload())
        )
        document["catalog_fingerprint"] = fingerprint(document["catalog_identity"])
        forged_text = json.dumps(document)
        forged = load_calibration_profile_catalog(forged_text)
        forged_target = (
            catalog_store.root
            / f"catalog-store-v{CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION}"
            / f"catalog-fingerprint-v{CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION}"
            / f"{forged.fingerprint}.json"
        )
        write_text_at(forged_target, forged_text)

        evaluation = runtime_evaluation()
        references = discover_calibration_profile_references_for_runtime(evaluation, forged)
        assert len(references) == 1
        loaded = profile_store.get(
            profile_fingerprint=references[0].profile_fingerprint,
            profile_fingerprint_version=references[0].profile_fingerprint_version,
        )
        assert loaded.fingerprint == real_profile.fingerprint
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (loaded,))

    def test_missing_referenced_profile_stays_profile_not_found(self, tmp_path: Path) -> None:
        profile_store, catalog_store = shared_stores(tmp_path)
        profile = eligible_profile()
        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        catalog_store.put(catalog)
        loaded_catalog = catalog_store.get(
            catalog_fingerprint=catalog.fingerprint,
            catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
        )
        references = discover_calibration_profile_references_for_runtime(
            runtime_evaluation(), loaded_catalog
        )
        assert len(references) == 1
        with pytest.raises(CalibrationProfileNotFoundError):
            profile_store.get(
                profile_fingerprint=references[0].profile_fingerprint,
                profile_fingerprint_version=references[0].profile_fingerprint_version,
            )

    def test_corrupt_referenced_profile_stays_profile_integrity(self, tmp_path: Path) -> None:
        profile_store, catalog_store = shared_stores(tmp_path)
        profile = eligible_profile()
        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        catalog_store.put(catalog)
        loaded_catalog = catalog_store.get(
            catalog_fingerprint=catalog.fingerprint,
            catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
        )
        references = discover_calibration_profile_references_for_runtime(
            runtime_evaluation(), loaded_catalog
        )
        target = (
            profile_store.root
            / f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}"
            / f"profile-fingerprint-v{CALIBRATION_PROFILE_FINGERPRINT_VERSION}"
            / f"{references[0].profile_fingerprint}.json"
        )
        write_text_at(target, "{ not json")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            profile_store.get(
                profile_fingerprint=references[0].profile_fingerprint,
                profile_fingerprint_version=references[0].profile_fingerprint_version,
            )

    def test_missing_catalog_is_not_found(self, tmp_path: Path) -> None:
        _, catalog_store = shared_stores(tmp_path)
        profile = eligible_profile()
        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        with pytest.raises(CalibrationProfileCatalogNotFoundError):
            catalog_store.get(
                catalog_fingerprint=catalog.fingerprint,
                catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
            )

    def test_corrupt_catalog_is_integrity(self, tmp_path: Path) -> None:
        _, catalog_store = shared_stores(tmp_path)
        catalog = CalibrationProfileCatalog.from_profiles((eligible_profile(),))
        target = (
            catalog_store.root
            / f"catalog-store-v{CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION}"
            / f"catalog-fingerprint-v{CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION}"
            / f"{catalog.fingerprint}.json"
        )
        write_text_at(target, "{ not json")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            catalog_store.get(
                catalog_fingerprint=catalog.fingerprint,
                catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
            )

    def test_both_stores_coexist_under_one_root(self, tmp_path: Path) -> None:
        profile_store, catalog_store = shared_stores(tmp_path)
        profile = eligible_profile()
        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        profile_store.put(profile)
        catalog_store.put(catalog)
        assert profile_store.root == catalog_store.root
        assert (
            profile_store.get(
                profile_fingerprint=profile.fingerprint,
                profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
            ).fingerprint
            == profile.fingerprint
        )
        assert (
            catalog_store.get(
                catalog_fingerprint=catalog.fingerprint,
                catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
            ).fingerprint
            == catalog.fingerprint
        )

    def test_a_catalog_may_reference_an_absent_profile(self, tmp_path: Path) -> None:
        # A snapshot is a claim, not a population: the catalog store must not
        # verify profile existence, because that would couple the two stores.
        _, catalog_store = shared_stores(tmp_path)
        catalog = CalibrationProfileCatalog.from_profiles((eligible_profile(),))
        catalog_store.put(catalog)
        assert (
            catalog_store.get(
                catalog_fingerprint=catalog.fingerprint,
                catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
            ).fingerprint
            == catalog.fingerprint
        )

    def test_unsupported_method_survives_to_the_application_boundary(self, tmp_path: Path) -> None:
        """A method the runtime cannot apply remains a uniquely eligible profile.

        Eligibility is an exact Binding plus the target and input identities, so the
        method is never a selection dimension. The profile has to survive the
        catalog, both stores, and selection, and only fail when application is
        attempted, because no earlier layer may learn a method preference.
        """
        profile_store, catalog_store = shared_stores(tmp_path)
        profile = unsupported_method_profile()
        profile_store.put(profile)
        catalog = CalibrationProfileCatalog.from_profiles((profile,))
        catalog_store.put(catalog)
        loaded_catalog = catalog_store.get(
            catalog_fingerprint=catalog.fingerprint,
            catalog_fingerprint_version=CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
        )

        evaluation = runtime_evaluation()
        references = discover_calibration_profile_references_for_runtime(evaluation, loaded_catalog)
        assert [reference.profile_fingerprint for reference in references] == [profile.fingerprint]

        loaded_profile = profile_store.get(
            profile_fingerprint=references[0].profile_fingerprint,
            profile_fingerprint_version=references[0].profile_fingerprint_version,
        )
        selected = select_calibration_profile_for_runtime(evaluation, (loaded_profile,))
        assert selected.fingerprint == profile.fingerprint

        with pytest.raises(InvalidDecisionError):
            apply_profile_to_runtime_evaluation(evaluation, selected)


# ---------------------------------------------------------------------------
# Profile loader bounded errors (PART 36)
# ---------------------------------------------------------------------------


class TestProfileLoaderBoundedErrors:
    """The Profile loader echoes attacker input, so it must bound what it echoes."""

    def test_huge_artifact_type_keeps_the_message_bounded(self) -> None:
        document = json.loads(serialize_calibration_profile(fitted_profile()))
        document["artifact_type"] = "x" * 1_000_000
        with pytest.raises(InvalidDecisionError) as info:
            load_calibration_profile(json.dumps(document))
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS

    def test_huge_rogue_key_keeps_the_message_bounded(self) -> None:
        document = json.loads(serialize_calibration_profile(fitted_profile()))
        document["k" * 1_000_000] = 1
        with pytest.raises(InvalidDecisionError) as info:
            load_calibration_profile(json.dumps(document))
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS

    def test_huge_malformed_identity_value_keeps_the_message_bounded(self) -> None:
        document = json.loads(serialize_calibration_profile(fitted_profile()))
        document["profile_fingerprint"] = "y" * 1_000_000
        with pytest.raises(InvalidDecisionError) as info:
            load_calibration_profile(json.dumps(document))
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS

    def test_huge_nested_value_keeps_the_message_bounded(self) -> None:
        document = json.loads(serialize_calibration_profile(fitted_profile()))
        document["fitted_parameters"] = {"slope": "z" * 1_000_000, "intercept": 0.0}
        with pytest.raises(InvalidDecisionError) as info:
            load_calibration_profile(json.dumps(document))
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS


# ---------------------------------------------------------------------------
# Loader field fuzz (PART 34, PART 36)
# ---------------------------------------------------------------------------

#: A 4000-digit integer serializes and re-renders without hitting the
#: interpreter's int-to-text limit, so it isolates pure message unboundedness.
#: The 6000-digit raw-text case in ``RAW_ATTACKS`` covers the limit case itself.
HUGE_INT = int("1" * 4000)
HUGE_STR = "x" * 1_000_000

PROFILE_FIELD_PATHS = [
    "artifact_type",
    "serialization_version",
    "profile_identity",
    "profile_identity.v",
    "profile_identity.binding",
    "profile_identity.binding.binding_fingerprint",
    "profile_identity.binding.binding_fingerprint_version",
    "profile_identity.ground_truth_semantics",
    "profile_identity.ground_truth_semantics.ground_truth_semantics_fingerprint",
    "profile_identity.ground_truth_semantics.ground_truth_semantics_fingerprint_version",
    "profile_identity.target_id",
    "profile_identity.target_version",
    "profile_identity.input_score_id",
    "profile_identity.input_score_version",
    "profile_identity.method_id",
    "profile_identity.method_version",
    "profile_identity.method_configuration",
    "profile_identity.fitted_parameters",
    "profile_identity.training_dataset_fingerprint",
    "profile_identity.training_dataset_fingerprint_version",
    "materialized_binding",
    "materialized_ground_truth_semantics",
    "profile_fingerprint",
]

CATALOG_FIELD_PATHS = [
    "artifact_type",
    "serialization_version",
    "catalog_identity",
    "catalog_identity.v",
    "catalog_identity.entries",
    "catalog_fingerprint",
    "catalog_identity.entries.0.profile_fingerprint",
    "catalog_identity.entries.0.profile_fingerprint_version",
    "catalog_identity.entries.0.binding",
    "catalog_identity.entries.0.target_id",
    "catalog_identity.entries.0.target_version",
    "catalog_identity.entries.0.input_score_id",
    "catalog_identity.entries.0.input_score_version",
]


def document_paths(value: Any, prefix: str = "") -> list[str]:
    """Return every dotted path in a JSON document; digits index lists."""
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(document_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            paths.append(path)
            paths.extend(document_paths(item, path))
    return paths


def set_at_path(document: dict[str, Any], path: str, value: Any) -> None:
    """Set ``path`` (dot separated, digits index lists) inside ``document``."""
    parts = path.split(".")
    node: Any = document
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


@pytest.mark.parametrize("path", PROFILE_FIELD_PATHS)
@pytest.mark.parametrize("value_kind", ["huge_string", "huge_int"])
class TestProfileLoaderFieldFuzz:
    """No sampled Profile document field may inflate a loader error message.

    ``PROFILE_FIELD_PATHS`` samples a subset; ``TestLoaderFuzzCoversEveryPath``
    walks every reachable path so the guarantee does not depend on the sample.
    """

    def test_huge_value_keeps_the_message_bounded(self, path: str, value_kind: str) -> None:
        document = json.loads(serialize_calibration_profile(fitted_profile()))
        set_at_path(document, path, HUGE_STR if value_kind == "huge_string" else HUGE_INT)
        serialized = json.dumps(document)
        with pytest.raises(InvalidDecisionError) as info:
            load_calibration_profile(serialized)
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS


@pytest.mark.parametrize("path", CATALOG_FIELD_PATHS)
@pytest.mark.parametrize("value_kind", ["huge_string", "huge_int"])
class TestCatalogLoaderFieldFuzz:
    """No sampled catalog document field may inflate a loader error message.

    ``CATALOG_FIELD_PATHS`` samples a subset; ``TestLoaderFuzzCoversEveryPath``
    walks every reachable path so the guarantee does not depend on the sample.
    """

    def test_huge_value_keeps_the_message_bounded(self, path: str, value_kind: str) -> None:
        document = json.loads(serialize_calibration_profile_catalog(catalog_of(eligible_profile())))
        set_at_path(document, path, HUGE_STR if value_kind == "huge_string" else HUGE_INT)
        serialized = json.dumps(document)
        with pytest.raises(InvalidDecisionError) as info:
            load_calibration_profile_catalog(serialized)
        assert len(str(info.value)) < _MAX_MESSAGE_CHARS


# ---------------------------------------------------------------------------
# Scope guards (PART 43, PART 57)
# ---------------------------------------------------------------------------


class TestScopeGuards:
    """No registry, enumeration, lifecycle, or automation may have crept in."""

    def test_stores_expose_only_exact_identity_operations(self) -> None:
        for store_class in (
            DirectoryCalibrationProfileStore,
            DirectoryCalibrationProfileCatalogStore,
        ):
            public = {name for name in dir(store_class) if not name.startswith("_")}
            assert public == {"get", "put", "root"}

    @pytest.mark.parametrize(
        "forbidden",
        [
            "list",
            "list_all",
            "list_profiles",
            "list_catalogs",
            "scan",
            "find",
            "search",
            "query",
            "all",
            "latest",
            "active",
            "default",
            "delete",
            "remove",
            "update",
            "replace",
            "rename",
            "activate",
            "deactivate",
            "promote",
            "supersede",
            "alias",
            "refresh",
            "sync",
            "synchronize",
        ],
    )
    def test_no_lifecycle_or_enumeration_member_exists(self, forbidden: str) -> None:
        for store_class in (
            DirectoryCalibrationProfileStore,
            DirectoryCalibrationProfileCatalogStore,
        ):
            assert not hasattr(store_class, forbidden)

    def test_store_classes_are_not_root_exported(self) -> None:
        import probvenance

        assert not hasattr(probvenance, "DirectoryCalibrationProfileStore")
        assert not hasattr(probvenance, "DirectoryCalibrationProfileCatalogStore")

    def test_calibration_errors_are_root_exported(self) -> None:
        import probvenance

        for name in (
            "CalibrationProfileStoreError",
            "CalibrationProfileNotFoundError",
            "CalibrationProfileStoreIntegrityError",
            "CalibrationProfileCatalogStoreError",
            "CalibrationProfileCatalogNotFoundError",
            "CalibrationProfileCatalogStoreIntegrityError",
            "CalibrationProfileSelectionError",
            "NoEligibleCalibrationProfileError",
            "AmbiguousCalibrationProfileSelectionError",
        ):
            assert hasattr(probvenance, name)
            assert name in probvenance.__all__

    def test_calibration_foundation_does_not_reach_into_the_runtime(self) -> None:
        # ``calibration`` may import the runtime TYPES it consumes, but the
        # runtime must never import the calibration foundation back.
        from probvenance import calibration

        imported = imported_modules(calibration)
        assert "probvenance.runtime" in imported
        assert "probvenance.calibration_evaluation" not in imported


class TestCallerArgumentBoundedErrors:
    """A public entry point must not echo a hostile caller argument unbounded.

    Bounding uses ``repr``-equivalent for short values, so these assertions only
    fail for the megabyte-long argument that motivated them: a caller passing a
    huge string where an artifact was expected previously produced a
    1,000,050-character ``InvalidDecisionError``. A valid artifact carrying a
    huge legal string (for example a long ``model`` name) is also bounded.
    """

    @staticmethod
    def _assert_bounded(call: Callable[[], object]) -> None:
        with pytest.raises(InvalidDecisionError) as info:
            call()
        length = len(str(info.value))
        assert length < 2000, length

    def test_serializer_bounds_a_wrong_typed_argument(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: serialize_calibration_profile(wrong))

    def test_loader_bounds_a_wrong_typed_argument(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: load_calibration_profile(wrong))

    def test_loader_bounds_an_unconvertible_integer_argument(self) -> None:
        self._assert_bounded(lambda: load_calibration_profile(10**5000))

    def test_catalog_serializer_bounds_a_wrong_typed_argument(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: serialize_calibration_profile_catalog(wrong))

    def test_catalog_loader_bounds_a_wrong_typed_argument(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: load_calibration_profile_catalog(wrong))

    def test_scorer_bounds_a_wrong_typed_profile(self) -> None:
        wrong: Any = HUGE_STR
        observation: Any = object()
        self._assert_bounded(lambda: predicted_winner_correctness(wrong, observation))

    def test_application_kernel_bounds_a_wrong_typed_profile(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: _apply_profile_to_selected_probability(wrong, 0.5))

    def test_runtime_application_bounds_a_wrong_typed_profile(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: apply_profile_to_runtime_evaluation(wrong, fitted_profile()))

    def test_fitter_bounds_a_wrong_typed_dataset(self) -> None:
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: fit_l2_logistic_selected_probability(wrong, l2_strength=0.01))

    def test_profile_store_put_bounds_a_wrong_typed_artifact(self, tmp_path: Any) -> None:
        store = DirectoryCalibrationProfileStore(tmp_path / "profiles")
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: store.put(wrong))

    def test_catalog_store_put_bounds_a_wrong_typed_artifact(self, tmp_path: Any) -> None:
        store = DirectoryCalibrationProfileCatalogStore(tmp_path / "catalogs")
        wrong: Any = HUGE_STR
        self._assert_bounded(lambda: store.put(wrong))

    def test_a_valid_artifact_with_a_huge_legal_string_stays_bounded(self) -> None:
        import dataclasses

        profile = fitted_profile()
        huge_binding = dataclasses.replace(profile.binding, model=HUGE_STR)
        self._assert_bounded(
            lambda: serialize_calibration_profile(
                dataclasses.replace(profile, binding=huge_binding)
            )
        )


class TestLoaderFuzzCoversEveryPath:
    """Every reachable document path is fuzzed, not only the sampled subset."""

    def _assert_path_bounded(self, path: str, serialized_of: Any, load: Any) -> None:
        for value in (HUGE_STR, HUGE_INT):
            document = json.loads(serialized_of())
            set_at_path(document, path, value)
            serialized = json.dumps(document)
            with pytest.raises(InvalidDecisionError) as info:
                load(serialized)
            assert len(str(info.value)) < _MAX_MESSAGE_CHARS

    def test_every_profile_document_path_stays_bounded(self) -> None:
        serialized_of = lambda: serialize_calibration_profile(fitted_profile())  # noqa: E731
        paths = document_paths(json.loads(serialized_of()))
        assert len(paths) > len(PROFILE_FIELD_PATHS)
        for path in paths:
            self._assert_path_bounded(path, serialized_of, load_calibration_profile)

    def test_every_catalog_document_path_stays_bounded(self) -> None:
        serialized_of = lambda: serialize_calibration_profile_catalog(  # noqa: E731
            catalog_of(eligible_profile())
        )
        paths = document_paths(json.loads(serialized_of()))
        assert len(paths) > len(CATALOG_FIELD_PATHS)
        for path in paths:
            self._assert_path_bounded(path, serialized_of, load_calibration_profile_catalog)
