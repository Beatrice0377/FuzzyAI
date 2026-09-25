"""Exact content-addressed directory store for calibration profiles.

The central contract: retrieval requires an exact profile fingerprint and
fingerprint schema version, derives exactly one path, and restores the artifact
through the identity-verified loader with the requested identity supplied as an
independent expected pin. There is no enumeration, matching, or fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    choice_observation,
    resolved_truth,
)
from test_profile_serialization import (
    evaluation_dataset,
    fitted_profile,
    runtime_evaluation,
)

from probvenance import (
    CalibrationProfileNotFoundError,
    CalibrationProfileStoreError,
    CalibrationProfileStoreIntegrityError,
    InvalidDecisionError,
)
from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationDataset,
    apply_profile_to_runtime_evaluation,
    fit_l2_logistic_selected_probability,
    predicted_winner_correctness,
    serialize_calibration_profile,
)
from probvenance.calibration_evaluation import apply_profile_to_evaluation_dataset
from probvenance.calibration_store import (
    CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION,
    DirectoryCalibrationProfileStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def store_at(tmp_path: Path) -> DirectoryCalibrationProfileStore:
    return DirectoryCalibrationProfileStore(tmp_path / "profiles")


def key_of(profile: Any) -> dict[str, Any]:
    return {
        "profile_fingerprint": profile.fingerprint,
        "profile_fingerprint_version": CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    }


def stored_path(store: DirectoryCalibrationProfileStore, profile: Any) -> Path:
    return (
        store.root
        / f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}"
        / f"profile-fingerprint-v{CALIBRATION_PROFILE_FINGERPRINT_VERSION}"
        / f"{profile.fingerprint}.json"
    )


def other_profile() -> Any:
    return fit_l2_logistic_selected_probability(
        CalibrationDataset.create([choice_observation()]), l2_strength=0.5
    )


# ---------------------------------------------------------------------------
# Round trip and layout (PART 8, PART 38)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_put_then_get_preserves_identity(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        loaded = store.get(**key_of(profile))
        assert loaded.fingerprint == profile.fingerprint
        assert loaded.canonical_payload() == profile.canonical_payload()
        assert loaded.binding.canonical_payload() == profile.binding.canonical_payload()
        assert (
            loaded.ground_truth_semantics.canonical_payload()
            == profile.ground_truth_semantics.canonical_payload()
        )
        assert serialize_calibration_profile(loaded) == serialize_calibration_profile(profile)

    def test_layout_is_deterministic(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        expected = stored_path(store, profile)
        assert expected.exists()
        assert expected.read_text(encoding="utf-8") == serialize_calibration_profile(profile)

    def test_stored_bytes_are_canonical(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        text = stored_path(store, profile).read_text(encoding="utf-8")
        assert text == serialize_calibration_profile(profile)
        assert not text.endswith("\n")


# ---------------------------------------------------------------------------
# Lookup semantics (PART 10, PART 11, PART 32, PART 44, PART 45)
# ---------------------------------------------------------------------------


class TestLookupSemantics:
    def test_requires_both_key_parts(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        with pytest.raises(TypeError):
            store.get(profile_fingerprint=profile.fingerprint)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "fingerprint",
        [
            "../x",
            "../../etc/passwd",
            "a/b",
            "a\\b",
            "abc",
            "0" * 63,
            "0" * 65,
            "A" * 64,
            " " + "0" * 64,
            "0" * 64 + " ",
            "0" * 63 + "/",
        ],
    )
    def test_malformed_fingerprint_rejected_before_filesystem(
        self, tmp_path: Path, fingerprint: str
    ) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError):
            store.get(
                profile_fingerprint=fingerprint,
                profile_fingerprint_version=1,
            )

    @pytest.mark.parametrize("version", [True, False, 1.0, "1", 0, -1, None])
    def test_malformed_version_rejected(self, tmp_path: Path, version: Any) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError):
            store.get(
                profile_fingerprint="0" * 64,
                profile_fingerprint_version=version,
            )

    def test_unsupported_version_is_not_a_miss(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        with pytest.raises(InvalidDecisionError, match="does not understand"):
            store.get(
                profile_fingerprint=profile.fingerprint,
                profile_fingerprint_version=2,
            )

    def test_exact_miss_is_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(profile_fingerprint="0" * 64, profile_fingerprint_version=1)

    def test_exact_miss_does_not_mutate_filesystem(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(profile_fingerprint="0" * 64, profile_fingerprint_version=1)
        assert not store.root.exists()

    def test_no_singleton_fallback(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(profile_fingerprint="f" * 64, profile_fingerprint_version=1)

    def test_two_profiles_remain_distinct(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        first = fitted_profile()
        second = other_profile()
        assert first.fingerprint != second.fingerprint
        store.put(first)
        store.put(second)
        assert store.get(**key_of(first)).fingerprint == first.fingerprint
        assert store.get(**key_of(second)).fingerprint == second.fingerprint


# ---------------------------------------------------------------------------
# Write semantics (PART 19, PART 22, PART 23, PART 24)
# ---------------------------------------------------------------------------


class TestWriteSemantics:
    def test_put_is_idempotent(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        before = path.read_text(encoding="utf-8")
        store.put(profile)
        assert path.read_text(encoding="utf-8") == before
        assert len(list(path.parent.glob("*.json"))) == 1

    def test_corrupt_existing_target_is_not_overwritten(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        path.write_text("{ partial", encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.put(profile)
        assert path.read_text(encoding="utf-8") == "{ partial"

    def test_failed_publication_leaves_no_partial_final(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        path = stored_path(store, profile)

        def explode(source: Any, destination: Any) -> None:
            raise OSError("simulated publication failure")

        monkeypatch.setattr(os, "replace", explode)
        with pytest.raises(CalibrationProfileStoreError):
            store.put(profile)
        assert not path.exists()
        assert list(path.parent.glob(".tmp-profile-*")) == []


# ---------------------------------------------------------------------------
# Tamper and integrity (PART 47, PART 48, PART 49, PART 50, PART 51, PART 52,
# PART 53, PART 54)
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_a_huge_integer_literal_is_an_integrity_failure(self, tmp_path: Path) -> None:
        # Python raises a plain ValueError from int() while parsing an integer
        # literal beyond the digit limit, and JSONDecodeError is a ValueError
        # subclass but that one is not. The store must still fail closed with a
        # store error rather than leaking a low-level parse exception.
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        stored = stored_path(store, profile)
        text = stored.read_text(encoding="utf-8")
        assert '"target_version"' in text
        stored.write_text(
            text.replace('"target_version":1', '"target_version":' + "9" * 6000, 1),
            encoding="utf-8",
        )
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.put(profile)

    def test_a_rogue_key_does_not_echo_megabytes(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        stored = stored_path(store, profile)
        text = stored.read_text(encoding="utf-8").rstrip()
        stored.write_text(text[:-1] + ',"' + "K" * 1_000_000 + '":1}', encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError) as caught:
            store.get(**key_of(profile))
        assert len(str(caught.value)) < 2000

    def test_a_huge_artifact_type_does_not_echo_megabytes(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        stored = stored_path(store, profile)
        stored.write_text(
            stored.read_text(encoding="utf-8").replace(
                '"artifact_type":"probvenance.calibration-profile"',
                '"artifact_type":"' + "Z" * 1_000_000 + '"',
                1,
            ),
            encoding="utf-8",
        )
        with pytest.raises(CalibrationProfileStoreIntegrityError) as caught:
            store.get(**key_of(profile))
        assert len(str(caught.value)) < 2000

    def test_a_huge_caller_fingerprint_does_not_echo_megabytes(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError) as caught:
            store.get(
                profile_fingerprint="a" * 1_000_000,
                profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
            )
        assert len(str(caught.value)) < 2000

    def test_different_valid_profile_substituted_is_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        first = fitted_profile()
        second = other_profile()
        store.put(first)
        stored_path(store, first).write_text(
            serialize_calibration_profile(second), encoding="utf-8"
        )
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(first))

    def test_embedded_fingerprint_tamper_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["profile_fingerprint"] = "0" * 64
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_binding_materialization_tamper_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["materialized_binding"]["model_revision"] = "tampered"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_ground_truth_materialization_tamper_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["materialized_ground_truth_semantics"]["labeling_rule"] = "tampered"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_fitted_state_tamper_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["profile_identity"]["fitted_parameters"]["slope"] = 999.0
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_truncated_file_is_integrity_not_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        stored_path(store, profile).write_text("{ partial", encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_deeply_nested_artifact_is_integrity_not_recursion_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        document = json.loads(path.read_text(encoding="utf-8"))
        nested: Any = "x"
        for _ in range(495):
            nested = {"n": nested}
        document["materialized_binding"]["rendering_semantics"] = nested
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_invalid_utf8_is_integrity_not_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        stored_path(store, profile).write_bytes(b"\xff\xfe\x00")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))

    def test_noncanonical_valid_json_is_rejected_by_managed_store(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        path = stored_path(store, profile)
        document = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))


# ---------------------------------------------------------------------------
# Root behavior (PART 62)
# ---------------------------------------------------------------------------


class TestRootBehavior:
    def test_put_creates_directories(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        assert store.root.is_dir()

    def test_root_as_file_is_store_error(self, tmp_path: Path) -> None:
        root = tmp_path / "root-file"
        root.write_text("not a directory", encoding="utf-8")
        store = DirectoryCalibrationProfileStore(root)
        with pytest.raises(CalibrationProfileStoreError):
            store.put(fitted_profile())
        with pytest.raises(CalibrationProfileStoreError):
            store.get(profile_fingerprint="0" * 64, profile_fingerprint_version=1)


# ---------------------------------------------------------------------------
# Layout corruption classification (Phase 4C.6a)
# ---------------------------------------------------------------------------


class TestLayoutCorruptionClassification:
    def test_store_version_component_as_file_is_integrity_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        store.root.mkdir(parents=True)
        (store.root / f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}").write_text(
            "not a directory", encoding="utf-8"
        )
        profile = fitted_profile()
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.put(profile)

    def test_profile_version_component_as_file_is_integrity_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        version_directory = (
            store.root
            / f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}"
            / f"profile-fingerprint-v{CALIBRATION_PROFILE_FINGERPRINT_VERSION}"
        )
        version_directory.parent.mkdir(parents=True)
        version_directory.write_text("not a directory", encoding="utf-8")
        profile = fitted_profile()
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(**key_of(profile))
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.put(profile)

    def test_missing_root_is_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(profile_fingerprint="0" * 64, profile_fingerprint_version=1)
        assert not store.root.exists()

    def test_missing_target_in_valid_layout_is_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        stored_path(store, profile).parent.mkdir(parents=True)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(**key_of(profile))

    def test_missing_store_directory_is_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        store.root.mkdir(parents=True)
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(profile_fingerprint="0" * 64, profile_fingerprint_version=1)

    @pytest.mark.skipif(os.getuid() == 0, reason="root bypasses directory permissions")
    def test_unreadable_managed_directory_is_an_operational_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        version_directory = stored_path(store, profile).parent
        version_directory.chmod(0o000)
        try:
            with pytest.raises(CalibrationProfileStoreError) as error:
                store.get(**key_of(profile))
        finally:
            version_directory.chmod(0o755)
        assert not isinstance(error.value, CalibrationProfileNotFoundError)


# ---------------------------------------------------------------------------
# Semantic equivalence (PART 39, PART 40, PART 41, PART 42, PART 43)
# ---------------------------------------------------------------------------


class TestSemanticEquivalence:
    def test_offline_scores_are_identical(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        loaded = store.get(**key_of(profile))
        dataset = evaluation_dataset((choice_observation(),))
        original = apply_profile_to_evaluation_dataset(dataset, profile)
        restored = apply_profile_to_evaluation_dataset(dataset, loaded)
        assert [row.predicted_correctness for row in original.rows] == [
            row.predicted_correctness for row in restored.rows
        ]
        assert original.profile_fingerprint == restored.profile_fingerprint

    def test_runtime_application_is_identical(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        loaded = store.get(**key_of(profile))
        evaluation = runtime_evaluation()
        original = apply_profile_to_runtime_evaluation(evaluation, profile)
        restored = apply_profile_to_runtime_evaluation(evaluation, loaded)
        assert original.result.predicted_correctness == restored.result.predicted_correctness
        assert (
            original.result.calibration_profile_fingerprint
            == restored.result.calibration_profile_fingerprint
        )
        assert original.trace.execution_fingerprint == restored.trace.execution_fingerprint

    def test_scorer_binding_gate_survives_retrieval(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        loaded = store.get(**key_of(profile))
        other = choice_observation(backend=FakeCategoricalBackend(model="other-model"))
        with pytest.raises(InvalidDecisionError):
            predicted_winner_correctness(loaded, other)

    def test_offline_ground_truth_gate_survives_retrieval(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        loaded = store.get(**key_of(profile))
        other_semantics = choice_observation(
            resolved_truth("billing", labeling_rule="a different labeling rule")
        )
        dataset = evaluation_dataset((other_semantics,))
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_evaluation_dataset(dataset, loaded)


# ---------------------------------------------------------------------------
# Scope proof (PART 5, PART 6, PART 30, PART 34, PART 35, PART 36)
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    @pytest.mark.parametrize(
        "name",
        [
            "list",
            "list_profiles",
            "all",
            "find",
            "search",
            "query",
            "scan",
            "lookup_by_binding",
            "lookup_by_method",
            "latest",
            "best",
            "contains_compatible",
            "delete",
            "update",
            "replace",
            "rename",
        ],
    )
    def test_no_enumeration_or_mutation_api(self, tmp_path: Path, name: str) -> None:
        store = store_at(tmp_path)
        assert not hasattr(store, name)

    def test_no_manifest_or_index_files(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        store.put(fitted_profile())
        names = {path.name for path in store.root.rglob("*")}
        assert not any(name in {"manifest.json", "index.json", "profiles.json"} for name in names)

    def test_runtime_provenance_carries_no_store_identity(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        profile = fitted_profile()
        store.put(profile)
        loaded = store.get(**key_of(profile))
        evaluation = runtime_evaluation()
        applied = apply_profile_to_runtime_evaluation(evaluation, loaded)
        field_names = set(applied.result.__dataclass_fields__)
        assert not any("store" in name or "serialization" in name for name in field_names)
        assert applied.result.calibration_profile_fingerprint == profile.fingerprint
        assert (
            applied.result.calibration_profile_fingerprint_version
            == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        )
