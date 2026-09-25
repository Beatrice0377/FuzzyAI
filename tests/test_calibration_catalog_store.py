"""Exact content-addressed directory store for calibration profile catalogs.

The central contract: retrieval requires an exact catalog snapshot fingerprint
and fingerprint schema version, derives exactly one path, and restores the
artifact through the identity-verified catalog loader with the requested
identity supplied as an independent expected pin. Placing a different but fully
self-consistent snapshot at the requested path is rejected. There is no
enumeration, lifecycle, or fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from test_calibration_catalog_serialization import (
    catalog_of,
    eligible_profile,
    other_binding_profile,
    other_semantics_profile,
    other_training_profile,
    runtime_evaluation,
    unsupported_method_profile,
)

from probvenance import (
    CalibrationProfileCatalogNotFoundError,
    CalibrationProfileCatalogStoreError,
    CalibrationProfileCatalogStoreIntegrityError,
    InvalidDecisionError,
)
from probvenance.calibration_catalog import (
    CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
    CalibrationProfileCatalog,
    discover_calibration_profile_references_for_runtime,
    load_calibration_profile_catalog,
    serialize_calibration_profile_catalog,
)
from probvenance.calibration_catalog_store import (
    CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION,
    DirectoryCalibrationProfileCatalogStore,
)
from probvenance.calibration_store import DirectoryCalibrationProfileStore

CATALOG_FP_VERSION = CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION


def store_at(tmp_path: Path) -> DirectoryCalibrationProfileCatalogStore:
    return DirectoryCalibrationProfileCatalogStore(tmp_path / "catalogs")


def key_of(catalog: CalibrationProfileCatalog) -> dict[str, Any]:
    return {
        "catalog_fingerprint": catalog.fingerprint,
        "catalog_fingerprint_version": CATALOG_FP_VERSION,
    }


def stored_path(store: DirectoryCalibrationProfileCatalogStore, catalog: Any) -> Path:
    return (
        store.root
        / f"catalog-store-v{CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION}"
        / f"catalog-fingerprint-v{CATALOG_FP_VERSION}"
        / f"{catalog.fingerprint}.json"
    )


def discover(catalog: CalibrationProfileCatalog, **declared: Any) -> Any:
    return discover_calibration_profile_references_for_runtime(
        runtime_evaluation(), catalog, **declared
    )


def write_at(store: DirectoryCalibrationProfileCatalogStore, catalog: Any, text: str) -> Path:
    target = stored_path(store, catalog)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Layout and identity
# ---------------------------------------------------------------------------


class TestLayoutAndIdentity:
    def test_layout_version_is_one(self) -> None:
        assert CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION == 1

    def test_path_derives_only_from_store_version_fingerprint_version_and_fingerprint(
        self, tmp_path: Path
    ) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        expected = (
            tmp_path
            / "catalogs"
            / "catalog-store-v1"
            / "catalog-fingerprint-v1"
            / f"{catalog.fingerprint}.json"
        )
        assert stored_path(store, catalog) == expected

    def test_root_must_be_str_or_pathlike(self) -> None:
        with pytest.raises(InvalidDecisionError):
            DirectoryCalibrationProfileCatalogStore(123)  # type: ignore[arg-type]

    def test_root_property_is_the_supplied_root(self, tmp_path: Path) -> None:
        store = DirectoryCalibrationProfileCatalogStore(tmp_path / "x")
        assert store.root == tmp_path / "x"


# ---------------------------------------------------------------------------
# Key validation (must fail before filesystem access)
# ---------------------------------------------------------------------------


class TestKeyValidation:
    @pytest.mark.parametrize(
        "bad_fingerprint",
        [
            "../x",
            "/absolute/path",
            "a/b",
            "a\\b",
            "A" * 64,
            "abcdef",
            "a" * 65,
            " " + "a" * 63,
            "a" * 63 + " ",
            "",
            "g" * 64,
        ],
    )
    def test_malformed_catalog_fingerprint_is_rejected(
        self, tmp_path: Path, bad_fingerprint: str
    ) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError):
            store.get(
                catalog_fingerprint=bad_fingerprint,
                catalog_fingerprint_version=CATALOG_FP_VERSION,
            )

    @pytest.mark.parametrize("bad_version", [True, False, 1.0, "1", 0, -1, None])
    def test_malformed_catalog_fingerprint_version_is_rejected(
        self, tmp_path: Path, bad_version: Any
    ) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError):
            store.get(
                catalog_fingerprint="a" * 64,
                catalog_fingerprint_version=bad_version,
            )

    def test_malformed_key_does_not_reach_the_filesystem(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError):
            store.get(catalog_fingerprint="../escape", catalog_fingerprint_version=1)
        assert not store.root.exists()

    def test_unsupported_fingerprint_version_is_not_a_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        with pytest.raises(InvalidDecisionError):
            store.get(
                catalog_fingerprint=catalog.fingerprint,
                catalog_fingerprint_version=CATALOG_FP_VERSION + 1,
            )

    def test_unsupported_fingerprint_version_message_states_the_support(self) -> None:
        store = DirectoryCalibrationProfileCatalogStore("unused")
        with pytest.raises(InvalidDecisionError) as error:
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=7)
        assert "7" in str(error.value)
        assert "1" in str(error.value)


# ---------------------------------------------------------------------------
# Put semantics
# ---------------------------------------------------------------------------


class TestPutSemantics:
    def test_put_requires_a_real_catalog(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError):
            store.put("not-a-catalog")  # type: ignore[arg-type]

    def test_put_stores_exactly_the_catalog_serializer_output(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        store.put(catalog)
        assert stored_path(store, catalog).read_text(encoding="utf-8") == (
            serialize_calibration_profile_catalog(catalog)
        )

    def test_put_stores_utf8_without_bom_or_trailing_newline(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        raw = stored_path(store, catalog).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert not raw.endswith(b"\n")

    def test_put_is_idempotent(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        before = stored_path(store, catalog).read_bytes()
        store.put(catalog)
        assert stored_path(store, catalog).read_bytes() == before

    def test_idempotent_put_leaves_exactly_one_artifact(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        store.put(catalog)
        artifacts = list(
            (store.root / "catalog-store-v1" / "catalog-fingerprint-v1").glob("*.json")
        )
        assert len(artifacts) == 1

    def test_corrupt_existing_target_is_not_overwritten(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        target = write_at(store, catalog, "{not json")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.put(catalog)
        assert target.read_text(encoding="utf-8") == "{not json"

    def test_substituted_existing_target_is_not_overwritten(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        other = catalog_of(other_binding_profile())
        target = write_at(store, catalog, serialize_calibration_profile_catalog(other))
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.put(catalog)
        assert target.read_text(encoding="utf-8") == serialize_calibration_profile_catalog(other)

    def test_put_leaves_no_temporary_files(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        store.put(catalog_of(eligible_profile()))
        directory = store.root / "catalog-store-v1" / "catalog-fingerprint-v1"
        assert not [path for path in directory.iterdir() if path.name.startswith(".tmp")]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_exact_round_trip(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        store.put(catalog)
        loaded = store.get(**key_of(catalog))
        assert loaded.fingerprint == catalog.fingerprint
        assert loaded.canonical_payload() == catalog.canonical_payload()
        assert loaded.references == catalog.references
        assert serialize_calibration_profile_catalog(loaded) == (
            serialize_calibration_profile_catalog(catalog)
        )

    def test_empty_catalog_round_trip(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = CalibrationProfileCatalog.from_profiles(())
        store.put(catalog)
        loaded = store.get(**key_of(catalog))
        assert loaded.fingerprint == catalog.fingerprint
        assert loaded.canonical_payload() == catalog.canonical_payload()
        assert loaded.references == ()
        assert discover(loaded) == ()

    def test_retrieved_bytes_equal_original_canonical_text(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(other_training_profile())
        store.put(catalog)
        store.get(**key_of(catalog))
        assert stored_path(store, catalog).read_text(encoding="utf-8") == (
            serialize_calibration_profile_catalog(catalog)
        )

    def test_two_distinct_snapshots_remain_distinct(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        first = catalog_of(eligible_profile())
        second = catalog_of(other_binding_profile(), other_training_profile())
        assert first.fingerprint != second.fingerprint
        assert stored_path(store, first) != stored_path(store, second)
        store.put(first)
        store.put(second)
        assert store.get(**key_of(first)).references == first.references
        assert store.get(**key_of(second)).references == second.references

    def test_repeated_get_is_deterministic(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        store.put(catalog)
        first = store.get(**key_of(catalog))
        second = store.get(**key_of(catalog))
        assert first.canonical_payload() == second.canonical_payload()


# ---------------------------------------------------------------------------
# Tamper matrix
# ---------------------------------------------------------------------------


class TestTamperMatrix:
    def test_different_valid_catalog_at_exact_path_is_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        other = catalog_of(other_binding_profile())
        write_at(store, catalog, serialize_calibration_profile_catalog(other))
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_fully_recomputed_snapshot_at_exact_path_is_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        malicious = catalog_of(other_training_profile(), other_binding_profile())
        text = serialize_calibration_profile_catalog(malicious)
        assert load_calibration_profile_catalog(text).fingerprint == malicious.fingerprint
        write_at(store, catalog, text)
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_embedded_fingerprint_tamper_is_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        target = stored_path(store, catalog)
        document = json.loads(target.read_text(encoding="utf-8"))
        document["catalog_fingerprint"] = "b" * 64
        target.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_binding_metadata_tamper_is_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        target = stored_path(store, catalog)
        document = json.loads(target.read_text(encoding="utf-8"))
        entries = document["catalog_identity"]["entries"]
        assert entries
        entries[0]["binding"]["model"] = "tampered-model"
        target.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_reference_tamper_is_rejected(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        store.put(catalog)
        target = stored_path(store, catalog)
        document = json.loads(target.read_text(encoding="utf-8"))
        document["catalog_identity"]["entries"][0]["profile_fingerprint"] = "c" * 64
        target.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_truncated_json_is_integrity_not_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        canonical = serialize_calibration_profile_catalog(catalog)
        stored_path(store, catalog).write_text(canonical[: len(canonical) // 2], encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_invalid_utf8_is_integrity_not_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        stored_path(store, catalog).write_bytes(b"\xff\xfe\x00broken")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_noncanonical_valid_json_is_rejected_by_the_store(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        canonical = serialize_calibration_profile_catalog(catalog)
        pretty = json.dumps(json.loads(canonical), indent=2)
        assert load_calibration_profile_catalog(pretty).fingerprint == catalog.fingerprint
        write_at(store, catalog, pretty)
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_deeply_nested_document_is_integrity_not_recursion_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        nested = '{"a": ' * 900 + "1" + "}" * 900
        document = (
            '{"artifact_type": "'
            + "a" * 64
            + '", "serialization_version": 1, "catalog_fingerprint": "'
            + "a" * 64
            + '", "catalog_identity": {"v": 1, "entries": [], "rogue": '
            + nested
            + "}}"
        )
        stored_path(store, catalog).write_text(document, encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_huge_integer_is_integrity_not_value_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        document = (
            '{"artifact_type": "x", "serialization_version": 1, "catalog_fingerprint": "'
            + "a" * 64
            + '", "catalog_identity": {"v": '
            + "9" * 6000
            + ', "entries": []}}'
        )
        stored_path(store, catalog).write_text(document, encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))

    def test_invalid_document_never_leaks_a_raw_exception(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        stored_path(store, catalog).write_text(
            json.dumps({"artifact_type": "x", "serialization_version": 1}), encoding="utf-8"
        )
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(**key_of(catalog))


class TestBoundedErrorMessages:
    def test_rogue_artifact_type_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        document = json.dumps(
            {
                "artifact_type": "a" * 1_000_000,
                "serialization_version": 1,
                "catalog_fingerprint": "a" * 64,
                "catalog_identity": {"v": 1, "entries": []},
            }
        )
        stored_path(store, catalog).write_text(document, encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError) as error:
            store.get(**key_of(catalog))
        assert len(str(error.value)) < 2000

    def test_huge_caller_fingerprint_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError) as error:
            store.get(catalog_fingerprint="z" * 1_000_000, catalog_fingerprint_version=1)
        assert len(str(error.value)) < 2000

    def test_rogue_document_key_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        document = json.dumps(
            {
                "artifact_type": "probvenance.calibration-profile-catalog",
                "serialization_version": 1,
                "catalog_fingerprint": "a" * 64,
                "catalog_identity": {"v": 1, "entries": []},
                "r" * 1_000_000: 1,
            }
        )
        stored_path(store, catalog).write_text(document, encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError) as error:
            store.get(**key_of(catalog))
        assert len(str(error.value)) < 2000

    def test_duplicate_huge_key_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        huge_key = "k" * 1_000_000
        document = f'{{"{huge_key}": 1, "{huge_key}": 2}}'
        stored_path(store, catalog).write_text(document, encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError) as error:
            store.get(**key_of(catalog))
        assert len(str(error.value)) < 2000

    def test_huge_number_literal_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        document = '{"x": ' + ("9" * 1_000_000) + ".0}"
        stored_path(store, catalog).write_text(document, encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError) as error:
            store.get(**key_of(catalog))
        assert len(str(error.value)) < 2000

    def test_huge_caller_version_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        with pytest.raises(InvalidDecisionError) as error:
            store.get(
                catalog_fingerprint=catalog.fingerprint,
                catalog_fingerprint_version=10**100_000,
            )
        assert len(str(error.value)) < 2000

    def test_huge_negative_caller_version_keeps_the_message_bounded(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(InvalidDecisionError) as error:
            store.get(
                catalog_fingerprint="a" * 64,
                catalog_fingerprint_version=-(10**100_000),
            )
        assert len(str(error.value)) < 2000


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_missing_exact_snapshot_is_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(CalibrationProfileCatalogNotFoundError):
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=1)

    def test_missing_get_does_not_create_the_root(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(CalibrationProfileCatalogNotFoundError):
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=1)
        assert not store.root.exists()

    def test_absent_version_directory_is_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        other = catalog_of(other_binding_profile())
        with pytest.raises(CalibrationProfileCatalogNotFoundError):
            store.get(**key_of(other))

    def test_root_as_file_is_an_operational_error(self, tmp_path: Path) -> None:
        root = tmp_path / "rootfile"
        root.write_text("x", encoding="utf-8")
        store = DirectoryCalibrationProfileCatalogStore(root)
        with pytest.raises(CalibrationProfileCatalogStoreError):
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=1)

    def test_store_layout_component_as_file_is_integrity(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        layout = store.root / "catalog-store-v1"
        layout.parent.mkdir(parents=True, exist_ok=True)
        layout.write_text("x", encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=1)

    def test_fingerprint_version_component_as_file_is_integrity(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        version_dir = store.root / "catalog-store-v1" / "catalog-fingerprint-v1"
        version_dir.mkdir(parents=True, exist_ok=True)
        version_dir.rmdir()
        version_dir.write_text("x", encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=1)

    def test_layout_corruption_is_not_masked_as_not_found(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        layout = store.root / "catalog-store-v1"
        layout.parent.mkdir(parents=True, exist_ok=True)
        layout.write_text("x", encoding="utf-8")
        with pytest.raises(CalibrationProfileCatalogStoreIntegrityError):
            store.get(catalog_fingerprint="a" * 64, catalog_fingerprint_version=1)

    @pytest.mark.skipif(os.getuid() == 0, reason="root bypasses directory permissions")
    def test_unreadable_managed_directory_is_an_operational_error(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        version_dir = stored_path(store, catalog).parent
        version_dir.chmod(0o000)
        try:
            with pytest.raises(CalibrationProfileCatalogStoreError) as error:
                store.get(**key_of(catalog))
        finally:
            version_dir.chmod(0o755)
        assert not isinstance(error.value, CalibrationProfileCatalogNotFoundError)


# ---------------------------------------------------------------------------
# Discovery equivalence and multiplicity
# ---------------------------------------------------------------------------


class TestDiscoveryEquivalence:
    def test_discovery_survives_store_round_trip(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        store.put(catalog)
        loaded = store.get(**key_of(catalog))
        for declared in ({}, {"task_id": "support-routing"}, {"taxonomy_id": "support"}):
            assert discover(loaded, **declared) == discover(catalog, **declared)

    def test_zero_one_and_many_cardinality_survive_store_round_trip(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        zero = catalog_of(other_binding_profile())
        one = catalog_of(eligible_profile())
        many = catalog_of(eligible_profile(), other_semantics_profile())
        for catalog in (zero, one, many):
            store.put(catalog)
        assert discover(store.get(**key_of(zero))) == ()
        assert len(discover(store.get(**key_of(one)))) == 1
        assert len(discover(store.get(**key_of(many)))) == 2

    def test_ground_truth_distinct_references_survive(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_semantics_profile())
        store.put(catalog)
        references = discover(store.get(**key_of(catalog)))
        assert len(references) == 2
        assert len({reference.profile_fingerprint for reference in references}) == 2

    def test_training_distinct_references_survive(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile(), other_training_profile())
        store.put(catalog)
        assert len(discover(store.get(**key_of(catalog)))) == 2

    def test_unsupported_method_reference_survives(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(unsupported_method_profile())
        store.put(catalog)
        references = discover(store.get(**key_of(catalog)))
        assert len(references) == 1
        assert references[0].profile_fingerprint == unsupported_method_profile().fingerprint


# ---------------------------------------------------------------------------
# Trust boundary and independence
# ---------------------------------------------------------------------------


class TestTrustBoundary:
    def test_catalog_may_reference_profiles_absent_from_the_profile_store(
        self, tmp_path: Path
    ) -> None:
        store = store_at(tmp_path)
        catalog = catalog_of(eligible_profile())
        store.put(catalog)
        assert store.get(**key_of(catalog)).references == catalog.references

    def test_retrieval_does_not_attest_profile_existence(self, tmp_path: Path) -> None:
        catalog_store = store_at(tmp_path)
        profile_store = DirectoryCalibrationProfileStore(tmp_path / "profiles")
        catalog = catalog_of(eligible_profile())
        catalog_store.put(catalog)
        reference = discover(catalog_store.get(**key_of(catalog)))[0]
        with pytest.raises(Exception) as error:
            profile_store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )
        assert type(error.value).__name__ == "CalibrationProfileNotFoundError"

    def test_profile_store_put_does_not_create_catalogs(self, tmp_path: Path) -> None:
        profile_store = DirectoryCalibrationProfileStore(tmp_path / "shared")
        profile_store.put(eligible_profile())
        assert not (tmp_path / "shared" / "catalog-store-v1").exists()

    def test_catalog_put_does_not_populate_the_profile_store(self, tmp_path: Path) -> None:
        catalog_store = DirectoryCalibrationProfileCatalogStore(tmp_path / "shared")
        catalog_store.put(catalog_of(eligible_profile()))
        assert not (tmp_path / "shared" / "store-v1").exists()

    def test_catalog_store_has_no_profile_store_dependency(self, tmp_path: Path) -> None:
        store = DirectoryCalibrationProfileCatalogStore(tmp_path / "root")
        with pytest.raises(TypeError):
            DirectoryCalibrationProfileCatalogStore(  # type: ignore[call-arg]
                tmp_path / "root", profile_store=None
            )
        assert store.root == tmp_path / "root"


class TestSharedRootCoexistence:
    def test_profile_and_catalog_stores_share_one_root(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        profile_store = DirectoryCalibrationProfileStore(shared)
        catalog_store = DirectoryCalibrationProfileCatalogStore(shared)
        profile = eligible_profile()
        catalog = catalog_of(profile)
        profile_store.put(profile)
        catalog_store.put(catalog)
        assert (
            profile_store.get(
                profile_fingerprint=profile.fingerprint,
                profile_fingerprint_version=1,
            ).fingerprint
            == profile.fingerprint
        )
        assert catalog_store.get(**key_of(catalog)).fingerprint == catalog.fingerprint

    def test_stores_do_not_interpret_each_others_files(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        profile_store = DirectoryCalibrationProfileStore(shared)
        catalog_store = DirectoryCalibrationProfileCatalogStore(shared)
        profile = eligible_profile()
        catalog = catalog_of(profile)
        profile_store.put(profile)
        catalog_store.put(catalog)
        assert (shared / "store-v1").is_dir()
        assert (shared / "catalog-store-v1").is_dir()
        loaded_catalog = catalog_store.get(**key_of(catalog))
        assert loaded_catalog.references[0].profile_fingerprint == profile.fingerprint


# ---------------------------------------------------------------------------
# Scope boundaries
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    @pytest.mark.parametrize(
        "forbidden",
        ["list", "list_catalogs", "scan", "find", "search", "query", "all"],
    )
    def test_no_enumeration_api(self, forbidden: str) -> None:
        assert not hasattr(DirectoryCalibrationProfileCatalogStore, forbidden)

    @pytest.mark.parametrize(
        "forbidden",
        ["latest", "active", "default", "activate", "deactivate", "promote", "supersede"],
    )
    def test_no_lifecycle_api(self, forbidden: str) -> None:
        assert not hasattr(DirectoryCalibrationProfileCatalogStore, forbidden)

    @pytest.mark.parametrize(
        "forbidden",
        ["delete", "update", "replace", "rename", "alias", "put_named", "get_by_name"],
    )
    def test_no_mutation_or_naming_api(self, forbidden: str) -> None:
        assert not hasattr(DirectoryCalibrationProfileCatalogStore, forbidden)

    def test_get_accepts_no_name_or_environment_argument(self, tmp_path: Path) -> None:
        store = store_at(tmp_path)
        with pytest.raises(TypeError):
            store.get(  # type: ignore[call-arg]
                catalog_fingerprint="a" * 64,
                catalog_fingerprint_version=1,
                name="prod",
            )

    def test_module_does_not_import_the_profile_store(self) -> None:
        import probvenance.calibration_catalog_store as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "from probvenance.calibration_store import" not in source
        assert "calibration_selection" not in source
