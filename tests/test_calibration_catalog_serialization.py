"""Contract tests for catalog snapshot identity, serialization, and loading.

A persisted catalog is a discovery snapshot, not an authorization artifact. Its
fingerprint proves the snapshot is internally self-consistent; it does NOT prove
that a referenced profile exists, that a store artifact is intact, that the
snapshot metadata still matches the real profile, or that any profile is
selectable. These tests pin that boundary and prove the downstream store and
selector remain the authorization path.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
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
from probvenance._strict_json import require_bounded_nesting
from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationDataset,
    CalibrationProfile,
    fit_l2_logistic_selected_probability,
)
from probvenance.calibration_catalog import (
    CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
    CALIBRATION_PROFILE_CATALOG_SERIALIZATION_TYPE,
    CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION,
    CalibrationProfileCatalog,
    CalibrationProfileReference,
    discover_calibration_profile_references_for_runtime,
    load_calibration_profile_catalog,
    serialize_calibration_profile_catalog,
)
from probvenance.calibration_selection import select_calibration_profile_for_runtime
from probvenance.calibration_store import DirectoryCalibrationProfileStore
from probvenance.errors import (
    CalibrationProfileNotFoundError,
    CalibrationProfileStoreIntegrityError,
    NoEligibleCalibrationProfileError,
)
from probvenance.fingerprint import canonical_json, fingerprint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def runtime_evaluation() -> Any:
    runtime, _ = make_choice_runtime()
    return runtime.evaluate_with_trace(make_choice_decision())


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


def unsupported_method_profile() -> CalibrationProfile:
    return forge_profile(eligible_profile(), method_id="some-future-method")


def catalog_of(*profiles: CalibrationProfile) -> CalibrationProfileCatalog:
    return CalibrationProfileCatalog.from_profiles(tuple(profiles))


def serialize(catalog: CalibrationProfileCatalog) -> str:
    return serialize_calibration_profile_catalog(catalog)


def document_of(catalog: CalibrationProfileCatalog) -> dict[str, Any]:
    return json.loads(serialize(catalog))


def plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def reload_document(document: dict[str, Any]) -> CalibrationProfileCatalog:
    return load_calibration_profile_catalog(json.dumps(document))


def assert_document_rejected(document: dict[str, Any]) -> None:
    with pytest.raises(InvalidDecisionError):
        reload_document(document)


def deep_value(value: Any, levels: int) -> Any:
    for _ in range(levels):
        value = {"n": value}
    return value


def tampered(mutate: Any) -> dict[str, Any]:
    document = document_of(catalog_of(eligible_profile()))
    mutate(document)
    return document


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


CATALOG_ENTRY_KEYS = {
    "profile_fingerprint",
    "profile_fingerprint_version",
    "binding",
    "target_id",
    "target_version",
    "input_score_id",
    "input_score_version",
}

BINDING_KEYS = {
    "probability_formulation_fingerprint",
    "probability_formulation_fingerprint_version",
    "model",
    "model_revision",
    "tokenizer",
    "tokenizer_revision",
    "rendering_semantics",
    "task_id",
    "domain_id",
    "taxonomy_id",
    "taxonomy_version",
}


# ---------------------------------------------------------------------------
# Catalog identity (PART 2, PART 6, PART 7, PART 8, PART 10)
# ---------------------------------------------------------------------------


class TestCatalogIdentity:
    def test_catalog_version_constants_are_v1(self) -> None:
        assert CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION == 1
        assert CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION == 1
        assert CALIBRATION_PROFILE_CATALOG_SERIALIZATION_TYPE == (
            "probvenance.calibration-profile-catalog"
        )

    def test_payload_shape_commits_reference_binding_and_companion_fields(self) -> None:
        profile = eligible_profile()
        payload = catalog_of(profile).canonical_payload()
        assert payload["v"] == CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION
        assert len(payload["entries"]) == 1
        entry = payload["entries"][0]
        assert set(entry) == CATALOG_ENTRY_KEYS
        assert entry["profile_fingerprint"] == profile.fingerprint
        assert entry["profile_fingerprint_version"] == CALIBRATION_PROFILE_FINGERPRINT_VERSION
        assert set(entry["binding"]) == BINDING_KEYS
        assert entry["binding"] == plain(profile.binding.canonical_payload())
        assert entry["target_id"] == profile.target_id
        assert entry["target_version"] == profile.target_version
        assert entry["input_score_id"] == profile.input_score_id
        assert entry["input_score_version"] == profile.input_score_version

    def test_fingerprint_is_the_canonical_payload_fingerprint(self) -> None:
        catalog = catalog_of(eligible_profile())
        assert catalog.fingerprint == fingerprint(catalog.canonical_payload())
        assert len(catalog.fingerprint) == 64
        assert catalog.fingerprint == catalog.fingerprint

    def test_identity_excludes_ground_truth_method_and_training_dimensions(self) -> None:
        payload = catalog_of(eligible_profile()).canonical_payload()
        entry = payload["entries"][0]
        for excluded in (
            "ground_truth_semantics",
            "labeling_rule",
            "ambiguity_policy",
            "method_id",
            "method_version",
            "method_configuration",
            "fitted_parameters",
            "slope",
            "intercept",
            "training_dataset_fingerprint",
            "profile_fingerprint_schema",
        ):
            assert excluded not in entry
        serialized = serialize(catalog_of(eligible_profile()))
        for excluded in ("method_id", "fitted_parameters", "slope", "labeling_rule"):
            assert excluded not in serialized

    def test_identity_does_not_change_a_profile_identity(self) -> None:
        profile = eligible_profile()
        before_fingerprint = profile.fingerprint
        before_payload = plain(profile.canonical_payload())
        catalog_of(profile, other_training_profile())
        assert profile.fingerprint == before_fingerprint
        assert plain(profile.canonical_payload()) == before_payload

    def test_empty_catalog_has_a_deterministic_identity(self) -> None:
        empty = catalog_of()
        assert empty.canonical_payload() == {
            "v": CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
            "entries": [],
        }
        assert empty.fingerprint == fingerprint(empty.canonical_payload())
        assert catalog_of().fingerprint == empty.fingerprint


# ---------------------------------------------------------------------------
# Snapshot semantics, not input-order semantics (PART 9)
# ---------------------------------------------------------------------------


class TestPermutationIdentity:
    def test_all_source_orders_produce_one_identity_and_one_json(self) -> None:
        profiles = (eligible_profile(), other_semantics_profile(), other_training_profile())
        baseline = catalog_of(*profiles)
        from itertools import permutations

        for ordering in permutations(profiles):
            candidate = catalog_of(*ordering)
            assert candidate.fingerprint == baseline.fingerprint
            assert serialize(candidate) == serialize(baseline)
            assert candidate.references == baseline.references


# ---------------------------------------------------------------------------
# Determinism and round-trip (PART 33, PART 34, PART 35, PART 36, PART 37)
# ---------------------------------------------------------------------------


class TestDeterminismAndRoundTrip:
    def test_serialize_is_deterministic(self) -> None:
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        assert serialize(catalog) == serialize(catalog)

    def test_serialize_load_serialize_is_an_exact_fixed_point(self) -> None:
        catalog = catalog_of(eligible_profile(), other_semantics_profile())
        first = serialize(catalog)
        loaded = load_calibration_profile_catalog(first)
        assert serialize(loaded) == first

    def test_loaded_catalog_preserves_identity_and_references(self) -> None:
        catalog = catalog_of(eligible_profile(), other_binding_profile())
        loaded = load_calibration_profile_catalog(serialize(catalog))
        assert loaded.fingerprint == catalog.fingerprint
        assert plain(loaded.canonical_payload()) == plain(catalog.canonical_payload())
        assert loaded.references == catalog.references

    def test_empty_catalog_round_trips(self) -> None:
        catalog = catalog_of()
        loaded = load_calibration_profile_catalog(serialize(catalog))
        assert loaded.fingerprint == catalog.fingerprint
        assert loaded.references == ()
        assert discover(runtime_evaluation(), loaded) == ()

    def test_noncanonical_json_is_normalized_on_reserialization(self) -> None:
        catalog = catalog_of(eligible_profile())
        document = document_of(catalog)
        spaced = json.dumps(document, indent=2)
        loaded = load_calibration_profile_catalog(spaced)
        assert serialize(loaded) == serialize(catalog)
        assert loaded.fingerprint == catalog.fingerprint


# ---------------------------------------------------------------------------
# Strict parsing (PART 22, PART 63)
# ---------------------------------------------------------------------------


class TestStrictParsing:
    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="duplicate object key"):
            load_calibration_profile_catalog('{"artifact_type": "a", "artifact_type": "b"}')

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_nonfinite_literals_are_rejected(self, literal: str) -> None:
        with pytest.raises(InvalidDecisionError, match="non-finite JSON number"):
            load_calibration_profile_catalog('{"artifact_type": ' + literal + "}")

    def test_overflowing_float_literal_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="non-finite JSON number"):
            load_calibration_profile_catalog('{"artifact_type": 1e9999}')

    def test_huge_integer_literal_is_rejected(self) -> None:
        text = '{"artifact_type": ' + "9" * 6000 + "}"
        with pytest.raises(InvalidDecisionError, match="not valid JSON"):
            load_calibration_profile_catalog(text)

    def test_deeply_nested_document_is_rejected(self) -> None:
        text = "[" * 2000 + "]" * 2000
        with pytest.raises(InvalidDecisionError, match="nested too deeply"):
            load_calibration_profile_catalog(text)

    def test_document_that_parses_but_nests_too_deeply_is_rejected(self) -> None:
        # A document can nest just deeply enough for json.loads to succeed and
        # still overflow the recursive freezing that binding restoration does
        # afterwards. That path used to leak a raw RecursionError, so pin the
        # bounded-depth rejection at a depth the parser itself accepts.
        document = document_of(catalog_of(eligible_profile()))
        document["catalog_identity"]["entries"][0]["binding"]["rendering_semantics"] = deep_value(
            "x", 495
        )
        assert_document_rejected(document)

    def test_nesting_within_the_bound_is_accepted(self) -> None:
        require_bounded_nesting(deep_value("x", 63), subject="test artifact")
        with pytest.raises(InvalidDecisionError, match="nested more than 64 levels"):
            require_bounded_nesting(deep_value("x", 64), subject="test artifact")

    def test_malformed_json_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="not valid JSON"):
            load_calibration_profile_catalog("{")

    def test_non_object_top_level_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="must be a JSON object"):
            load_calibration_profile_catalog("[]")

    def test_non_string_input_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="must be a str"):
            load_calibration_profile_catalog(b"{}")  # type: ignore[arg-type]

    def test_error_messages_never_echo_megabytes(self) -> None:
        huge = "x" * 1_000_000
        for document in (
            tampered(lambda d: d.__setitem__("artifact_type", huge)),
            tampered(lambda d: d.__setitem__("rogue", huge)),
            tampered(lambda d: d.__setitem__("catalog_fingerprint", huge)),
            tampered(lambda d: d["catalog_identity"].__setitem__("v", huge)),
        ):
            with pytest.raises(InvalidDecisionError) as error:
                reload_document(document)
            assert len(str(error.value)) < 2000


# ---------------------------------------------------------------------------
# Exact schema (PART 24, PART 25, PART 26)
# ---------------------------------------------------------------------------


class TestSchemaEnforcement:
    def test_missing_envelope_key_is_rejected(self) -> None:
        for key in (
            "artifact_type",
            "serialization_version",
            "catalog_fingerprint",
            "catalog_identity",
        ):
            document = tampered(lambda d, k=key: d.pop(k))
            assert_document_rejected(document)

    def test_unknown_envelope_key_is_rejected(self) -> None:
        assert_document_rejected(tampered(lambda d: d.__setitem__("extra", 1)))

    def test_unknown_identity_key_is_rejected(self) -> None:
        assert_document_rejected(tampered(lambda d: d["catalog_identity"].__setitem__("extra", 1)))

    def test_unknown_entry_key_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(lambda d: d["catalog_identity"]["entries"][0].__setitem__("extra", 1))
        )

    @pytest.mark.parametrize(
        "excluded_key",
        ["method_id", "fitted_parameters", "training_dataset_fingerprint", "labeling_rule"],
    )
    def test_excluded_metadata_key_is_not_part_of_the_schema(self, excluded_key: str) -> None:
        assert_document_rejected(
            tampered(
                lambda d, k=excluded_key: d["catalog_identity"]["entries"][0].__setitem__(k, 1)
            )
        )

    def test_wrong_artifact_type_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(lambda d: d.__setitem__("artifact_type", "probvenance.calibration-profile"))
        )

    @pytest.mark.parametrize("version", [0, True, 1.0, "1", 2, None])
    def test_unsupported_serialization_version_is_rejected(self, version: Any) -> None:
        assert_document_rejected(
            tampered(lambda d: d.__setitem__("serialization_version", version))
        )

    def test_unsupported_catalog_identity_version_is_rejected(self) -> None:
        assert_document_rejected(tampered(lambda d: d["catalog_identity"].__setitem__("v", 2)))

    def test_entries_must_be_an_array(self) -> None:
        assert_document_rejected(
            tampered(lambda d: d["catalog_identity"].__setitem__("entries", {}))
        )

    def test_malformed_profile_fingerprint_is_rejected(self) -> None:
        for value in ("../x", "A" * 64, "a" * 63, "a" * 65, "", 12):
            assert_document_rejected(
                tampered(
                    lambda d, v=value: d["catalog_identity"]["entries"][0].__setitem__(
                        "profile_fingerprint", v
                    )
                )
            )

    def test_malformed_binding_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(lambda d: d["catalog_identity"]["entries"][0]["binding"].pop("model"))
        )
        assert_document_rejected(
            tampered(lambda d: d["catalog_identity"]["entries"][0].__setitem__("binding", {}))
        )

    def test_malformed_target_and_input_fields_are_rejected(self) -> None:
        for key, value in (
            ("target_id", ""),
            ("target_version", 0),
            ("target_version", True),
            ("input_score_id", ""),
            ("input_score_version", 0),
        ):
            assert_document_rejected(
                tampered(
                    lambda d, k=key, v=value: d["catalog_identity"]["entries"][0].__setitem__(k, v)
                )
            )


# ---------------------------------------------------------------------------
# Duplicate and ordering semantics (PART 38, PART 39)
# ---------------------------------------------------------------------------


class TestDuplicateAndOrdering:
    def test_duplicate_entry_identity_is_rejected(self) -> None:
        document = tampered(
            lambda d: d["catalog_identity"]["entries"].append(
                dict(d["catalog_identity"]["entries"][0])
            )
        )
        with pytest.raises(InvalidDecisionError, match="more than once"):
            reload_document(document)

    def test_out_of_order_entries_do_not_form_a_second_encoding(self) -> None:
        profiles = (eligible_profile(), other_semantics_profile())
        document = document_of(catalog_of(*profiles))
        canonical_entries = sorted(
            document["catalog_identity"]["entries"],
            key=lambda entry: (entry["profile_fingerprint_version"], entry["profile_fingerprint"]),
        )
        document["catalog_identity"]["entries"] = list(reversed(canonical_entries))
        document["catalog_fingerprint"] = fingerprint(document["catalog_identity"])
        with pytest.raises(InvalidDecisionError, match="does not match the serialized"):
            reload_document(document)


# ---------------------------------------------------------------------------
# Tamper detection (PART 40)
# ---------------------------------------------------------------------------


class TestTamperDetection:
    def test_binding_field_tamper_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(
                lambda d: d["catalog_identity"]["entries"][0]["binding"].__setitem__(
                    "model", "attacker-model"
                )
            )
        )

    def test_target_version_tamper_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(lambda d: d["catalog_identity"]["entries"][0].__setitem__("target_version", 2))
        )

    def test_input_score_version_tamper_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(
                lambda d: d["catalog_identity"]["entries"][0].__setitem__("input_score_version", 2)
            )
        )

    def test_profile_reference_tamper_is_rejected(self) -> None:
        assert_document_rejected(
            tampered(
                lambda d: d["catalog_identity"]["entries"][0].__setitem__(
                    "profile_fingerprint", "b" * 64
                )
            )
        )

    def test_embedded_catalog_fingerprint_is_not_trusted(self) -> None:
        assert_document_rejected(tampered(lambda d: d.__setitem__("catalog_fingerprint", "c" * 64)))


# ---------------------------------------------------------------------------
# Expected identity pin (PART 20, PART 21, PART 42)
# ---------------------------------------------------------------------------


class TestExpectedPin:
    def test_matching_pin_is_accepted(self) -> None:
        catalog = catalog_of(eligible_profile())
        loaded = load_calibration_profile_catalog(
            serialize(catalog),
            expected_catalog_fingerprint=catalog.fingerprint,
            expected_catalog_fingerprint_version=1,
        )
        assert loaded.fingerprint == catalog.fingerprint

    def test_wrong_expected_fingerprint_is_rejected(self) -> None:
        catalog = catalog_of(eligible_profile())
        with pytest.raises(InvalidDecisionError, match="does not match the expected"):
            load_calibration_profile_catalog(
                serialize(catalog),
                expected_catalog_fingerprint="d" * 64,
                expected_catalog_fingerprint_version=1,
            )

    def test_unsupported_expected_version_is_rejected(self) -> None:
        catalog = catalog_of(eligible_profile())
        with pytest.raises(InvalidDecisionError, match="not supported"):
            load_calibration_profile_catalog(
                serialize(catalog),
                expected_catalog_fingerprint=catalog.fingerprint,
                expected_catalog_fingerprint_version=2,
            )

    @pytest.mark.parametrize("version", [True, 1.0, "1"])
    def test_non_integer_expected_version_is_rejected(self, version: Any) -> None:
        catalog = catalog_of(eligible_profile())
        with pytest.raises(InvalidDecisionError, match="must be an integer"):
            load_calibration_profile_catalog(
                serialize(catalog),
                expected_catalog_fingerprint=catalog.fingerprint,
                expected_catalog_fingerprint_version=version,
            )

    def test_half_supplied_pin_is_rejected(self) -> None:
        catalog = catalog_of(eligible_profile())
        with pytest.raises(InvalidDecisionError, match="together or not at all"):
            load_calibration_profile_catalog(
                serialize(catalog), expected_catalog_fingerprint=catalog.fingerprint
            )
        with pytest.raises(InvalidDecisionError, match="together or not at all"):
            load_calibration_profile_catalog(
                serialize(catalog), expected_catalog_fingerprint_version=1
            )

    def test_pin_detects_a_fully_recomputed_snapshot(self) -> None:
        catalog = catalog_of(eligible_profile())
        trusted = catalog.fingerprint
        document = document_of(catalog)
        document["catalog_identity"]["entries"][0]["binding"]["model"] = "attacker-model"
        document["catalog_fingerprint"] = fingerprint(document["catalog_identity"])
        modified = json.dumps(document)
        assert load_calibration_profile_catalog(modified).fingerprint != trusted
        with pytest.raises(InvalidDecisionError, match="does not match the expected"):
            load_calibration_profile_catalog(
                modified,
                expected_catalog_fingerprint=trusted,
                expected_catalog_fingerprint_version=1,
            )


# ---------------------------------------------------------------------------
# Trust boundary and downstream authorization (PART 43-PART 49)
# ---------------------------------------------------------------------------


class TestTrustBoundary:
    def test_loaded_catalog_is_a_snapshot_claim_not_verified_profile_metadata(self) -> None:
        real_profile = other_binding_profile()
        document = document_of(catalog_of(real_profile))
        document["catalog_identity"]["entries"][0]["binding"] = plain(
            eligible_profile().binding.canonical_payload()
        )
        document["catalog_fingerprint"] = fingerprint(document["catalog_identity"])
        forged = reload_document(document)
        entry = forged.canonical_payload()["entries"][0]
        assert entry["binding"]["model"] == "fake-model"
        assert real_profile.binding.model == "other-model"
        assert forged.fingerprint == fingerprint(forged.canonical_payload())

    def test_forged_catalog_metadata_cannot_authorize_a_mismatched_profile(
        self, tmp_path: Any
    ) -> None:
        evaluation = runtime_evaluation()
        real_profile = other_binding_profile()
        document = document_of(catalog_of(real_profile))
        document["catalog_identity"]["entries"][0]["binding"] = plain(
            eligible_profile().binding.canonical_payload()
        )
        document["catalog_fingerprint"] = fingerprint(document["catalog_identity"])
        forged = reload_document(document)

        references = discover(evaluation, forged)
        assert len(references) == 1
        assert references[0].profile_fingerprint == real_profile.fingerprint

        store = DirectoryCalibrationProfileStore(tmp_path / "forged")
        store.put(real_profile)
        loaded = store.get(
            profile_fingerprint=references[0].profile_fingerprint,
            profile_fingerprint_version=references[0].profile_fingerprint_version,
        )
        assert loaded.fingerprint == real_profile.fingerprint
        assert loaded.binding.model == "other-model"

        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (loaded,))

    def test_missing_referenced_profile_remains_a_store_not_found(self, tmp_path: Any) -> None:
        evaluation = runtime_evaluation()
        catalog = load_calibration_profile_catalog(serialize(catalog_of(eligible_profile())))
        (reference,) = discover(evaluation, catalog)
        store = DirectoryCalibrationProfileStore(tmp_path / "absent")
        with pytest.raises(CalibrationProfileNotFoundError):
            store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )

    def test_corrupt_referenced_profile_remains_a_store_integrity_error(
        self, tmp_path: Any
    ) -> None:
        evaluation = runtime_evaluation()
        profile = eligible_profile()
        catalog = load_calibration_profile_catalog(serialize(catalog_of(profile)))
        (reference,) = discover(evaluation, catalog)
        store = DirectoryCalibrationProfileStore(tmp_path / "corrupt")
        store.put(profile)
        target = next(store.root.rglob("*.json"))
        target.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(CalibrationProfileStoreIntegrityError):
            store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )


# ---------------------------------------------------------------------------
# Discovery equivalence after persistence (PART 50, PART 51, PART 52, PART 53)
# ---------------------------------------------------------------------------


class TestDiscoveryEquivalence:
    def test_discovery_is_identical_before_and_after_serialization(self) -> None:
        evaluation = runtime_evaluation()
        second_evaluation = runtime_evaluation()
        profiles = (eligible_profile(), other_binding_profile())
        catalog = catalog_of(*profiles)
        loaded = load_calibration_profile_catalog(serialize(catalog))
        for declared in ({}, {"task_id": "support-routing"}, {"taxonomy_id": "support"}):
            assert discover(evaluation, loaded, **declared) == discover(
                second_evaluation, catalog, **declared
            )

    def test_zero_one_and_many_cardinality_survive_persistence(self) -> None:
        evaluation = runtime_evaluation()
        zero = load_calibration_profile_catalog(serialize(catalog_of(other_binding_profile())))
        one = load_calibration_profile_catalog(serialize(catalog_of(eligible_profile())))
        many = load_calibration_profile_catalog(
            serialize(catalog_of(eligible_profile(), other_semantics_profile()))
        )
        assert discover(evaluation, zero) == ()
        assert len(discover(evaluation, one)) == 1
        assert len(discover(evaluation, many)) == 2

    def test_ground_truth_multiplicity_survives_persistence(self) -> None:
        evaluation = runtime_evaluation()
        catalog = load_calibration_profile_catalog(
            serialize(catalog_of(eligible_profile(), other_semantics_profile()))
        )
        references = discover(evaluation, catalog)
        assert len(references) == 2
        assert len({reference.profile_fingerprint for reference in references}) == 2

    def test_training_multiplicity_survives_persistence(self) -> None:
        evaluation = runtime_evaluation()
        catalog = load_calibration_profile_catalog(
            serialize(catalog_of(eligible_profile(), other_training_profile()))
        )
        assert len(discover(evaluation, catalog)) == 2

    def test_unsupported_method_reference_survives_persistence(self) -> None:
        evaluation = runtime_evaluation()
        catalog = load_calibration_profile_catalog(
            serialize(catalog_of(unsupported_method_profile()))
        )
        references = discover(evaluation, catalog)
        assert len(references) == 1
        assert references[0].profile_fingerprint == unsupported_method_profile().fingerprint

    def test_multiple_eligible_profiles_remain_selector_ambiguity(self, tmp_path: Any) -> None:
        evaluation = runtime_evaluation()
        first = eligible_profile()
        second = other_semantics_profile()
        catalog = load_calibration_profile_catalog(serialize(catalog_of(first, second)))
        store = DirectoryCalibrationProfileStore(tmp_path / "ambiguous")
        for profile in (first, second):
            store.put(profile)
        profiles = tuple(
            store.get(
                profile_fingerprint=reference.profile_fingerprint,
                profile_fingerprint_version=reference.profile_fingerprint_version,
            )
            for reference in discover(evaluation, catalog)
        )
        assert len(profiles) == 2
        from probvenance.errors import AmbiguousCalibrationProfileSelectionError

        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(evaluation, profiles)

    def test_unique_case_composes_all_four_layers(self, tmp_path: Any) -> None:
        evaluation = runtime_evaluation()
        eligible = eligible_profile()
        catalog = load_calibration_profile_catalog(
            serialize(catalog_of(eligible, other_binding_profile()))
        )
        store = DirectoryCalibrationProfileStore(tmp_path / "unique")
        store.put(eligible)
        (reference,) = discover(evaluation, catalog)
        loaded = store.get(
            profile_fingerprint=reference.profile_fingerprint,
            profile_fingerprint_version=reference.profile_fingerprint_version,
        )
        selected = select_calibration_profile_for_runtime(evaluation, (loaded,))
        from probvenance.calibration import apply_profile_to_runtime_evaluation

        calibrated = apply_profile_to_runtime_evaluation(evaluation, selected)
        assert calibrated.result.calibrated is True
        assert calibrated.result.predicted_correctness is not None


# ---------------------------------------------------------------------------
# Scope boundaries (PART 13, PART 15, PART 16, PART 54-PART 62)
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    def test_no_filesystem_or_store_api_exists(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        for absent in (
            "save_calibration_profile_catalog",
            "load_calibration_profile_catalog_from_path",
            "DirectoryCalibrationProfileCatalogStore",
            "CalibrationProfileCatalogStore",
            "register_calibration_profile",
            "lookup_calibration_profile",
            "select_from_catalog_json",
        ):
            assert not hasattr(catalog_module, absent), absent

    def test_loader_does_not_access_a_store_or_the_filesystem(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in (
            "calibration_store",
            "DirectoryCalibrationProfileStore",
            "load_calibration_profile(",
            "environ",
            "getenv",
            "expanduser",
            "Path(",
            "open(",
            "listdir",
            "rglob",
            "glob",
            "hmac",
            "sign",
            "secret",
            "keyring",
        ):
            assert forbidden not in code, forbidden

    def test_serialized_catalog_contains_no_profile_state(self) -> None:
        profiles = (eligible_profile(), other_semantics_profile())
        serialized = serialize(catalog_of(*profiles))
        for forbidden in (
            "fitted_parameters",
            "slope",
            "intercept",
            "method_configuration",
            "labeling_rule",
            "training_dataset_fingerprint",
            "observations",
        ):
            assert forbidden not in serialized, forbidden

    def test_no_registry_ranking_or_policy_vocabulary_in_source(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in (
            "registry",
            "ranking",
            "latest",
            "default_profile",
            "priority",
            "quality_score",
            "staging",
            "CALIBRATION_PROFILE_SELECTION_VERSION",
        ):
            assert forbidden not in code, forbidden

    def test_no_new_dependency(self) -> None:
        import probvenance.calibration_catalog as catalog_module

        code = module_code_without_docstrings(catalog_module)
        for forbidden in ("import numpy", "import torch", "import requests", "import httpx"):
            assert forbidden not in code, forbidden

    def test_catalog_serialization_names_are_not_exported_at_package_root(self) -> None:
        import probvenance

        for absent in (
            "serialize_calibration_profile_catalog",
            "load_calibration_profile_catalog",
            "CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION",
        ):
            assert not hasattr(probvenance, absent), absent

    def test_reference_type_is_unchanged(self) -> None:
        profile = eligible_profile()
        (reference,) = catalog_of(profile).references
        assert isinstance(reference, CalibrationProfileReference)
        assert reference.profile_fingerprint == profile.fingerprint
        assert reference.profile_fingerprint_version == CALIBRATION_PROFILE_FINGERPRINT_VERSION
