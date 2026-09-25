"""Explicit calibration profile catalog and non-authoritative runtime discovery.

This module implements the first supported profile catalog: an immutable,
in-memory discovery index of known exact profile identities.

The architectural chain is deliberately explicit and layered:

    CalibrationProfile objects
        -> explicit catalog construction
    CalibrationProfileCatalog
        -> runtime discovery
    exact CalibrationProfileReference values
        -> caller performs exact store.get(...)
    CalibrationProfile objects
        -> explicit Phase 4C.7 selection
    unique eligible CalibrationProfile or fail closed
        -> explicit runtime application
    calibrated runtime Evaluation

Discovery proposes candidate identities. Selection authorizes one loaded
profile. Application applies that exact profile. The catalog is therefore not a
second selector: it never ranks, prefers, authorizes, loads, or applies a
profile, and it never accesses a store, the filesystem, the environment, or a
registry.

A discovered reference is not authorization to apply the referenced profile.
Catalog metadata is discovery-index metadata, not runtime authorization
provenance, and may be stale relative to an external store. The Phase 4C.7
selector re-validates the actual loaded profiles, so catalog discovery never
weakens selection.

Dependency direction: this module depends on :mod:`probvenance.calibration` for
the profile, binding, and shared eligibility projection, and on the runtime and
error foundations. The calibration, runtime, results, trace, and store modules
never import this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from probvenance._strict_json import parse_strict_json_object
from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationBinding,
    CalibrationProfile,
    _abbreviate,
    _require_exact_keys,
    _require_non_empty_str,
    _require_profile_fingerprint_identity,
    _require_uncalibrated_runtime_evaluation,
    _restore_calibration_binding,
    _runtime_calibration_binding,
    _runtime_profile_semantics_are_eligible,
)
from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint
from probvenance.runtime import Evaluation

_OPERATION = "calibration profile discovery"

#: Identity version of the exact catalog discovery snapshot. This versions the
#: fingerprint payload shape, not the wire representation, and is independent of
#: the profile, binding, and ground-truth-semantics fingerprint versions.
CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION = 1

#: Version of the catalog JSON wire representation. Deliberately independent of
#: the catalog fingerprint version: one versions the identity schema, the other
#: versions its serialized form.
CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION = 1

CALIBRATION_PROFILE_CATALOG_SERIALIZATION_TYPE = "probvenance.calibration-profile-catalog"

_CATALOG_ENVELOPE_KEYS = frozenset(
    {
        "artifact_type",
        "serialization_version",
        "catalog_fingerprint",
        "catalog_identity",
    }
)

_CATALOG_IDENTITY_KEYS = frozenset({"v", "entries"})

_CATALOG_ENTRY_KEYS = frozenset(
    {
        "profile_fingerprint",
        "profile_fingerprint_version",
        "binding",
        "target_id",
        "target_version",
        "input_score_id",
        "input_score_version",
    }
)


@dataclass(frozen=True, slots=True)
class CalibrationProfileReference:
    """An exact reference to one calibration profile artifact.

    The reference identifies exactly one profile by its existing profile
    fingerprint and fingerprint schema version. It carries no path, store
    layout version, serialization version, alias, name, priority, quality
    score, or creation time, and it is not a second profile identity.
    """

    profile_fingerprint: str
    profile_fingerprint_version: int

    def __post_init__(self) -> None:
        _require_profile_fingerprint_identity(
            self.profile_fingerprint, self.profile_fingerprint_version
        )


@dataclass(frozen=True, slots=True)
class _CalibrationProfileCatalogEntry:
    """Private discovery projection for one catalogued profile.

    The projection retains only the exact profile reference and the metadata
    needed for runtime eligibility discovery. Fitted parameters, method
    configuration, training dataset, ground-truth semantics, quality metrics,
    and serialization bytes are deliberately not retained.
    """

    reference: CalibrationProfileReference
    binding: CalibrationBinding
    target_id: str
    target_version: int
    input_score_id: str
    input_score_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CalibrationProfileReference):
            raise InvalidDecisionError(
                "a catalog entry reference must be a CalibrationProfileReference, got "
                f"{type(self.reference).__name__}"
            )
        if not isinstance(self.binding, CalibrationBinding):
            raise InvalidDecisionError(
                "a catalog entry binding must be a CalibrationBinding, got "
                f"{type(self.binding).__name__}"
            )
        _require_non_empty_str("target_id", self.target_id)
        _require_non_empty_str("input_score_id", self.input_score_id)
        for name, version in (
            ("target_version", self.target_version),
            ("input_score_version", self.input_score_version),
        ):
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise InvalidDecisionError(
                    f"a catalog entry {name} must be an integer greater than or equal "
                    f"to 1, got {version!r}"
                )


@dataclass(frozen=True, slots=True, init=False)
class CalibrationProfileCatalog:
    """An immutable explicit index of known exact profile identities.

    A catalog is built only from a caller-supplied tuple of real
    :class:`~probvenance.calibration.CalibrationProfile` artifacts through
    :meth:`from_profiles`. It does not scan a store, walk a directory, read the
    environment, or consult a registry, and it retains no full profiles.

    Catalog membership is not part of profile identity: building a catalog never
    changes a profile fingerprint, its canonical payload, or its serialization.
    """

    _entries: tuple[_CalibrationProfileCatalogEntry, ...]
    _CONSTRUCTION_TOKEN: ClassVar[object] = object()

    def __init__(
        self,
        entries: tuple[_CalibrationProfileCatalogEntry, ...],
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not CalibrationProfileCatalog._CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "CalibrationProfileCatalog must be built with "
                "CalibrationProfileCatalog.from_profiles(...); hand-declared "
                "catalog entries would be a second, unverified identity projection"
            )
        object.__setattr__(self, "_entries", entries)

    @classmethod
    def from_profiles(cls, profiles: tuple[CalibrationProfile, ...]) -> CalibrationProfileCatalog:
        """Build an immutable catalog from an explicit tuple of profiles.

        Every member must be a real ``CalibrationProfile`` and every exact
        profile identity must be distinct. The caller supplies the profiles
        explicitly; construction never discovers them.
        """
        if type(profiles) is not tuple:
            raise InvalidDecisionError(
                "profiles must be a plain tuple of CalibrationProfile objects, got "
                f"{type(profiles).__name__}; a subclassed, mutable, or lazy container "
                "would make catalog contents iteration-dependent"
            )
        seen: set[tuple[int, str]] = set()
        entries: list[_CalibrationProfileCatalogEntry] = []
        for profile in profiles:
            if not isinstance(profile, CalibrationProfile):
                raise InvalidDecisionError(
                    "every catalog member must be a CalibrationProfile, got "
                    f"{type(profile).__name__} ({_abbreviate(profile)})"
                )
            identity = (CALIBRATION_PROFILE_FINGERPRINT_VERSION, profile.fingerprint)
            if identity in seen:
                raise InvalidDecisionError(
                    "the catalog input contains the same exact calibration profile "
                    f"identity more than once: {profile.fingerprint!r}; a malformed "
                    "catalog input is not legitimate discovery multiplicity"
                )
            seen.add(identity)
            entries.append(
                _CalibrationProfileCatalogEntry(
                    reference=CalibrationProfileReference(
                        profile_fingerprint=profile.fingerprint,
                        profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
                    ),
                    binding=profile.binding,
                    target_id=profile.target_id,
                    target_version=profile.target_version,
                    input_score_id=profile.input_score_id,
                    input_score_version=profile.input_score_version,
                )
            )
        entries.sort(
            key=lambda entry: (
                entry.reference.profile_fingerprint_version,
                entry.reference.profile_fingerprint,
            )
        )
        return cls(tuple(entries), _construction_token=cls._CONSTRUCTION_TOKEN)

    @classmethod
    def _from_serialized_entries(
        cls, entries: tuple[_CalibrationProfileCatalogEntry, ...]
    ) -> CalibrationProfileCatalog:
        """Restore a catalog from entries reconstructed by the loader.

        This is a restoration path, not a derivation path: entry metadata comes
        from a persisted snapshot rather than from live profiles, so a loaded
        catalog remains a set of snapshot claims and stays non-authoritative.
        Entries are sorted into canonical order so a permuted encoding of one
        logical catalog cannot produce a second catalog identity.
        """
        ordered = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    entry.reference.profile_fingerprint_version,
                    entry.reference.profile_fingerprint,
                ),
            )
        )
        return cls(ordered, _construction_token=cls._CONSTRUCTION_TOKEN)

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the frozen catalog discovery-snapshot identity payload."""
        return {
            "v": CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
            "entries": [
                {
                    "profile_fingerprint": entry.reference.profile_fingerprint,
                    "profile_fingerprint_version": entry.reference.profile_fingerprint_version,
                    "binding": entry.binding.canonical_payload(),
                    "target_id": entry.target_id,
                    "target_version": entry.target_version,
                    "input_score_id": entry.input_score_id,
                    "input_score_version": entry.input_score_version,
                }
                for entry in self._entries
            ],
        }

    @property
    def fingerprint(self) -> str:
        """Versioned fingerprint of the exact catalog discovery snapshot."""
        return fingerprint(self.canonical_payload())

    @property
    def references(self) -> tuple[CalibrationProfileReference, ...]:
        """Return every catalogued reference in deterministic exact-identity order."""
        return tuple(entry.reference for entry in self._entries)


def discover_calibration_profile_references_for_runtime(
    evaluation: Evaluation,
    catalog: CalibrationProfileCatalog,
    *,
    task_id: str | None = None,
    domain_id: str | None = None,
    taxonomy_id: str | None = None,
    taxonomy_version: int | None = None,
) -> tuple[CalibrationProfileReference, ...]:
    """Return exact references whose discovery metadata matches the runtime.

    Discovery is non-authoritative. It returns a deterministic tuple of exact
    profile references, never profiles, and never accesses a store. Zero
    references returns ``()`` and many references returns all of them; discovery
    never raises a selection no-eligible or ambiguity error merely because of
    cardinality. Selection owns those policy outcomes once the referenced
    profiles are loaded by exact identity.

    Eligibility uses the same exact binding plus winner-correctness
    target/input projection as the Phase 4C.7 selector, reconstructed from the
    evaluation trace plus the CALLER's declarations. Ground-truth semantics,
    method, method configuration, fitted parameters, training dataset, quality
    metrics, recency, and catalog input order are deliberately not filters.
    """
    if not isinstance(catalog, CalibrationProfileCatalog):
        raise InvalidDecisionError(
            f"catalog must be a CalibrationProfileCatalog, got {type(catalog).__name__}"
        )
    _, trace = _require_uncalibrated_runtime_evaluation(evaluation, operation=_OPERATION)
    runtime_binding = _runtime_calibration_binding(
        trace,
        task_id=task_id,
        domain_id=domain_id,
        taxonomy_id=taxonomy_id,
        taxonomy_version=taxonomy_version,
    )
    return tuple(
        entry.reference
        for entry in catalog._entries
        if _runtime_profile_semantics_are_eligible(
            profile_binding=entry.binding,
            target_id=entry.target_id,
            target_version=entry.target_version,
            input_score_id=entry.input_score_id,
            input_score_version=entry.input_score_version,
            runtime_binding=runtime_binding,
        )
    )


def serialize_calibration_profile_catalog(catalog: CalibrationProfileCatalog) -> str:
    """Serialize a catalog discovery snapshot to canonical JSON text.

    The result is :func:`probvenance.fingerprint.canonical_json` of the
    envelope, so repeated serialization of one catalog is byte-for-byte
    identical: there is no timestamp, no random identifier, and no host or path
    metadata. The envelope carries the frozen catalog identity payload, which
    already embeds each entry's full binding discovery projection, so no
    separate materialization block is needed. No profile is serialized, and no
    filesystem, database, or store I/O is performed.
    """
    if not isinstance(catalog, CalibrationProfileCatalog):
        raise InvalidDecisionError(
            f"catalog must be a CalibrationProfileCatalog, got {type(catalog).__name__}"
        )
    envelope: dict[str, JSONValue] = {
        "artifact_type": CALIBRATION_PROFILE_CATALOG_SERIALIZATION_TYPE,
        "serialization_version": CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION,
        "catalog_fingerprint": catalog.fingerprint,
        "catalog_identity": catalog.canonical_payload(),
    }
    return canonical_json(envelope)


def load_calibration_profile_catalog(
    serialized: str,
    *,
    expected_catalog_fingerprint: str | None = None,
    expected_catalog_fingerprint_version: int | None = None,
) -> CalibrationProfileCatalog:
    """Restore a catalog snapshot from JSON text, verifying its identity.

    No claimed hash is trusted merely because it is present. The loader parses
    strictly, enforces the exact v1 envelope and entry schema, reconstructs
    typed references and typed bindings, rejects a duplicate exact profile
    identity, sorts entries into canonical order, and finally requires both the
    restored canonical identity payload and the restored catalog fingerprint to
    match the document.

    A loaded catalog remains a set of snapshot CLAIMS. Its fingerprint proves
    that the snapshot is internally self-consistent; it does NOT prove that a
    referenced profile still exists, that a store artifact is intact, or that
    the snapshot metadata still matches the actual profile. Those remain the
    responsibilities of exact store retrieval and Phase 4C.7 selection, which
    revalidate the real loaded profiles.

    The optional ``expected_catalog_fingerprint`` /
    ``expected_catalog_fingerprint_version`` must be supplied together or not at
    all, and support a caller that obtained the expected identity through a
    separate trusted channel. Embedded fingerprint consistency is NOT
    cryptographic authenticity, and this loader performs no signature or MAC
    verification.
    """
    if not isinstance(serialized, str):
        raise InvalidDecisionError(
            f"serialized must be a str, got {type(serialized).__name__} ({serialized!r})"
        )
    if (expected_catalog_fingerprint is None) != (expected_catalog_fingerprint_version is None):
        raise InvalidDecisionError(
            "expected_catalog_fingerprint and expected_catalog_fingerprint_version "
            "must be supplied together or not at all"
        )

    document = parse_strict_json_object(serialized, subject="calibration profile catalog")
    _require_exact_keys("serialized calibration profile catalog", document, _CATALOG_ENVELOPE_KEYS)

    artifact_type = document["artifact_type"]
    if artifact_type != CALIBRATION_PROFILE_CATALOG_SERIALIZATION_TYPE:
        raise InvalidDecisionError(
            "the serialized calibration profile catalog has an unsupported artifact "
            f"type: expected {CALIBRATION_PROFILE_CATALOG_SERIALIZATION_TYPE!r}, got "
            f"{_abbreviate(artifact_type)}"
        )
    serialization_version = document["serialization_version"]
    if (
        isinstance(serialization_version, bool)
        or not isinstance(serialization_version, int)
        or serialization_version != CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION
    ):
        raise InvalidDecisionError(
            "the serialized calibration profile catalog has an unsupported "
            "serialization version: expected "
            f"{CALIBRATION_PROFILE_CATALOG_SERIALIZATION_VERSION}, got "
            f"{_abbreviate(serialization_version)}"
        )

    identity = _require_exact_keys(
        "catalog_identity", document["catalog_identity"], _CATALOG_IDENTITY_KEYS
    )
    if identity["v"] != CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION:
        raise InvalidDecisionError(
            "the serialized calibration profile catalog has an unsupported catalog "
            f"identity version: expected {CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION}, "
            f"got {_abbreviate(identity['v'])}"
        )
    entry_payloads = identity["entries"]
    if not isinstance(entry_payloads, list):
        raise InvalidDecisionError(
            f"catalog_identity['entries'] must be a JSON array, got {type(entry_payloads).__name__}"
        )

    seen: set[tuple[int, str]] = set()
    entries: list[_CalibrationProfileCatalogEntry] = []
    for index, entry_payload in enumerate(entry_payloads):
        field = f"catalog_identity['entries'][{index}]"
        entry = _require_exact_keys(field, entry_payload, _CATALOG_ENTRY_KEYS)
        reference = CalibrationProfileReference(
            profile_fingerprint=entry["profile_fingerprint"],
            profile_fingerprint_version=entry["profile_fingerprint_version"],
        )
        identity_key = (reference.profile_fingerprint_version, reference.profile_fingerprint)
        if identity_key in seen:
            raise InvalidDecisionError(
                "the serialized calibration profile catalog contains the same exact "
                f"profile identity more than once: {reference.profile_fingerprint!r}"
            )
        seen.add(identity_key)
        entries.append(
            _CalibrationProfileCatalogEntry(
                reference=reference,
                binding=_restore_calibration_binding(entry["binding"], name=f"{field}['binding']"),
                target_id=entry["target_id"],
                target_version=entry["target_version"],
                input_score_id=entry["input_score_id"],
                input_score_version=entry["input_score_version"],
            )
        )

    restored = CalibrationProfileCatalog._from_serialized_entries(tuple(entries))
    if canonical_json(restored.canonical_payload()) != canonical_json(identity):
        raise InvalidDecisionError(
            "the restored calibration profile catalog identity does not match the "
            "serialized catalog identity payload"
        )
    if restored.fingerprint != document["catalog_fingerprint"]:
        raise InvalidDecisionError(
            "the restored calibration profile catalog fingerprint does not match the "
            f"serialized catalog fingerprint: restored {restored.fingerprint!r} vs "
            f"serialized {_abbreviate(document['catalog_fingerprint'])}"
        )

    if expected_catalog_fingerprint is not None:
        if isinstance(expected_catalog_fingerprint_version, bool) or not isinstance(
            expected_catalog_fingerprint_version, int
        ):
            raise InvalidDecisionError(
                "the expected catalog fingerprint version must be an integer, got "
                f"{expected_catalog_fingerprint_version!r}"
            )
        if expected_catalog_fingerprint_version != CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION:
            raise InvalidDecisionError(
                "the expected catalog fingerprint version is not supported: expected "
                f"{CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION}, got "
                f"{expected_catalog_fingerprint_version!r}"
            )
        if restored.fingerprint != expected_catalog_fingerprint:
            raise InvalidDecisionError(
                "the restored calibration profile catalog does not match the expected "
                f"catalog fingerprint: restored {restored.fingerprint!r} vs expected "
                f"{_abbreviate(expected_catalog_fingerprint)}"
            )

    return restored
