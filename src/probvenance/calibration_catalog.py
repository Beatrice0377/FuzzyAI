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

from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationBinding,
    CalibrationProfile,
    _abbreviate,
    _require_profile_fingerprint_identity,
    _require_uncalibrated_runtime_evaluation,
    _runtime_calibration_binding,
    _runtime_profile_semantics_are_eligible,
)
from probvenance.errors import InvalidDecisionError
from probvenance.runtime import Evaluation

_OPERATION = "calibration profile discovery"


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
