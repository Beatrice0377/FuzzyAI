"""Explicit runtime profile eligibility and unique selection.

Selection is authorization, not recommendation. The selector may establish
that a calibration profile is semantically eligible for a declared runtime
population. It never claims that a profile is statistically best, newest, has
the best Brier score or log loss, generalizes best, or should be preferred over
another eligible profile.

The v1 policy is deliberately conservative and has no tie-break:

* zero eligible unique profiles raises an explicit no-eligible error;
* exactly one eligible unique profile is returned;
* more than one distinct eligible profile raises an ambiguity error.

Candidates are supplied explicitly by the caller as an immutable tuple. The
selector performs no store enumeration, discovery, registry lookup, directory
scanning, or ambient profile search, and it is storage-agnostic. The caller may
retrieve known profile identities from
:class:`~probvenance.calibration_store.DirectoryCalibrationProfileStore` and
pass the resulting objects here.

Selection does not apply, load, store, search, rank, or score profile quality.
Applying the selected profile remains a separate explicit call to
:func:`probvenance.calibration.apply_profile_to_runtime_evaluation`.
"""

from __future__ import annotations

from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    UNCALIBRATED_SELECTED_PROBABILITY_ID,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    WINNER_CORRECTNESS_TARGET_ID,
    WINNER_CORRECTNESS_TARGET_VERSION,
    CalibrationBinding,
    CalibrationProfile,
    _abbreviate,
    _binding_payloads_equal,
    _require_uncalibrated_runtime_evaluation,
    _runtime_calibration_binding,
)
from probvenance.errors import (
    AmbiguousCalibrationProfileSelectionError,
    InvalidDecisionError,
    NoEligibleCalibrationProfileError,
)
from probvenance.runtime import Evaluation

_OPERATION = "calibration profile selection"


def _require_candidate_tuple(candidates: object) -> tuple[CalibrationProfile, ...]:
    """Return the candidate set, requiring the frozen tuple container shape."""
    if not isinstance(candidates, tuple):
        raise InvalidDecisionError(
            "candidates must be a tuple of CalibrationProfile objects, got "
            f"{type(candidates).__name__}; a mutable or lazy container would make "
            "selection order- or iteration-dependent"
        )
    return candidates


def _require_distinct_candidate_identities(
    candidates: tuple[CalibrationProfile, ...],
) -> None:
    """Require every candidate to be a distinct exact profile identity."""
    seen: set[tuple[int, str]] = set()
    for candidate in candidates:
        identity = (CALIBRATION_PROFILE_FINGERPRINT_VERSION, candidate.fingerprint)
        if identity in seen:
            raise InvalidDecisionError(
                "the candidate set contains the same exact calibration profile "
                f"identity more than once: {candidate.fingerprint!r}; a malformed "
                "candidate set is not selection ambiguity"
            )
        seen.add(identity)


def _require_profile_candidates(
    candidates: tuple[CalibrationProfile, ...],
) -> None:
    """Require every candidate to be a real, distinct CalibrationProfile."""
    for candidate in candidates:
        if not isinstance(candidate, CalibrationProfile):
            raise InvalidDecisionError(
                "every candidate must be a CalibrationProfile, got "
                f"{type(candidate).__name__} ({_abbreviate(candidate)})"
            )
    _require_distinct_candidate_identities(candidates)


def _is_eligible(
    profile: CalibrationProfile,
    runtime_binding: CalibrationBinding,
) -> bool:
    """Return whether a profile is eligible for the runtime population.

    Eligibility requires an exact binding match plus the exact target and input
    score semantics of the existing runtime winner-correctness application.
    Method, fitted parameters, training dataset, ground-truth semantics, and
    every quality metric are deliberately not eligibility dimensions.
    """
    if not _binding_payloads_equal(profile.binding, runtime_binding):
        return False
    if (
        profile.target_id != WINNER_CORRECTNESS_TARGET_ID
        or profile.target_version != WINNER_CORRECTNESS_TARGET_VERSION
    ):
        return False
    return (
        profile.input_score_id == UNCALIBRATED_SELECTED_PROBABILITY_ID
        and profile.input_score_version == UNCALIBRATED_SELECTED_PROBABILITY_VERSION
    )


def select_calibration_profile_for_runtime(
    evaluation: Evaluation,
    candidates: tuple[CalibrationProfile, ...],
    *,
    task_id: str | None = None,
    domain_id: str | None = None,
    taxonomy_id: str | None = None,
    taxonomy_version: int | None = None,
) -> CalibrationProfile:
    """Return the one eligible profile for an uncalibrated runtime evaluation.

    The runtime population identity is reconstructed from the evaluation trace
    plus the CALLER's declarations, using the same reconstruction the runtime
    application path uses. A candidate is eligible only when its exact
    :class:`CalibrationBinding` equals that reconstructed binding and its target
    and input-score semantics match the runtime winner-correctness contract.
    Declarations are never read from a candidate profile, and ``None`` is never
    a wildcard.

    Selection is unique-or-fail-closed: an empty or fully ineligible candidate
    set raises :class:`NoEligibleCalibrationProfileError`, and more than one
    distinct eligible profile raises
    :class:`AmbiguousCalibrationProfileSelectionError` with no tie-break. The
    evaluation must be uncalibrated and coherent, and it is never modified.
    """
    candidate_set = _require_candidate_tuple(candidates)
    _, trace = _require_uncalibrated_runtime_evaluation(evaluation, operation=_OPERATION)
    _require_profile_candidates(candidate_set)
    runtime_binding = _runtime_calibration_binding(
        trace,
        task_id=task_id,
        domain_id=domain_id,
        taxonomy_id=taxonomy_id,
        taxonomy_version=taxonomy_version,
    )
    eligible = [
        candidate for candidate in candidate_set if _is_eligible(candidate, runtime_binding)
    ]
    if not eligible:
        raise NoEligibleCalibrationProfileError(
            "no candidate calibration profile is eligible for the runtime population; "
            f"{len(candidate_set)} candidate(s) were supplied and the runtime binding "
            f"fingerprint is {runtime_binding.fingerprint!r}"
        )
    if len(eligible) > 1:
        identities = ", ".join(sorted(profile.fingerprint for profile in eligible))
        raise AmbiguousCalibrationProfileSelectionError(
            f"more than one candidate calibration profile is eligible for the runtime "
            f"population; {len(eligible)} distinct eligible profiles: {identities}"
        )
    return eligible[0]
