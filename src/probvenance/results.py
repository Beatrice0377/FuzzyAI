"""Result value objects: certainty measures and decision outcomes.

Terminology note: the "how peaked" number is called *concentration*
(``1 - entropy``). It is never called "confidence".
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from probvenance.errors import InvalidProbabilityError

_SUM_TOLERANCE = 1e-9


def _validate_distribution(probabilities: Sequence[float]) -> None:
    if len(probabilities) == 0:
        raise InvalidProbabilityError("probabilities must be a non-empty sequence")
    for i, p in enumerate(probabilities):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise InvalidProbabilityError(
                f"probabilities[{i}] must be a real number, got {type(p).__name__} ({p!r})"
            )
        if not math.isfinite(p):
            raise InvalidProbabilityError(f"probabilities[{i}] must be finite, got {p!r}")
        if p < 0.0 or p > 1.0:
            raise InvalidProbabilityError(f"probabilities[{i}] must be in [0, 1], got {p!r}")
    total = float(sum(probabilities))
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise InvalidProbabilityError(
            f"probabilities must sum to 1.0 (tolerance {_SUM_TOLERANCE}), got sum {total!r}"
        )


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Normalized Shannon entropy of a probability distribution (base e).

    ``H = -sum(p * ln(p))`` over ``p > 0`` (a probability of exactly 0
    contributes 0), normalized by ``ln(n)`` so the result lies in [0, 1]:
    0 means fully concentrated on one outcome, 1 means uniform over ``n``
    outcomes. For ``n == 1`` the result is defined as ``0.0``. The result is
    clamped into [0, 1] to absorb floating-point error.
    """
    _validate_distribution(probabilities)
    n = len(probabilities)
    if n == 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    value = entropy / math.log(n)
    return min(1.0, max(0.0, value))


def probability_margin(probabilities: Sequence[float]) -> float:
    """Margin between the two most probable outcomes: ``p[0] - p[1]``.

    Probabilities are sorted in descending order; the margin is the difference
    between the top two. For ``n == 1`` the margin is defined as ``p[0]``
    (i.e. ``1.0`` for a valid single-outcome distribution). Deterministic.
    """
    _validate_distribution(probabilities)
    ordered = sorted(probabilities, reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


@dataclass(frozen=True, slots=True)
class Certainty:
    """How certain a decision outcome is, independent of whether it is correct.

    ``entropy`` is the NORMALIZED entropy in [0, 1]: 0 = fully concentrated,
    1 = uniform. ``margin`` is ``top1_probability - top2_probability`` in
    [0, 1].
    """

    entropy: float
    margin: float

    def __post_init__(self) -> None:
        for name in ("entropy", "margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidProbabilityError(
                    f"{name} must be a real number, got {type(value).__name__} ({value!r})"
                )
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise InvalidProbabilityError(
                    f"{name} must be a finite float in [0, 1], got {value!r}"
                )

    @property
    def concentration(self) -> float:
        """How peaked the distribution is: ``1 - entropy`` (never "confidence")."""
        return 1.0 - self.entropy

    @classmethod
    def from_probabilities(cls, probabilities: Sequence[float]) -> "Certainty":
        """Compute certainty from a probability distribution (e.g. ``[p, 1 - p]`` for bool)."""
        return cls(
            entropy=normalized_entropy(probabilities),
            margin=probability_margin(probabilities),
        )


def _validate_calibration_provenance(
    calibration_profile_fingerprint: Any,
    calibration_profile_fingerprint_version: Any,
) -> None:
    """Validate the shared ``(profile fingerprint, version)`` provenance pair.

    Both must be ``None`` (uncalibrated) or both set, with a non-empty string
    fingerprint and an ``int >= 1`` (non-bool) version. ``DecisionResult`` and
    ``DecisionTrace`` share this one rule so their provenance mirrors cannot
    drift apart.
    """
    if calibration_profile_fingerprint is not None and (
        not isinstance(calibration_profile_fingerprint, str)
        or not calibration_profile_fingerprint.strip()
    ):
        raise InvalidProbabilityError(
            "calibration_profile_fingerprint must be None or a non-empty string, "
            f"got {calibration_profile_fingerprint!r}"
        )
    if calibration_profile_fingerprint_version is not None and (
        isinstance(calibration_profile_fingerprint_version, bool)
        or not isinstance(calibration_profile_fingerprint_version, int)
        or calibration_profile_fingerprint_version < 1
    ):
        raise InvalidProbabilityError(
            "calibration_profile_fingerprint_version must be None or an int >= 1, "
            f"got {calibration_profile_fingerprint_version!r}"
        )
    if (calibration_profile_fingerprint is None) != (
        calibration_profile_fingerprint_version is None
    ):
        raise InvalidProbabilityError(
            "calibration_profile_fingerprint and calibration_profile_fingerprint_version "
            "must be set together, got fingerprint "
            f"{calibration_profile_fingerprint!r} and version "
            f"{calibration_profile_fingerprint_version!r}"
        )


def _validate_result_fields(
    certainty: Any,
    method: Any,
    trace_id: Any,
    predicted_correctness: Any,
    calibrated: Any,
    calibration_profile_fingerprint: Any,
    calibration_profile_fingerprint_version: Any,
) -> None:
    """Shared validation for all :class:`DecisionResult` subclasses.

    Called explicitly from each concrete ``__post_init__`` because dataclass
    inheritance does not chain ``__post_init__``.

    The calibration fields form a small state machine. A calibrated result
    carries a finite ``predicted_correctness`` in [0, 1] AND the exact
    ``(fingerprint, version)`` pair of the applied profile. An uncalibrated
    result carries none of the three. Half-states are rejected.

    This is STRUCTURAL validity only: a coherent hand-built calibrated result
    is accepted, and acceptance is NOT attestation that the named profile
    really produced the number. The supported runtime-linked application path
    is the provenance-producing one.
    """
    if not isinstance(certainty, Certainty):
        raise InvalidProbabilityError(
            f"certainty must be a Certainty, got {type(certainty).__name__} ({certainty!r})"
        )
    if not isinstance(method, str) or not method.strip():
        raise InvalidProbabilityError(f"method must be a non-empty string, got {method!r}")
    if trace_id is not None and (not isinstance(trace_id, str) or not trace_id.strip()):
        raise InvalidProbabilityError(
            f"trace_id must be None or a non-empty string, got {trace_id!r}"
        )
    if not isinstance(calibrated, bool):
        raise InvalidProbabilityError(
            f"calibrated must be a bool, got {type(calibrated).__name__} ({calibrated!r})"
        )
    if predicted_correctness is not None and (
        isinstance(predicted_correctness, bool)
        or not isinstance(predicted_correctness, (int, float))
        or not math.isfinite(predicted_correctness)
        or predicted_correctness < 0.0
        or predicted_correctness > 1.0
    ):
        raise InvalidProbabilityError(
            "predicted_correctness must be None or a finite float in [0, 1], "
            f"got {predicted_correctness!r}"
        )
    _validate_calibration_provenance(
        calibration_profile_fingerprint,
        calibration_profile_fingerprint_version,
    )
    if calibrated:
        if predicted_correctness is None:
            raise InvalidProbabilityError(
                "a calibrated result must carry predicted_correctness; "
                "calibrated=True with predicted_correctness=None is a forbidden half-state"
            )
        if calibration_profile_fingerprint is None:
            raise InvalidProbabilityError(
                "a calibrated result must carry the exact calibration profile identity "
                "(calibration_profile_fingerprint and "
                "calibration_profile_fingerprint_version); a calibrated result without "
                "profile provenance is a forbidden half-state"
            )
    else:
        if predicted_correctness is not None:
            raise InvalidProbabilityError(
                "predicted_correctness may only be set when calibrated is True "
                "(uncalibrated results must leave predicted_correctness as None)"
            )
        if calibration_profile_fingerprint is not None:
            raise InvalidProbabilityError(
                "an uncalibrated result must not carry calibration profile provenance "
                "(calibration_profile_fingerprint/version must both be None when "
                "calibrated is False)"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionResult:
    """Provenance base for all decision results.

    Phase 1 never auto-fills ``predicted_correctness`` and never sets
    ``calibrated=True`` on its own: an uncalibrated result has
    ``predicted_correctness=None`` and ``calibrated=False``.

    ``calibrated`` records APPLICATION STATE, not quality: it means a supported
    :class:`~probvenance.calibration.CalibrationProfile` was applied through the
    declared calibration application contract and ``predicted_correctness`` is
    populated. It does NOT mean the result is empirically well calibrated, that
    Brier or log loss improved, or that any statistical guarantee holds.

    ``calibration_profile_fingerprint`` and
    ``calibration_profile_fingerprint_version`` are the EXACT identity of the
    profile that produced ``predicted_correctness``. The profile object itself
    is never stored on a result, and none of its internals (method
    configuration, training dataset fingerprint, fitted parameters,
    ground-truth semantics, binding) are duplicated here. The fingerprint is
    the identity link; the profile is looked up elsewhere if needed.
    """

    certainty: Certainty
    method: str
    trace_id: str | None = None
    predicted_correctness: float | None = None
    calibrated: bool = False
    calibration_profile_fingerprint: str | None = None
    calibration_profile_fingerprint_version: int | None = None

    def __post_init__(self) -> None:
        _validate_result_fields(
            self.certainty,
            self.method,
            self.trace_id,
            self.predicted_correctness,
            self.calibrated,
            self.calibration_profile_fingerprint,
            self.calibration_profile_fingerprint_version,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolResult(DecisionResult):
    """Outcome of a :class:`~probvenance.decisions.BoolDecision`."""

    probability_true: float

    def __post_init__(self) -> None:
        # With ``slots=True`` the zero-arg ``super()`` cell refers to the stale
        # pre-recreation class object, so the shared module-level helper is
        # called directly (dataclass inheritance does not chain
        # ``__post_init__`` anyway).
        _validate_result_fields(
            self.certainty,
            self.method,
            self.trace_id,
            self.predicted_correctness,
            self.calibrated,
            self.calibration_profile_fingerprint,
            self.calibration_profile_fingerprint_version,
        )
        p = self.probability_true
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise InvalidProbabilityError(
                f"probability_true must be a real number, got {type(p).__name__} ({p!r})"
            )
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise InvalidProbabilityError(
                f"probability_true must be a finite float in [0, 1], got {p!r}"
            )

    @property
    def probability_false(self) -> float:
        return 1.0 - self.probability_true


@dataclass(frozen=True, slots=True, kw_only=True)
class ChoiceResult(DecisionResult):
    """Outcome of a :class:`~probvenance.decisions.ChoiceDecision`.

    ``probabilities`` is a read-only snapshot: a plain ``dict`` deep-copied at
    construction, preserving the given key order. Key order is
    semantics-bearing for tie-breaking in :attr:`value`.
    """

    # ``__hash__ = None`` needs a ``type: ignore`` under ``mypy --strict``
    # (banned here), so raise instead; ``@dataclass`` will not overwrite this.
    def __hash__(self) -> int:
        raise TypeError(f"{type(self).__name__} is unhashable: probabilities is a mapping field")

    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_result_fields(
            self.certainty,
            self.method,
            self.trace_id,
            self.predicted_correctness,
            self.calibrated,
            self.calibration_profile_fingerprint,
            self.calibration_profile_fingerprint_version,
        )
        probabilities = self.probabilities
        if not isinstance(probabilities, Mapping):
            raise InvalidProbabilityError(
                f"probabilities must be a mapping, got {type(probabilities).__name__}"
            )
        if len(probabilities) < 2:
            raise InvalidProbabilityError(
                f"probabilities must contain at least 2 entries, got {len(probabilities)}"
            )
        for key, value in probabilities.items():
            if not isinstance(key, str) or not key:
                raise InvalidProbabilityError(
                    f"probability keys must be non-empty str, got {key!r}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidProbabilityError(
                    f"probabilities[{key!r}] must be a real number, "
                    f"got {type(value).__name__} ({value!r})"
                )
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise InvalidProbabilityError(
                    f"probabilities[{key!r}] must be a finite float in [0, 1], got {value!r}"
                )
        total = float(sum(probabilities.values()))
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise InvalidProbabilityError(
                f"probabilities must sum to 1.0 (tolerance {_SUM_TOLERANCE}), got sum {total!r}"
            )
        # Values are floats (immutable) and the mapping is wrapped read-only, so
        # this result is a true snapshot: the recorded selection can never drift
        # after construction. Key order is preserved for tie-breaking.
        object.__setattr__(self, "probabilities", MappingProxyType(dict(probabilities)))

    @property
    def value(self) -> str:
        """Deterministic argmax over the ordered probabilities.

        On ties the entry appearing FIRST in the stored order wins. The result
        is always one of the keys.
        """
        best_key: str | None = None
        best_value = -1.0
        for key, value in self.probabilities.items():
            if value > best_value:
                best_key = key
                best_value = value
        assert best_key is not None  # at least 2 validated entries exist
        return best_key

    @property
    def choice_names(self) -> tuple[str, ...]:
        """The probability keys in stored (candidate) order."""
        return tuple(self.probabilities.keys())
