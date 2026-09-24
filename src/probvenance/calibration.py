"""Calibration data foundation and one offline fitting method.

This module implements the ground-truth and observation data model that a
calibration harness fits against: ground-truth provenance, a ground-truth
record with resolution semantics, a calibration binding, a calibration
observation with deterministically derived status and correctness, an
observation fingerprint, a calibration dataset with structural pooling
constraints, a dataset fingerprint, and a calibration profile identity that
composes its binding, ground-truth semantics, target, input-score, method, and
training-dataset provenance.

It also implements ONE offline fitting method: an L2-regularized logistic
regression of winner correctness on the raw selected semantic probability
(:func:`fit_l2_logistic_selected_probability`), plus the single
predicted-correctness scoring function
(:func:`predicted_winner_correctness`). The taxonomy constraints are enforced
at the fitting boundary: a fit-eligible dataset rejects taxonomy-miss,
unresolved, and unadjudicated rows rather than silently filtering them, and the
profile construction path refuses a profile whose ground-truth semantics or
binding do not match its training data.

Still NOT implemented here (out of scope): alternative calibration methods
(temperature scaling, isotonic regression), profile registries, nearest-profile
matching, profile serialization or cross-process loading, runtime profile
application, and evaluation metrics (which live in
``probvenance.calibration_evaluation``).

Public API note: this foundation is intentionally NOT frozen as public API
yet. Nothing from this module is exported through ``probvenance.__all__`` or
imported into ``probvenance/__init__.py``; the module is importable as
``probvenance.calibration`` for tests and internal use only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint
from probvenance.results import BoolResult, ChoiceResult
from probvenance.runtime import Evaluation
from probvenance.trace import DecisionTrace

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

CALIBRATION_OBSERVATION_FINGERPRINT_VERSION = 1
"""Version of the calibration observation fingerprint payload schema."""

CALIBRATION_BINDING_FINGERPRINT_VERSION = 1
"""Version of the calibration binding fingerprint payload schema."""

CALIBRATION_DATASET_FINGERPRINT_VERSION = 2
"""Version of the calibration dataset fingerprint payload schema."""

CALIBRATION_PROFILE_FINGERPRINT_VERSION = 1
"""Version of the calibration profile fingerprint payload schema."""

GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION = 1
"""Version of the ground-truth semantics fingerprint payload schema."""

RENDERING_SEMANTICS_VERSION = 1
"""Version of the centralized rendering-semantics canonical projection."""

# ---------------------------------------------------------------------------
# Calibration target identity
# ---------------------------------------------------------------------------

#: The single implemented calibration target: whether the recorded selected
#: semantic value equals the resolved ground truth. This target is shared by
#: every winner-correctness evaluation artifact and is owned by no single
#: metric; the target no longer belongs to Brier. A target with the same id but
#: a different version is a different target semantics identity, because the
#: meaning of the label could have changed. Exact identity only: no
#: target-version compatibility policy is implemented.
#:
#: It lives here, in the calibration foundation, rather than in the evaluation
#: layer, because the future CalibrationProfile identity composes it together
#: with CalibrationBinding and GroundTruthSemanticsIdentity, which are defined
#: in this module. Owning it in the evaluation layer would force this module to
#: import the evaluation layer, which already imports this module.
#:
#: Two constants are deliberately sufficient here. A CalibrationTargetIdentity
#: type becomes justified only when several targets exist, when a target
#: carries structured configuration, or when target compatibility needs
#: behaviour; none of those is true yet.
WINNER_CORRECTNESS_TARGET_ID = "winner_correctness"
WINNER_CORRECTNESS_TARGET_VERSION = 1

# ---------------------------------------------------------------------------
# Calibrator input-score identity
# ---------------------------------------------------------------------------

#: The input score a calibrator consumes: the probability the *uncalibrated*
#: semantic distribution assigns to the recorded selected semantic value. It is
#: NOT ``predicted_correctness``, NOT a confidence, and NOT a calibrated
#: probability. The runtime never sets ``predicted_correctness`` and never marks
#: a result ``calibrated=True``.
#:
#: The input-score identity is not redundant with the calibration target: the
#: same target admits different calibration problems, because a calibrator may
#: consume the selected probability, the whole restricted distribution, or
#: richer deterministic features. Only the selected probability is implemented.
#:
#: It is owned here, beside the target identity, for the same dependency reason:
#: the CalibrationProfile identity composes it with the binding and
#: ground-truth semantics identities, all of which live in this module.
UNCALIBRATED_SELECTED_PROBABILITY_ID = "uncalibrated-selected-probability"
UNCALIBRATED_SELECTED_PROBABILITY_VERSION = 1

#: The post-calibration predicted-correctness score identity: the score
#: produced by applying ONE exact fitted :class:`CalibrationProfile` to an
#: observation under the profile's declared input-score semantics. It is
#: intended to estimate ``P(winner_correctness = 1)`` under the profile's
#: target and ground-truth semantics, which makes it a genuinely different
#: score semantics from :data:`UNCALIBRATED_SELECTED_PROBABILITY_ID`; the two
#: must never be conflated, so this score has its own identity instead of
#: reusing the uncalibrated one.
#:
#: It is owned here, in the calibration foundation, rather than in the
#: evaluation layer, because future runtime calibration must consume the same
#: score semantics.
#:
#: A calibrated-score SEMANTICS is not evidence that the score is empirically
#: well calibrated, that the calibrator improved anything, or that the model is
#: globally calibrated. Those are evaluation conclusions, not score identity.
PREDICTED_WINNER_CORRECTNESS_ID = "predicted-winner-correctness"
PREDICTED_WINNER_CORRECTNESS_VERSION = 1

_OBSERVATION_CONSTRUCTION_TOKEN: Final[object] = object()
"""Construction capability held only by ``CalibrationObservation.from_evaluation``.

The token is deliberately NOT a dataclass field: it is a keyword-only
``__init__`` parameter that defaults to ``None``, so ``dataclasses.replace``
cannot smuggle it (``replace`` re-supplies only the dataclass field values)
and it is therefore not readable from an instance, not present in
``dataclasses.fields()``, and not part of ``repr``, equality, hashing, or
pickle state. Direct field construction and ``replace`` reconstruction both
fail the capability check because neither passes through the supported
construction path that requires matching runtime linkage identities between
the result and the trace.
"""

_PROFILE_CONSTRUCTION_TOKEN: Final[object] = object()
"""Construction capability held only by ``CalibrationProfile._from_fitted_state``.

The token is deliberately NOT a dataclass field: it is a keyword-only
``__init__`` parameter that defaults to ``None``, so ``dataclasses.replace``
cannot smuggle it (``replace`` re-supplies only the dataclass field values)
and it is therefore not readable from an instance, not present in
``dataclasses.fields()``, and not part of ``repr``, equality, hashing, or
pickle state. Direct field construction and ``replace`` reconstruction both
fail the capability check because neither passes through the supported
construction path that derives the profile identity from a fitted
:class:`CalibrationDataset`.
"""

_BOOL_TRUE_NAME: str = "true"
_BOOL_FALSE_NAME: str = "false"

_BOOL_OUTCOME_ORDER: tuple[str, ...] = (_BOOL_FALSE_NAME, _BOOL_TRUE_NAME)
"""Semantic Bool outcome order, matching the formulation identity order.

The first entry wins ties, so a 0.5 / 0.5 Bool tie selects ``False``.
"""


# ---------------------------------------------------------------------------
# Rendering semantics (Part I5)
# ---------------------------------------------------------------------------


def canonical_rendering_semantics(
    rendering_config: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    """Project a rendering config onto its scoring-relevant semantics.

    One centralized, versioned projection instead of scattered ``.get()``
    calls. A missing known key is an EXPLICIT UNKNOWN (``None``), never
    ``False``: an absent ``enable_thinking`` is not the same decision
    population as ``enable_thinking=False``. A non-mapping or absent input
    yields an all-unknown projection. The projection version is included
    under the ``"v"`` key so the projection itself is versionable.
    """
    if rendering_config is None or not isinstance(rendering_config, Mapping):
        return {"v": RENDERING_SEMANTICS_VERSION, "enable_thinking": None}
    raw_thinking = rendering_config.get("enable_thinking")
    enable_thinking: bool | None
    if raw_thinking is None or isinstance(raw_thinking, bool):
        enable_thinking = raw_thinking
    else:
        raise InvalidDecisionError(
            "rendering_config['enable_thinking'] must be None or bool, got "
            f"{type(raw_thinking).__name__} ({raw_thinking!r})"
        )
    return {"v": RENDERING_SEMANTICS_VERSION, "enable_thinking": enable_thinking}


# ---------------------------------------------------------------------------
# Ground truth (Part F)
# ---------------------------------------------------------------------------


def _require_json_value(name: str, value: Any) -> None:
    """Reject values outside the canonical JSON domain.

    The accepted domain is exactly the canonical JSON domain: ``None``,
    ``bool``, ``int``, finite ``float``, ``str``, ``list``, and ``dict``
    with ``str`` keys. Tuples and non-finite floats (``nan``, ``inf``,
    ``-inf``) are rejected because they cannot be canonicalized by
    :func:`probvenance.fingerprint.canonical_json`, so accepting them here
    would allow constructing a value whose fingerprint is not computable.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidDecisionError(f"{name} must be a finite float, got {value!r}")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(f"{name}[]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidDecisionError(
                    f"{name} keys must be str, got {type(key).__name__} ({key!r})"
                )
            _require_json_value(f"{name}[{key!r}]", item)
        return
    raise InvalidDecisionError(
        f"{name} must be a JSON-compatible value, got {type(value).__name__} ({value!r})"
    )


def _require_non_empty_str(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidDecisionError(
            f"{name} must be a non-empty string, got {type(value).__name__} ({value!r})"
        )


def _require_real_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise InvalidDecisionError(f"{name} must be a bool, got {type(value).__name__} ({value!r})")


def _freeze_json_value(value: Any) -> Any:
    """Return a deeply immutable copy of a canonical JSON value."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """Return a plain JSON-compatible copy of a frozen JSON value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class GroundTruthProvenance:
    """Provenance of how a ground-truth label was established.

    ``label_source`` names where the label came from (for example a human
    annotator or a deterministic rule), ``labeling_rule`` states the rule
    that was applied, ``adjudicated`` records whether a human adjudicated
    the label, and ``ambiguity_policy`` states how ambiguous cases were
    handled. The optional taxonomy pair identifies the label taxonomy and
    may be partially known: each of ``taxonomy_id`` and ``taxonomy_version``
    may independently be ``None`` (a known taxonomy with an unknown version
    is a real state, so the atomic identity rule is deliberately not applied
    here).
    """

    label_source: str
    labeling_rule: str
    adjudicated: bool
    ambiguity_policy: str
    taxonomy_id: str | None = None
    taxonomy_version: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str("label_source", self.label_source)
        _require_non_empty_str("labeling_rule", self.labeling_rule)
        _require_real_bool("adjudicated", self.adjudicated)
        _require_non_empty_str("ambiguity_policy", self.ambiguity_policy)
        if self.taxonomy_id is not None:
            _require_non_empty_str("taxonomy_id", self.taxonomy_id)
        if self.taxonomy_version is not None and (
            isinstance(self.taxonomy_version, bool)
            or not isinstance(self.taxonomy_version, int)
            or self.taxonomy_version < 1
        ):
            raise InvalidDecisionError(
                "taxonomy_version must be an int >= 1, got "
                f"{type(self.taxonomy_version).__name__} "
                f"({self.taxonomy_version!r})"
            )

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible payload for fingerprinting."""
        return {
            "label_source": self.label_source,
            "labeling_rule": self.labeling_rule,
            "adjudicated": self.adjudicated,
            "ambiguity_policy": self.ambiguity_policy,
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_version": self.taxonomy_version,
        }


@dataclass(frozen=True, slots=True)
class GroundTruthSemanticsIdentity:
    """What a ground-truth label MEANS, as a statistical target.

    This is the semantics axis of the ground-truth identity, deliberately
    separated from :class:`GroundTruthProvenance` (the full label lineage).
    Two observations may be pooled into one fitting dataset only when their
    ground truth was established under the SAME semantics, regardless of
    which specific producer supplied the labels.

    Fields:

    ``labeling_rule``      the rule that established the label (human
                           adjudication rule, downstream business outcome,
                           heuristic rule, ...). Two different rules can
                           assign the same string label while measuring
                           different statistical targets.
    ``ambiguity_policy``   how ambiguous cases were handled when the label
                           was established.
    ``taxonomy_id``        the candidate taxonomy the label belongs to, or
                           ``None`` when no taxonomy was declared.
    ``taxonomy_version``   the taxonomy version, or ``None`` when unknown.

    Deliberately EXCLUDED:

    ``label_source``  different annotators (or other producers) with
        identical labeling semantics measure the same statistical target
        and must remain poolable; the specific producer stays in
        :class:`GroundTruthProvenance` and in the observation identity.
    ``adjudicated``  this is an existing fit-eligibility gate, not an
        adjudication-protocol identity; no adjudication-protocol field is
        derived from ``label_source``.

    The taxonomy pair may be independently ``None`` (a known taxonomy with
    an unknown version is a real state), so the atomic identity rule is
    deliberately not applied here, mirroring
    :class:`GroundTruthProvenance`.

    Construct instances ONLY through :meth:`from_provenance`: the semantics
    identity is deterministically derived from a provenance, and callers
    must never hand-declare a second semantics set for one
    :class:`GroundTruthProvenance`.
    """

    labeling_rule: str
    ambiguity_policy: str
    taxonomy_id: str | None
    taxonomy_version: int | None

    def __post_init__(self) -> None:
        _require_non_empty_str("labeling_rule", self.labeling_rule)
        _require_non_empty_str("ambiguity_policy", self.ambiguity_policy)
        if self.taxonomy_id is not None:
            _require_non_empty_str("taxonomy_id", self.taxonomy_id)
        if self.taxonomy_version is not None and (
            isinstance(self.taxonomy_version, bool)
            or not isinstance(self.taxonomy_version, int)
            or self.taxonomy_version < 1
        ):
            raise InvalidDecisionError(
                "taxonomy_version must be an int >= 1, got "
                f"{type(self.taxonomy_version).__name__} "
                f"({self.taxonomy_version!r})"
            )

    @classmethod
    def from_provenance(cls, provenance: GroundTruthProvenance) -> GroundTruthSemanticsIdentity:
        """Deterministically derive the semantics identity from a provenance.

        The single centralized derivation: callers must never hand-declare
        a second semantics set for one :class:`GroundTruthProvenance`.
        ``label_source`` is deliberately not carried over (see the class
        docstring).
        """
        return cls(
            labeling_rule=provenance.labeling_rule,
            ambiguity_policy=provenance.ambiguity_policy,
            taxonomy_id=provenance.taxonomy_id,
            taxonomy_version=provenance.taxonomy_version,
        )

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the canonical semantic projection (no version key).

        The canonical semantic projection and the versioned fingerprint
        payload are conceptually separate: this payload carries ONLY the
        four semantic fields, while :attr:`fingerprint` hashes a payload
        that additionally carries the fingerprint schema version.
        """
        return {
            "labeling_rule": self.labeling_rule,
            "ambiguity_policy": self.ambiguity_policy,
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_version": self.taxonomy_version,
        }

    @property
    def fingerprint(self) -> str:
        """Versioned fingerprint of the ground-truth semantics identity."""
        payload: dict[str, JSONValue] = {
            "v": GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
            **self.canonical_payload(),
        }
        return fingerprint(payload)


class GroundTruthResolutionStatus(StrEnum):
    """Whether a ground-truth record carries a resolved value.

    Deliberately does NOT contain ``taxonomy_miss``: a taxonomy miss is
    DERIVED from a resolved ground truth plus the declared outcome space,
    and must never be set by the labeler.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class GroundTruthRecord:
    """A ground-truth value with its resolution status and provenance.

    Exactly three responsibilities: the ``value``, the
    ``resolution_status``, and the ``provenance``. The record does not know
    the decision family, so it validates only JSON compatibility and
    "not None" for a resolved value; the family-specific rule (real ``bool``
    for Bool, non-empty candidate name for Choice) is validated in the
    observation. An unresolved record carries ``value=None``: there is no
    resolved truth to hold.
    """

    value: JSONValue
    resolution_status: GroundTruthResolutionStatus
    provenance: GroundTruthProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.resolution_status, GroundTruthResolutionStatus):
            raise InvalidDecisionError(
                "resolution_status must be a GroundTruthResolutionStatus, got "
                f"{type(self.resolution_status).__name__} "
                f"({self.resolution_status!r})"
            )
        if self.resolution_status is GroundTruthResolutionStatus.RESOLVED:
            if self.value is None:
                raise InvalidDecisionError(
                    "value must not be None when resolution_status is RESOLVED"
                )
            _require_json_value("value", self.value)
        elif self.value is not None:
            raise InvalidDecisionError(
                "value must be None when resolution_status is UNRESOLVED, got "
                f"{type(self.value).__name__} ({self.value!r})"
            )

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible payload for fingerprinting."""
        return {
            "value": self.value,
            "resolution_status": str(self.resolution_status),
            "provenance": self.provenance.canonical_payload(),
        }


# ---------------------------------------------------------------------------
# Calibration binding (Part I)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationBinding:
    """Which probability population an observation belongs to.

    Sourced from the runtime's existing provenance (the ``DecisionTrace``),
    never a second source of truth. The exact probability formulation
    fingerprint commits the formulation internals, so this binding must not
    duplicate them: no ``compiler_id`` / ``compiler_version`` /
    ``assembler_id`` / ``assembler_version`` / ``doctrine_*`` /
    ``candidate_mapping`` fields live here.

    ``model_revision=None`` means UNKNOWN REVISION (INV-26), serialized as
    an explicit JSON null, never a wildcard marker string.

    The optional ``task_id`` / ``domain_id`` / ``taxonomy_id`` /
    ``taxonomy_version`` fields are caller declarations. A declaration is
    provenance, NOT proof of statistical interchangeability.

    This class defines NO policy about whether two bindings with unknown
    dimensions may share a profile; profile matching is out of scope.
    """

    probability_formulation_fingerprint: str
    probability_formulation_fingerprint_version: int
    model: str | None
    model_revision: str | None
    tokenizer: str | None
    tokenizer_revision: str | None
    rendering_semantics: Mapping[str, JSONValue]
    task_id: str | None = None
    domain_id: str | None = None
    taxonomy_id: str | None = None
    taxonomy_version: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(
            "probability_formulation_fingerprint",
            self.probability_formulation_fingerprint,
        )
        if (
            isinstance(self.probability_formulation_fingerprint_version, bool)
            or not isinstance(self.probability_formulation_fingerprint_version, int)
            or self.probability_formulation_fingerprint_version < 1
        ):
            raise InvalidDecisionError(
                "probability_formulation_fingerprint_version must be an int >= 1, "
                "got "
                f"{type(self.probability_formulation_fingerprint_version).__name__} "
                f"({self.probability_formulation_fingerprint_version!r})"
            )
        for name in ("model", "model_revision", "tokenizer", "tokenizer_revision"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty_str(name, value)
        if not isinstance(self.rendering_semantics, Mapping):
            raise InvalidDecisionError(
                "rendering_semantics must be a Mapping, got "
                f"{type(self.rendering_semantics).__name__} "
                f"({self.rendering_semantics!r})"
            )
        for key, value in self.rendering_semantics.items():
            if not isinstance(key, str):
                raise InvalidDecisionError(
                    f"rendering_semantics keys must be str, got {type(key).__name__} ({key!r})"
                )
            _require_json_value(f"rendering_semantics[{key!r}]", value)
        if "enable_thinking" in self.rendering_semantics:
            # Mirrors canonical_rendering_semantics: an int like 1 must not
            # masquerade as True under Python equality or canonical JSON.
            enable_thinking = self.rendering_semantics["enable_thinking"]
            if enable_thinking is not None and not isinstance(enable_thinking, bool):
                raise InvalidDecisionError(
                    "rendering_semantics['enable_thinking'] must be None or a bool, got "
                    f"{type(enable_thinking).__name__} ({enable_thinking!r})"
                )
        if "v" in self.rendering_semantics:
            version = self.rendering_semantics["v"]
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise InvalidDecisionError(
                    "rendering_semantics['v'] must be an int >= 1, got "
                    f"{type(version).__name__} ({version!r})"
                )
        for name in ("task_id", "domain_id", "taxonomy_id"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty_str(name, value)
        if self.taxonomy_version is not None and (
            isinstance(self.taxonomy_version, bool)
            or not isinstance(self.taxonomy_version, int)
            or self.taxonomy_version < 1
        ):
            raise InvalidDecisionError(
                "taxonomy_version must be an int >= 1, got "
                f"{type(self.taxonomy_version).__name__} "
                f"({self.taxonomy_version!r})"
            )
        object.__setattr__(
            self,
            "rendering_semantics",
            _freeze_json_value(self.rendering_semantics),
        )

    @classmethod
    def from_trace(
        cls,
        trace: DecisionTrace,
        *,
        task_id: str | None = None,
        domain_id: str | None = None,
        taxonomy_id: str | None = None,
        taxonomy_version: int | None = None,
    ) -> CalibrationBinding:
        """Build a binding from a runtime ``DecisionTrace``.

        The trace is the single source of truth for the must-bind fields;
        the optional keyword arguments are caller declarations.
        """
        return cls(
            probability_formulation_fingerprint=trace.probability_formulation_fingerprint,
            probability_formulation_fingerprint_version=trace.probability_formulation_fingerprint_version,
            model=trace.model,
            model_revision=trace.model_revision,
            tokenizer=trace.tokenizer,
            tokenizer_revision=trace.tokenizer_revision,
            rendering_semantics=canonical_rendering_semantics(trace.rendering_config),
            task_id=task_id,
            domain_id=domain_id,
            taxonomy_id=taxonomy_id,
            taxonomy_version=taxonomy_version,
        )

    @property
    def is_fully_resolved(self) -> bool:
        """Are all must-bind source dimensions concretely known?

        Answers only that question. It implies nothing about whether two
        bindings may share a profile; no such policy exists here.
        """
        return (
            self.model is not None
            and self.model_revision is not None
            and self.tokenizer is not None
            and self.tokenizer_revision is not None
            and all(
                value is not None for key, value in self.rendering_semantics.items() if key != "v"
            )
        )

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible payload for fingerprinting."""
        return {
            "probability_formulation_fingerprint": (self.probability_formulation_fingerprint),
            "probability_formulation_fingerprint_version": (
                self.probability_formulation_fingerprint_version
            ),
            "model": self.model,
            "model_revision": self.model_revision,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
            "rendering_semantics": _thaw_json_value(self.rendering_semantics),
            "task_id": self.task_id,
            "domain_id": self.domain_id,
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_version": self.taxonomy_version,
        }

    @property
    def fingerprint(self) -> str:
        """Versioned representation of the binding identity.

        The binding is a composite identity (formulation identity + source
        identity + task/domain declarations); the fingerprint is an
        internal, versioned representation of that identity. It does not
        replace the component formulation/source identities.
        """
        payload: dict[str, JSONValue] = {
            "v": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "binding": self.canonical_payload(),
        }
        return fingerprint(payload)


# ---------------------------------------------------------------------------
# Calibration observation (Parts G, H, J, K)
# ---------------------------------------------------------------------------


class CalibrationObservationStatus(StrEnum):
    """Derived observation status.

    ``TAXONOMY_MISS`` is derived, never labeled: it arises when a resolved
    ground truth falls outside the declared semantic outcome space.
    """

    RESOLVED = "resolved"
    TAXONOMY_MISS = "taxonomy_miss"
    UNRESOLVED = "unresolved"


def _require_coherent_linkage(
    result: BoolResult | ChoiceResult,
    trace: DecisionTrace,
) -> None:
    """Require that a result and trace carry matching runtime linkage identities.

    The runtime stamps the same trace id onto the assembled result and the
    trace it reports, so a mismatch means the pair cannot be verified as
    coming from a single evaluation under the supported construction
    contract. Linkage is never inferred from probability values: two
    different executions can emit identical distributions.
    """
    result_linkage = result.trace_id
    if result_linkage is None:
        raise InvalidDecisionError(
            "CalibrationObservation requires a result with execution linkage: "
            "result.trace_id is None while the trace linkage identity is "
            f"{trace.trace_id!r}, so the pair cannot be verified"
        )
    if result_linkage != trace.trace_id:
        raise InvalidDecisionError(
            "CalibrationObservation requires matching runtime linkage "
            f"identities: result linkage identity {result_linkage!r} does not "
            f"match trace linkage identity {trace.trace_id!r}"
        )


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One uncalibrated semantic decision paired with ground truth.

    Status and correctness are deterministically DERIVED from the semantic
    result and the ground-truth record; the constructor accepts no
    ``correct`` argument, so a caller cannot lie about correctness. A
    taxonomy miss is never represented as ``correct=False``: it is not an
    ordinary in-taxonomy model error and must not be mixable into
    winner-correctness fitting by ignoring a status field.

    Probabilities stay SEMANTIC and ORDER-PRESERVING: a tuple of
    ``(name, probability)`` pairs, so a candidate-order difference survives
    fingerprinting (the canonical JSON serializer sorts dict keys, which
    would erase candidate order in a plain dict).

    ``execution_fingerprint`` is optional audit metadata only; it is
    deliberately NOT part of the fingerprint payload. The execution
    fingerprint identifies an execution instance, and a deterministic replay
    that yields a different trace id must not thereby become a different
    semantic observation.

    The only supported construction path is
    :meth:`CalibrationObservation.from_evaluation`, which requires matching
    runtime linkage identities between the result and the trace before any
    provenance is derived. Direct field construction is rejected, and so is
    ``dataclasses.replace`` reconstruction: both rebuild an observation
    through the constructor without the construction capability, and neither
    passes through the supported construction path that requires matching
    runtime linkage identities. Lower-level Python escape hatches such as
    ``object.__new__``, ``copy``, and ``pickle`` are inherent to the
    language, are not supported construction paths, and are not defended
    against. A future persistence or reconstruction entry point must be
    added explicitly and validated separately, and is NOT in scope here.
    """

    decision_family: str
    outcome_order: tuple[str, ...]
    probabilities: tuple[tuple[str, float], ...]
    selected_value: JSONValue
    ground_truth: GroundTruthRecord
    binding: CalibrationBinding
    decision_fingerprint: str
    execution_fingerprint: str | None = None
    status: CalibrationObservationStatus = field(init=False)
    correct: bool | None = field(init=False)

    def __init__(
        self,
        decision_family: str,
        outcome_order: tuple[str, ...],
        probabilities: tuple[tuple[str, float], ...],
        selected_value: JSONValue,
        ground_truth: GroundTruthRecord,
        binding: CalibrationBinding,
        decision_fingerprint: str,
        execution_fingerprint: str | None = None,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _OBSERVATION_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "CalibrationObservation must be constructed via "
                "CalibrationObservation.from_evaluation(...), which requires "
                "matching runtime linkage identities between the result and "
                "the trace before calibration provenance is derived"
            )
        object.__setattr__(self, "decision_family", decision_family)
        object.__setattr__(self, "outcome_order", outcome_order)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "selected_value", selected_value)
        object.__setattr__(self, "ground_truth", ground_truth)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "decision_fingerprint", decision_fingerprint)
        object.__setattr__(self, "execution_fingerprint", execution_fingerprint)
        # A hand-written __init__ means dataclasses does not call __post_init__
        # for us, so the field validation and the derivation of status and
        # correct are invoked explicitly here.
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_non_empty_str("decision_family", self.decision_family)
        if self.decision_family not in ("bool", "choice"):
            raise InvalidDecisionError(
                f"decision_family must be 'bool' or 'choice', got {self.decision_family!r}"
            )
        _require_non_empty_str("decision_fingerprint", self.decision_fingerprint)
        if not self.outcome_order:
            raise InvalidDecisionError("outcome_order must be non-empty")
        if len(set(self.outcome_order)) != len(self.outcome_order):
            raise InvalidDecisionError(
                f"outcome_order entries must be unique, got {self.outcome_order!r}"
            )
        for name, probability in self.probabilities:
            _require_non_empty_str("probability name", name)
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise InvalidDecisionError(
                    f"probability for {name!r} must be a float, got "
                    f"{type(probability).__name__} ({probability!r})"
                )
        if tuple(name for name, _ in self.probabilities) != self.outcome_order:
            raise InvalidDecisionError(
                "probability names must match outcome_order exactly and in order, "
                f"got probabilities {tuple(n for n, _ in self.probabilities)!r} "
                f"against outcome_order {self.outcome_order!r}"
            )
        if not isinstance(self.binding, CalibrationBinding):
            raise InvalidDecisionError(
                "binding must be a CalibrationBinding, got "
                f"{type(self.binding).__name__} ({self.binding!r})"
            )
        if self.execution_fingerprint is not None:
            _require_non_empty_str("execution_fingerprint", self.execution_fingerprint)
        if self.decision_family == "bool":
            if not isinstance(self.selected_value, bool):
                raise InvalidDecisionError(
                    "a Bool observation's selected_value must be a real bool, got "
                    f"{type(self.selected_value).__name__} ({self.selected_value!r})"
                )
            _validate_bool_ground_truth(self.ground_truth)
        else:
            if (
                not isinstance(self.selected_value, str)
                or self.selected_value not in self.outcome_order
            ):
                raise InvalidDecisionError(
                    "a Choice observation's selected_value must be a semantic "
                    f"candidate name in the outcome space, got {self.selected_value!r}"
                )
            _validate_choice_ground_truth(self.ground_truth)
        status = _derive_status(self.decision_family, self.ground_truth, self.outcome_order)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "correct",
            _derive_correct(status, self.selected_value, self.ground_truth),
        )

    @classmethod
    def from_evaluation(
        cls,
        evaluation: Evaluation | tuple[BoolResult | ChoiceResult, DecisionTrace],
        ground_truth: GroundTruthRecord,
        *,
        task_id: str | None = None,
        domain_id: str | None = None,
        taxonomy_id: str | None = None,
        taxonomy_version: int | None = None,
    ) -> CalibrationObservation:
        """Deterministically build an observation from a runtime evaluation.

        Accepts a runtime ``Evaluation`` or a ``(result, trace)`` pair. The
        caller never hand-supplies ``selected_value``, the probability
        formulation fingerprint, ``model_revision``, or ``correct``: the
        runtime already knows them.
        """
        if isinstance(evaluation, Evaluation):
            result = evaluation.result
            trace = evaluation.trace
        elif isinstance(evaluation, tuple) and len(evaluation) == 2:
            result, trace = evaluation
        else:
            raise InvalidDecisionError(
                "evaluation must be an Evaluation or a (result, trace) pair, got "
                f"{type(evaluation).__name__} ({evaluation!r})"
            )
        if not isinstance(trace, DecisionTrace):
            raise InvalidDecisionError(f"trace must be a DecisionTrace, got {type(trace).__name__}")
        if not isinstance(result, (BoolResult, ChoiceResult)):
            raise InvalidDecisionError(
                f"result must be a BoolResult or ChoiceResult, got {type(result).__name__}"
            )
        _require_coherent_linkage(result, trace)

        binding = CalibrationBinding.from_trace(
            trace,
            task_id=task_id,
            domain_id=domain_id,
            taxonomy_id=taxonomy_id,
            taxonomy_version=taxonomy_version,
        )

        if trace.decision_family == "bool":
            outcome_order = _BOOL_OUTCOME_ORDER
            if not isinstance(result, BoolResult):
                raise InvalidDecisionError(
                    f"a bool decision_family requires a BoolResult, got {type(result).__name__}"
                )
            probabilities: tuple[tuple[str, float], ...] = (
                ("false", result.probability_false),
                ("true", result.probability_true),
            )
            selected_value: JSONValue = _select_bool_value(result.probability_true)
            _validate_bool_ground_truth(ground_truth)
        elif trace.decision_family == "choice":
            if not isinstance(result, ChoiceResult):
                raise InvalidDecisionError(
                    f"a choice decision_family requires a ChoiceResult, got {type(result).__name__}"
                )
            outcome_order = result.choice_names
            probabilities = tuple((name, result.probabilities[name]) for name in outcome_order)
            selected_value = result.value
            _validate_choice_ground_truth(ground_truth)
        else:
            raise InvalidDecisionError(f"unsupported decision_family {trace.decision_family!r}")

        return cls(
            decision_family=trace.decision_family,
            outcome_order=outcome_order,
            probabilities=probabilities,
            selected_value=selected_value,
            ground_truth=ground_truth,
            binding=binding,
            decision_fingerprint=trace.decision_fingerprint,
            execution_fingerprint=trace.execution_fingerprint,
            _construction_token=_OBSERVATION_CONSTRUCTION_TOKEN,
        )

    @property
    def fit_eligible(self) -> bool:
        """May this observation enter a calibration fitting dataset?

        True only when the status is RESOLVED, the ground-truth provenance
        is adjudicated, and the required provenance is valid. Scoring
        diagnostics (label mass, model confidence, score gaps) deliberately play no
        part: the Phase 3A design froze the absence of any such empirical
        threshold.
        """
        if self.status is not CalibrationObservationStatus.RESOLVED:
            return False
        provenance = self.ground_truth.provenance
        try:
            provenance.canonical_payload()
        except InvalidDecisionError:
            return False
        return provenance.adjudicated is True

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible payload for fingerprinting.

        Commits the semantic identity (decision fingerprint), the
        probability formulation identity plus its version, the binding
        payload, the ordered semantic outcome probabilities, the selected
        value, the full ground-truth record, the derived status, and the
        derived correctness. Excludes ``trace_id``, wall-clock timestamps,
        execution attempt counts, logging metadata, and the execution
        fingerprint (audit metadata only).
        """
        return {
            "v": CALIBRATION_OBSERVATION_FINGERPRINT_VERSION,
            "decision_family": self.decision_family,
            "decision_fingerprint": self.decision_fingerprint,
            "probability_formulation_fingerprint": (
                self.binding.probability_formulation_fingerprint
            ),
            "probability_formulation_fingerprint_version": (
                self.binding.probability_formulation_fingerprint_version
            ),
            "binding": self.binding.canonical_payload(),
            "outcome_order": list(self.outcome_order),
            "probabilities": [[name, probability] for name, probability in self.probabilities],
            "selected_value": self.selected_value,
            "ground_truth": self.ground_truth.canonical_payload(),
            "status": str(self.status),
            "correct": self.correct,
        }

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint of the observation's canonical payload."""
        return fingerprint(self.canonical_payload())


def _select_bool_value(probability_true: float) -> bool:
    """Derive the Bool selected value with the existing tie-break doctrine.

    First in declared outcome order wins; the Bool semantic outcome order is
    ``("false", "true")`` (the same order the formulation identity uses), so
    a 0.5 / 0.5 tie selects ``False``.
    """
    return probability_true > 0.5


def _selected_probability(observation: CalibrationObservation) -> float:
    """Extract the uncalibrated selected semantic probability of an observation.

    The recorded ``selected_value`` is the selection carried by the supplied
    runtime-linked result: the winner is NOT recomputed and tie-breaking is
    NOT re-run. For Choice decisions the probability is looked up by semantic
    candidate name (never by a scoring label such as ``A``/``B``/``C`` and
    never by a token id). For Bool decisions the name is chosen from the
    recorded boolean over the Bool semantic outcome order.

    This is the single owner of the calibrator input-score semantics: the
    fitting objective and the evaluation metrics both consume exactly this
    function, so no second mapping, argmax, or tie-break implementation may
    exist anywhere else.
    """
    selected_value = observation.selected_value
    if observation.decision_family == "bool":
        if selected_value is True:
            name = _BOOL_TRUE_NAME
        elif selected_value is False:
            name = _BOOL_FALSE_NAME
        else:
            raise InvalidDecisionError(
                "cannot map the recorded selected_value "
                f"{selected_value!r} of a bool observation to an outcome name; "
                "the recorded selected_value must be a real bool"
            )
    else:
        if not isinstance(selected_value, str):
            raise InvalidDecisionError(
                "cannot map the recorded selected_value "
                f"{selected_value!r} of a choice observation to a semantic "
                "candidate name; the recorded selected_value must be a "
                "candidate name string"
            )
        name = selected_value
    for candidate_name, probability in observation.probabilities:
        if candidate_name == name:
            return probability
    raise InvalidDecisionError(
        f"the recorded selected_value {selected_value!r} of observation "
        f"{observation.fingerprint} has no recorded probability under outcome "
        f"order {observation.outcome_order!r}; the uncalibrated selected "
        "semantic probability cannot be extracted"
    )


def _validate_bool_ground_truth(ground_truth: GroundTruthRecord) -> None:
    """A resolved Bool ground truth must be a real bool; no coercion."""
    if ground_truth.resolution_status is not GroundTruthResolutionStatus.RESOLVED:
        return
    if not isinstance(ground_truth.value, bool):
        raise InvalidDecisionError(
            "a resolved Bool ground truth must be a real bool, got "
            f"{type(ground_truth.value).__name__} ({ground_truth.value!r})"
        )


def _validate_choice_ground_truth(ground_truth: GroundTruthRecord) -> None:
    """A resolved Choice ground truth must be a non-empty candidate name.

    A scoring label like ``A`` / ``B`` / ``C`` is never a valid ground-truth
    outcome; the semantic candidate name is required.
    """
    if ground_truth.resolution_status is not GroundTruthResolutionStatus.RESOLVED:
        return
    _require_non_empty_str("ground truth value (candidate name)", ground_truth.value)


def _derive_status(
    decision_family: str,
    ground_truth: GroundTruthRecord,
    outcome_order: Sequence[str],
) -> CalibrationObservationStatus:
    """Deterministically derive the observation status."""
    if ground_truth.resolution_status is GroundTruthResolutionStatus.UNRESOLVED:
        return CalibrationObservationStatus.UNRESOLVED
    if decision_family == "bool":
        # Bool branch: the outcome space is the real bools {False, True} and
        # a resolved Bool ground truth has already been validated to be a
        # real bool, so Bool can never be a taxonomy miss.
        return CalibrationObservationStatus.RESOLVED
    if ground_truth.value not in outcome_order:
        # Choice branch: the resolved ground truth is a semantic candidate
        # name that falls outside the declared outcome space.
        return CalibrationObservationStatus.TAXONOMY_MISS
    return CalibrationObservationStatus.RESOLVED


def _derive_correct(
    status: CalibrationObservationStatus,
    selected_value: JSONValue,
    ground_truth: GroundTruthRecord,
) -> bool | None:
    """Deterministically derive correctness; never caller-supplied."""
    if status is not CalibrationObservationStatus.RESOLVED:
        return None
    return bool(selected_value == ground_truth.value)


# ---------------------------------------------------------------------------
# Calibration dataset (Parts L, M)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    """The observations actually prepared for calibration fitting.

    Pooling requires three things, all enforced in ``__post_init__`` so
    direct construction cannot bypass what :meth:`create` checks:

    1. every member is fit-eligible; non-fit-eligible observations
       (taxonomy miss, unresolved, unadjudicated) are explicitly rejected,
       never silently filtered;
    2. every member's ``CalibrationBinding`` canonical payload is
       structurally equal to the dataset binding's canonical payload:
       different exact formulation or source bindings are never silently
       pooled (INV-23), and shared formulation-family membership never
       authorises pooling (INV-24);
    3. every member's :class:`GroundTruthSemanticsIdentity` (derived from
       its ground-truth provenance) is identical: observations whose ground
       truth was established under different labeling rules, ambiguity
       policies, or taxonomies measure different statistical targets and
       are never pooled, even when the binding matches.

    Full :class:`GroundTruthProvenance` equality is deliberately NOT
    required: a specific label producer is not part of the statistical
    target, so different ``label_source`` values may coexist in one dataset
    while their observation identities remain distinct.

    The probability binding and the ground-truth semantics stay orthogonal:
    conceptually a fitting population is ``CalibrationBinding +
    GroundTruthSemanticsIdentity``, but they remain separate classes and
    the semantics identity is never placed inside the binding.

    The dataset fingerprint answers ONLY "which fit-eligible observations
    does this fitting dataset contain?". It does not answer whether the data
    represents the deployment population, whether the sample size is
    enough, or whether calibration is good.
    """

    binding: CalibrationBinding
    observations: tuple[CalibrationObservation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.observations:
            raise InvalidDecisionError(
                "a calibration fitting dataset must contain at least one observation"
            )
        binding_payload = self.binding.canonical_payload()
        binding_payload_json = canonical_json(binding_payload)
        dataset_semantics = GroundTruthSemanticsIdentity.from_provenance(
            self.observations[0].ground_truth.provenance
        )
        dataset_semantics_payload = dataset_semantics.canonical_payload()
        dataset_semantics_payload_json = canonical_json(dataset_semantics_payload)
        rejected: list[str] = []
        for index, observation in enumerate(self.observations):
            if not observation.fit_eligible:
                rejected.append(
                    f"observation {index}: status {str(observation.status)!r} is not fit-eligible"
                )
            observation_binding_payload = observation.binding.canonical_payload()
            if canonical_json(observation_binding_payload) != binding_payload_json:
                rejected.append(
                    f"observation {index}: binding canonical payload "
                    f"{observation_binding_payload!r} does not match the "
                    f"dataset binding canonical payload {binding_payload!r} "
                    f"(dataset binding fingerprint {self.binding.fingerprint!r}, "
                    f"observation binding fingerprint "
                    f"{observation.binding.fingerprint!r})"
                )
            observation_semantics = GroundTruthSemanticsIdentity.from_provenance(
                observation.ground_truth.provenance
            )
            observation_semantics_payload = observation_semantics.canonical_payload()
            if canonical_json(observation_semantics_payload) != dataset_semantics_payload_json:
                rejected.append(
                    f"observation {index}: ground-truth semantics identity "
                    f"{observation_semantics_payload!r} (fingerprint "
                    f"{observation_semantics.fingerprint!r}) does not match the "
                    f"dataset ground-truth semantics identity "
                    f"{dataset_semantics_payload!r} (fingerprint "
                    f"{dataset_semantics.fingerprint!r})"
                )
        if rejected:
            raise InvalidDecisionError(
                "calibration fitting dataset rejected "
                f"{len(rejected)} of {len(self.observations)} observations: " + "; ".join(rejected)
            )

    @classmethod
    def create(
        cls,
        observations: Sequence[CalibrationObservation],
    ) -> CalibrationDataset:
        """Build a fitting dataset from fit-eligible, same-binding observations.

        The binding is derived from the first observation; every other
        validation (fit eligibility, binding structural equality,
        ground-truth semantics equality) is enforced in ``__post_init__``,
        so this classmethod gains no way to bypass the pooling checks.
        """
        if not observations:
            raise InvalidDecisionError(
                "a calibration fitting dataset must contain at least one observation"
            )
        binding = observations[0].binding
        return cls(binding=binding, observations=tuple(observations))

    @property
    def ground_truth_semantics(self) -> GroundTruthSemanticsIdentity:
        """The dataset's derived ground-truth semantics identity.

        This is a read-only DERIVED property, not a constructor field: a
        dataset can never claim semantics A while its observations say B.
        It is computed deterministically from the observations: because
        ``__post_init__`` rejects any observation whose derived semantics
        identity differs, all observations agree by the time this property
        can be read, so deriving it from the first observation's
        ground-truth provenance is safe and deterministic.
        """
        return GroundTruthSemanticsIdentity.from_provenance(
            self.observations[0].ground_truth.provenance
        )

    @property
    def fingerprint(self) -> str:
        """Row-order independent, multiplicity-preserving dataset fingerprint.

        Implemented as a hash over the canonical ORDERED list of sorted
        observation fingerprints: a multiset, not a set hash. The payload
        commits the binding fingerprint and the ground-truth semantics
        fingerprint, each with its own fingerprint schema version.
        """
        observation_fingerprints: list[JSONValue] = [
            observation.fingerprint for observation in self.observations
        ]
        observation_fingerprints.sort(key=str)
        semantics = self.ground_truth_semantics
        payload: dict[str, JSONValue] = {
            "v": CALIBRATION_DATASET_FINGERPRINT_VERSION,
            "binding_fingerprint": self.binding.fingerprint,
            "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "ground_truth_semantics_fingerprint": semantics.fingerprint,
            "ground_truth_semantics_fingerprint_version": (
                GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
            ),
            "observation_fingerprints": observation_fingerprints,
        }
        return fingerprint(payload)


# ---------------------------------------------------------------------------
# Calibration profile (Phase 4C.1)
# ---------------------------------------------------------------------------


def _require_method_state_mapping(
    name: str,
    value: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    """Validate a method-state mapping and return a plain mutable copy.

    ``None`` is treated as an empty mapping. The value must be a ``Mapping``
    with ``str`` keys whose values are canonical JSON values (the existing
    ``_require_json_value`` domain: no ``NaN``, no infinities, no tuples, no
    sets, no callables, no arbitrary objects). The returned copy is plain and
    mutable; the caller deep-freezes it before storing.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidDecisionError(
            f"{name} must be a Mapping or None, got {type(value).__name__} ({value!r})"
        )
    for key, item in value.items():
        if not isinstance(key, str):
            raise InvalidDecisionError(
                f"{name} keys must be str, got {type(key).__name__} ({key!r})"
            )
        _require_json_value(f"{name}[{key!r}]", item)
    return dict(value)


def _freeze_method_state(value: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
    """Deep-freeze a validated method-state mapping."""
    return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})


def _require_coherent_taxonomy_identity(
    binding: CalibrationBinding,
    ground_truth_semantics: GroundTruthSemanticsIdentity,
) -> None:
    """Fail closed when the binding and ground-truth taxonomies contradict.

    The binding's declared taxonomy and the ground-truth semantics' taxonomy
    describe the same label space, so two concrete but different taxonomies
    (or two concrete but different versions of one taxonomy) cannot both be
    true. Cross-taxonomy calibration would require an explicit taxonomy
    mapping identity, which does not exist. An unknown taxonomy on either
    side is allowed: absence is not a contradiction, and no equality is
    invented for it. Neither identity is ever rewritten or defaulted.
    """
    binding_taxonomy_id = binding.taxonomy_id
    semantics_taxonomy_id = ground_truth_semantics.taxonomy_id
    if binding_taxonomy_id is None or semantics_taxonomy_id is None:
        return
    if binding_taxonomy_id != semantics_taxonomy_id:
        raise InvalidDecisionError(
            "the binding taxonomy and the ground-truth taxonomy are different: "
            f"binding taxonomy_id {binding_taxonomy_id!r} vs ground-truth "
            f"taxonomy_id {semantics_taxonomy_id!r}. Cross-taxonomy calibration "
            "requires an explicit taxonomy mapping identity, which does not exist"
        )
    binding_taxonomy_version = binding.taxonomy_version
    semantics_taxonomy_version = ground_truth_semantics.taxonomy_version
    if binding_taxonomy_version is None or semantics_taxonomy_version is None:
        return
    if binding_taxonomy_version != semantics_taxonomy_version:
        raise InvalidDecisionError(
            "the binding taxonomy version and the ground-truth taxonomy version "
            f"are different for taxonomy_id {binding_taxonomy_id!r}: binding "
            f"taxonomy_version {binding_taxonomy_version!r} vs ground-truth "
            f"taxonomy_version {semantics_taxonomy_version!r}"
        )


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """A reusable fitted calibration artifact and its full identity.

    A profile records WHICH probability population it was fitted for (the
    :class:`CalibrationBinding`), WHAT the fitted labels mean (the
    :class:`GroundTruthSemanticsIdentity`), WHICH statistical target and
    calibrator input score it assumes (the winner-correctness target and the
    uncalibrated selected probability), WHICH fitting method produced it
    (``method_id`` / ``method_version`` / ``method_configuration``), the
    fitted numbers themselves (``fitted_parameters``), and WHICH fitting
    dataset it was fitted on (the dataset fingerprint and its schema
    version).

    The profile is a reusable fitted artifact. It is never a
    ``DecisionResult``, an ``EvaluationResult``, a registry entry, a runtime
    trace, or a metric result.

    A valid profile proves ONLY that its provenance and identity are
    structurally coherent. It does NOT prove that the calibrator improves
    Brier or log loss, that it generalizes beyond its training data, that
    the training data is representative of the deployment population, or
    that a training/evaluation split is independent. The training dataset
    fingerprint says which observations were fitted, NOT that they were a
    good fitting population.

    A supported public fitter now exists:
    :func:`fit_l2_logistic_selected_probability` produces a profile from a
    fitting :class:`CalibrationDataset`. That fitter proves only that its
    declared fitting problem was solved; evaluating the fitted mapping on
    held-out data, and runtime profile application, are later phases.
    Applying a profile does not require a runtime ground-truth record: the
    profile's ground-truth semantics identity describes what the fitted
    ``predicted_correctness`` refers to, and the event being predicted
    normally has no ground truth yet.

    Internally, construction goes through the module-private
    :meth:`_from_fitted_state`, which derives the binding, the ground-truth
    semantics identity, the target identity, the input-score identity, and
    the training dataset fingerprint from a fitted
    :class:`CalibrationDataset`. Direct field construction is rejected, and
    so is ``dataclasses.replace`` reconstruction: both rebuild a profile
    through the constructor without the construction capability, and neither
    passes through the internal construction path that derives the
    identity from a dataset. Lower-level Python escape hatches such as
    ``object.__new__``, ``copy``, and ``pickle`` are inherent to the
    language, are not supported construction paths, and are not defended
    against.
    """

    binding: CalibrationBinding
    ground_truth_semantics: GroundTruthSemanticsIdentity
    target_id: str
    target_version: int
    input_score_id: str
    input_score_version: int
    method_id: str
    method_version: int
    method_configuration: Mapping[str, JSONValue]
    fitted_parameters: Mapping[str, JSONValue]
    training_dataset_fingerprint: str
    training_dataset_fingerprint_version: int

    def __init__(
        self,
        binding: CalibrationBinding,
        ground_truth_semantics: GroundTruthSemanticsIdentity,
        target_id: str,
        target_version: int,
        input_score_id: str,
        input_score_version: int,
        method_id: str,
        method_version: int,
        method_configuration: Mapping[str, JSONValue],
        fitted_parameters: Mapping[str, JSONValue],
        training_dataset_fingerprint: str,
        training_dataset_fingerprint_version: int,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _PROFILE_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "CalibrationProfile cannot be constructed directly: a profile is "
                "produced by a supported calibration fitter "
                "(fit_l2_logistic_selected_probability), which derives its "
                "identity from a fitted CalibrationDataset"
            )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "ground_truth_semantics", ground_truth_semantics)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "target_version", target_version)
        object.__setattr__(self, "input_score_id", input_score_id)
        object.__setattr__(self, "input_score_version", input_score_version)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "method_version", method_version)
        object.__setattr__(self, "method_configuration", method_configuration)
        object.__setattr__(self, "fitted_parameters", fitted_parameters)
        object.__setattr__(self, "training_dataset_fingerprint", training_dataset_fingerprint)
        object.__setattr__(
            self, "training_dataset_fingerprint_version", training_dataset_fingerprint_version
        )
        # A hand-written __init__ means dataclasses does not call __post_init__
        # for us, so the field validation is invoked explicitly here.
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CalibrationBinding):
            raise InvalidDecisionError(
                "binding must be a CalibrationBinding, got "
                f"{type(self.binding).__name__} ({self.binding!r})"
            )
        if not isinstance(self.ground_truth_semantics, GroundTruthSemanticsIdentity):
            raise InvalidDecisionError(
                "ground_truth_semantics must be a GroundTruthSemanticsIdentity, got "
                f"{type(self.ground_truth_semantics).__name__} "
                f"({self.ground_truth_semantics!r})"
            )
        _require_non_empty_str("target_id", self.target_id)
        if (
            isinstance(self.target_version, bool)
            or not isinstance(self.target_version, int)
            or self.target_version < 1
        ):
            raise InvalidDecisionError(
                "target_version must be an int >= 1, got "
                f"{type(self.target_version).__name__} ({self.target_version!r})"
            )
        _require_non_empty_str("input_score_id", self.input_score_id)
        if (
            isinstance(self.input_score_version, bool)
            or not isinstance(self.input_score_version, int)
            or self.input_score_version < 1
        ):
            raise InvalidDecisionError(
                "input_score_version must be an int >= 1, got "
                f"{type(self.input_score_version).__name__} "
                f"({self.input_score_version!r})"
            )
        _require_non_empty_str("method_id", self.method_id)
        if (
            isinstance(self.method_version, bool)
            or not isinstance(self.method_version, int)
            or self.method_version < 1
        ):
            raise InvalidDecisionError(
                "method_version must be an int >= 1, got "
                f"{type(self.method_version).__name__} ({self.method_version!r})"
            )
        if not isinstance(self.method_configuration, Mapping):
            raise InvalidDecisionError(
                "method_configuration must be a Mapping, got "
                f"{type(self.method_configuration).__name__} "
                f"({self.method_configuration!r})"
            )
        if not isinstance(self.fitted_parameters, Mapping):
            raise InvalidDecisionError(
                "fitted_parameters must be a Mapping, got "
                f"{type(self.fitted_parameters).__name__} ({self.fitted_parameters!r})"
            )
        _require_non_empty_str("training_dataset_fingerprint", self.training_dataset_fingerprint)
        if (
            isinstance(self.training_dataset_fingerprint_version, bool)
            or not isinstance(self.training_dataset_fingerprint_version, int)
            or self.training_dataset_fingerprint_version < 1
        ):
            raise InvalidDecisionError(
                "training_dataset_fingerprint_version must be an int >= 1, got "
                f"{type(self.training_dataset_fingerprint_version).__name__} "
                f"({self.training_dataset_fingerprint_version!r})"
            )

    @classmethod
    def _from_fitted_state(
        cls,
        dataset: CalibrationDataset,
        *,
        method_id: str,
        method_version: int,
        method_configuration: Mapping[str, JSONValue] | None = None,
        fitted_parameters: Mapping[str, JSONValue] | None = None,
    ) -> CalibrationProfile:
        """Build a profile from a fitted dataset and a method's fitted state.

        Internal producer for a future fitting harness. The binding, the
        ground-truth semantics identity, the target identity, the input-score
        identity, and the training dataset fingerprint are all derived from
        the dataset and the module constants; a caller cannot override any
        of them.
        """
        if not isinstance(dataset, CalibrationDataset):
            raise InvalidDecisionError(
                f"dataset must be a CalibrationDataset, got {type(dataset).__name__} ({dataset!r})"
            )
        _require_non_empty_str("method_id", method_id)
        if (
            isinstance(method_version, bool)
            or not isinstance(method_version, int)
            or (method_version < 1)
        ):
            raise InvalidDecisionError(
                "method_version must be an int >= 1, got "
                f"{type(method_version).__name__} ({method_version!r})"
            )
        configuration = _freeze_method_state(
            _require_method_state_mapping("method_configuration", method_configuration)
        )
        parameters = _freeze_method_state(
            _require_method_state_mapping("fitted_parameters", fitted_parameters)
        )
        ground_truth_semantics = dataset.ground_truth_semantics
        _require_coherent_taxonomy_identity(dataset.binding, ground_truth_semantics)
        return cls(
            binding=dataset.binding,
            ground_truth_semantics=ground_truth_semantics,
            target_id=WINNER_CORRECTNESS_TARGET_ID,
            target_version=WINNER_CORRECTNESS_TARGET_VERSION,
            input_score_id=UNCALIBRATED_SELECTED_PROBABILITY_ID,
            input_score_version=UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
            method_id=method_id,
            method_version=method_version,
            method_configuration=configuration,
            fitted_parameters=parameters,
            training_dataset_fingerprint=dataset.fingerprint,
            training_dataset_fingerprint_version=CALIBRATION_DATASET_FINGERPRINT_VERSION,
            _construction_token=_PROFILE_CONSTRUCTION_TOKEN,
        )

    def require_binding_match(self, binding: CalibrationBinding) -> None:
        """Require an exact binding match or raise.

        The comparison uses the full canonical binding identity (the
        canonical JSON of the binding's canonical payload), not a
        hand-picked subset of dimensions. There is no fallback: formulation
        family similarity, a shared model, a shared task declaration, or a
        shared taxonomy never authorize a partial match.
        """
        if not isinstance(binding, CalibrationBinding):
            raise InvalidDecisionError(
                f"binding must be a CalibrationBinding, got {type(binding).__name__} ({binding!r})"
            )
        profile_binding_json = canonical_json(self.binding.canonical_payload())
        candidate_binding_json = canonical_json(binding.canonical_payload())
        if profile_binding_json != candidate_binding_json:
            raise InvalidDecisionError(
                "the profile binding does not match the supplied binding: profile "
                f"binding fingerprint {self.binding.fingerprint!r} vs supplied "
                f"binding fingerprint {binding.fingerprint!r}"
            )

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible payload for fingerprinting.

        The binding and the ground-truth semantics identity are committed by
        fingerprint plus schema version only; their constituent fields are
        deliberately not duplicated here. ``method_configuration`` and
        ``fitted_parameters`` are distinct identity components and are
        emitted as plain thawed dicts.
        """
        return {
            "v": CALIBRATION_PROFILE_FINGERPRINT_VERSION,
            "binding": {
                "binding_fingerprint": self.binding.fingerprint,
                "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            },
            "ground_truth_semantics": {
                "ground_truth_semantics_fingerprint": self.ground_truth_semantics.fingerprint,
                "ground_truth_semantics_fingerprint_version": (
                    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
                ),
            },
            "target_id": self.target_id,
            "target_version": self.target_version,
            "input_score_id": self.input_score_id,
            "input_score_version": self.input_score_version,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "method_configuration": _thaw_json_value(self.method_configuration),
            "fitted_parameters": _thaw_json_value(self.fitted_parameters),
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "training_dataset_fingerprint_version": self.training_dataset_fingerprint_version,
        }

    @property
    def fingerprint(self) -> str:
        """Versioned fingerprint of the calibration profile identity."""
        return fingerprint(self.canonical_payload())


# ---------------------------------------------------------------------------
# L2-regularized logistic scaling on the selected probability (Phase 4C.2)
# ---------------------------------------------------------------------------

#: The first supported calibration fitting method. Its v1 mapping is
#: ``q(p) = sigmoid(slope * p + intercept)``, where ``p`` is the uncalibrated
#: selected semantic probability produced by :func:`_selected_probability`.
#: ``slope`` and ``intercept`` are the only fitted parameters, and both are
#: L2-regularized.
#:
#: This is deliberately NOT temperature scaling: temperature scaling divides a
#: logit by a fitted scalar and leaves no intercept, whereas this method is an
#: affine map in the raw probability domain.
#:
#: The mapping has the same functional form as Platt scaling, so the
#: distinction is not the formula: classic Platt scaling fits a sigmoid over a
#: decision-function value and commonly targets the signed distance to an SVM
#: separating hyperplane; here the single feature IS the uncalibrated selected
#: probability itself, and the positive L2 term is mandatory rather than
#: optional. "L2-regularized logistic scaling on selected probability" is the
#: precise name.
L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID = "l2-logistic-selected-probability"
L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION = 1

_L2_LOGISTIC_OBJECTIVE_ID = "mean-bernoulli-nll-plus-l2"
_L2_LOGISTIC_OBJECTIVE_VERSION = 1
_L2_LOGISTIC_INPUT_TRANSFORM_ID = "identity-selected-probability"
_L2_LOGISTIC_INPUT_TRANSFORM_VERSION = 1
_L2_LOGISTIC_ENDPOINT_POLICY_ID = "exact-raw-selected-probability"
_L2_LOGISTIC_ENDPOINT_POLICY_VERSION = 1
_L2_LOGISTIC_SOLVER_ID = "newton-backtracking"
_L2_LOGISTIC_SOLVER_VERSION = 2
_L2_LOGISTIC_CONVERGENCE_ID = "strong-convexity-objective-gap"
_L2_LOGISTIC_CONVERGENCE_VERSION = 1
_L2_LOGISTIC_REGULARIZED_PARAMETERS: tuple[str, ...] = ("slope", "intercept")

_L2_LOGISTIC_INITIAL_SLOPE = 0.0
_L2_LOGISTIC_INITIAL_INTERCEPT = 0.0
#: Declared objective-gap tolerance of the convergence certificate. The
#: objective is ``l2_strength``-strongly convex, so a gradient norm bound
#: certifies suboptimality; see :func:`_l2_logistic_certificate_threshold`.
#: Justified against the three required factors: float64 carries about 2.2e-16
#: relative precision, so a 1e-16 objective gap sits at the representable
#: resolution of an order-one objective and a tighter tolerance cannot be
#: defended; the two-parameter objective reaches its float plateau long before
#: 1e-14, so tightening to 1e-16, 1e-18, 1e-20, or 1e-21 changes no fitted
#: parameter; and against a 60-digit reference solution 1e-14 restores the
#: ordinary-strength fitted parameters exactly while 1e-12 stops the solver one
#: Newton step early and pushes ``l2_strength = 0.01`` about 7e-7 away from the
#: optimum. Among the tested candidates (1e-12, 1e-14, 1e-16, 1e-18, 1e-20,
#: 1e-21), 1e-14 was the largest that preserved the declared ordinary-strength
#: parameter-fidelity criterion on the recorded regression cases; the claim does
#: not extend to untested tolerances or datasets.
_L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE = 1e-14
_L2_LOGISTIC_MAX_ITERATIONS = 100
_L2_LOGISTIC_BACKTRACKING_FACTOR = 0.5
_L2_LOGISTIC_ARMIJO_COEFFICIENT = 1e-4
_L2_LOGISTIC_MAX_BACKTRACKING_STEPS = 60


def _stable_sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid.

    The branch keeps the exponential argument non-positive, so the result is
    always finite for finite ``z``.

    Mathematically the logistic sigmoid lies in ``(0, 1)`` for finite ``z``.
    The binary64 result may round to exactly ``0.0`` or ``1.0`` for
    sufficiently large ``|z|``: ``exp(-z)`` underflows to ``0.0`` once ``z``
    exceeds roughly ``745``, so ``sigmoid(z)`` becomes exactly ``1.0``, and
    symmetrically it becomes exactly ``0.0`` for sufficiently negative ``z``.
    Label-aware loss, residual, and curvature calculations therefore must not
    rely on subtracting the rounded endpoint from ``1``; they are written in
    the algebraically equivalent form that stays significant at the endpoints
    (see :func:`_binary_logistic_terms`).
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _stable_softplus(z: float) -> float:
    """Numerically stable ``log(1 + exp(z))``.

    Written as ``z + log1p(exp(-z))`` for ``z >= 0`` so the exponential
    argument is never positive. ``log(sigmoid(z))`` and ``log(1 - sigmoid(z))``
    are never formed, so the training objective has no hidden approximation.
    """
    if z >= 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _binary_logistic_terms(z: float, correct: bool) -> tuple[float, float, float]:
    """Label-aware ``(loss, residual, curvature)`` of one Bernoulli row.

    ``loss`` is the row's Bernoulli NLL, ``residual`` is ``d(loss)/dz``, and
    ``curvature`` is ``d2(loss)/dz2``. The objective, gradient, and Hessian all
    consume this one function so no two of them can disagree numerically.

    Each branch is written in the form that stays significant when ``q`` itself
    rounds to exactly ``0.0`` or ``1.0``. For a correct row at large positive
    ``z`` the algebraically equivalent ``softplus(z) - z`` cancels to exactly
    ``0.0`` although the true loss is a representable tiny positive number, and
    ``q * (1 - q)`` cancels the curvature for the same reason. Neither
    ``log(sigmoid(z))`` nor ``log(1 - sigmoid(z))`` is ever formed.
    """
    q = _stable_sigmoid(z)
    complement = _stable_sigmoid(-z)
    curvature = q * complement
    if correct:
        return _stable_softplus(-z), -complement, curvature
    return _stable_softplus(z), q, curvature


def _l2_logistic_objective(
    rows: Sequence[tuple[float, float]],
    slope: float,
    intercept: float,
    l2_strength: float,
) -> float:
    """Mean Bernoulli NLL plus the L2 penalty, at the given parameters."""
    terms = []
    for p, y in rows:
        loss, _residual, _curvature = _binary_logistic_terms(slope * p + intercept, y == 1.0)
        terms.append(loss)
    mean_nll = math.fsum(terms) / len(rows)
    return mean_nll + (l2_strength / 2.0) * (slope * slope + intercept * intercept)


def _l2_logistic_gradient(
    rows: Sequence[tuple[float, float]],
    slope: float,
    intercept: float,
    l2_strength: float,
) -> tuple[float, float]:
    """Objective gradient: mean(residual * p) + lambda * slope, and the intercept twin."""
    slope_terms = []
    intercept_terms = []
    for p, y in rows:
        _loss, residual, _curvature = _binary_logistic_terms(slope * p + intercept, y == 1.0)
        slope_terms.append(residual * p)
        intercept_terms.append(residual)
    count = len(rows)
    grad_slope = math.fsum(slope_terms) / count + l2_strength * slope
    grad_intercept = math.fsum(intercept_terms) / count + l2_strength * intercept
    return grad_slope, grad_intercept


def _l2_logistic_data_hessian(
    rows: Sequence[tuple[float, float]],
    slope: float,
    intercept: float,
) -> tuple[float, float, float]:
    """Symmetric 2x2 Hessian ``(H_aa, H_ab, H_bb)`` of the mean NLL alone.

    The L2 penalty is deliberately excluded so the caller can assemble the
    regularized matrix from the data term and the penalty term separately.
    Adding ``l2_strength`` to a diagonal entry before the determinant is formed
    loses the penalty entirely when ``l2_strength`` is below the unit in the
    last place of the data term, which would make a well-posed fit look
    singular.
    """
    aa_terms = []
    ab_terms = []
    bb_terms = []
    for p, y in rows:
        _loss, _residual, curvature = _binary_logistic_terms(slope * p + intercept, y == 1.0)
        aa_terms.append(curvature * p * p)
        ab_terms.append(curvature * p)
        bb_terms.append(curvature)
    count = len(rows)
    return (
        math.fsum(aa_terms) / count,
        math.fsum(ab_terms) / count,
        math.fsum(bb_terms) / count,
    )


def _l2_logistic_hessian(
    rows: Sequence[tuple[float, float]],
    slope: float,
    intercept: float,
    l2_strength: float,
) -> tuple[float, float, float]:
    """Symmetric 2x2 Hessian ``(H_aa, H_ab, H_bb)`` at the given parameters.

    This is the Hessian of the regularized objective: the mean-NLL Hessian with
    ``l2_strength`` added to both diagonal entries.
    """
    data_aa, data_ab, data_bb = _l2_logistic_data_hessian(rows, slope, intercept)
    return (data_aa + l2_strength, data_ab, data_bb + l2_strength)


def _l2_logistic_certificate_threshold(l2_strength: float) -> float:
    """Gradient norm bound equivalent to the declared objective-gap certificate.

    The declared objective is ``l2_strength``-strongly convex because both
    parameters carry a positive L2 penalty, so for the unique minimizer
    ``theta*``::

        J(theta) - J(theta*) <= ||grad J(theta)||^2 / (2 * l2_strength)

    Therefore ``J - J* <= objective_suboptimality_tolerance`` is exactly
    ``||grad J|| <= sqrt(2 * l2_strength * tolerance)``. The bound is
    assembled as ``sqrt(2) * sqrt(l2_strength) * sqrt(tolerance)`` because
    forming ``2 * l2_strength * tolerance`` underflows to ``0.0`` for tiny
    subnormal strengths, which would make the certificate demand an exact zero
    gradient, whereas each factor here stays representable far deeper.
    """
    return (
        math.sqrt(2.0)
        * math.sqrt(l2_strength)
        * math.sqrt(_L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE)
    )


def _require_l2_strength(l2_strength: Any) -> float:
    """Validate and normalize the L2 strength to an accepted finite float.

    The accepted domain is: a finite real number strictly above zero whose
    declared objective coefficient ``l2_strength / 2.0`` is itself still
    representable as a positive binary64 float. That second condition is not a
    statistical recommendation, and it is not an arbitrary practical lower
    bound: it is a numerical representability precondition for the actual v1
    objective implementation ``mean_nll + (l2_strength / 2.0) * (slope^2 +
    intercept^2)``. If ``l2_strength / 2.0`` rounds to ``0.0``, the objective
    would silently lose its L2 term while the gradient (``l2_strength * slope``)
    and the Hessian (``data + l2_strength``) would still carry it, so the three
    would no longer describe one declared fitting problem. Rejecting such a
    strength fails closed instead of optimizing a different objective than the
    one the artifact advertises.

    A bool is rejected even though it is an ``int`` subclass. No hidden maximum
    is imposed.
    """
    if isinstance(l2_strength, bool) or not isinstance(l2_strength, (int, float)):
        raise InvalidDecisionError(
            "l2_strength must be a finite real number > 0, got "
            f"{type(l2_strength).__name__} ({l2_strength!r})"
        )
    try:
        value = float(l2_strength)
    except OverflowError:
        raise InvalidDecisionError(
            f"l2_strength must be a finite real number > 0, got {l2_strength!r}"
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise InvalidDecisionError(
            f"l2_strength must be a finite real number > 0, got {l2_strength!r}"
        )
    if value / 2.0 == 0.0:
        raise InvalidDecisionError(
            "l2_strength is positive and finite, but too small to preserve a "
            "nonzero (l2_strength / 2) coefficient under the fitter's float "
            f"numerical contract, got {value!r}"
        )
    return value


def _ordered_fitting_rows(dataset: CalibrationDataset) -> list[tuple[float, float]]:
    """Build the fitting rows in a deterministic canonical order.

    The dataset fingerprint is row-order independent, so the fitting arithmetic
    must be too. The deterministic canonical ordering below, sorted by
    observation fingerprint, is the formal row-order-independence mechanism;
    ``math.fsum`` is used for every accumulation to reduce rounding error and
    support stable deterministic arithmetic. Multiplicity is preserved because
    a sorted list is used, never a set.

    Each row is exactly ``(uncalibrated selected probability, winner
    correctness)``. No other observation attribute may enter the model: not
    entropy, not any scoring diagnostic statistic, not the model name, and not
    the candidate count. The binding identity determines the population and is
    not a numerical feature.
    """
    ordered = sorted(dataset.observations, key=lambda observation: observation.fingerprint)
    rows: list[tuple[float, float]] = []
    for observation in ordered:
        correct = observation.correct
        if not isinstance(correct, bool):
            raise InvalidDecisionError(
                "a fitting row requires a real bool winner-correctness label, got "
                f"{type(correct).__name__} ({correct!r}); the fitter never coerces "
                "an unknown or non-bool correctness into a label"
            )
        probability = _selected_probability(observation)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise InvalidDecisionError(
                f"the selected probability {probability!r} of observation "
                f"{observation.fingerprint} is not a finite value in [0, 1]"
            )
        rows.append((probability, 1.0 if correct else 0.0))
    return rows


def _l2_logistic_method_configuration(l2_strength: float) -> dict[str, JSONValue]:
    """The identity-bearing v1 method configuration.

    Everything needed to reproduce the intended fitting problem is recorded:
    what was optimized, which parameters were regularized, whether the input
    probability was transformed, clipped, or smoothed, and which solver
    contract produced the parameters. Solver iteration telemetry is
    deliberately absent because it is execution telemetry, not semantics.
    """
    return {
        "objective": {
            "id": _L2_LOGISTIC_OBJECTIVE_ID,
            "version": _L2_LOGISTIC_OBJECTIVE_VERSION,
        },
        "l2_strength": l2_strength,
        "regularized_parameters": list(_L2_LOGISTIC_REGULARIZED_PARAMETERS),
        "input_transform": {
            "id": _L2_LOGISTIC_INPUT_TRANSFORM_ID,
            "version": _L2_LOGISTIC_INPUT_TRANSFORM_VERSION,
        },
        "endpoint_policy": {
            "id": _L2_LOGISTIC_ENDPOINT_POLICY_ID,
            "version": _L2_LOGISTIC_ENDPOINT_POLICY_VERSION,
            "epsilon": None,
            "clipping": False,
            "label_smoothing": False,
        },
        "solver": {
            "id": _L2_LOGISTIC_SOLVER_ID,
            "version": _L2_LOGISTIC_SOLVER_VERSION,
            "initial_slope": _L2_LOGISTIC_INITIAL_SLOPE,
            "initial_intercept": _L2_LOGISTIC_INITIAL_INTERCEPT,
            "max_iterations": _L2_LOGISTIC_MAX_ITERATIONS,
            "backtracking_factor": _L2_LOGISTIC_BACKTRACKING_FACTOR,
            "armijo_coefficient": _L2_LOGISTIC_ARMIJO_COEFFICIENT,
            "max_backtracking_steps": _L2_LOGISTIC_MAX_BACKTRACKING_STEPS,
            "convergence": {
                "id": _L2_LOGISTIC_CONVERGENCE_ID,
                "version": _L2_LOGISTIC_CONVERGENCE_VERSION,
                "objective_suboptimality_tolerance": (
                    _L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE
                ),
            },
        },
    }


def _solve_l2_logistic(
    rows: Sequence[tuple[float, float]],
    l2_strength: float,
) -> tuple[float, float]:
    """Deterministic 2-parameter Newton fit with a backtracking line search.

    The objective is strictly convex and coercive whenever ``l2_strength > 0``
    and the fitting rows are non-empty, because the L2 penalty regularizes BOTH
    ``slope`` and ``intercept``. That makes the Hessian positive definite, so
    the Newton direction is always a descent direction and the unique finite
    global optimum is reachable from the fixed zero initialization.

    The frozen solver contract does not contain a best-effort mode: a parameter
    pair is returned only when the strong-convexity objective-gap certificate
    is satisfied. Any other exit, including the iteration budget running out or
    the line search failing, raises instead of returning a partially converged
    parameter pair. The certificate is the sole success condition; no separate
    gradient tolerance can authorize a profile on its own. For every
    ``l2_strength > 0`` the regularized Hessian is positive definite, so a
    numerically degenerate Hessian solve uses a deterministically scaled
    gradient direction instead of failing the fit; the line search still has to
    accept every step, and the fallback never authorizes convergence by itself.
    """
    slope = _L2_LOGISTIC_INITIAL_SLOPE
    intercept = _L2_LOGISTIC_INITIAL_INTERCEPT
    objective = _l2_logistic_objective(rows, slope, intercept, l2_strength)
    certificate_threshold = _l2_logistic_certificate_threshold(l2_strength)
    for _iteration in range(_L2_LOGISTIC_MAX_ITERATIONS):
        grad_slope, grad_intercept = _l2_logistic_gradient(rows, slope, intercept, l2_strength)
        if math.hypot(grad_slope, grad_intercept) <= certificate_threshold:
            return slope, intercept
        data_aa, data_ab, data_bb = _l2_logistic_data_hessian(rows, slope, intercept)
        hessian_aa = data_aa + l2_strength
        hessian_bb = data_bb + l2_strength
        # ``det(l2_strength * I + M)`` assembled from the data term and the
        # penalty term separately. Forming ``(data_aa + lambda) *
        # (data_bb + lambda)`` directly cancels the penalty below the unit in
        # the last place of the data term (tiny ``l2_strength``) and overflows
        # to infinity above roughly ``1e154``, even though the regularized
        # Hessian is positive definite by construction for every
        # ``l2_strength > 0``. The data term is a curvature-weighted second
        # moment, so it is positive semi-definite and its determinant is
        # mathematically non-negative; a negative computed value is pure
        # rounding (it appears when every selected probability is equal, which
        # makes the data term rank one) and is clamped so it cannot flip the
        # sign of a penalty-dominated determinant.
        data_determinant = data_aa * data_bb - data_ab * data_ab
        if data_determinant < 0.0:
            data_determinant = 0.0
        determinant = (
            data_determinant + l2_strength * (data_aa + data_bb) + l2_strength * l2_strength
        )
        if math.isfinite(determinant) and determinant > 0.0:
            # The numerators keep ``l2_strength`` algebraically separate for the
            # same reason as the determinant: folding it into a diagonal entry
            # first rounds it away when it is below that entry's unit in the
            # last place, which would zero the whole Newton step.
            delta_slope = (
                -(data_bb * grad_slope - data_ab * grad_intercept + l2_strength * grad_slope)
                / determinant
            )
            delta_intercept = (
                -(-data_ab * grad_slope + data_aa * grad_intercept + l2_strength * grad_intercept)
                / determinant
            )
        else:
            # ``l2_strength * I`` plus a positive semi-definite data term is
            # positive definite for every ``l2_strength > 0``, so a non-finite
            # or non-positive assembled determinant is a scaling artifact
            # rather than genuine indefiniteness. Fall back to a
            # deterministically scaled gradient direction, which stays a
            # descent direction, and let the line search enforce decrease.
            scale = max(abs(hessian_aa), abs(hessian_bb), abs(data_ab), 1.0)
            delta_slope = -grad_slope / scale
            delta_intercept = -grad_intercept / scale
        directional_derivative = grad_slope * delta_slope + grad_intercept * delta_intercept
        step = 1.0
        accepted = False
        for _backtrack in range(_L2_LOGISTIC_MAX_BACKTRACKING_STEPS):
            candidate_slope = slope + step * delta_slope
            candidate_intercept = intercept + step * delta_intercept
            candidate_objective = _l2_logistic_objective(
                rows, candidate_slope, candidate_intercept, l2_strength
            )
            if candidate_objective <= (
                objective + _L2_LOGISTIC_ARMIJO_COEFFICIENT * step * directional_derivative
            ):
                slope = candidate_slope
                intercept = candidate_intercept
                objective = candidate_objective
                accepted = True
                break
            step *= _L2_LOGISTIC_BACKTRACKING_FACTOR
        if not accepted:
            raise InvalidDecisionError(
                "the fitting line search could not find a sufficient objective "
                f"decrease within {_L2_LOGISTIC_MAX_BACKTRACKING_STEPS} reductions, "
                "so the solver cannot certify an objective gap within "
                f"{_L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE!r} for "
                f"l2_strength {l2_strength!r}; no calibration profile is produced "
                "from an uncertified fit"
            )
    raise InvalidDecisionError(
        "the fitting solver could not certify an objective gap within "
        f"{_L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE!r} for l2_strength "
        f"{l2_strength!r} within {_L2_LOGISTIC_MAX_ITERATIONS} iterations; no "
        "calibration profile is produced from an uncertified fit"
    )


def fit_l2_logistic_selected_probability(
    dataset: CalibrationDataset,
    *,
    l2_strength: float,
) -> CalibrationProfile:
    """Fit the v1 L2-regularized logistic calibrator for winner correctness.

    This is the first SUPPORTED producer of a :class:`CalibrationProfile`. It
    deterministically maps one fitting :class:`CalibrationDataset` to a profile
    by minimizing

        ``mean_i(softplus(z_i) - y_i * z_i) + (l2_strength / 2) * (a^2 + b^2)``

    with ``z_i = a * p_i + b``, ``p_i`` the frozen uncalibrated selected
    semantic probability, and ``y_i`` the winner-correctness label. Nothing is
    returned unless the declared solver contract reports convergence.

    The mapping is ``q(p) = sigmoid(a * p + b)`` with ``slope = a`` and
    ``intercept = b``. ``p = 0`` and ``p = 1`` are ordinary finite inputs: no
    epsilon, clipping, label smoothing, or logit transform is used.

    Fitting proves only that the declared fitting problem was solved. It does
    NOT establish that the calibrator is good, that it generalizes, or that
    anything is calibrated; a fitted profile is evaluated on held-out data in a
    later phase. The profile this returns carries the dataset's exact binding
    and ground-truth semantics identity, so no profile matching or compatibility
    policy is added here.
    """
    if not isinstance(dataset, CalibrationDataset):
        raise InvalidDecisionError(
            f"dataset must be a CalibrationDataset, got {type(dataset).__name__} ({dataset!r})"
        )
    strength = _require_l2_strength(l2_strength)
    rows = _ordered_fitting_rows(dataset)
    slope, intercept = _solve_l2_logistic(rows, strength)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise InvalidDecisionError(
            f"the fitted parameters must be finite, got slope {slope!r} and intercept {intercept!r}"
        )
    return CalibrationProfile._from_fitted_state(
        dataset,
        method_id=L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID,
        method_version=L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION,
        method_configuration=_l2_logistic_method_configuration(strength),
        fitted_parameters={"slope": slope, "intercept": intercept},
    )


def _require_l2_logistic_fitted_parameters(
    fitted_parameters: Mapping[str, JSONValue],
) -> tuple[float, float]:
    expected = {"slope", "intercept"}
    if set(fitted_parameters) != expected:
        raise InvalidDecisionError(
            "the L2 logistic calibration method requires fitted parameters "
            f"{sorted(expected)}, got {sorted(fitted_parameters)}"
        )
    slope_value = fitted_parameters["slope"]
    intercept_value = fitted_parameters["intercept"]
    if isinstance(slope_value, bool) or not isinstance(slope_value, (int, float)):
        raise InvalidDecisionError(
            f"the L2 logistic calibration method requires a real slope, got {slope_value!r}"
        )
    if isinstance(intercept_value, bool) or not isinstance(intercept_value, (int, float)):
        raise InvalidDecisionError(
            f"the L2 logistic calibration method requires a real intercept, got {intercept_value!r}"
        )
    slope = float(slope_value)
    intercept = float(intercept_value)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise InvalidDecisionError(
            "the L2 logistic calibration method requires finite parameters, got "
            f"slope {slope!r} and intercept {intercept!r}"
        )
    return slope, intercept


def _apply_profile_to_selected_probability(
    profile: CalibrationProfile,
    selected_probability: float,
) -> float:
    """Apply one exact fitted profile to one already-selected probability.

    This is the SINGLE numerical Profile application truth source. Both the
    offline path (:func:`predicted_winner_correctness`, which extracts the
    probability from a :class:`CalibrationObservation`) and the runtime-linked
    path (:func:`apply_profile_to_runtime_evaluation`, which extracts it from a
    :class:`~probvenance.results.DecisionResult`) consume exactly this function,
    so the two paths can never drift.

    Application is a pure function of the profile's fitted state: the profile
    is never refitted, the optimizer is never invoked, and ``l2_strength``
    plays no role. Dispatch is on the profile's explicit ``method_id`` /
    ``method_version``; an unsupported method fails closed with no
    parameter-shape guessing and no identity fallback, and the fitted state
    must contain exactly the coefficients the supported method declares, so a
    malformed internal profile also fails closed.

    ``p = 0`` and ``p = 1`` are ordinary finite inputs and the output is never
    clipped: for a large enough ``|slope * p + intercept|`` the binary64
    logistic map rounds to exactly ``0.0`` or ``1.0``, and such an endpoint is
    returned as produced.
    """
    if not isinstance(profile, CalibrationProfile):
        raise InvalidDecisionError(
            f"profile must be a CalibrationProfile, got {type(profile).__name__} ({profile!r})"
        )
    if profile.target_id != WINNER_CORRECTNESS_TARGET_ID or (
        profile.target_version != WINNER_CORRECTNESS_TARGET_VERSION
    ):
        raise InvalidDecisionError(
            "the profile does not declare the winner-correctness target "
            f"{WINNER_CORRECTNESS_TARGET_ID!r} v{WINNER_CORRECTNESS_TARGET_VERSION}; got "
            f"{profile.target_id!r} v{profile.target_version}"
        )
    if profile.input_score_id != UNCALIBRATED_SELECTED_PROBABILITY_ID or (
        profile.input_score_version != UNCALIBRATED_SELECTED_PROBABILITY_VERSION
    ):
        raise InvalidDecisionError(
            "the profile does not declare the uncalibrated selected probability as its "
            f"input score; got {profile.input_score_id!r} v{profile.input_score_version}"
        )
    if profile.method_id != L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID or (
        profile.method_version != L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION
    ):
        raise InvalidDecisionError(
            "the profile method is not supported by profile application: "
            f"got {profile.method_id!r} v{profile.method_version}, supported "
            f"{L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID!r} "
            f"v{L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION}"
        )
    slope, intercept = _require_l2_logistic_fitted_parameters(profile.fitted_parameters)
    if isinstance(selected_probability, bool) or not isinstance(selected_probability, (int, float)):
        raise InvalidDecisionError(
            "selected_probability must be a real number, got "
            f"{type(selected_probability).__name__} ({selected_probability!r})"
        )
    if not math.isfinite(selected_probability) or not (0.0 <= selected_probability <= 1.0):
        raise InvalidDecisionError(
            f"selected_probability must be a finite float in [0, 1], got {selected_probability!r}"
        )
    return _stable_sigmoid(slope * float(selected_probability) + intercept)


def predicted_winner_correctness(
    profile: CalibrationProfile,
    observation: CalibrationObservation,
) -> float:
    """Apply one exact fitted profile to one observation's selected score.

    This is the single truth source for the post-calibration
    :data:`PREDICTED_WINNER_CORRECTNESS_ID` score. It returns the profile
    produced estimate of the probability that the observation's RECORDED
    selected semantic value is correct under the profile's winner-correctness
    ground-truth semantics.

    Application is a pure function of the profile's fitted state: the profile
    is never refitted, the optimizer is never invoked, ``l2_strength`` plays no
    role, and the winner is never recomputed. The selected probability is the
    probability attached to the recorded ``selected_value`` (retrieved through
    the SAME :func:`_selected_probability` extractor the fitter and the
    uncalibrated evaluators use), never a fresh ``argmax`` over the recorded
    distribution, so no tie-breaking is re-run. The actual mapping is applied by
    :func:`_apply_profile_to_selected_probability`, the shared numerical kernel
    that runtime-linked application also consumes.

    ``p = 0`` and ``p = 1`` are ordinary finite inputs; the mapping output is
    never clipped. It is not guaranteed to be strictly interior: for a fitted
    ``slope * p + intercept`` large enough in magnitude the binary64 result of
    the logistic map rounds to exactly ``0.0`` or ``1.0`` (see
    :func:`_stable_sigmoid`), and such an endpoint is returned as produced.
    """
    if not isinstance(profile, CalibrationProfile):
        raise InvalidDecisionError(
            f"profile must be a CalibrationProfile, got {type(profile).__name__} ({profile!r})"
        )
    if not isinstance(observation, CalibrationObservation):
        raise InvalidDecisionError(
            "observation must be a CalibrationObservation, got "
            f"{type(observation).__name__} ({observation!r})"
        )
    return _apply_profile_to_selected_probability(profile, _selected_probability(observation))


def _runtime_selected_probability(result: BoolResult | ChoiceResult) -> float:
    """Return the probability attached to a runtime result's selected value.

    The selected value is read from the result, never recomputed: for a
    :class:`~probvenance.results.BoolResult` the selected semantic value is
    ``true`` when ``probability_true > 0.5`` (matching the binary tie rule that
    ``0.5`` selects ``false``), and for a
    :class:`~probvenance.results.ChoiceResult` it is the recorded
    ``result.value`` whose deterministic tie-breaking already happened at
    result construction. No ``argmax`` is re-run, no tie is re-broken, and no
    scoring label or token id is consulted.
    """
    if isinstance(result, BoolResult):
        if _select_bool_value(result.probability_true):
            return result.probability_true
        return result.probability_false
    if isinstance(result, ChoiceResult):
        return result.probabilities[result.value]
    raise InvalidDecisionError(
        "runtime-linked calibration application supports BoolResult and ChoiceResult, got "
        f"{type(result).__name__}"
    )


def apply_profile_to_runtime_evaluation(
    evaluation: Evaluation,
    profile: CalibrationProfile,
    *,
    task_id: str | None = None,
    domain_id: str | None = None,
    taxonomy_id: str | None = None,
    taxonomy_version: int | None = None,
) -> Evaluation:
    """Return a calibrated copy of an uncalibrated runtime ``Evaluation``.

    This is the only supported path that may legitimately produce a
    ``DecisionResult`` with ``calibrated=True`` and a populated
    ``predicted_correctness``. It is an explicit, caller-driven transformation:
    the caller supplies the exact :class:`CalibrationProfile` to apply and
    declares the task/domain/taxonomy identity it knows. There is no registry,
    no lookup, no nearest profile, and no fallback.

    The profile must already be compatible with the evaluation, established by
    reconstructing a :class:`CalibrationBinding` from the trace's provenance
    plus the CALLER's declarations and requiring an exact match against the
    profile's binding. Optional declaration values are never copied from the
    profile to manufacture a match. Ground truth is not required at runtime:
    the profile's ground-truth semantics define what ``predicted_correctness``
    means, and runtime application does not prove those label semantics match a
    future observed label.

    The numerical mapping is :func:`_apply_profile_to_selected_probability`,
    the same kernel the offline path consumes, so the two can never drift. The
    input evaluation must be uncalibrated, and the original result and trace
    are never mutated: a new result of the same concrete type and a new trace
    are returned, with the execution fingerprint and the semantic outcome
    distribution unchanged. ``calibrated=True`` records that a profile was
    applied, not that the profile is statistically valid or improves any
    metric.
    """
    if not isinstance(evaluation, Evaluation):
        raise InvalidDecisionError(
            f"evaluation must be an Evaluation, got {type(evaluation).__name__} ({evaluation!r})"
        )
    if not isinstance(profile, CalibrationProfile):
        raise InvalidDecisionError(
            f"profile must be a CalibrationProfile, got {type(profile).__name__} ({profile!r})"
        )
    result = evaluation.result
    trace = evaluation.trace
    if (
        result.calibrated
        or result.predicted_correctness is not None
        or result.calibration_profile_fingerprint is not None
        or result.calibration_profile_fingerprint_version is not None
    ):
        raise InvalidDecisionError(
            "runtime-linked calibration application requires an uncalibrated result; "
            "the supplied evaluation is already calibrated and is never recalibrated, "
            "stacked, or overwritten"
        )
    if (
        trace.calibration_profile_fingerprint is not None
        or trace.calibration_profile_fingerprint_version is not None
    ):
        raise InvalidDecisionError(
            "runtime-linked calibration application requires an uncalibrated trace; "
            "the supplied trace already carries calibration provenance"
        )
    if result.trace_id is None or result.trace_id != trace.trace_id:
        raise InvalidDecisionError(
            "the result and trace must share one non-null trace id before calibration "
            f"application, got result.trace_id={result.trace_id!r} and "
            f"trace.trace_id={trace.trace_id!r}"
        )
    if trace.decision_family == "bool":
        if not isinstance(result, BoolResult):
            raise InvalidDecisionError(
                f"a bool decision_family requires a BoolResult, got {type(result).__name__}"
            )
    elif trace.decision_family == "choice":
        if not isinstance(result, ChoiceResult):
            raise InvalidDecisionError(
                f"a choice decision_family requires a ChoiceResult, got {type(result).__name__}"
            )
    else:
        raise InvalidDecisionError(f"unsupported decision_family {trace.decision_family!r}")
    runtime_binding = CalibrationBinding.from_trace(
        trace,
        task_id=task_id,
        domain_id=domain_id,
        taxonomy_id=taxonomy_id,
        taxonomy_version=taxonomy_version,
    )
    profile.require_binding_match(runtime_binding)
    predicted_correctness = _apply_profile_to_selected_probability(
        profile,
        _runtime_selected_probability(result),
    )
    calibrated_result = replace(
        result,
        predicted_correctness=predicted_correctness,
        calibrated=True,
        calibration_profile_fingerprint=profile.fingerprint,
        calibration_profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    )
    calibrated_trace = replace(
        trace,
        calibration_profile_fingerprint=profile.fingerprint,
        calibration_profile_fingerprint_version=CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    )
    if (
        calibrated_result.calibration_profile_fingerprint
        != calibrated_trace.calibration_profile_fingerprint
        or calibrated_result.calibration_profile_fingerprint_version
        != calibrated_trace.calibration_profile_fingerprint_version
    ):
        raise InvalidDecisionError(
            "the calibrated result and trace must carry the same calibration provenance"
        )
    return Evaluation(result=calibrated_result, trace=calibrated_trace)
