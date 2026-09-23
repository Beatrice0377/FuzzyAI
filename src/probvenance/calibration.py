"""Calibration data foundation (Phase 4A, Parts D through M).

This module implements the ground-truth and observation data model that a
future calibration harness will fit against: ground-truth provenance, a
ground-truth record with resolution semantics, a calibration binding, a
calibration observation with deterministically derived status and correctness,
an observation fingerprint, a calibration dataset with structural pooling
constraints, and a dataset fingerprint.

Deliberately NOT implemented here (out of scope for this round):
``CalibrationProfile``, profile registries, nearest-profile matching,
fitting algorithms, evaluation metrics, and one-vs-rest handling.

Public API note: this foundation is intentionally NOT frozen as public API
yet. Nothing from this module is exported through ``probvenance.__all__`` or
imported into ``probvenance/__init__.py``; the module is importable as
``probvenance.calibration`` for tests and internal use only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, fingerprint
from probvenance.results import BoolResult, ChoiceResult
from probvenance.runtime import Evaluation
from probvenance.trace import DecisionTrace

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

CALIBRATION_OBSERVATION_FINGERPRINT_VERSION = 1
"""Version of the calibration observation fingerprint payload schema."""

CALIBRATION_DATASET_FINGERPRINT_VERSION = 1
"""Version of the calibration dataset fingerprint payload schema."""

RENDERING_SEMANTICS_VERSION = 1
"""Version of the centralized rendering-semantics canonical projection."""

_OBSERVATION_CONSTRUCTION_TOKEN: Final[object] = object()
"""Construction capability held only by ``CalibrationObservation.from_evaluation``.

The token is deliberately NOT a dataclass field: it is a keyword-only
``__init__`` parameter that defaults to ``None``, so ``dataclasses.replace``
cannot smuggle it (``replace`` re-supplies only the dataclass field values)
and it is therefore not readable from an instance, not present in
``dataclasses.fields()``, and not part of ``repr``, equality, hashing, or
pickle state. Direct field construction and ``replace`` reconstruction both
fail the capability check because neither can prove that its fields come
from one coherent execution lineage.
"""

_BOOL_OUTCOME_ORDER: tuple[str, ...] = ("false", "true")
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
    """Reject values that cannot appear in a canonical JSON payload."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (list, tuple)):
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
            "rendering_semantics": dict(self.rendering_semantics),
            "task_id": self.task_id,
            "domain_id": self.domain_id,
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_version": self.taxonomy_version,
        }

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint of the binding's canonical payload."""
        return fingerprint(self.canonical_payload())


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
    """Require that a result and trace come from one coherent execution.

    The runtime stamps the same trace id onto the assembled result and the
    trace it reports, so a mismatch proves the pair was not produced by a
    single evaluation. Linkage is never inferred from probability values:
    two different executions can emit identical distributions.
    """
    result_linkage = result.trace_id
    if result_linkage is None:
        raise InvalidDecisionError(
            "CalibrationObservation requires a result with execution linkage: "
            "result.trace_id is None while the trace linkage identity is "
            f"{trace.trace_id!r}, so the pair cannot be proven coherent"
        )
    if result_linkage != trace.trace_id:
        raise InvalidDecisionError(
            "CalibrationObservation requires coherent execution lineage: "
            f"result linkage identity {result_linkage!r} does not match "
            f"trace linkage identity {trace.trace_id!r}"
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
    :meth:`CalibrationObservation.from_evaluation`, which proves that the
    result and trace share one coherent execution lineage before any
    provenance is derived. Direct field construction is rejected, and so is
    ``dataclasses.replace`` reconstruction: both rebuild an observation
    through the constructor without the construction capability, and neither
    can prove that the fields it supplies come from one coherent execution
    lineage. Lower-level Python escape hatches such as ``object.__new__``,
    ``copy``, and ``pickle`` are inherent to the language, are not supported
    construction paths, and are not defended against. A future persistence or
    reconstruction entry point must be added explicitly and validated
    separately, and is NOT in scope here.
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
                "CalibrationObservation.from_evaluation(...), which proves that the "
                "result and trace share one coherent execution lineage"
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

    Every member must be fit-eligible; non-fit-eligible observations
    (taxonomy miss, unresolved, unadjudicated) are explicitly rejected at
    construction, never silently filtered. All members must share an
    identical ``CalibrationBinding`` canonical payload: different exact
    formulation or source bindings are never silently pooled (INV-23), and
    shared formulation-family membership never authorises pooling (INV-24).

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
        binding_fingerprint = self.binding.fingerprint
        rejected: list[str] = []
        for index, observation in enumerate(self.observations):
            if not observation.fit_eligible:
                rejected.append(
                    f"observation {index}: status {str(observation.status)!r} is not fit-eligible"
                )
            if observation.binding.fingerprint != binding_fingerprint:
                rejected.append(
                    f"observation {index}: binding fingerprint "
                    f"{observation.binding.fingerprint!r} does not match the "
                    f"dataset binding fingerprint {binding_fingerprint!r}"
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
        """Build a fitting dataset from fit-eligible, same-binding observations."""
        if not observations:
            raise InvalidDecisionError(
                "a calibration fitting dataset must contain at least one observation"
            )
        binding = observations[0].binding
        return cls(binding=binding, observations=tuple(observations))

    @property
    def fingerprint(self) -> str:
        """Row-order independent, multiplicity-preserving dataset fingerprint.

        Implemented as a hash over the canonical ORDERED list of sorted
        observation fingerprints: a multiset, not a set hash.
        """
        observation_fingerprints: list[JSONValue] = [
            observation.fingerprint for observation in self.observations
        ]
        observation_fingerprints.sort(key=str)
        payload: dict[str, JSONValue] = {
            "v": CALIBRATION_DATASET_FINGERPRINT_VERSION,
            "binding": self.binding.fingerprint,
            "observation_fingerprints": observation_fingerprints,
        }
        return fingerprint(payload)
