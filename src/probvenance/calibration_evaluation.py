"""Pre-calibration winner-correctness evaluation.

This module is the evaluation layer of the calibration work. It is deliberately
separate from the fitting layer in :mod:`probvenance.calibration`:

- :class:`CalibrationDataset` (Phase 4A) is the *fitting* dataset contract. It
  stays exactly as it is; this module does not widen it.
- :class:`CalibrationEvaluationDataset` is the *evaluation* dataset contract. It
  reuses the same observations but adds a declared evaluation split role and
  split identity, and derives its own identity.

Evaluation measures empirical behaviour. Evaluation does not create
calibration. Everything here evaluates the **uncalibrated selected semantic
probability** (see ``UNCALIBRATED_SELECTED_PROBABILITY_ID``) against the derived
winner-correctness label ``Y_correct in {0, 1}``. The runtime keeps
``DecisionResult.predicted_correctness = None`` and ``DecisionResult.calibrated
= False``; nothing in this module changes that, and no
:class:`~probvenance.calibration.CalibrationProfile` exists or is implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from probvenance.calibration import (
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    CalibrationBinding,
    CalibrationObservation,
    GroundTruthSemanticsIdentity,
    _require_non_empty_str,
)
from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint

__all__ = [
    "BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION",
    "BRIER_METRIC_ID",
    "BRIER_METRIC_VERSION",
    "CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION",
    "LOG_LOSS_BOUNDARY_POLICY",
    "LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION",
    "LOG_LOSS_LOG_BASE",
    "LOG_LOSS_METRIC_ID",
    "LOG_LOSS_METRIC_VERSION",
    "LOG_LOSS_TARGET",
    "UNCALIBRATED_SELECTED_PROBABILITY_ID",
    "UNCALIBRATED_SELECTED_PROBABILITY_VERSION",
    "BrierEvaluationResult",
    "CalibrationEvaluationDataset",
    "EvaluationSplitRole",
    "LogLossEvaluationResult",
    "evaluate_uncalibrated_winner_brier",
    "evaluate_uncalibrated_winner_log_loss",
]

# ---------------------------------------------------------------------------
# Fingerprint schema versions (new in this module; existing versions unchanged)
# ---------------------------------------------------------------------------

CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION = 1
BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION = 1
LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION = 1

# ---------------------------------------------------------------------------
# Score semantics identity
# ---------------------------------------------------------------------------

#: The evaluated input score is the probability the *uncalibrated* semantic
#: distribution assigns to the semantic value the runtime actually selected.
#: It is NOT ``predicted_correctness``, NOT a confidence, and NOT a calibrated
#: probability. The runtime never sets ``predicted_correctness`` and never
#: marks a result ``calibrated=True``; this module never does either.
UNCALIBRATED_SELECTED_PROBABILITY_ID = "uncalibrated-selected-probability"
UNCALIBRATED_SELECTED_PROBABILITY_VERSION = 1

# ---------------------------------------------------------------------------
# Metric identity
# ---------------------------------------------------------------------------

BRIER_METRIC_ID = "brier"
BRIER_METRIC_VERSION = 1
#: The empirical target the Brier score is computed against.
BRIER_TARGET = "winner_correctness"

LOG_LOSS_METRIC_ID = "log-loss"
LOG_LOSS_METRIC_VERSION = 1
#: The empirical target the log loss is computed against.
LOG_LOSS_TARGET = "winner_correctness"
#: The boundary policy of the implemented log loss: exact natural-log
#: endpoints, no epsilon, no clipping, no smoothing.
LOG_LOSS_BOUNDARY_POLICY = "exact"
#: The logarithm base of the implemented log loss: the natural logarithm.
LOG_LOSS_LOG_BASE = "e"

# ---------------------------------------------------------------------------
# Construction token for the supported-path-only result artifact
# ---------------------------------------------------------------------------

_BRIER_RESULT_CONSTRUCTION_TOKEN = object()
_LOG_LOSS_RESULT_CONSTRUCTION_TOKEN = object()

_BOOL_TRUE_NAME = "true"
_BOOL_FALSE_NAME = "false"


class EvaluationSplitRole(StrEnum):
    """Declared role of an evaluation split.

    Only evaluation roles exist here. ``TRAIN`` and ``FITTING`` are fitting
    concepts and belong to :class:`~probvenance.calibration.CalibrationDataset`,
    not to an evaluation dataset.
    """

    VALIDATION = "validation"
    TEST = "test"


def _selected_probability(
    observation: CalibrationObservation,
) -> float:
    """Extract the uncalibrated selected semantic probability.

    The recorded ``selected_value`` is treated as an execution fact: the winner
    is NOT recomputed and tie-breaking is NOT re-run. For Choice decisions the
    probability is looked up by semantic candidate name (never by a scoring
    label such as ``A``/``B``/``C`` and never by a token id). For Bool
    decisions the outcome order is ``("false", "true")`` and the name is chosen
    from the recorded boolean.
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


def _binary_log_loss_term(probability: float, correct: bool) -> float:
    """One exact natural-log binary log-loss term.

    ``correct=True`` contributes ``-ln(p)``; ``correct=False`` contributes
    ``-ln(1 - p)``. The endpoints are exact: a correct deterministic endpoint
    scores ``0.0`` and an impossible observed outcome scores ``+inf``. There
    is no epsilon, no clipping, and no smoothing; the branches are explicit so
    a ``0 * log(0)`` style product (which would produce NaN) is never formed.
    The ``correct=False`` branch uses :func:`math.log1p` instead of
    ``log(1.0 - p)`` to avoid precision loss near ``p = 1``.
    """
    if correct:
        if probability == 0.0:
            return math.inf
        if probability == 1.0:
            return 0.0
        return -math.log(probability)
    if probability == 1.0:
        return math.inf
    if probability == 0.0:
        return 0.0
    return -math.log1p(-probability)


@dataclass(frozen=True, slots=True)
class CalibrationEvaluationDataset:
    """A declared evaluation split over fit-eligible calibration observations.

    All identity fields are derived. Callers must not (and cannot) supply a
    binding, ground-truth semantics, correctness labels, or a dataset
    fingerprint: the binding and semantics are derived from the observations,
    correctness is the already-derived ``CalibrationObservation.correct``, and
    the fingerprint is computed from the admitted rows plus the declared split
    metadata.

    Admission is all-or-nothing: reasons accumulate and nothing is silently
    filtered. A taxonomy miss or an unresolved or unadjudicated ground truth is
    never encoded as ``correct = False``; those observations are rejected
    because they are not fit-eligible.

    The declared ``split_role`` and ``split_id`` are provenance, not proof:
    they record which role the caller assigned to this split and do not prove
    statistical independence from any data used for future fitting.
    """

    observations: tuple[CalibrationObservation, ...]
    split_role: EvaluationSplitRole
    split_id: str

    def __post_init__(self) -> None:
        rejected: list[str] = []
        declared = self.observations
        observations: tuple[CalibrationObservation, ...] = ()
        if isinstance(declared, tuple) and declared:
            observations = declared
        else:
            rejected.append("an evaluation dataset must contain at least one observation")
        if not isinstance(self.split_role, EvaluationSplitRole):
            rejected.append(
                "split_role must be an EvaluationSplitRole (validation or test), "
                f"got {self.split_role!r}"
            )
        try:
            _require_non_empty_str("split_id", self.split_id)
        except InvalidDecisionError as exc:
            rejected.append(str(exc))

        # Type-check elements before touching any attribute: a malformed
        # container must fail closed with InvalidDecisionError, not AttributeError.
        typed = tuple(
            observation
            for observation in observations
            if isinstance(observation, CalibrationObservation)
        )

        binding_payload_json: str | None = None
        semantics_payload_json: str | None = None
        if typed:
            binding_payload_json = canonical_json(typed[0].binding.canonical_payload())
            semantics_payload_json = canonical_json(
                GroundTruthSemanticsIdentity.from_provenance(
                    typed[0].ground_truth.provenance
                ).canonical_payload()
            )
        for index, observation in enumerate(observations):
            if not isinstance(observation, CalibrationObservation):
                rejected.append(
                    f"observation {index}: expected a CalibrationObservation, "
                    f"got {type(observation).__name__}"
                )
                continue
            if not observation.fit_eligible:
                rejected.append(
                    f"observation {index}: status {observation.status!r} is not "
                    "fit-eligible (the ground truth must be resolved and "
                    "adjudicated)"
                )
            elif observation.correct is None:
                # Belt and braces: fit_eligible already implies a resolved
                # observation with a derived correctness label.
                rejected.append(
                    f"observation {index}: derived correctness is None even "
                    "though the observation is fit-eligible"
                )
            observation_binding_json = canonical_json(observation.binding.canonical_payload())
            if observation_binding_json != binding_payload_json:
                rejected.append(
                    f"observation {index}: binding canonical payload "
                    f"{observation_binding_json} does not match the dataset "
                    f"binding canonical payload {binding_payload_json}"
                )
            observation_semantics_json = canonical_json(
                GroundTruthSemanticsIdentity.from_provenance(
                    observation.ground_truth.provenance
                ).canonical_payload()
            )
            if observation_semantics_json != semantics_payload_json:
                rejected.append(
                    f"observation {index}: ground-truth semantics identity "
                    f"{observation_semantics_json} does not match the dataset "
                    f"ground-truth semantics identity {semantics_payload_json}"
                )
        if rejected:
            raise InvalidDecisionError(
                f"calibration evaluation dataset rejected "
                f"{len(rejected)} reason(s) for "
                f"{len(observations)} observation(s): " + "; ".join(rejected)
            )

    @property
    def binding(self) -> CalibrationBinding:
        """The binding derived from the admitted observations."""
        return self.observations[0].binding

    @property
    def ground_truth_semantics(self) -> GroundTruthSemanticsIdentity:
        """The ground-truth semantics identity derived from the observations."""
        return GroundTruthSemanticsIdentity.from_provenance(
            self.observations[0].ground_truth.provenance
        )

    @property
    def fingerprint(self) -> str:
        """Row-order independent identity of the declared evaluation split.

        The payload commits the split metadata, the derived binding and
        ground-truth semantics identities, and the sorted observation
        fingerprints (multiplicity preserved, never a set). A different
        ``split_role`` or a different ``split_id`` yields a different
        fingerprint even when the observations are identical.
        """
        observation_fingerprints: list[JSONValue] = [
            observation.fingerprint for observation in self.observations
        ]
        observation_fingerprints.sort(key=str)
        payload: dict[str, JSONValue] = {
            "v": CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
            "split_role": str(self.split_role.value),
            "split_id": self.split_id,
            "binding_fingerprint": self.binding.fingerprint,
            "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "ground_truth_semantics_fingerprint": (self.ground_truth_semantics.fingerprint),
            "ground_truth_semantics_fingerprint_version": (
                GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
            ),
            "observation_fingerprints": observation_fingerprints,
        }
        return fingerprint(payload)


@dataclass(frozen=True, slots=True)
class BrierEvaluationResult:
    """Immutable, provenance-rich artifact of one Brier evaluation run.

    The supported construction contract is the module-level evaluator
    :func:`evaluate_uncalibrated_winner_brier`. Direct construction and
    ``dataclasses.replace`` reconstruction are rejected so a caller cannot
    combine one dataset with a fabricated ``value`` or ``count``. This is an
    API discipline within the supported construction contract, not a security
    boundary; low-level Python escape hatches such as ``object.__new__`` are
    not supported construction paths and are not defended against.

    ``dataclasses.replace(result)`` reaches ``__init__`` with no construction
    token and is rejected with :class:`InvalidDecisionError`.
    ``dataclasses.replace(result, value=...)`` is rejected earlier by
    ``dataclasses`` itself (a ``ValueError``, because ``value`` is declared
    with ``init=False``) and never reaches ``__init__``; that rejection is
    intentional and is NOT an :class:`InvalidDecisionError`.

    ``count`` and ``value`` are derived from the dataset by the evaluator and
    are stored as immutable scalars. ``canonical_payload`` emits
    ``"configuration": {}`` as a constant: the metric has no free parameters,
    and no caller-controlled configuration mapping exists.
    """

    metric_id: str = field(init=False, repr=False)
    metric_version: int = field(init=False, repr=False)
    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    count: int = field(init=False, repr=False)
    value: float = field(init=False, repr=False)

    def __init__(
        self,
        dataset: CalibrationEvaluationDataset | None = None,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _BRIER_RESULT_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "BrierEvaluationResult cannot be constructed directly or with "
                "dataclasses.replace; use "
                "evaluate_uncalibrated_winner_brier(dataset), the only "
                "supported construction path"
            )
        if not isinstance(dataset, CalibrationEvaluationDataset):
            raise InvalidDecisionError(
                "the supported BrierEvaluationResult construction path requires "
                "a CalibrationEvaluationDataset, got "
                f"{type(dataset).__name__}"
            )
        count = len(dataset.observations)
        squared_errors: list[float] = []
        for observation in dataset.observations:
            probability = _selected_probability(observation)
            label = 1.0 if observation.correct else 0.0
            squared_errors.append((probability - label) ** 2)
        value = math.fsum(squared_errors) / count
        object.__setattr__(self, "metric_id", BRIER_METRIC_ID)
        object.__setattr__(self, "metric_version", BRIER_METRIC_VERSION)
        object.__setattr__(self, "target", BRIER_TARGET)
        object.__setattr__(self, "input_score_id", UNCALIBRATED_SELECTED_PROBABILITY_ID)
        object.__setattr__(self, "input_score_version", UNCALIBRATED_SELECTED_PROBABILITY_VERSION)
        object.__setattr__(self, "evaluation_dataset_fingerprint", dataset.fingerprint)
        object.__setattr__(
            self,
            "evaluation_dataset_fingerprint_version",
            CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "value", value)

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the metric identity, the input-score identity, the evaluation
        dataset identity with its payload schema version, the (empty)
        configuration, ``count``, and ``value``. Timestamps, wall-clock time,
        hostnames, and logging metadata are deliberately excluded.
        """
        return {
            "v": BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION,
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "target": self.target,
            "input_score_id": self.input_score_id,
            "input_score_version": self.input_score_version,
            "evaluation_dataset_fingerprint": self.evaluation_dataset_fingerprint,
            "evaluation_dataset_fingerprint_version": (self.evaluation_dataset_fingerprint_version),
            "configuration": {},
            "count": self.count,
            "value": self.value,
        }

    @property
    def fingerprint(self) -> str:
        """Stable identity of the evaluation artifact.

        The hash of :meth:`canonical_payload`, deterministic for the object
        lifetime: every payload field is an immutable scalar or string fixed
        at construction.
        """
        return fingerprint(self.canonical_payload())


def evaluate_uncalibrated_winner_brier(
    dataset: CalibrationEvaluationDataset,
) -> BrierEvaluationResult:
    """Evaluate the pre-calibration winner-correctness Brier score.

    ``Brier = mean((p_i - y_i)^2)`` where ``p_i`` is the uncalibrated selected
    semantic probability of observation ``i`` and ``y_i`` is the derived
    winner-correctness label (``1`` when the observation is correct, ``0``
    otherwise). Accumulation uses :func:`math.fsum` for stable deterministic
    summation.

    This is a read-only offline measurement: it does not mutate the dataset or
    any observation, does not set ``calibrated`` or ``predicted_correctness``
    anywhere, does not create a :class:`~probvenance.calibration.CalibrationProfile`,
    and does not touch the runtime.
    """
    if not isinstance(dataset, CalibrationEvaluationDataset):
        raise InvalidDecisionError(
            "evaluate_uncalibrated_winner_brier requires a "
            f"CalibrationEvaluationDataset, got {type(dataset).__name__}"
        )
    return BrierEvaluationResult(dataset, _construction_token=_BRIER_RESULT_CONSTRUCTION_TOKEN)


@dataclass(frozen=True, slots=True)
class LogLossEvaluationResult:
    """Immutable, provenance-rich artifact of one log-loss evaluation run.

    The supported construction contract is the module-level evaluator
    :func:`evaluate_uncalibrated_winner_log_loss`. Direct construction and
    ``dataclasses.replace`` reconstruction are rejected so a caller cannot
    combine one dataset with a fabricated ``value`` or ``count``. This is an
    API discipline within the supported construction contract, not a security
    boundary; low-level Python escape hatches such as ``object.__new__`` are
    not supported construction paths and are not defended against.

    ``dataclasses.replace(result)`` reaches ``__init__`` with no construction
    token and is rejected with :class:`InvalidDecisionError`.
    ``dataclasses.replace(result, value=...)`` is rejected earlier by
    ``dataclasses`` itself (a ``ValueError``, because ``value`` is declared
    with ``init=False``) and never reaches ``__init__``; that rejection is
    intentional and is NOT an :class:`InvalidDecisionError`.

    ``value`` is the exact natural-log binary log loss over the dataset and
    MAY be ``math.inf``: an impossible observed outcome (a deterministic
    endpoint assigned to the wrong label) scores positive infinity. It is
    never NaN, never negative infinity, and never a negative finite log loss;
    the evaluator fails closed if such a value could arise. Because canonical
    JSON forbids non-finite floats, :meth:`canonical_payload` encodes ``value``
    structurally (a discriminator plus an optional finite number), never as a
    bare non-finite JSON number.

    ``count`` and ``value`` are derived from the dataset by the evaluator and
    are stored as immutable scalars. The metric configuration is fixed and
    derived from the module constants; ``canonical_payload`` emits it
    literally as ``{"boundary_policy": "exact", "log_base": "e"}``. No
    caller-controlled configuration mapping exists.
    """

    metric_id: str = field(init=False, repr=False)
    metric_version: int = field(init=False, repr=False)
    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    count: int = field(init=False, repr=False)
    value: float = field(init=False, repr=False)

    def __init__(
        self,
        dataset: CalibrationEvaluationDataset | None = None,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _LOG_LOSS_RESULT_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "LogLossEvaluationResult cannot be constructed directly or with "
                "dataclasses.replace; use "
                "evaluate_uncalibrated_winner_log_loss(dataset), the only "
                "supported construction path"
            )
        if not isinstance(dataset, CalibrationEvaluationDataset):
            raise InvalidDecisionError(
                "the supported LogLossEvaluationResult construction path requires "
                "a CalibrationEvaluationDataset, got "
                f"{type(dataset).__name__}"
            )
        count = len(dataset.observations)
        terms: list[float] = []
        for observation in dataset.observations:
            probability = _selected_probability(observation)
            if observation.correct is None:
                raise InvalidDecisionError(
                    f"observation {observation.fingerprint} has no derived "
                    "correctness label; the winner-correctness log loss cannot "
                    "be computed"
                )
            terms.append(_binary_log_loss_term(probability, observation.correct))
        value = math.inf if any(term == math.inf for term in terms) else math.fsum(terms) / count
        if math.isnan(value) or value == -math.inf or (math.isfinite(value) and value < 0.0):
            raise InvalidDecisionError(
                "the computed log loss value is not a valid log loss "
                f"(got {value!r}); a log loss is never NaN, never negative "
                "infinity, and never a negative finite number"
            )
        object.__setattr__(self, "metric_id", LOG_LOSS_METRIC_ID)
        object.__setattr__(self, "metric_version", LOG_LOSS_METRIC_VERSION)
        object.__setattr__(self, "target", LOG_LOSS_TARGET)
        object.__setattr__(self, "input_score_id", UNCALIBRATED_SELECTED_PROBABILITY_ID)
        object.__setattr__(self, "input_score_version", UNCALIBRATED_SELECTED_PROBABILITY_VERSION)
        object.__setattr__(self, "evaluation_dataset_fingerprint", dataset.fingerprint)
        object.__setattr__(
            self,
            "evaluation_dataset_fingerprint_version",
            CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "value", value)

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the metric identity, the input-score identity, the evaluation
        dataset identity with its payload schema version, the fixed metric
        configuration, ``count``, and a structural encoding of ``value``.
        Timestamps, wall-clock time, hostnames, and logging metadata are
        deliberately excluded.

        The structural value encoding exists because canonical JSON forbids
        non-finite floats: a finite value is encoded as
        ``{"kind": "finite", "number": <finite float>}`` and positive infinity
        as ``{"kind": "positive_infinity", "number": None}``. A finite value
        and positive infinity therefore have different canonical payloads and
        different fingerprints, and a missing value is never confused with
        mathematical positive infinity.
        """
        if self.value == math.inf:
            encoded_value: dict[str, JSONValue] = {
                "kind": "positive_infinity",
                "number": None,
            }
        else:
            encoded_value = {"kind": "finite", "number": self.value}
        return {
            "v": LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION,
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "target": self.target,
            "input_score_id": self.input_score_id,
            "input_score_version": self.input_score_version,
            "evaluation_dataset_fingerprint": self.evaluation_dataset_fingerprint,
            "evaluation_dataset_fingerprint_version": (self.evaluation_dataset_fingerprint_version),
            "configuration": {
                "boundary_policy": LOG_LOSS_BOUNDARY_POLICY,
                "log_base": LOG_LOSS_LOG_BASE,
            },
            "count": self.count,
            "value": encoded_value,
        }

    @property
    def fingerprint(self) -> str:
        """Stable identity of the evaluation artifact.

        The hash of :meth:`canonical_payload`, deterministic for the object
        lifetime: every payload field is an immutable scalar, string, or
        structural value encoding fixed at construction.
        """
        return fingerprint(self.canonical_payload())


def evaluate_uncalibrated_winner_log_loss(
    dataset: CalibrationEvaluationDataset,
) -> LogLossEvaluationResult:
    """Evaluate the pre-calibration winner-correctness log loss.

    ``log_loss = mean(-ln(p_i) if y_i = 1 else -ln(1 - p_i))`` where ``p_i``
    is the uncalibrated selected semantic probability of observation ``i`` and
    ``y_i`` is the derived winner-correctness label (``1`` when the
    observation is correct, ``0`` otherwise). The boundary policy is exact:
    there is no epsilon, no clipping, and no smoothing. A correct
    deterministic endpoint scores ``0.0``; an impossible observed outcome
    scores ``+inf``, and any infinite term makes the aggregate ``+inf``.
    Accumulation of finite terms uses :func:`math.fsum` for stable
    deterministic summation; infinite terms are detected before summation so
    ``fsum`` never has to guess semantics from a mix of non-finite values.

    The evaluated input score remains the uncalibrated selected semantic
    probability. It is NOT ``predicted_correctness``, NOT ``P(Y_correct = 1)``,
    NOT a calibrated probability, and NOT a confidence. Log loss is a proper
    scoring rule when its input is interpreted as a probability forecast of
    the scored binary target; this pre-calibration run does not grant that
    interpretation to the raw selected semantic probability (section 14.0 of
    ``docs/calibration-semantics.md``).

    This is a read-only offline measurement: it does not mutate the dataset or
    any observation, does not set ``calibrated`` or ``predicted_correctness``
    anywhere, does not create a :class:`~probvenance.calibration.CalibrationProfile`,
    and does not touch the runtime.
    """
    if not isinstance(dataset, CalibrationEvaluationDataset):
        raise InvalidDecisionError(
            "evaluate_uncalibrated_winner_log_loss requires a "
            f"CalibrationEvaluationDataset, got {type(dataset).__name__}"
        )
    return LogLossEvaluationResult(dataset, _construction_token=_LOG_LOSS_RESULT_CONSTRUCTION_TOKEN)
