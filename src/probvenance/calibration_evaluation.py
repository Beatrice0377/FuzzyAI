"""Pre-calibration winner-correctness evaluation.

This module is the evaluation layer of the calibration work. It is deliberately
separate from the fitting layer in :mod:`probvenance.calibration`:

- :class:`CalibrationDataset` (Phase 4A) is the *fitting* dataset contract. It
  stays exactly as it is; this module does not widen it.
- :class:`CalibrationEvaluationCohort` is the *declared evaluation source
  cohort* contract: the full set of observations supplied as one declared
  evaluation split, BEFORE metric eligibility is applied. Metric exclusion is
  provenance: taxonomy-miss, unresolved, and resolved-but-unadjudicated rows
  are retained and explicitly accounted, never silently dropped.
- :class:`CalibrationEvaluationDataset` is the *metric-eligible projection* of
  one declared source cohort. It is derived from the cohort; it is not an
  independently constructable container.

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
from fractions import Fraction

from probvenance.calibration import (
    CALIBRATION_BINDING_FINGERPRINT_VERSION,
    GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION,
    CalibrationBinding,
    CalibrationObservation,
    CalibrationObservationStatus,
    GroundTruthSemanticsIdentity,
    _require_non_empty_str,
)
from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint

__all__ = [
    "BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION",
    "BRIER_METRIC_ID",
    "BRIER_METRIC_VERSION",
    "CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION",
    "CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION",
    "EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID",
    "EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION",
    "EMPIRICAL_CORRECTNESS_RATE_ID",
    "EMPIRICAL_CORRECTNESS_RATE_VERSION",
    "EQUAL_WIDTH_BINNING_ID",
    "EQUAL_WIDTH_BINNING_VERSION",
    "LOG_LOSS_BOUNDARY_POLICY",
    "LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION",
    "LOG_LOSS_LOG_BASE",
    "LOG_LOSS_METRIC_ID",
    "LOG_LOSS_METRIC_VERSION",
    "LOG_LOSS_TARGET",
    "MEAN_SELECTED_PROBABILITY_ID",
    "MEAN_SELECTED_PROBABILITY_VERSION",
    "UNCALIBRATED_SELECTED_PROBABILITY_ID",
    "UNCALIBRATED_SELECTED_PROBABILITY_VERSION",
    "WINNER_BINNED_ABSOLUTE_GAP_ID",
    "WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION",
    "WINNER_BINNED_ABSOLUTE_GAP_VERSION",
    "WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION",
    "WINNER_RELIABILITY_CURVE_ID",
    "WINNER_RELIABILITY_CURVE_VERSION",
    "WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION",
    "BrierEvaluationResult",
    "CalibrationEvaluationCohort",
    "CalibrationEvaluationDataset",
    "EvaluationSplitRole",
    "LogLossEvaluationResult",
    "ReliabilityBinSummary",
    "WinnerBinnedAbsoluteGapResult",
    "WinnerCorrectnessDiagnosticsResult",
    "WinnerReliabilityResult",
    "evaluate_uncalibrated_winner_brier",
    "evaluate_uncalibrated_winner_diagnostics",
    "evaluate_uncalibrated_winner_log_loss",
    "evaluate_uncalibrated_winner_reliability",
    "evaluate_winner_binned_absolute_gap",
]

# ---------------------------------------------------------------------------
# Fingerprint schema versions (new in this module; existing versions unchanged)
# ---------------------------------------------------------------------------

CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION = 1
CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION = 2
BRIER_EVALUATION_RESULT_FINGERPRINT_VERSION = 2
LOG_LOSS_EVALUATION_RESULT_FINGERPRINT_VERSION = 2
WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION = 1

# ---------------------------------------------------------------------------
# Score semantics identity
# ---------------------------------------------------------------------------

#: The evaluated input score is the probability the *uncalibrated* semantic
#: distribution assigns to the recorded selected semantic value carried by the
#: supplied runtime-linked result. It is NOT ``predicted_correctness``, NOT a
#: confidence, and NOT a calibrated probability. The runtime never sets
#: ``predicted_correctness`` and never marks a result ``calibrated=True``; this
#: module never does either.
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
# Companion diagnostics identity
# ---------------------------------------------------------------------------

#: The empirical winner-correctness rate over the metric-eligible projection:
#: ``correct_count / count``. On this binary winner-correctness target this ONE
#: quantity is both the empirical correctness rate and the ordinary decision
#: accuracy on the evaluated rows; it is deliberately NOT stored twice under
#: two names. It is a rate over the evaluated population, NOT a per-example
#: predicted probability, and it is never written into
#: ``predicted_correctness``.
EMPIRICAL_CORRECTNESS_RATE_ID = "empirical-winner-correctness-rate"
EMPIRICAL_CORRECTNESS_RATE_VERSION = 1

#: The mean of the uncalibrated selected semantic probabilities over the
#: metric-eligible projection. Its semantics remain the UNCALIBRATED selected
#: semantic probability in aggregate; it is NOT a confidence, NOT predicted
#: correctness, and NOT a calibrated probability.
MEAN_SELECTED_PROBABILITY_ID = "mean-uncalibrated-selected-probability"
MEAN_SELECTED_PROBABILITY_VERSION = 1

#: The Brier value that a constant score equal to the evaluated population's
#: empirical correctness rate would obtain on these SAME evaluated rows. This
#: is a hindsight, in-sample, descriptive reference point derived from the
#: evaluation outcomes themselves; it is NOT an operational predictor, NOT a
#: training-derived baseline, and NOT calibration evidence.
EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID = "empirical-correctness-rate-constant-brier-reference"
EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION = 1

# ---------------------------------------------------------------------------
# Equal-width reliability summary identity
# ---------------------------------------------------------------------------

#: The only binning policy implemented by the reliability summary: equal-width
#: intervals over ``[0, 1]`` with the frozen boundary contract documented on
#: :class:`ReliabilityBinSummary`. Equal-mass (quantile) binning remains
#: unimplemented future work: it needs its own tie, duplicate-score, and
#: deterministic-partition contract before it can carry an identity.
EQUAL_WIDTH_BINNING_ID = "equal-width"
EQUAL_WIDTH_BINNING_VERSION = 1

#: The reliability summary semantic identity: a pre-calibration
#: winner-correctness reliability summary over raw-score regions. It is NOT
#: ECE, NOT a calibration curve, and NOT evidence of calibration.
WINNER_RELIABILITY_CURVE_ID = "winner-reliability-curve"
WINNER_RELIABILITY_CURVE_VERSION = 1

#: The canonical-payload schema version of the reliability result artifact.
#: It is deliberately distinct from the reliability semantic version and from
#: the binning-policy semantic version.
WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION = 1

# ---------------------------------------------------------------------------
# Binned absolute-gap aggregate identity (ECE-form, derived consumer)
# ---------------------------------------------------------------------------

#: The derived-aggregate semantic identity: a sample-weighted absolute
#: discrepancy between raw selected-score bin means and empirical
#: winner-correctness rates, computed over the bins of one exact
#: :class:`WinnerReliabilityResult`. Its formula equals the conventional
#: equal-width ECE estimator form, but the canonical semantics are the binned
#: absolute-gap diagnostic: this is NOT a calibration-error claim, NOT "model
#: ECE", and NOT evidence that the raw selected semantic probability is
#: calibrated.
WINNER_BINNED_ABSOLUTE_GAP_ID = "winner-correctness-equal-width-binned-absolute-gap"
WINNER_BINNED_ABSOLUTE_GAP_VERSION = 1

#: The canonical-payload schema version of the binned absolute-gap result
#: artifact. Deliberately distinct from the aggregate semantic version and
#: from every upstream identity version.
WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION = 1

# ---------------------------------------------------------------------------
# Construction tokens for the supported-path-only artifacts
# ---------------------------------------------------------------------------

_COHORT_CONSTRUCTION_TOKEN = object()
_DATASET_CONSTRUCTION_TOKEN = object()
_BRIER_RESULT_CONSTRUCTION_TOKEN = object()
_LOG_LOSS_RESULT_CONSTRUCTION_TOKEN = object()
_DIAGNOSTICS_RESULT_CONSTRUCTION_TOKEN = object()
_RELIABILITY_RESULT_CONSTRUCTION_TOKEN = object()
_BINNED_ABSOLUTE_GAP_RESULT_CONSTRUCTION_TOKEN = object()

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

    The recorded ``selected_value`` is the selection carried by the supplied
    runtime-linked result: the winner is NOT recomputed and tie-breaking is
    NOT re-run. For Choice decisions the probability is looked up by semantic
    candidate name (never by a scoring label such as ``A``/``B``/``C`` and
    never by a token id). For Bool decisions the outcome order is
    ``("false", "true")`` and the name is chosen from the recorded boolean.
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


# ---------------------------------------------------------------------------
# Declared evaluation source cohort
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationEvaluationCohort:
    """The FULL set of observations supplied as one declared evaluation split.

    A cohort is declared BEFORE metric eligibility is applied. It may contain
    fit-eligible resolved rows, taxonomy-miss rows, unresolved rows, and
    resolved-but-unadjudicated rows; those rows are retained as real rows and
    explicitly accounted, never silently filtered. A cohort is NOT a fitting
    dataset, NOT a metric result, and NOT proof that the caller supplied every
    real deployment event: it records the source cohort supplied to the
    evaluation harness, and it cannot prove that a caller did not discard
    events before cohort construction.

    All identity fields are derived. Callers must not (and cannot) supply a
    binding, ground-truth semantics, counts, or a fingerprint: the binding and
    the ground-truth semantics are derived from the observations, the
    exclusion accounting is deterministically derived from the observation
    statuses, and the fingerprint is computed from the admitted rows plus the
    declared split metadata.

    Admission is all-or-nothing: reasons accumulate and nothing is silently
    filtered. Every element must be a real :class:`CalibrationObservation`
    (duck-typed look-alikes and non-tuple containers are rejected), all
    observations must share one ``CalibrationBinding`` canonical payload and
    one :class:`GroundTruthSemanticsIdentity` (canonical-JSON string
    equality, never Python ``dict ==``), and the split metadata must be a
    valid :class:`EvaluationSplitRole` plus a non-empty ``split_id``. Mixed
    bindings or mixed ground-truth statistical meanings remain separate
    evaluation populations and are rejected.

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
            rejected.append("an evaluation cohort must contain at least one observation")
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
            observation_binding_json = canonical_json(observation.binding.canonical_payload())
            if observation_binding_json != binding_payload_json:
                rejected.append(
                    f"observation {index}: binding canonical payload "
                    f"{observation_binding_json} does not match the cohort "
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
                    f"{observation_semantics_json} does not match the cohort "
                    f"ground-truth semantics identity {semantics_payload_json}"
                )
        if rejected:
            raise InvalidDecisionError(
                f"calibration evaluation cohort rejected "
                f"{len(rejected)} reason(s) for "
                f"{len(observations)} observation(s): " + "; ".join(rejected)
            )

    # -- Deterministic metric-eligibility partition -------------------------

    def _partition(self) -> dict[str, tuple[CalibrationObservation, ...]]:
        """Partition cohort rows by metric-eligibility precedence.

        The precedence is fixed and mutually exclusive, so no row is counted
        twice and no unresolved-plus-unadjudicated row is double-counted:

        1. ``status == TAXONOMY_MISS`` -> ``taxonomy_miss``
        2. ``status == UNRESOLVED`` -> ``unresolved``
        3. ``status == RESOLVED`` and ``provenance.adjudicated is not True``
           -> ``unadjudicated_resolved``
        4. otherwise -> ``eligible``
        """
        buckets: dict[str, list[CalibrationObservation]] = {
            "eligible": [],
            "taxonomy_miss": [],
            "unresolved": [],
            "unadjudicated_resolved": [],
        }
        for observation in self.observations:
            if observation.status is CalibrationObservationStatus.TAXONOMY_MISS:
                buckets["taxonomy_miss"].append(observation)
            elif observation.status is CalibrationObservationStatus.UNRESOLVED:
                buckets["unresolved"].append(observation)
            elif (
                observation.status is CalibrationObservationStatus.RESOLVED
                and observation.ground_truth.provenance.adjudicated is not True
            ):
                buckets["unadjudicated_resolved"].append(observation)
            else:
                buckets["eligible"].append(observation)
        return {name: tuple(rows) for name, rows in buckets.items()}

    @property
    def source_count(self) -> int:
        """The number of rows in the declared source cohort."""
        return len(self.observations)

    @property
    def eligible_observations(self) -> tuple[CalibrationObservation, ...]:
        """The metric-eligible rows of the deterministic partition."""
        return self._partition()["eligible"]

    @property
    def eligible_count(self) -> int:
        """The number of metric-eligible rows."""
        return len(self.eligible_observations)

    @property
    def taxonomy_miss_count(self) -> int:
        """The number of taxonomy-miss rows retained in the cohort."""
        return len(self._partition()["taxonomy_miss"])

    @property
    def unresolved_count(self) -> int:
        """The number of unresolved rows retained in the cohort."""
        return len(self._partition()["unresolved"])

    @property
    def unadjudicated_resolved_count(self) -> int:
        """The number of resolved-but-unadjudicated rows retained in the cohort."""
        return len(self._partition()["unadjudicated_resolved"])

    @property
    def binding(self) -> CalibrationBinding:
        """The binding derived from the cohort observations."""
        return self.observations[0].binding

    @property
    def ground_truth_semantics(self) -> GroundTruthSemanticsIdentity:
        """The ground-truth semantics identity derived from the observations."""
        return GroundTruthSemanticsIdentity.from_provenance(
            self.observations[0].ground_truth.provenance
        )

    @property
    def fingerprint(self) -> str:
        """Row-order independent identity of the declared source cohort.

        The payload commits the split metadata, the derived binding and
        ground-truth semantics identities, and the sorted fingerprints of ALL
        cohort rows (multiplicity preserved, never a set). Because each
        observation fingerprint already commits the derived status,
        correctness, the ground-truth record, and the provenance, the cohort
        identity includes excluded rows as real rows, not just counters. A
        different ``split_role`` or a different ``split_id`` yields a
        different fingerprint even when the observations are identical.
        """
        observation_fingerprints: list[JSONValue] = [
            observation.fingerprint for observation in self.observations
        ]
        observation_fingerprints.sort(key=str)
        payload: dict[str, JSONValue] = {
            "v": CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
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


# ---------------------------------------------------------------------------
# Metric-eligible projection of one declared source cohort
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationEvaluationDataset:
    """The metric-eligible projection of one declared evaluation cohort.

    A dataset is a DERIVED view of exactly one
    :class:`CalibrationEvaluationCohort`: it holds the cohort's
    metric-eligible rows plus the cohort's split metadata, source cohort
    identity, and explicit exclusion accounting. It is not an independently
    constructable container: the only supported construction path is
    :meth:`from_cohort`, so a caller cannot present an arbitrary
    already-filtered tuple as an evaluation dataset with no cohort
    provenance.

    All identity fields are derived. Callers must not (and cannot) supply a
    binding, ground-truth semantics, correctness labels, counts, or a dataset
    fingerprint: the binding and semantics are derived from the cohort
    observations, correctness is the already-derived
    ``CalibrationObservation.correct``, and the fingerprint is computed from
    the eligible rows, the source cohort identity, and the exclusion
    accounting.

    A cohort with zero eligible rows is still a valid provenance artifact (an
    all-taxonomy-miss cohort is legitimate), but the metric-dataset projection
    fails clearly, because a Brier score or a log loss cannot average zero
    evaluated rows.

    The declared ``split_role`` and ``split_id`` are provenance, not proof:
    they record which role the caller assigned to this split and do not prove
    statistical independence from any data used for future fitting.
    """

    observations: tuple[CalibrationObservation, ...]
    split_role: EvaluationSplitRole
    split_id: str
    source_cohort_fingerprint: str = field(init=False, repr=False)
    source_count: int = field(init=False, repr=False)
    taxonomy_miss_count: int = field(init=False, repr=False)
    unresolved_count: int = field(init=False, repr=False)
    unadjudicated_resolved_count: int = field(init=False, repr=False)

    def __init__(
        self,
        observations: tuple[CalibrationObservation, ...] | None = None,
        split_role: EvaluationSplitRole | None = None,
        split_id: str | None = None,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _DATASET_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "CalibrationEvaluationDataset cannot be constructed directly or "
                "with dataclasses.replace; build a CalibrationEvaluationCohort "
                "and use CalibrationEvaluationDataset.from_cohort(cohort), the "
                "only supported construction path"
            )
        if not isinstance(observations, tuple) or not observations:
            raise InvalidDecisionError(
                "an evaluation dataset must contain at least one observation"
            )
        if not isinstance(split_role, EvaluationSplitRole):
            raise InvalidDecisionError(
                "split_role must be an EvaluationSplitRole (validation or test), "
                f"got {split_role!r}"
            )
        _require_non_empty_str("split_id", split_id)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "split_role", split_role)
        object.__setattr__(self, "split_id", split_id)

    @classmethod
    def from_cohort(cls, cohort: CalibrationEvaluationCohort) -> CalibrationEvaluationDataset:
        """Project the metric-eligible rows of one declared source cohort.

        This is the only supported construction path. The projection derives
        the eligible rows, the binding, the ground-truth semantics, the split
        metadata, the source cohort identity, and the exclusion accounting
        from the cohort; nothing is caller-supplied. A cohort with zero
        eligible rows is rejected here: the cohort itself remains a valid
        provenance artifact, but a metric dataset over zero evaluated rows
        cannot support a Brier score or a log loss.
        """
        if not isinstance(cohort, CalibrationEvaluationCohort):
            raise InvalidDecisionError(
                "CalibrationEvaluationDataset.from_cohort requires a "
                f"CalibrationEvaluationCohort, got {type(cohort).__name__}"
            )
        eligible = cohort.eligible_observations
        if not eligible:
            raise InvalidDecisionError(
                "the declared evaluation cohort has zero metric-eligible rows "
                f"(source_count={cohort.source_count}, "
                f"taxonomy_miss_count={cohort.taxonomy_miss_count}, "
                f"unresolved_count={cohort.unresolved_count}, "
                "unadjudicated_resolved_count="
                f"{cohort.unadjudicated_resolved_count}); a metric-eligible "
                "projection cannot be formed because a Brier score or a log "
                "loss cannot average zero evaluated rows"
            )
        dataset = cls(
            eligible,
            cohort.split_role,
            cohort.split_id,
            _construction_token=_DATASET_CONSTRUCTION_TOKEN,
        )
        object.__setattr__(dataset, "source_cohort_fingerprint", cohort.fingerprint)
        object.__setattr__(dataset, "source_count", cohort.source_count)
        object.__setattr__(dataset, "taxonomy_miss_count", cohort.taxonomy_miss_count)
        object.__setattr__(dataset, "unresolved_count", cohort.unresolved_count)
        object.__setattr__(
            dataset, "unadjudicated_resolved_count", cohort.unadjudicated_resolved_count
        )
        return dataset

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
        """Row-order independent identity of the metric-eligible projection.

        The payload commits the split metadata, the derived binding and
        ground-truth semantics identities, the sorted eligible observation
        fingerprints (multiplicity preserved, never a set), the source cohort
        identity, and the explicit exclusion accounting. Two cohorts with
        identical scored rows but different taxonomy-miss, unresolved, or
        unadjudicated exclusions therefore never collapse to the same
        evaluation dataset fingerprint.
        """
        observation_fingerprints: list[JSONValue] = [
            observation.fingerprint for observation in self.observations
        ]
        observation_fingerprints.sort(key=str)
        payload: dict[str, JSONValue] = {
            "v": CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
            "source_cohort_fingerprint": self.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": (
                CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION
            ),
            "split_role": str(self.split_role.value),
            "split_id": self.split_id,
            "binding_fingerprint": self.binding.fingerprint,
            "binding_fingerprint_version": CALIBRATION_BINDING_FINGERPRINT_VERSION,
            "ground_truth_semantics_fingerprint": (self.ground_truth_semantics.fingerprint),
            "ground_truth_semantics_fingerprint_version": (
                GROUND_TRUTH_SEMANTICS_FINGERPRINT_VERSION
            ),
            "observation_fingerprints": observation_fingerprints,
            "source_count": self.source_count,
            "eligible_count": len(self.observations),
            "taxonomy_miss_count": self.taxonomy_miss_count,
            "unresolved_count": self.unresolved_count,
            "unadjudicated_resolved_count": self.unadjudicated_resolved_count,
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

    ``count`` is the number of observations actually scored (the evaluated
    count), NOT the source cohort size. The cohort provenance fields record
    the declared source cohort identity and the explicit exclusion
    accounting, so the artifact is auditable without pretending that
    ``count`` equals the original cohort size. ``value`` and ``count`` are
    derived from the dataset by the evaluator and are stored as immutable
    scalars. ``canonical_payload`` emits ``"configuration": {}`` as a
    constant: the metric has no free parameters, and no caller-controlled
    configuration mapping exists.
    """

    metric_id: str = field(init=False, repr=False)
    metric_version: int = field(init=False, repr=False)
    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    source_cohort_fingerprint: str = field(init=False, repr=False)
    source_cohort_fingerprint_version: int = field(init=False, repr=False)
    source_count: int = field(init=False, repr=False)
    taxonomy_miss_count: int = field(init=False, repr=False)
    unresolved_count: int = field(init=False, repr=False)
    unadjudicated_resolved_count: int = field(init=False, repr=False)
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
        object.__setattr__(self, "source_cohort_fingerprint", dataset.source_cohort_fingerprint)
        object.__setattr__(
            self,
            "source_cohort_fingerprint_version",
            CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "source_count", dataset.source_count)
        object.__setattr__(self, "taxonomy_miss_count", dataset.taxonomy_miss_count)
        object.__setattr__(self, "unresolved_count", dataset.unresolved_count)
        object.__setattr__(
            self, "unadjudicated_resolved_count", dataset.unadjudicated_resolved_count
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "value", value)

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the metric identity, the input-score identity, the evaluation
        dataset identity with its payload schema version, the source cohort
        identity, the explicit exclusion accounting, the (empty)
        configuration, the evaluated ``count``, and ``value``. Timestamps,
        wall-clock time, hostnames, and logging metadata are deliberately
        excluded.
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
            "source_cohort_fingerprint": self.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": self.source_cohort_fingerprint_version,
            "source_count": self.source_count,
            "taxonomy_miss_count": self.taxonomy_miss_count,
            "unresolved_count": self.unresolved_count,
            "unadjudicated_resolved_count": self.unadjudicated_resolved_count,
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

    ``count`` is the number of observations actually scored (the evaluated
    count), NOT the source cohort size. The cohort provenance fields record
    the declared source cohort identity and the explicit exclusion
    accounting, so the artifact is auditable without pretending that
    ``count`` equals the original cohort size. ``value`` and ``count`` are
    derived from the dataset by the evaluator and are stored as immutable
    scalars. The metric configuration is fixed and derived from the module
    constants; ``canonical_payload`` emits it literally as
    ``{"boundary_policy": "exact", "log_base": "e"}``. No caller-controlled
    configuration mapping exists.
    """

    metric_id: str = field(init=False, repr=False)
    metric_version: int = field(init=False, repr=False)
    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    source_cohort_fingerprint: str = field(init=False, repr=False)
    source_cohort_fingerprint_version: int = field(init=False, repr=False)
    source_count: int = field(init=False, repr=False)
    taxonomy_miss_count: int = field(init=False, repr=False)
    unresolved_count: int = field(init=False, repr=False)
    unadjudicated_resolved_count: int = field(init=False, repr=False)
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
        object.__setattr__(self, "source_cohort_fingerprint", dataset.source_cohort_fingerprint)
        object.__setattr__(
            self,
            "source_cohort_fingerprint_version",
            CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "source_count", dataset.source_count)
        object.__setattr__(self, "taxonomy_miss_count", dataset.taxonomy_miss_count)
        object.__setattr__(self, "unresolved_count", dataset.unresolved_count)
        object.__setattr__(
            self, "unadjudicated_resolved_count", dataset.unadjudicated_resolved_count
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "value", value)

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the metric identity, the input-score identity, the evaluation
        dataset identity with its payload schema version, the source cohort
        identity, the explicit exclusion accounting, the fixed metric
        configuration, the evaluated ``count``, and a structural encoding of
        ``value``. Timestamps, wall-clock time, hostnames, and logging
        metadata are deliberately excluded.

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
            "source_cohort_fingerprint": self.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": self.source_cohort_fingerprint_version,
            "source_count": self.source_count,
            "taxonomy_miss_count": self.taxonomy_miss_count,
            "unresolved_count": self.unresolved_count,
            "unadjudicated_resolved_count": self.unadjudicated_resolved_count,
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


@dataclass(frozen=True, slots=True)
class WinnerCorrectnessDiagnosticsResult:
    """Immutable, provenance-rich companion diagnostics for one evaluation run.

    This artifact answers "what do these evaluated rows look like" next to the
    Brier and exact log-loss metrics. It commits three diagnostics over the
    SAME metric-eligible projection the metrics score:

    - the empirical winner-correctness rate (``correct_count / count``). On
      this binary winner-correctness target this ONE quantity is also the
      ordinary decision accuracy on the evaluated rows; it is stored once,
      under one identity, never duplicated as a second ``accuracy`` or
      ``base_rate`` field.
    - the mean uncalibrated selected semantic probability: aggregate raw-score
      behaviour, NOT a confidence and NOT predicted correctness.
    - the empirical constant Brier reference: the Brier value a constant score
      equal to the empirical correctness rate would obtain on these SAME
      evaluated rows. It is a hindsight, in-sample, descriptive reference
      derived from the evaluation outcomes themselves, NOT an operational
      predictor available before observing the outcomes, NOT a
      training-derived baseline, and NOT calibration evidence. All-correct
      gives rate ``1`` and reference ``0``; all-wrong gives rate ``0`` and
      reference ``0``; that endpoint behaviour is deliberate and is exactly
      why the reference is a hindsight prevalence reference. No Brier skill
      score, relative improvement, or winner/loser verdict is derived from it.

    The supported construction contract is the module-level evaluator
    :func:`evaluate_uncalibrated_winner_diagnostics`. Direct construction and
    ``dataclasses.replace`` reconstruction are rejected so a caller cannot
    combine one dataset with fabricated diagnostics. This is an API discipline
    within the supported construction contract, not a security boundary;
    low-level Python escape hatches such as ``object.__new__`` are not
    supported construction paths and are not defended against.

    ``dataclasses.replace(result)`` reaches ``__init__`` with no construction
    token and is rejected with :class:`InvalidDecisionError`.
    ``dataclasses.replace(result, value=...)`` is rejected earlier by
    ``dataclasses`` itself (a ``ValueError``, because every field is declared
    with ``init=False``) and never reaches ``__init__``; that rejection is
    intentional and is NOT an :class:`InvalidDecisionError`.

    ``count`` is the number of observations actually scored (the evaluated
    count), NOT the source cohort size. Excluded cohort rows never enter
    ``count``, the correctness rate, the mean selected probability, or the
    constant reference. The cohort provenance fields record the declared
    source cohort identity and the explicit exclusion accounting. No
    caller-supplied derived value exists, and ``canonical_payload`` generates
    every nested object fresh; no caller-owned mapping is stored.
    """

    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    source_cohort_fingerprint: str = field(init=False, repr=False)
    source_cohort_fingerprint_version: int = field(init=False, repr=False)
    source_count: int = field(init=False, repr=False)
    taxonomy_miss_count: int = field(init=False, repr=False)
    unresolved_count: int = field(init=False, repr=False)
    unadjudicated_resolved_count: int = field(init=False, repr=False)
    count: int = field(init=False, repr=False)
    correct_count: int = field(init=False, repr=False)
    incorrect_count: int = field(init=False, repr=False)
    empirical_correctness_rate: float = field(init=False, repr=False)
    mean_selected_probability: float = field(init=False, repr=False)
    empirical_constant_brier_reference: float = field(init=False, repr=False)

    def __init__(
        self,
        dataset: CalibrationEvaluationDataset | None = None,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _DIAGNOSTICS_RESULT_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "WinnerCorrectnessDiagnosticsResult cannot be constructed "
                "directly or with dataclasses.replace; use "
                "evaluate_uncalibrated_winner_diagnostics(dataset), the only "
                "supported construction path"
            )
        if not isinstance(dataset, CalibrationEvaluationDataset):
            raise InvalidDecisionError(
                "the supported WinnerCorrectnessDiagnosticsResult construction "
                "path requires a CalibrationEvaluationDataset, got "
                f"{type(dataset).__name__}"
            )
        count = len(dataset.observations)
        labels: list[float] = []
        probabilities: list[float] = []
        for observation in dataset.observations:
            labels.append(1.0 if observation.correct else 0.0)
            probabilities.append(_selected_probability(observation))
        correct_count = math.fsum(labels)
        if not correct_count.is_integer():
            raise InvalidDecisionError(
                "the derived winner-correctness labels are not binary; the "
                "winner-correctness diagnostics cannot be computed"
            )
        correct_count = int(correct_count)
        incorrect_count = count - correct_count
        empirical_correctness_rate = correct_count / count
        mean_selected_probability = math.fsum(probabilities) / count
        empirical_constant_brier_reference = (
            math.fsum((empirical_correctness_rate - label) ** 2 for label in labels) / count
        )
        object.__setattr__(self, "target", BRIER_TARGET)
        object.__setattr__(self, "input_score_id", UNCALIBRATED_SELECTED_PROBABILITY_ID)
        object.__setattr__(self, "input_score_version", UNCALIBRATED_SELECTED_PROBABILITY_VERSION)
        object.__setattr__(self, "evaluation_dataset_fingerprint", dataset.fingerprint)
        object.__setattr__(
            self,
            "evaluation_dataset_fingerprint_version",
            CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "source_cohort_fingerprint", dataset.source_cohort_fingerprint)
        object.__setattr__(
            self,
            "source_cohort_fingerprint_version",
            CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "source_count", dataset.source_count)
        object.__setattr__(self, "taxonomy_miss_count", dataset.taxonomy_miss_count)
        object.__setattr__(self, "unresolved_count", dataset.unresolved_count)
        object.__setattr__(
            self, "unadjudicated_resolved_count", dataset.unadjudicated_resolved_count
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "correct_count", correct_count)
        object.__setattr__(self, "incorrect_count", incorrect_count)
        object.__setattr__(self, "empirical_correctness_rate", empirical_correctness_rate)
        object.__setattr__(self, "mean_selected_probability", mean_selected_probability)
        object.__setattr__(
            self, "empirical_constant_brier_reference", empirical_constant_brier_reference
        )

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the diagnostics identity, the input-score identity, the
        evaluation dataset identity with its payload schema version, the
        source cohort identity, the explicit exclusion accounting, the label
        and rate accounting, and each diagnostic with its own identity and
        version. Every nested object is generated fresh here; no caller-owned
        mapping is stored or emitted. Timestamps, wall-clock time, hostnames,
        and logging metadata are deliberately excluded.
        """
        return {
            "v": WINNER_CORRECTNESS_DIAGNOSTICS_FINGERPRINT_VERSION,
            "target": self.target,
            "input_score_id": self.input_score_id,
            "input_score_version": self.input_score_version,
            "evaluation_dataset_fingerprint": self.evaluation_dataset_fingerprint,
            "evaluation_dataset_fingerprint_version": (self.evaluation_dataset_fingerprint_version),
            "source_cohort_fingerprint": self.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": self.source_cohort_fingerprint_version,
            "source_count": self.source_count,
            "count": self.count,
            "exclusions": {
                "taxonomy_miss": self.taxonomy_miss_count,
                "unresolved": self.unresolved_count,
                "unadjudicated_resolved": self.unadjudicated_resolved_count,
            },
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "diagnostics": {
                "empirical_correctness_rate": {
                    "id": EMPIRICAL_CORRECTNESS_RATE_ID,
                    "version": EMPIRICAL_CORRECTNESS_RATE_VERSION,
                    "value": self.empirical_correctness_rate,
                },
                "mean_selected_probability": {
                    "id": MEAN_SELECTED_PROBABILITY_ID,
                    "version": MEAN_SELECTED_PROBABILITY_VERSION,
                    "value": self.mean_selected_probability,
                },
                "empirical_constant_brier_reference": {
                    "id": EMPIRICAL_CONSTANT_BRIER_REFERENCE_ID,
                    "version": EMPIRICAL_CONSTANT_BRIER_REFERENCE_VERSION,
                    "constant_probability": self.empirical_correctness_rate,
                    "value": self.empirical_constant_brier_reference,
                },
            },
        }

    @property
    def fingerprint(self) -> str:
        """Stable identity of the diagnostics artifact.

        The hash of :meth:`canonical_payload`, deterministic for the object
        lifetime: every payload field is an immutable scalar, string, or
        freshly generated nested object fixed at construction.
        """
        return fingerprint(self.canonical_payload())


def evaluate_uncalibrated_winner_diagnostics(
    dataset: CalibrationEvaluationDataset,
) -> WinnerCorrectnessDiagnosticsResult:
    """Evaluate the winner-correctness companion diagnostics.

    Over the metric-eligible projection of the dataset (excluded cohort rows
    never enter any denominator):

    - ``y_i = 1`` when the observation is correct, ``0`` otherwise;
      ``correct_count = sum(y_i)``, ``incorrect_count = count -
      correct_count``, and ``correct_count + incorrect_count == count``.
    - ``empirical_correctness_rate = correct_count / count``. On this binary
      winner-correctness target this is also the ordinary decision accuracy
      on the evaluated rows. It is a rate over the evaluated population, NOT
      a per-example predicted probability, and it is never written into
      ``predicted_correctness``.
    - ``mean_selected_probability = fsum(p_i) / count`` where ``p_i`` is the
      uncalibrated selected semantic probability extracted by the SAME
      ``_selected_probability`` helper the metrics use. Its semantics remain
      the uncalibrated selected semantic probability in aggregate; it is NOT
      a confidence, NOT predicted correctness, and NOT a calibrated
      probability.
    - ``empirical_constant_brier_reference = fsum((q - y_i)^2) / count``
      where ``q`` is the empirical correctness rate; algebraically
      ``q * (1 - q)``. It answers "what Brier value would a constant score
      equal to this evaluated population's empirical correctness rate obtain
      on these same evaluated rows?" It is a hindsight, in-sample, descriptive
      reference derived from the evaluation outcomes themselves, NOT an
      operational predictor, NOT a training-derived baseline, and NOT
      calibration evidence. No Brier skill score, relative improvement, or
      winner/loser verdict is derived from it, and no constant log-loss,
      entropy, or cross-entropy reference exists.

    This is a read-only offline measurement: it does not mutate the dataset or
    any observation, does not set ``calibrated`` or ``predicted_correctness``
    anywhere, does not create a :class:`~probvenance.calibration.CalibrationProfile`,
    and does not touch the runtime.
    """
    if not isinstance(dataset, CalibrationEvaluationDataset):
        raise InvalidDecisionError(
            "evaluate_uncalibrated_winner_diagnostics requires a "
            f"CalibrationEvaluationDataset, got {type(dataset).__name__}"
        )
    return WinnerCorrectnessDiagnosticsResult(
        dataset,
        _construction_token=_DIAGNOSTICS_RESULT_CONSTRUCTION_TOKEN,
    )


# ---------------------------------------------------------------------------
# Equal-width winner-correctness reliability summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReliabilityBinSummary:
    """One equal-width raw-score region of a reliability summary.

    The interval contract is frozen. For ``B = bin_count`` and
    ``i = 0 .. B - 1`` the region owned by bin ``i`` is::

        bin 0:      [0/B, 1/B)
        bin i:      [i/B, (i+1)/B)
        last bin:   [(B-1)/B, 1]

    The lower bound is inclusive and the upper bound is exclusive, EXCEPT the
    final upper bound ``1`` is inclusive, so ``p = 0`` falls in the first bin
    and ``p = 1`` falls in the last bin. An exact interior boundary
    ``p = i/B`` belongs to bin ``i``, NOT to bin ``i - 1``. The boundaries
    themselves are NOT stored per bin: they are fully determined by
    ``bin_count`` and ``index``, so identity carries the binning id and
    version, ``bin_count``, and ``index``.

    Empty bins are retained, never omitted: ``count = 0``,
    ``correct_count = 0``, both statistics are ``None``, and the membership
    tuple is empty. ``None`` means "no observation fell in this interval"; it
    is never ``NaN`` and never ``0.0``, because an empty interval is not the
    same thing as an interval whose observed rate is zero.

    ``observation_fingerprints`` is sorted and multiplicity preserving: a
    duplicated dataset row appears twice, never deduplicated to a set.
    """

    index: int
    count: int
    correct_count: int
    mean_selected_probability: float | None
    empirical_correctness_rate: float | None
    observation_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InvalidDecisionError(
                f"ReliabilityBinSummary index must be a non-negative real int, got {self.index!r}"
            )
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise InvalidDecisionError(
                f"ReliabilityBinSummary count must be a non-negative real int, got {self.count!r}"
            )
        if (
            isinstance(self.correct_count, bool)
            or not isinstance(self.correct_count, int)
            or self.correct_count < 0
            or self.correct_count > self.count
        ):
            raise InvalidDecisionError(
                "ReliabilityBinSummary correct_count must be a real int with "
                f"0 <= correct_count <= count, got {self.correct_count!r} "
                f"for count {self.count}"
            )
        if len(self.observation_fingerprints) != self.count:
            raise InvalidDecisionError(
                "ReliabilityBinSummary observation_fingerprints must carry "
                f"exactly one entry per counted observation: got "
                f"{len(self.observation_fingerprints)} fingerprints for "
                f"count {self.count}"
            )
        if self.count == 0:
            if (
                self.mean_selected_probability is not None
                or self.empirical_correctness_rate is not None
            ):
                raise InvalidDecisionError(
                    "an empty ReliabilityBinSummary must leave both statistics "
                    "undefined (None); an empty interval is never an observed "
                    "zero"
                )
            return
        for name, value in (
            ("mean_selected_probability", self.mean_selected_probability),
            ("empirical_correctness_rate", self.empirical_correctness_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidDecisionError(
                    f"a non-empty ReliabilityBinSummary requires a finite {name}, got {value!r}"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise InvalidDecisionError(
                    f"a non-empty ReliabilityBinSummary requires a finite "
                    f"{name} in [0, 1], got {number!r}"
                )


@dataclass(frozen=True, slots=True)
class WinnerReliabilityResult:
    """Immutable, provenance-rich pre-calibration reliability summary.

    This artifact answers "what is the observed winner-correctness rate inside
    each raw selected-probability region?" over the SAME metric-eligible
    projection the Brier, log-loss, and diagnostics artifacts describe. The
    x-axis score is the UNCALIBRATED selected semantic probability extracted by
    the SAME ``_selected_probability`` helper the metrics use. It is still NOT
    ``predicted_correctness``, NOT ``P(Y_correct = 1)``, NOT a confidence, and
    NOT a calibrated probability: the summary describes raw-score regions
    versus empirical winner correctness without claiming that the raw score
    carries correctness-probability semantics.

    The binning policy is equal-width only
    (``EQUAL_WIDTH_BINNING_ID``/``EQUAL_WIDTH_BINNING_VERSION``) with the
    frozen interval contract documented on :class:`ReliabilityBinSummary`.
    Equal-mass (quantile) binning, adaptive binning, minimum-count merging,
    smoothing, and kernel curves are NOT implemented; equal-mass remains
    future work because it needs its own tie, duplicate-score, and
    deterministic-partition contract. No ECE, gap, or calibration-error
    aggregate is derived here: a bin showing mean raw score ``0.8`` with an
    observed rate of ``0.6`` is a description of that region, not a licensed
    "20-point overconfidence" measurement, and no ECE ingredient is stored.

    Every ``bin_count`` bins are retained, including empty ones, so
    ``len(bins) == bin_count`` always holds and a reader can distinguish an
    empty interval from an interval that never existed.

    The supported construction contract is the module-level evaluator
    :func:`evaluate_uncalibrated_winner_reliability`. Direct construction and
    ``dataclasses.replace`` reconstruction are rejected so a caller cannot
    combine one dataset with fabricated bins. ``dataclasses.replace(result)``
    reaches ``__init__`` with no construction token and is rejected with
    :class:`InvalidDecisionError`; ``dataclasses.replace(result, bins=...)``
    is rejected earlier by ``dataclasses`` itself (a ``ValueError``, because
    every field is declared with ``init=False``) and never reaches
    ``__init__``. This is an API discipline within the supported construction
    contract, not a security boundary; low-level Python escape hatches such as
    ``object.__new__`` are not supported construction paths and are not
    defended against.

    ``count`` is the number of observations actually scored (the evaluated
    count), NOT the source cohort size. Excluded cohort rows never enter any
    bin. The provenance fields record the declared source cohort identity and
    the explicit exclusion accounting. No caller-owned mapping is stored.
    """

    reliability_id: str = field(init=False, repr=False)
    reliability_version: int = field(init=False, repr=False)
    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    binning_id: str = field(init=False, repr=False)
    binning_version: int = field(init=False, repr=False)
    bin_count: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    source_cohort_fingerprint: str = field(init=False, repr=False)
    source_cohort_fingerprint_version: int = field(init=False, repr=False)
    source_count: int = field(init=False, repr=False)
    taxonomy_miss_count: int = field(init=False, repr=False)
    unresolved_count: int = field(init=False, repr=False)
    unadjudicated_resolved_count: int = field(init=False, repr=False)
    count: int = field(init=False, repr=False)
    bins: tuple[ReliabilityBinSummary, ...] = field(init=False, repr=False)

    def __init__(
        self,
        dataset: CalibrationEvaluationDataset | None = None,
        *,
        bin_count: int | None = None,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _RELIABILITY_RESULT_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "WinnerReliabilityResult cannot be constructed directly or "
                "with dataclasses.replace; use "
                "evaluate_uncalibrated_winner_reliability(dataset, "
                "bin_count=...), the only supported construction path"
            )
        if not isinstance(dataset, CalibrationEvaluationDataset):
            raise InvalidDecisionError(
                "the supported WinnerReliabilityResult construction path "
                "requires a CalibrationEvaluationDataset, got "
                f"{type(dataset).__name__}"
            )
        if isinstance(bin_count, bool) or not isinstance(bin_count, int):
            raise InvalidDecisionError(
                f"bin_count must be a real int (a bool is not acceptable), got {bin_count!r}"
            )
        if bin_count < 1:
            raise InvalidDecisionError(f"bin_count must be at least 1, got {bin_count!r}")
        members: list[list[tuple[float, str, bool]]] = [[] for _ in range(bin_count)]
        for observation in dataset.observations:
            probability = _selected_probability(observation)
            # The boundary contract compares the STORED probability value
            # against the mathematical rational boundaries i/B. The exact
            # rational value of the stored float decides ownership, so binary
            # floating multiplication never silently defines a boundary: a
            # rational boundary with no exact float representation (for
            # example 1/3) is handled by comparing against the exact
            # rational, and the stored float falls deterministically on one
            # side of it.
            rational_index = Fraction.from_float(probability) * bin_count
            index = int(rational_index)
            if index >= bin_count:
                index = bin_count - 1
            members[index].append((probability, observation.fingerprint, bool(observation.correct)))
        bins: list[ReliabilityBinSummary] = []
        for index, bucket in enumerate(members):
            bucket.sort(key=lambda item: (item[0], item[1]))
            bucket_count = len(bucket)
            if bucket_count == 0:
                bins.append(
                    ReliabilityBinSummary(
                        index=index,
                        count=0,
                        correct_count=0,
                        mean_selected_probability=None,
                        empirical_correctness_rate=None,
                        observation_fingerprints=(),
                    )
                )
                continue
            bucket_mean = math.fsum(item[0] for item in bucket) / bucket_count
            bucket_correct = sum(1 for item in bucket if item[2])
            bucket_rate = bucket_correct / bucket_count
            bins.append(
                ReliabilityBinSummary(
                    index=index,
                    count=bucket_count,
                    correct_count=bucket_correct,
                    mean_selected_probability=bucket_mean,
                    empirical_correctness_rate=bucket_rate,
                    observation_fingerprints=tuple(sorted(item[1] for item in bucket)),
                )
            )
        object.__setattr__(self, "reliability_id", WINNER_RELIABILITY_CURVE_ID)
        object.__setattr__(self, "reliability_version", WINNER_RELIABILITY_CURVE_VERSION)
        object.__setattr__(self, "target", BRIER_TARGET)
        object.__setattr__(self, "input_score_id", UNCALIBRATED_SELECTED_PROBABILITY_ID)
        object.__setattr__(self, "input_score_version", UNCALIBRATED_SELECTED_PROBABILITY_VERSION)
        object.__setattr__(self, "binning_id", EQUAL_WIDTH_BINNING_ID)
        object.__setattr__(self, "binning_version", EQUAL_WIDTH_BINNING_VERSION)
        object.__setattr__(self, "bin_count", bin_count)
        object.__setattr__(self, "evaluation_dataset_fingerprint", dataset.fingerprint)
        object.__setattr__(
            self,
            "evaluation_dataset_fingerprint_version",
            CALIBRATION_EVALUATION_DATASET_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "source_cohort_fingerprint", dataset.source_cohort_fingerprint)
        object.__setattr__(
            self,
            "source_cohort_fingerprint_version",
            CALIBRATION_EVALUATION_COHORT_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "source_count", dataset.source_count)
        object.__setattr__(self, "taxonomy_miss_count", dataset.taxonomy_miss_count)
        object.__setattr__(self, "unresolved_count", dataset.unresolved_count)
        object.__setattr__(
            self, "unadjudicated_resolved_count", dataset.unadjudicated_resolved_count
        )
        object.__setattr__(self, "count", len(dataset.observations))
        object.__setattr__(self, "bins", tuple(bins))

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the reliability identity, the input-score identity, the
        binning identity with its own semantic version and configuration, the
        evaluation dataset identity with its payload schema version, the
        source cohort identity, the explicit exclusion accounting, and every
        bin with its membership provenance. Empty bins serialize their two
        statistics as ``null``; no NaN and no infinity is ever emitted, and
        every nested object is generated fresh here (no caller-owned mapping
        is stored or emitted).
        """
        return {
            "v": WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION,
            "reliability_id": self.reliability_id,
            "reliability_version": self.reliability_version,
            "target": self.target,
            "input_score_id": self.input_score_id,
            "input_score_version": self.input_score_version,
            "binning": {
                "id": self.binning_id,
                "version": self.binning_version,
                "bin_count": self.bin_count,
            },
            "evaluation_dataset_fingerprint": self.evaluation_dataset_fingerprint,
            "evaluation_dataset_fingerprint_version": (self.evaluation_dataset_fingerprint_version),
            "source_cohort_fingerprint": self.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": self.source_cohort_fingerprint_version,
            "source_count": self.source_count,
            "count": self.count,
            "exclusions": {
                "taxonomy_miss": self.taxonomy_miss_count,
                "unresolved": self.unresolved_count,
                "unadjudicated_resolved": self.unadjudicated_resolved_count,
            },
            "bins": [
                {
                    "index": bin.index,
                    "count": bin.count,
                    "correct_count": bin.correct_count,
                    "mean_selected_probability": bin.mean_selected_probability,
                    "empirical_correctness_rate": bin.empirical_correctness_rate,
                    "observation_fingerprints": list(bin.observation_fingerprints),
                }
                for bin in self.bins
            ],
        }

    @property
    def fingerprint(self) -> str:
        """Stable identity of the reliability artifact.

        The hash of :meth:`canonical_payload`, deterministic for the object
        lifetime: every payload field is an immutable scalar, string, or
        freshly generated nested object fixed at construction.
        """
        return fingerprint(self.canonical_payload())


def evaluate_uncalibrated_winner_reliability(
    dataset: CalibrationEvaluationDataset,
    *,
    bin_count: int,
) -> WinnerReliabilityResult:
    """Evaluate the pre-calibration equal-width reliability summary.

    Over the metric-eligible projection of the dataset (excluded cohort rows
    never enter any bin), partition the uncalibrated selected semantic
    probabilities into ``bin_count`` equal-width regions with the frozen
    interval contract documented on :class:`ReliabilityBinSummary`:

    - ``bin 0`` owns ``[0/B, 1/B)``, bin ``i`` owns ``[i/B, (i+1)/B)``, and
      the last bin owns ``[(B-1)/B, 1]`` with an inclusive upper bound, so
      ``p = 0`` falls in the first bin, ``p = 1`` falls in the last bin, and
      an exact interior boundary ``p = i/B`` belongs to bin ``i``.
    - Boundary ownership is decided by comparing the STORED probability value
      against the mathematical rational boundaries ``i/B`` (via
      :meth:`fractions.Fraction.from_float`), never by binary floating
      multiplication; a rational boundary with no exact float representation
      is handled by that exact rational comparison.
    - Each bin reports ``count``, ``correct_count``, the mean selected
      probability (``math.fsum`` over the deterministically sorted member
      probabilities, so dataset row order cannot change a bin mean), the
      empirical correctness rate, and the sorted, multiplicity-preserving
      member observation fingerprints. Empty bins are retained with ``None``
      statistics, never NaN and never a fake ``0.0``.

    ``bin_count`` must be a real ``int`` (a ``bool`` is NOT acceptable) with
    ``bin_count >= 1``; ``True``, ``False``, ``0``, negatives, floats,
    strings, and ``None`` are rejected with :class:`InvalidDecisionError`.
    There is no hidden maximum.

    The summary answers "what is the observed winner-correctness rate inside
    each raw selected-probability region?" It is a pre-calibration
    description of raw-score regions, NOT ECE, NOT a calibration curve, and
    NOT evidence of calibration: a bin showing mean raw score ``0.8`` with an
    observed rate of ``0.6`` does not license calling that a "20-point
    overconfidence". No per-bin gap or calibration-error aggregate is
    derived, and no ECE is computed in any form.

    This is a read-only offline measurement: it does not mutate the dataset
    or any observation, does not set ``calibrated`` or
    ``predicted_correctness`` anywhere, does not create a
    :class:`~probvenance.calibration.CalibrationProfile`, and does not touch
    the runtime.
    """
    if not isinstance(dataset, CalibrationEvaluationDataset):
        raise InvalidDecisionError(
            "evaluate_uncalibrated_winner_reliability requires a "
            f"CalibrationEvaluationDataset, got {type(dataset).__name__}"
        )
    return WinnerReliabilityResult(
        dataset,
        bin_count=bin_count,
        _construction_token=_RELIABILITY_RESULT_CONSTRUCTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class WinnerBinnedAbsoluteGapResult:
    """Derived aggregate over one exact :class:`WinnerReliabilityResult`.

    For every NON-EMPTY bin of the source reliability summary the aggregate
    computes ``abs(bin.mean_selected_probability -
    bin.empirical_correctness_rate)`` and combines the per-bin gaps with the
    sample weights ``bin.count / reliability.count`` via ``math.fsum`` in
    fixed bin-index order. The formula equals the conventional equal-width
    ECE estimator form, but the canonical semantics are the binned
    absolute-gap diagnostic: the value is the sample-weighted absolute
    discrepancy between raw selected-score bin means and empirical
    winner-correctness rates. It is NOT a calibration-error claim, NOT "model
    ECE", and NOT evidence that the raw selected semantic probability is
    calibrated. The absolute value carries no direction, so no
    overconfidence or underconfidence inference is possible.

    The aggregate never re-bins observations, never re-extracts
    probabilities, never re-assigns bin indices, and never re-implements the
    boundary logic: the source reliability summary is the only binning truth
    source. Empty bins contribute zero mass (they are skipped; ``None`` bin
    statistics are never treated as ``0`` and no fake observed gap is
    constructed). No per-bin gaps, largest gap, worst bin, direction, or
    over/underconfidence counts are stored.
    """

    aggregate_id: str = field(init=False, repr=False)
    aggregate_version: int = field(init=False, repr=False)
    target: str = field(init=False, repr=False)
    input_score_id: str = field(init=False, repr=False)
    input_score_version: int = field(init=False, repr=False)
    reliability_fingerprint: str = field(init=False, repr=False)
    reliability_fingerprint_version: int = field(init=False, repr=False)
    reliability_id: str = field(init=False, repr=False)
    reliability_version: int = field(init=False, repr=False)
    binning_id: str = field(init=False, repr=False)
    binning_version: int = field(init=False, repr=False)
    bin_count: int = field(init=False, repr=False)
    evaluation_dataset_fingerprint: str = field(init=False, repr=False)
    evaluation_dataset_fingerprint_version: int = field(init=False, repr=False)
    source_cohort_fingerprint: str = field(init=False, repr=False)
    source_cohort_fingerprint_version: int = field(init=False, repr=False)
    source_count: int = field(init=False, repr=False)
    count: int = field(init=False, repr=False)
    taxonomy_miss_count: int = field(init=False, repr=False)
    unresolved_count: int = field(init=False, repr=False)
    unadjudicated_resolved_count: int = field(init=False, repr=False)
    value: float = field(init=False, repr=False)

    def __init__(
        self,
        reliability: WinnerReliabilityResult | None = None,
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _BINNED_ABSOLUTE_GAP_RESULT_CONSTRUCTION_TOKEN:
            raise InvalidDecisionError(
                "WinnerBinnedAbsoluteGapResult cannot be constructed directly "
                "or with dataclasses.replace; use "
                "evaluate_winner_binned_absolute_gap(reliability), the only "
                "supported construction path"
            )
        if not isinstance(reliability, WinnerReliabilityResult):
            raise InvalidDecisionError(
                "the supported WinnerBinnedAbsoluteGapResult construction path "
                "requires a WinnerReliabilityResult, got "
                f"{type(reliability).__name__}"
            )
        total = reliability.count
        weighted_terms: list[float] = []
        for bin_summary in reliability.bins:
            if bin_summary.count == 0:
                continue
            mean_score = bin_summary.mean_selected_probability
            observed_rate = bin_summary.empirical_correctness_rate
            if mean_score is None or observed_rate is None:
                raise InvalidDecisionError(
                    "the source reliability summary reports a non-empty bin "
                    f"(index {bin_summary.index}) with missing statistics; the "
                    "upstream reliability invariant is broken and the binned "
                    "absolute-gap aggregate cannot be computed"
                )
            absolute_gap = abs(mean_score - observed_rate)
            weighted_term = bin_summary.count / total * absolute_gap
            if not math.isfinite(absolute_gap) or not math.isfinite(weighted_term):
                raise InvalidDecisionError(
                    "the source reliability summary produced a non-finite bin "
                    f"statistic (index {bin_summary.index}); the upstream "
                    "reliability invariant is broken and the binned "
                    "absolute-gap aggregate cannot be computed"
                )
            if weighted_term < 0.0:
                raise InvalidDecisionError(
                    "the source reliability summary produced a negative "
                    f"weighted term (index {bin_summary.index}); the upstream "
                    "reliability invariant is broken and the binned "
                    "absolute-gap aggregate cannot be computed"
                )
            weighted_terms.append(weighted_term)
        value = math.fsum(weighted_terms)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InvalidDecisionError(
                "the binned absolute-gap aggregate is not a finite value in "
                f"[0, 1] (got {value!r}); the upstream reliability invariant "
                "is broken and the aggregate cannot be computed"
            )
        object.__setattr__(self, "aggregate_id", WINNER_BINNED_ABSOLUTE_GAP_ID)
        object.__setattr__(self, "aggregate_version", WINNER_BINNED_ABSOLUTE_GAP_VERSION)
        object.__setattr__(self, "target", reliability.target)
        object.__setattr__(self, "input_score_id", reliability.input_score_id)
        object.__setattr__(self, "input_score_version", reliability.input_score_version)
        object.__setattr__(self, "reliability_fingerprint", reliability.fingerprint)
        object.__setattr__(
            self,
            "reliability_fingerprint_version",
            WINNER_RELIABILITY_RESULT_FINGERPRINT_VERSION,
        )
        object.__setattr__(self, "reliability_id", reliability.reliability_id)
        object.__setattr__(self, "reliability_version", reliability.reliability_version)
        object.__setattr__(self, "binning_id", reliability.binning_id)
        object.__setattr__(self, "binning_version", reliability.binning_version)
        object.__setattr__(self, "bin_count", reliability.bin_count)
        object.__setattr__(
            self,
            "evaluation_dataset_fingerprint",
            reliability.evaluation_dataset_fingerprint,
        )
        object.__setattr__(
            self,
            "evaluation_dataset_fingerprint_version",
            reliability.evaluation_dataset_fingerprint_version,
        )
        object.__setattr__(self, "source_cohort_fingerprint", reliability.source_cohort_fingerprint)
        object.__setattr__(
            self,
            "source_cohort_fingerprint_version",
            reliability.source_cohort_fingerprint_version,
        )
        object.__setattr__(self, "source_count", reliability.source_count)
        object.__setattr__(self, "count", reliability.count)
        object.__setattr__(self, "taxonomy_miss_count", reliability.taxonomy_miss_count)
        object.__setattr__(self, "unresolved_count", reliability.unresolved_count)
        object.__setattr__(
            self, "unadjudicated_resolved_count", reliability.unadjudicated_resolved_count
        )
        object.__setattr__(self, "value", value)

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Return the exact payload the fingerprint hashes.

        Commits the aggregate identity, the input-score identity, the source
        reliability identity with its fingerprint and payload schema version,
        the binning identity with its semantic version and configuration, the
        evaluation dataset identity, the source cohort identity, the explicit
        exclusion accounting, and the aggregate value. Every nested object is
        generated fresh here; no caller-owned mapping is stored or emitted,
        and no NaN or infinity is ever serialized.
        """
        return {
            "v": WINNER_BINNED_ABSOLUTE_GAP_RESULT_FINGERPRINT_VERSION,
            "aggregate_id": self.aggregate_id,
            "aggregate_version": self.aggregate_version,
            "target": self.target,
            "input_score_id": self.input_score_id,
            "input_score_version": self.input_score_version,
            "reliability_fingerprint": self.reliability_fingerprint,
            "reliability_fingerprint_version": self.reliability_fingerprint_version,
            "reliability": {
                "reliability_id": self.reliability_id,
                "reliability_version": self.reliability_version,
            },
            "binning": {
                "binning_id": self.binning_id,
                "binning_version": self.binning_version,
                "bin_count": self.bin_count,
            },
            "evaluation_dataset_fingerprint": self.evaluation_dataset_fingerprint,
            "evaluation_dataset_fingerprint_version": (self.evaluation_dataset_fingerprint_version),
            "source_cohort_fingerprint": self.source_cohort_fingerprint,
            "source_cohort_fingerprint_version": self.source_cohort_fingerprint_version,
            "source_count": self.source_count,
            "count": self.count,
            "exclusions": {
                "taxonomy_miss_count": self.taxonomy_miss_count,
                "unresolved_count": self.unresolved_count,
                "unadjudicated_resolved_count": self.unadjudicated_resolved_count,
            },
            "value": self.value,
        }

    @property
    def fingerprint(self) -> str:
        """Stable identity of the aggregate artifact.

        The hash of :meth:`canonical_payload`, deterministic for the object
        lifetime. Two reliability artifacts that happen to produce the same
        numeric value carry different reliability fingerprints and therefore
        different aggregate fingerprints: numeric equality is not provenance
        identity.
        """
        return fingerprint(self.canonical_payload())


def evaluate_winner_binned_absolute_gap(
    reliability: WinnerReliabilityResult,
) -> WinnerBinnedAbsoluteGapResult:
    """Compute the binned absolute-gap aggregate over a reliability summary.

    The single supported input is an exact :class:`WinnerReliabilityResult`
    built by :func:`evaluate_uncalibrated_winner_reliability`; the source
    reliability summary already fixes the partition, so this evaluator takes
    no ``bin_count`` argument and accepts the reliability partition as truth.
    Anything else (``None``, a mapping, a duck-typed look-alike, a
    :class:`WinnerCorrectnessDiagnosticsResult`, a
    :class:`CalibrationEvaluationDataset`) fails closed with
    :class:`InvalidDecisionError`.

    For each non-empty bin ``b`` (``count > 0``) of the source summary, in
    fixed bin-index order:

    ```text
    absolute_gap_b = abs(mean_selected_probability_b
                         - empirical_correctness_rate_b)
    weighted_term_b = (count_b / count) * absolute_gap_b
    value = math.fsum(weighted_term_b for non-empty bins)
    ```

    This is mathematically the conventional equal-width ECE estimator form,
    frozen here as the sample-weighted absolute discrepancy between raw
    selected-score bin means and empirical winner-correctness rates. Empty
    bins contribute zero mass: they are skipped, their ``None`` statistics
    are never treated as ``0``, and no fake observed gap is constructed; the
    presence of an empty bin is not an error. A non-finite statistic, a
    negative weighted term, or an out-of-range aggregate means the upstream
    reliability invariant is broken and fails closed with
    :class:`InvalidDecisionError`.

    The interpretation stays pre-calibration and directional claims stay
    impossible: the absolute value carries no direction, so no
    overconfidence or underconfidence inference can be drawn, and the number
    is not a calibration-quality claim about the model.

    This is a read-only offline derivation: it does not mutate the source
    reliability summary, the dataset, or any observation, does not set
    ``calibrated`` or ``predicted_correctness`` anywhere, and does not
    create a :class:`~probvenance.calibration.CalibrationProfile`.
    """
    return WinnerBinnedAbsoluteGapResult(
        reliability,
        _construction_token=_BINNED_ABSOLUTE_GAP_RESULT_CONSTRUCTION_TOKEN,
    )
