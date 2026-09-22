"""ProbabilityAssembler: converts raw evidence into decision results.

The assembler is the ONLY place where raw logits become a probability. It
performs no calibration and never claims calibratedness: every result it
produces is explicitly ``calibrated=False`` with ``predicted_correctness=None``.

Two entry points exist:

- :func:`assemble_bool_probability` for the binary token-logit strategy.
- :func:`assemble_choice_probability` for the categorical token-logit
  strategy, which maps label-space probabilities back to semantic candidate
  names through the plan's ``candidate_mapping``.
"""

import math

from fuzzyai.diagnostics import (
    BINARY_EVIDENCE_LABELS,
    ChoiceScoringDiagnostics,
    diagnose_choice_evidence,
)
from fuzzyai.errors import InvalidDecisionError
from fuzzyai.plans import EvidenceKind, InferencePlan, RawEvidence, ScoringStrategy
from fuzzyai.results import BoolResult, Certainty, ChoiceResult


def assemble_bool_probability(
    evidence: RawEvidence,
    *,
    trace_id: str | None = None,
    method: str = ScoringStrategy.BINARY_TOKEN_LOGITS.value,
) -> BoolResult:
    """Convert binary token-logit evidence into an uncalibrated
    :class:`~fuzzyai.results.BoolResult`.

    The probability of ``True`` is the stable softmax over the two logits,
    which equals ``sigmoid(l_true - l_false)``. Extreme logits (e.g.
    ``+/-1000``) are handled via max-shift subtraction, never via
    ``exp`` of a raw value, so no overflow can occur.

    Raises:
        InvalidDecisionError: if the evidence kind is not
            :attr:`EvidenceKind.LOGITS`, or its labels are not exactly
            ``("false", "true")`` in either order.
    """
    if evidence.kind is not EvidenceKind.LOGITS:
        raise InvalidDecisionError(
            f"binary assembly requires {EvidenceKind.LOGITS.value!r} evidence, "
            f"got {evidence.kind.value!r}"
        )
    if len(evidence.labels) != 2 or set(evidence.labels) != set(BINARY_EVIDENCE_LABELS):
        raise InvalidDecisionError(
            f"binary evidence labels must be exactly {BINARY_EVIDENCE_LABELS} in either "
            f"order, got {evidence.labels!r}"
        )
    l_true = evidence.values[evidence.labels.index("true")]
    l_false = evidence.values[evidence.labels.index("false")]
    # Max-shift stable softmax over exactly two logits: subtracting the max
    # keeps every exponent in (-inf, 0], so exp() never overflows.
    shift = max(l_true, l_false)
    exp_true = math.exp(l_true - shift)
    exp_false = math.exp(l_false - shift)
    probability_true = exp_true / (exp_true + exp_false)
    return BoolResult(
        probability_true=probability_true,
        certainty=Certainty.from_probabilities([probability_true, 1.0 - probability_true]),
        method=method,
        trace_id=trace_id,
        predicted_correctness=None,
        calibrated=False,
    )


def assemble_choice_probability(
    evidence: RawEvidence,
    *,
    plan: InferencePlan,
    trace_id: str | None = None,
    method: str = ScoringStrategy.CATEGORICAL_TOKEN_LOGITS.value,
) -> tuple[ChoiceResult, ChoiceScoringDiagnostics]:
    """Convert categorical token-logit evidence into an uncalibrated
    :class:`~fuzzyai.results.ChoiceResult` plus its scoring diagnostics.

    The label-space probabilities are the stable N-way softmax over the
    candidate logits: ``p_i = exp(l_i - m) / sum_j exp(l_j - m)`` with
    ``m = max(logits)``, so extreme logits never overflow. The probabilities
    are then mapped to semantic candidate names through
    ``plan.candidate_mapping``: entry ``i`` maps ``targets[i]`` to
    ``candidate_mapping[i].candidate_name``. The evidence labels must equal
    ``plan.targets`` EXACTLY, in the same order; no reordering ever happens.

    Raises:
        InvalidDecisionError: if the plan strategy is not
            :attr:`ScoringStrategy.CATEGORICAL_TOKEN_LOGITS`, the evidence
            kind is not :attr:`EvidenceKind.LOGITS`, the evidence labels do
            not equal ``plan.targets`` in order, or the mapping length does
            not match the evidence length.
    """
    if plan.strategy is not ScoringStrategy.CATEGORICAL_TOKEN_LOGITS:
        raise InvalidDecisionError(
            "choice assembly requires the "
            f"{ScoringStrategy.CATEGORICAL_TOKEN_LOGITS.value} strategy, got "
            f"{plan.strategy.value!r}"
        )
    if evidence.kind is not EvidenceKind.LOGITS:
        raise InvalidDecisionError(
            f"choice assembly requires {EvidenceKind.LOGITS.value!r} evidence, "
            f"got {evidence.kind.value!r}"
        )
    if evidence.labels != plan.targets:
        raise InvalidDecisionError(
            "choice evidence labels must equal plan.targets exactly, in the same "
            f"order: expected {plan.targets!r}, got {evidence.labels!r}"
        )
    if len(plan.candidate_mapping) != len(evidence.values):
        raise InvalidDecisionError(
            "candidate_mapping must have exactly one entry per evidence value, got "
            f"{len(plan.candidate_mapping)} entries for {len(evidence.values)} values"
        )
    logits = evidence.values
    # Max-shift stable N-way softmax: subtracting the max keeps every exponent
    # in (-inf, 0], so exp() never overflows.
    shift = max(logits)
    exps = [math.exp(logit - shift) for logit in logits]
    total = sum(exps)
    label_probabilities = [value / total for value in exps]
    semantic_probabilities = {
        entry.candidate_name: label_probabilities[index]
        for index, entry in enumerate(plan.candidate_mapping)
    }
    diagnostics = diagnose_choice_evidence(evidence, plan=plan)
    result = ChoiceResult(
        probabilities=semantic_probabilities,
        certainty=Certainty.from_probabilities(list(semantic_probabilities.values())),
        method=method,
        trace_id=trace_id,
        predicted_correctness=None,
        calibrated=False,
    )
    return result, diagnostics
