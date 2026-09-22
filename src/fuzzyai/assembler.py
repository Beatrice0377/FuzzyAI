"""ProbabilityAssembler: converts raw evidence into decision results.

The assembler is the ONLY place where raw logits become a probability. It
performs no calibration and never claims calibratedness: every result it
produces is explicitly ``calibrated=False`` with ``predicted_correctness=None``.
"""

import math

from fuzzyai.errors import InvalidDecisionError
from fuzzyai.plans import EvidenceKind, RawEvidence, ScoringStrategy
from fuzzyai.results import BoolResult, Certainty

BINARY_EVIDENCE_LABELS = ("false", "true")


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
