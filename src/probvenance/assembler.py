"""ProbabilityAssembler: converts raw evidence into decision results.

The assembler is the ONLY place where raw logits become a probability. It
performs no calibration and never claims calibratedness: every result it
produces is explicitly ``calibrated=False`` with ``predicted_correctness=None``.

Two concrete assemblers exist:

- :func:`assemble_bool_probability` for the binary token-logit strategy.
- :func:`assemble_choice_probability` for the categorical token-logit
  strategy, which maps label-space probabilities back to semantic candidate
  names through the plan's ``candidate_mapping``.

:func:`assemble_probability` selects one of them from a plan. Selection is by
the plan's FULL ``(strategy, assembler_id, assembler_version)`` declaration,
never by strategy alone, so a plan cannot execute one implementation while
declaring another. The set of known implementations is closed: there is no
registry, no plugin loading, and no fallback.
"""

import math
from collections.abc import Callable, Mapping

from probvenance.diagnostics import (
    BINARY_EVIDENCE_LABELS,
    ChoiceScoringDiagnostics,
    ScoringDiagnostics,
    diagnose_bool_evidence,
    diagnose_choice_evidence,
)
from probvenance.errors import InvalidDecisionError, UnsupportedAssemblerError
from probvenance.plans import EvidenceKind, InferencePlan, RawEvidence, ScoringStrategy
from probvenance.results import BoolResult, Certainty, ChoiceResult

BINARY_ASSEMBLER_ID = "binary-restricted-softmax"
CATEGORICAL_ASSEMBLER_ID = "categorical-restricted-softmax"
BINARY_ASSEMBLER_VERSION = 1
CATEGORICAL_ASSEMBLER_VERSION = 1


def assemble_bool_probability(
    evidence: RawEvidence,
    *,
    trace_id: str | None = None,
    method: str = ScoringStrategy.BINARY_TOKEN_LOGITS.value,
) -> BoolResult:
    """Convert binary token-logit evidence into an uncalibrated
    :class:`~probvenance.results.BoolResult`.

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
    :class:`~probvenance.results.ChoiceResult` plus its scoring diagnostics.

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


AssembledProbability = tuple[
    BoolResult | ChoiceResult, ScoringDiagnostics | ChoiceScoringDiagnostics
]
ProbabilityAssembler = Callable[[InferencePlan, RawEvidence, str | None], AssembledProbability]


def _assemble_binary_for_plan(
    _plan: InferencePlan, evidence: RawEvidence, trace_id: str | None
) -> AssembledProbability:
    result = assemble_bool_probability(evidence, trace_id=trace_id)
    return result, diagnose_bool_evidence(evidence)


def _assemble_categorical_for_plan(
    plan: InferencePlan, evidence: RawEvidence, trace_id: str | None
) -> AssembledProbability:
    return assemble_choice_probability(evidence, plan=plan, trace_id=trace_id)


_ASSEMBLER_IMPLEMENTATIONS: Mapping[tuple[ScoringStrategy, str, int], ProbabilityAssembler] = {
    (
        ScoringStrategy.BINARY_TOKEN_LOGITS,
        BINARY_ASSEMBLER_ID,
        BINARY_ASSEMBLER_VERSION,
    ): _assemble_binary_for_plan,
    (
        ScoringStrategy.CATEGORICAL_TOKEN_LOGITS,
        CATEGORICAL_ASSEMBLER_ID,
        CATEGORICAL_ASSEMBLER_VERSION,
    ): _assemble_categorical_for_plan,
}


def _supported_assembler_declarations() -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (strategy.value, assembler_id, version)
        for strategy, assembler_id, version in _ASSEMBLER_IMPLEMENTATIONS
    )


def resolve_probability_assembler(
    *,
    strategy: ScoringStrategy,
    assembler_id: str | None,
    assembler_version: int | None,
) -> ProbabilityAssembler:
    """Return the one implementation an exact assembler declaration identifies.

    Resolution is by the full ``(strategy, assembler_id, assembler_version)``
    tuple, never by strategy alone: a known assembler paired with the wrong
    strategy, or an unsupported version, does not match.

    An undeclared assembler identity is an explicit unknown (INV-26). It is
    rejected as-is rather than coerced into a sentinel such as an empty id or a
    zero version, because a sentinel would be indistinguishable from a real
    declaration that merely failed to match.

    Raises:
        UnsupportedAssemblerError: if the declaration is an unknown identity, or
            if no implementation matches the tuple.
    """
    if assembler_id is None or assembler_version is None:
        raise UnsupportedAssemblerError(
            "the plan declares no probability assembler identity: "
            f"strategy={strategy.value!r}, assembler_id={assembler_id!r}, "
            f"assembler_version={assembler_version!r}; supported declarations are "
            f"{_supported_assembler_declarations()}"
        )
    implementation = _ASSEMBLER_IMPLEMENTATIONS.get((strategy, assembler_id, assembler_version))
    if implementation is None:
        raise UnsupportedAssemblerError(
            "no probability assembler matches the plan declaration: "
            f"strategy={strategy.value!r}, assembler_id={assembler_id!r}, "
            f"assembler_version={assembler_version!r}; supported declarations are "
            f"{_supported_assembler_declarations()}"
        )
    return implementation


def assemble_probability(
    plan: InferencePlan,
    evidence: RawEvidence,
    *,
    trace_id: str | None = None,
) -> AssembledProbability:
    """Assemble the result and diagnostics the plan's declaration selects.

    The implementation that runs is the one ``plan`` names, so the assembled
    probability and the recorded provenance cannot disagree.

    Raises:
        UnsupportedAssemblerError: if the plan's declaration matches nothing.
    """
    implementation = resolve_probability_assembler(
        strategy=plan.strategy,
        assembler_id=plan.assembler_id,
        assembler_version=plan.assembler_version,
    )
    return implementation(plan, evidence, trace_id)
