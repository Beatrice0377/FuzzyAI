"""Assembler dispatch tests: a plan executes only the implementation it declares.

The runtime must run the exact assembler named by
``(strategy, assembler_id, assembler_version)``. An unknown id, a known id
paired with the wrong strategy, and an unsupported version must all be rejected
before any probability result or trace exists. No model, no network, no GPU.
"""

import math
from dataclasses import replace
from typing import Any

import pytest

from fuzzyai import (
    BINARY_EVIDENCE_LABELS,
    BackendCapabilities,
    BoolCompiler,
    BoolDecision,
    BoolResult,
    ChoiceCompiler,
    ChoiceDecision,
    ChoiceResult,
    ChoiceScoringDiagnostics,
    EvidenceKind,
    FuzzyAI,
    InferencePlan,
    RawEvidence,
    ScoringDiagnostics,
    ScoringStrategy,
)
from fuzzyai.assembler import (
    BINARY_ASSEMBLER_ID,
    BINARY_ASSEMBLER_VERSION,
    CATEGORICAL_ASSEMBLER_ID,
    CATEGORICAL_ASSEMBLER_VERSION,
    assemble_probability,
    resolve_probability_assembler,
)
from fuzzyai.errors import UnsupportedAssemblerError

BINARY_METADATA: dict[str, Any] = {
    "positive_token_id": 9642,
    "negative_token_id": 3134,
    "vocab_logsumexp": math.log(4.0),
    "top_token_id": 9642,
    "top_token_logit": math.log(3.0),
}


def binary_plan() -> InferencePlan:
    decision = BoolDecision(
        "Was the parcel delivered on time?", context="The courier confirmed it."
    )
    return BoolCompiler().compile(decision, BackendCapabilities(binary_token_logits=True))


def categorical_plan() -> InferencePlan:
    decision = ChoiceDecision(
        "Which team owns this ticket?",
        context="The customer asks about a refund.",
        choices={
            "billing": "Payment and billing issues",
            "shipping": "Delivery and logistics",
            "returns": "Returns and refunds",
        },
    )
    return ChoiceCompiler().compile(decision, BackendCapabilities(categorical_token_logits=True))


def binary_evidence(plan: InferencePlan) -> RawEvidence:
    return RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=BINARY_EVIDENCE_LABELS,
        values=(0.0, math.log(3.0)),
        plan_fingerprint=plan.fingerprint,
        metadata=dict(BINARY_METADATA),
    )


def categorical_evidence(plan: InferencePlan) -> RawEvidence:
    return RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=plan.targets,
        values=(0.0, math.log(3.0), 0.0),
        plan_fingerprint=plan.fingerprint,
        metadata={
            "vocab_logsumexp": math.log(6.0),
            "top_token_id": 101,
            "top_token_logit": math.log(3.0),
            "resolved_target_token_ids": [
                [label, 100 + index] for index, label in enumerate(plan.targets)
            ],
        },
    )


def bool_decision() -> BoolDecision:
    return BoolDecision("Was the parcel delivered on time?", context="The courier confirmed it.")


class TestResolveProbabilityAssembler:
    def test_valid_binary_tuple_resolves(self) -> None:
        plan = binary_plan()
        implementation = resolve_probability_assembler(
            strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
            assembler_id=BINARY_ASSEMBLER_ID,
            assembler_version=BINARY_ASSEMBLER_VERSION,
        )
        result, diagnostics = implementation(plan, binary_evidence(plan), "trace-1")
        assert isinstance(result, BoolResult)
        assert isinstance(diagnostics, ScoringDiagnostics)

    def test_valid_categorical_tuple_resolves(self) -> None:
        plan = categorical_plan()
        implementation = resolve_probability_assembler(
            strategy=ScoringStrategy.CATEGORICAL_TOKEN_LOGITS,
            assembler_id=CATEGORICAL_ASSEMBLER_ID,
            assembler_version=CATEGORICAL_ASSEMBLER_VERSION,
        )
        result, diagnostics = implementation(plan, categorical_evidence(plan), "trace-1")
        assert isinstance(result, ChoiceResult)
        assert isinstance(diagnostics, ChoiceScoringDiagnostics)

    def test_unknown_assembler_id_rejected(self) -> None:
        with pytest.raises(UnsupportedAssemblerError, match="my-magical-assembler"):
            resolve_probability_assembler(
                strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
                assembler_id="my-magical-assembler",
                assembler_version=999,
            )

    def test_known_assembler_with_wrong_strategy_rejected(self) -> None:
        with pytest.raises(UnsupportedAssemblerError, match=CATEGORICAL_ASSEMBLER_ID):
            resolve_probability_assembler(
                strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
                assembler_id=CATEGORICAL_ASSEMBLER_ID,
                assembler_version=CATEGORICAL_ASSEMBLER_VERSION,
            )

    def test_unsupported_assembler_version_rejected(self) -> None:
        with pytest.raises(UnsupportedAssemblerError, match=str(BINARY_ASSEMBLER_VERSION + 1)):
            resolve_probability_assembler(
                strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
                assembler_id=BINARY_ASSEMBLER_ID,
                assembler_version=BINARY_ASSEMBLER_VERSION + 1,
            )

    def test_rejection_message_names_the_whole_declaration(self) -> None:
        with pytest.raises(UnsupportedAssemblerError) as excinfo:
            resolve_probability_assembler(
                strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
                assembler_id="my-magical-assembler",
                assembler_version=999,
            )
        message = str(excinfo.value)
        assert "binary_token_logits" in message
        assert "my-magical-assembler" in message
        assert "999" in message


class TestAssembleProbability:
    def test_binary_plan_assembles_a_bool_result(self) -> None:
        plan = binary_plan()
        result, diagnostics = assemble_probability(plan, binary_evidence(plan), trace_id="trace-1")
        assert isinstance(result, BoolResult)
        assert isinstance(diagnostics, ScoringDiagnostics)

    def test_categorical_plan_assembles_a_choice_result(self) -> None:
        plan = categorical_plan()
        result, diagnostics = assemble_probability(
            plan, categorical_evidence(plan), trace_id="trace-1"
        )
        assert isinstance(result, ChoiceResult)
        assert isinstance(diagnostics, ChoiceScoringDiagnostics)
        assert set(result.probabilities) == {"billing", "shipping", "returns"}

    def test_lying_custom_plan_is_rejected_before_assembly(self) -> None:
        plan = replace(binary_plan(), assembler_id="my-magical-assembler", assembler_version=999)
        with pytest.raises(UnsupportedAssemblerError, match="my-magical-assembler"):
            assemble_probability(plan, binary_evidence(plan), trace_id="trace-1")


class LyingBoolCompiler(BoolCompiler):
    """A compiler emitting a plan that declares a non-existent assembler."""

    def compile(self, decision: Any, capabilities: Any) -> InferencePlan:
        plan = super().compile(decision, capabilities)
        return replace(plan, assembler_id="my-magical-assembler", assembler_version=999)


class RecordingBackend:
    def __init__(self) -> None:
        self.capabilities = BackendCapabilities(binary_token_logits=True)
        self.executed_plans: list[InferencePlan] = []

    def execute(self, plan: InferencePlan) -> RawEvidence:
        self.executed_plans.append(plan)
        return RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=BINARY_EVIDENCE_LABELS,
            values=(0.0, math.log(3.0)),
            plan_fingerprint=plan.fingerprint,
            metadata=dict(BINARY_METADATA),
        )


class TestRuntimeAssemblerProvenance:
    def test_runtime_rejects_a_lying_plan_without_result_or_trace(self) -> None:
        ai = FuzzyAI(backend=RecordingBackend(), compiler=LyingBoolCompiler())
        with pytest.raises(UnsupportedAssemblerError, match="my-magical-assembler"):
            ai.evaluate_with_trace(bool_decision())

    def test_runtime_trace_records_the_verified_declaration(self) -> None:
        ai = FuzzyAI(backend=RecordingBackend())
        evaluation = ai.evaluate_with_trace(bool_decision())
        assert isinstance(evaluation.result, BoolResult)
        assert evaluation.trace.assembler_id == BINARY_ASSEMBLER_ID
        assert evaluation.trace.assembler_version == BINARY_ASSEMBLER_VERSION
