"""End-to-end runtime tests: ChoiceDecision -> ChoiceResult -> DecisionTrace.

Uses a tiny in-memory fake backend with deterministic logits, honest
metadata, and resolved scoring token ids. No model downloads, no GPU.
"""

import math
from typing import Any

import pytest

from fuzzyai import (
    Backend,
    BackendCapabilities,
    BoolDecision,
    Choice,
    ChoiceCompiler,
    ChoiceDecision,
    ChoiceResult,
    ChoiceScoringDiagnostics,
    Evaluation,
    EvidenceKind,
    FuzzyAI,
    InferencePlan,
    InvalidDecisionError,
    RawEvidence,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
)

CAPS = BackendCapabilities(categorical_token_logits=True)

LOGIT_BILLING = 0.0
LOGIT_SHIPPING = math.log(3.0)
LOGIT_RETURNS = 0.0
VOCAB_LOGSUMEXP = math.log(6.0)


def honest_metadata(plan: InferencePlan) -> dict[str, Any]:
    # A fake, deterministic "tokenizer": label i resolves to token id 100 + i.
    resolved = [[label, 100 + index] for index, label in enumerate(plan.targets)]
    return {
        "vocab_logsumexp": VOCAB_LOGSUMEXP,
        "top_token_id": 101,
        "top_token_logit": LOGIT_SHIPPING,
        "resolved_target_token_ids": resolved,
        "model": "fake-model",
        "model_revision": "rev-1",
        "input_token_count": 42,
    }


class FakeCategoricalBackend:
    """Deterministic backend handling both strategies with honest metadata."""

    def __init__(self, *, categorical_token_logits: bool = True) -> None:
        self.capabilities = BackendCapabilities(
            binary_token_logits=True,
            categorical_token_logits=categorical_token_logits,
        )
        self.executed_plans: list[InferencePlan] = []

    def execute(self, plan: InferencePlan) -> RawEvidence:
        self.executed_plans.append(plan)
        if plan.strategy.value == "binary_token_logits":
            return RawEvidence(
                kind=EvidenceKind.LOGITS,
                labels=("false", "true"),
                values=(0.0, math.log(3.0)),
                plan_fingerprint=plan.fingerprint,
                metadata={
                    "vocab_logsumexp": math.log(4.0),
                    "top_token_id": 9642,
                    "top_token_logit": math.log(3.0),
                    "positive_token_id": 9642,
                    "negative_token_id": 3134,
                },
            )
        values = tuple(
            LOGIT_BILLING if label == "A" else LOGIT_SHIPPING if label == "B" else LOGIT_RETURNS
            for label in plan.targets
        )
        return RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=plan.targets,
            values=values,
            plan_fingerprint=plan.fingerprint,
            metadata=honest_metadata(plan),
        )


class CountingBackend(FakeCategoricalBackend):
    """Counts forward passes: the runtime must trigger exactly one."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_passes = 0

    def execute(self, plan: InferencePlan) -> RawEvidence:
        self.forward_passes += 1
        return super().execute(plan)


def make_runtime(
    backend: FakeCategoricalBackend | None = None,
) -> tuple[FuzzyAI, FakeCategoricalBackend]:
    active_backend = backend if backend is not None else FakeCategoricalBackend()
    return FuzzyAI(backend=active_backend), active_backend


def make_decision() -> ChoiceDecision:
    return ChoiceDecision(
        "Which department should handle this request?",
        context="The customer asks about a refund for a damaged parcel.",
        choices={
            "billing": "Payment, charges, invoices",
            "shipping": "Delivery, couriers, parcels",
            "returns": "Refunds, exchanges, warranty",
        },
    )


class TestChoiceEndToEnd:
    def test_evaluate_with_trace_full_pipeline(self) -> None:
        runtime, backend = make_runtime()
        decision = make_decision()
        evaluation = runtime.evaluate_with_trace(decision)
        assert isinstance(evaluation, Evaluation)
        assert isinstance(evaluation.result, ChoiceResult)
        # Hand-computed: exps are (1, 3, 1), so (0.2, 0.6, 0.2).
        assert evaluation.result.probabilities["billing"] == pytest.approx(0.2)
        assert evaluation.result.probabilities["shipping"] == pytest.approx(0.6)
        assert evaluation.result.probabilities["returns"] == pytest.approx(0.2)
        assert evaluation.result.value == "shipping"
        assert evaluation.result.method == "categorical_token_logits"
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None
        assert evaluation.result.trace_id == evaluation.trace.trace_id
        assert backend.executed_plans, "backend was never executed"
        plan = backend.executed_plans[0]
        assert plan.decision_fingerprint == decision.fingerprint
        assert plan.targets == ("A", "B", "C")

    def test_trace_records_categorical_provenance(self) -> None:
        runtime, _ = make_runtime()
        trace = runtime.evaluate_with_trace(make_decision()).trace
        assert trace.scoring_strategy == "categorical_token_logits"
        assert trace.probability_true is None
        assert trace.positive_token_id == -1
        assert trace.negative_token_id == -1
        assert trace.resolved_target_token_ids == (("A", 100), ("B", 101), ("C", 102))
        assert trace.model == "fake-model"
        assert trace.input_token_count == 42
        assert isinstance(trace.scoring_diagnostics, ChoiceScoringDiagnostics)
        assert trace.scoring_diagnostics.scoring_label_mass == pytest.approx(5.0 / 6.0)

    def test_exactly_one_forward_pass(self) -> None:
        backend = CountingBackend()
        runtime = FuzzyAI(backend=backend)
        runtime.evaluate_with_trace(make_decision())
        assert backend.forward_passes == 1

    def test_evaluate_matches_evaluate_with_trace(self) -> None:
        runtime, _ = make_runtime()
        decision = make_decision()
        result = runtime.evaluate(decision)
        evaluation = runtime.evaluate_with_trace(decision)
        assert isinstance(result, ChoiceResult)
        traced_result = evaluation.result
        assert isinstance(traced_result, ChoiceResult)
        assert result.probabilities == traced_result.probabilities
        assert result.value == traced_result.value
        assert result.certainty == traced_result.certainty
        assert result.method == traced_result.method
        assert result.trace_id != traced_result.trace_id

    def test_evidence_labels_are_execution_labels_not_semantics(self) -> None:
        runtime, _ = make_runtime()
        evaluation = runtime.evaluate_with_trace(make_decision())
        assert evaluation.trace.evidence.labels == ("A", "B", "C")
        traced_result = evaluation.result
        assert isinstance(traced_result, ChoiceResult)
        assert traced_result.probabilities["billing"] is not None
        assert "billing" not in evaluation.trace.evidence.labels

    def test_bool_path_still_works_on_same_facade(self) -> None:
        runtime, _ = make_runtime()
        result = runtime.evaluate(BoolDecision("Was it delivered?"))
        assert result.method == "binary_token_logits"


class TestChoiceDispatchRejections:
    def test_non_decision_object_rejected(self) -> None:
        runtime, backend = make_runtime()
        not_a_decision: Any = "compile me if you can"
        with pytest.raises(UnsupportedDecisionError, match="BoolDecision and ChoiceDecision"):
            runtime.evaluate_with_trace(not_a_decision)
        assert backend.executed_plans == []

    def test_missing_capability_rejected_before_execution(self) -> None:
        runtime, backend = make_runtime(
            backend=FakeCategoricalBackend(categorical_token_logits=False)
        )
        with pytest.raises(UnsupportedCapabilityError, match="categorical_token_logits"):
            runtime.evaluate_with_trace(make_decision())
        assert backend.executed_plans == []

    def test_choice_compiler_injection_used(self) -> None:
        backend = FakeCategoricalBackend()
        compiler = ChoiceCompiler()
        runtime = FuzzyAI(backend=backend, choice_compiler=compiler)
        assert runtime.choice_compiler is compiler
        runtime.evaluate_with_trace(make_decision())
        assert backend.executed_plans


class TestChoiceLineage:
    def test_foreign_plan_fingerprint_rejected(self) -> None:
        other_plan = ChoiceCompiler().compile(
            ChoiceDecision("A different question.", choices=[Choice(name="x"), Choice(name="y")]),
            CAPS,
        )
        backend = FakeCategoricalBackend()
        original_execute = backend.execute

        def execute_with_foreign_lineage(plan: InferencePlan) -> RawEvidence:
            evidence = original_execute(plan)
            return RawEvidence(
                kind=evidence.kind,
                labels=evidence.labels,
                values=evidence.values,
                plan_fingerprint=other_plan.fingerprint,
                metadata=dict(evidence.metadata),
            )

        backend.execute = execute_with_foreign_lineage
        runtime = FuzzyAI(backend=backend)
        with pytest.raises(InvalidDecisionError, match="plan_fingerprint"):
            runtime.evaluate_with_trace(make_decision())

    def test_backend_satisfies_protocol(self) -> None:
        backend: Backend = FakeCategoricalBackend()
        assert backend.capabilities.categorical_token_logits is True
