"""End-to-end runtime tests: BoolDecision -> ... -> BoolResult -> DecisionTrace.

Uses a tiny in-memory fake backend with deterministic logits and honest
metadata. No model downloads, no network, no GPU.
"""

import math
import uuid
from typing import Any

import pytest

from probvenance import (
    BINARY_DOCTRINE_ID,
    BINARY_EVIDENCE_LABELS,
    Backend,
    BackendCapabilities,
    BoolCompiler,
    BoolDecision,
    BoolResult,
    ChoiceDecision,
    DecisionTrace,
    Evaluation,
    EvidenceKind,
    InferencePlan,
    InvalidDecisionError,
    Probvenance,
    RawEvidence,
    ScoringDiagnostics,
    UnsupportedCapabilityError,
    normalized_entropy,
    probability_margin,
)

LOGIT_FALSE = 0.0
LOGIT_TRUE = math.log(3.0)
VOCAB_LOGSUMEXP = math.log(4.0)

HONEST_METADATA: dict[str, Any] = {
    "positive_token_id": 9642,
    "negative_token_id": 3134,
    "model": "fake-model",
    "model_revision": "rev-1",
    "input_token_count": 42,
    "vocab_logsumexp": VOCAB_LOGSUMEXP,
    "top_token_id": 9642,
    "top_token_logit": LOGIT_TRUE,
}


class FakeBackend:
    """Deterministic backend: fixed logits, lineage-honest, honest metadata."""

    def __init__(
        self,
        *,
        binary_token_logits: bool = True,
        values: tuple[float, float] = (LOGIT_FALSE, LOGIT_TRUE),
        plan_fingerprint_override: str | None = None,
        omit_plan_fingerprint: bool = False,
    ) -> None:
        self.capabilities = BackendCapabilities(binary_token_logits=binary_token_logits)
        self._values = values
        self._plan_fingerprint_override = plan_fingerprint_override
        self._omit_plan_fingerprint = omit_plan_fingerprint
        self.executed_plans: list[InferencePlan] = []

    def execute(self, plan: InferencePlan) -> RawEvidence:
        self.executed_plans.append(plan)
        if self._omit_plan_fingerprint:
            plan_fingerprint = None
        elif self._plan_fingerprint_override is not None:
            plan_fingerprint = self._plan_fingerprint_override
        else:
            plan_fingerprint = plan.fingerprint
        return RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=BINARY_EVIDENCE_LABELS,
            values=self._values,
            plan_fingerprint=plan_fingerprint,
            metadata=dict(HONEST_METADATA),
        )


def make_runtime(**backend_kwargs: Any) -> tuple[Probvenance, FakeBackend]:
    backend = FakeBackend(**backend_kwargs)
    return Probvenance(backend=backend), backend


def make_decision() -> BoolDecision:
    return BoolDecision(
        "Was the parcel delivered on the promised day?",
        context="The courier confirmed Thursday in writing.",
    )


class TestEndToEnd:
    def test_evaluate_with_trace_full_pipeline(self) -> None:
        runtime, backend = make_runtime()
        decision = make_decision()
        evaluation = runtime.evaluate_with_trace(decision)
        assert isinstance(evaluation, Evaluation)
        assert isinstance(evaluation.result, BoolResult)
        assert evaluation.result.probability_true == pytest.approx(0.75)
        assert evaluation.result.method == "binary_token_logits"
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None
        p = evaluation.result.probability_true
        assert evaluation.result.certainty.entropy == pytest.approx(
            normalized_entropy([p, 1.0 - p])
        )
        assert evaluation.result.certainty.margin == pytest.approx(probability_margin([p, 1.0 - p]))
        assert backend.executed_plans, "backend was never executed"
        plan = backend.executed_plans[0]
        assert plan.decision_fingerprint == decision.fingerprint

    def test_result_and_trace_share_trace_id(self) -> None:
        runtime, _ = make_runtime()
        evaluation = runtime.evaluate_with_trace(make_decision())
        assert evaluation.result.trace_id == evaluation.trace.trace_id
        parsed = uuid.UUID(evaluation.trace.trace_id)
        assert parsed.version == 4

    def test_trace_lineage_matches_decision_plan_and_evidence(self) -> None:
        runtime, backend = make_runtime()
        decision = make_decision()
        evaluation = runtime.evaluate_with_trace(decision)
        trace: DecisionTrace = evaluation.trace
        assert trace.decision_fingerprint == decision.fingerprint
        assert trace.plan_fingerprint == backend.executed_plans[0].fingerprint
        assert trace.evidence.plan_fingerprint == trace.plan_fingerprint

    def test_trace_fields_from_fake_backend_metadata(self) -> None:
        runtime, backend = make_runtime()
        evaluation = runtime.evaluate_with_trace(make_decision())
        trace = evaluation.trace
        assert trace.positive_token_id == 9642
        assert trace.negative_token_id == 3134
        assert trace.model == "fake-model"
        assert trace.model_revision == "rev-1"
        assert trace.input_token_count == 42
        assert trace.backend_type == "FakeBackend"
        assert trace.scoring_strategy == "binary_token_logits"
        assert trace.doctrine_id == BINARY_DOCTRINE_ID
        assert trace.positive_verbalizer == "yes"
        assert trace.negative_verbalizer == "no"
        assert trace.latency_ms >= 0.0
        assert trace.timestamp
        assert trace.rendered_input is None
        assert backend.executed_plans

    def test_trace_carries_scoring_diagnostics(self) -> None:
        runtime, _ = make_runtime()
        trace = runtime.evaluate_with_trace(make_decision()).trace
        assert isinstance(trace.scoring_diagnostics, ScoringDiagnostics)
        # The fake logits are (0, ln 3) over a uniform 4-token vocabulary, so
        # the two verbalizers hold exactly 3/4 of the full-vocabulary mass.
        # math.logaddexp does not exist in the stdlib; use the stable form.
        log_mass = LOGIT_TRUE + math.log1p(math.exp(LOGIT_FALSE - LOGIT_TRUE))
        assert trace.scoring_diagnostics.verbalizer_mass == pytest.approx(
            math.exp(log_mass - VOCAB_LOGSUMEXP)
        )
        assert trace.scoring_diagnostics.top_token_id == 9642
        assert trace.scoring_diagnostics.top_token_probability == pytest.approx(
            math.exp(LOGIT_TRUE - VOCAB_LOGSUMEXP)
        )
        assert trace.scoring_diagnostics.positive_token_probability == pytest.approx(
            math.exp(LOGIT_TRUE - VOCAB_LOGSUMEXP)
        )
        assert trace.scoring_diagnostics.negative_token_probability == pytest.approx(
            math.exp(LOGIT_FALSE - VOCAB_LOGSUMEXP)
        )

    def test_execution_fingerprint_is_64_lowercase_hex(self) -> None:
        runtime, _ = make_runtime()
        trace = runtime.evaluate_with_trace(make_decision()).trace
        fp = trace.execution_fingerprint
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert fp == fp.lower()
        int(fp, 16)

    def test_evaluate_matches_evaluate_with_trace(self) -> None:
        runtime, _ = make_runtime()
        decision = make_decision()
        result = runtime.evaluate(decision)
        evaluation = runtime.evaluate_with_trace(decision)
        assert result.probability_true == pytest.approx(evaluation.result.probability_true)
        assert result.certainty == evaluation.result.certainty
        assert result.method == evaluation.result.method
        assert result.calibrated == evaluation.result.calibrated
        assert result.predicted_correctness == evaluation.result.predicted_correctness
        assert result.trace_id != evaluation.result.trace_id

    def test_capture_rendered_input_flag_gates_capture(self) -> None:
        backend = FakeBackend()
        runtime = Probvenance(backend=backend, capture_rendered_input=True)
        evaluation = runtime.evaluate_with_trace(make_decision())
        assert evaluation.trace.rendered_input is None

        metadata_backend = FakeBackendWithRenderedInput()
        capturing = Probvenance(backend=metadata_backend, capture_rendered_input=True)
        captured = capturing.evaluate_with_trace(make_decision())
        assert captured.trace.rendered_input == "RENDERED PROMPT"


class FakeBackendWithRenderedInput(FakeBackend):
    def execute(self, plan: InferencePlan) -> RawEvidence:
        evidence = super().execute(plan)
        with_rendered = dict(evidence.metadata)
        with_rendered["rendered_input"] = "RENDERED PROMPT"
        return RawEvidence(
            kind=evidence.kind,
            labels=evidence.labels,
            values=evidence.values,
            plan_fingerprint=evidence.plan_fingerprint,
            metadata=with_rendered,
        )


class TestRuntimeRejections:
    def test_missing_capability_rejected_before_execution(self) -> None:
        runtime, backend = make_runtime(binary_token_logits=False)
        with pytest.raises(UnsupportedCapabilityError, match="binary_token_logits"):
            runtime.evaluate_with_trace(make_decision())
        assert backend.executed_plans == []

    def test_choice_decision_without_categorical_capability_rejected_before_execution(
        self,
    ) -> None:
        runtime, backend = make_runtime()
        decision = ChoiceDecision(
            "Which carrier delivered?",
            choices={"dhl": "DHL", "ups": "UPS"},
        )
        # Phase 2B dispatches ChoiceDecision to the choice path; a backend
        # without the categorical capability fails at compile time.
        with pytest.raises(UnsupportedCapabilityError, match="categorical_token_logits"):
            runtime.evaluate_with_trace(decision)
        assert backend.executed_plans == []

    def test_foreign_plan_fingerprint_rejected(self) -> None:
        other_plan_plan = BoolCompiler().compile(
            BoolDecision("A different decision, compiled earlier."),
            BackendCapabilities(binary_token_logits=True),
        )
        runtime, backend = make_runtime(plan_fingerprint_override=other_plan_plan.fingerprint)
        with pytest.raises(InvalidDecisionError, match="plan_fingerprint"):
            runtime.evaluate_with_trace(make_decision())
        assert len(backend.executed_plans) == 1

    def test_evidence_without_lineage_is_rejected(self) -> None:
        runtime, backend = make_runtime(omit_plan_fingerprint=True)
        with pytest.raises(InvalidDecisionError, match="plan_fingerprint") as exc_info:
            runtime.evaluate_with_trace(make_decision())
        expected = backend.executed_plans[0].fingerprint
        assert expected in str(exc_info.value)
        assert "received none" in str(exc_info.value)


class TestFacadeProperties:
    def test_default_compiler_and_flag(self) -> None:
        runtime, backend = make_runtime()
        assert runtime.backend is backend
        assert isinstance(runtime.compiler, BoolCompiler)
        assert runtime.capture_rendered_input is False

    def test_custom_compiler_used(self) -> None:
        compiler = BoolCompiler(positive_verbalizer="affirm", negative_verbalizer="deny")
        backend = FakeBackend()
        runtime = Probvenance(backend=backend, compiler=compiler)
        assert runtime.compiler is compiler
        evaluation = runtime.evaluate_with_trace(make_decision())
        assert evaluation.trace.positive_verbalizer == "affirm"
        assert evaluation.trace.negative_verbalizer == "deny"

    def test_backend_satisfies_protocol(self) -> None:
        backend: Backend = FakeBackend()
        assert backend.capabilities.binary_token_logits is True
