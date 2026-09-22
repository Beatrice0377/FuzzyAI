"""Tests for categorical decision traces: resolved token ids and provenance."""

import json
import math
from typing import Any

import pytest

from fuzzyai import (
    BackendCapabilities,
    Certainty,
    ChoiceCompiler,
    ChoiceDecision,
    ChoiceResult,
    ChoiceScoringDiagnostics,
    EvidenceKind,
    InferencePlan,
    InvalidDecisionError,
    RawEvidence,
    ScoringDiagnostics,
    ScoringStrategy,
    assemble_choice_probability,
    build_decision_trace,
    diagnose_choice_evidence,
)

CAPS = BackendCapabilities(categorical_token_logits=True)
TIMESTAMP = "2025-01-01T00:00:00+00:00"

LOGIT_BILLING = 0.0
LOGIT_SHIPPING = math.log(3.0)
LOGIT_RETURNS = 0.0
VOCAB_LOGSUMEXP = math.log(6.0)

RESOLVED_IDS: list[list[Any]] = [["A", 11], ["B", 22], ["C", 33]]


def make_plan() -> InferencePlan:
    decision = ChoiceDecision(
        "Which department?",
        choices={"billing": "money", "shipping": "boxes", "returns": "warranty"},
    )
    return ChoiceCompiler().compile(decision, CAPS)


def make_evidence(
    metadata: dict[str, Any] | None = None, plan: InferencePlan | None = None
) -> RawEvidence:
    active_plan = plan if plan is not None else make_plan()
    return RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=active_plan.targets,
        values=(LOGIT_BILLING, LOGIT_SHIPPING, LOGIT_RETURNS),
        plan_fingerprint=active_plan.fingerprint,
        metadata=metadata if metadata is not None else honest_metadata(),
    )


def honest_metadata() -> dict[str, Any]:
    return {
        "vocab_logsumexp": VOCAB_LOGSUMEXP,
        "top_token_id": 22,
        "top_token_logit": LOGIT_SHIPPING,
        "resolved_target_token_ids": RESOLVED_IDS,
    }


def make_result_and_diagnostics(
    evidence: RawEvidence, plan: InferencePlan
) -> tuple[ChoiceResult, ChoiceScoringDiagnostics]:
    result, diagnostics = assemble_choice_probability(evidence, plan=plan)
    assert isinstance(diagnostics, ChoiceScoringDiagnostics)
    return result, diagnostics


def build_trace(
    *,
    plan: InferencePlan | None = None,
    evidence: RawEvidence | None = None,
    result: Any = None,
    diagnostics: Any = None,
) -> Any:
    active_plan = plan if plan is not None else make_plan()
    active_evidence = evidence if evidence is not None else make_evidence(plan=active_plan)
    if result is None or diagnostics is None:
        built_result, built_diagnostics = make_result_and_diagnostics(active_evidence, active_plan)
        result = result if result is not None else built_result
        diagnostics = diagnostics if diagnostics is not None else built_diagnostics
    return build_decision_trace(
        trace_id="trace-1",
        timestamp=TIMESTAMP,
        decision_fingerprint=active_plan.decision_fingerprint,
        plan=active_plan,
        evidence=active_evidence,
        result=result,
        diagnostics=diagnostics,
        backend_type="FakeBackend",
        latency_ms=1.5,
    )


class TestCategoricalTraceFields:
    def test_strategy_doctrine_and_verbalizers(self) -> None:
        trace = build_trace()
        assert trace.scoring_strategy == ScoringStrategy.CATEGORICAL_TOKEN_LOGITS.value
        assert trace.doctrine_id == "categorical-semantic-judgment-v1"
        assert trace.positive_verbalizer == "none"
        assert trace.negative_verbalizer == "none"

    def test_binary_token_id_slots_are_unused_sentinels(self) -> None:
        trace = build_trace()
        assert trace.positive_token_id == -1
        assert trace.negative_token_id == -1

    def test_probability_true_is_none_for_categorical(self) -> None:
        trace = build_trace()
        assert trace.probability_true is None

    def test_diagnostics_are_choice_diagnostics(self) -> None:
        trace = build_trace()
        assert isinstance(trace.scoring_diagnostics, ChoiceScoringDiagnostics)
        assert not isinstance(trace.scoring_diagnostics, ScoringDiagnostics)
        assert trace.scoring_diagnostics.candidate_mass == pytest.approx(5.0 / 6.0)

    def test_candidate_mapping_mirrors_plan(self) -> None:
        plan = make_plan()
        trace = build_trace(plan=plan)
        assert trace.candidate_mapping == plan.candidate_mapping
        assert [entry.scoring_label for entry in trace.candidate_mapping] == ["A", "B", "C"]

    def test_resolved_target_token_ids_in_target_order(self) -> None:
        trace = build_trace()
        assert trace.resolved_target_token_ids == (("A", 11), ("B", 22), ("C", 33))

    def test_lineage_and_result_type(self) -> None:
        plan = make_plan()
        trace = build_trace(plan=plan)
        assert trace.plan_fingerprint == plan.fingerprint
        assert trace.decision_fingerprint == plan.decision_fingerprint
        assert trace.evidence.plan_fingerprint == plan.fingerprint


class TestExecutionFingerprint:
    def test_is_64_lowercase_hex(self) -> None:
        fp = build_trace().execution_fingerprint
        assert len(fp) == 64
        assert fp == fp.lower()
        int(fp, 16)

    def test_deterministic_for_same_inputs(self) -> None:
        assert build_trace().execution_fingerprint == build_trace().execution_fingerprint

    def test_different_resolved_token_ids_change_execution_fingerprint(self) -> None:
        other_metadata = honest_metadata()
        other_metadata["resolved_target_token_ids"] = [["A", 111], ["B", 222], ["C", 333]]
        other_trace = build_trace(evidence=make_evidence(metadata=other_metadata))
        assert other_trace.execution_fingerprint != build_trace().execution_fingerprint

    def test_same_resolved_ids_same_execution_fingerprint_regardless_of_label_values(self) -> None:
        # Only the RESOLVED ids enter the execution fingerprint; the evidence
        # logit values do not (they are the output, not the configuration).
        first = build_trace()
        second = build_trace(
            evidence=make_evidence(metadata={**honest_metadata(), "top_token_text": "shipping"})
        )
        assert first.execution_fingerprint == second.execution_fingerprint


class TestCategoricalTraceRejections:
    def test_missing_resolved_ids_rejected(self) -> None:
        metadata = {
            "vocab_logsumexp": VOCAB_LOGSUMEXP,
            "top_token_id": 22,
            "top_token_logit": LOGIT_SHIPPING,
        }
        evidence = make_evidence(metadata=metadata)
        result, diagnostics = make_result_and_diagnostics(evidence, make_plan())
        with pytest.raises(InvalidDecisionError, match="resolved_target_token_ids"):
            build_trace(evidence=evidence, result=result, diagnostics=diagnostics)

    def test_binary_diagnostics_variant_rejected_for_categorical_plan(self) -> None:
        wrong_diagnostics = ScoringDiagnostics(
            verbalizer_mass=0.5,
            top_token_id=22,
            top_token_probability=0.5,
            positive_token_probability=0.2,
            negative_token_probability=0.2,
        )
        with pytest.raises(InvalidDecisionError, match="ChoiceScoringDiagnostics"):
            build_trace(diagnostics=wrong_diagnostics)

    def test_wrong_pair_shape_rejected(self) -> None:
        metadata = {**honest_metadata(), "resolved_target_token_ids": [["A", 11], ["B"]]}
        evidence = make_evidence(metadata=metadata)
        result, diagnostics = make_result_and_diagnostics(evidence, make_plan())
        with pytest.raises(InvalidDecisionError, match=r"resolved_target_token_ids.*pair"):
            build_trace(evidence=evidence, result=result, diagnostics=diagnostics)

    def test_negative_token_id_rejected(self) -> None:
        metadata = {**honest_metadata(), "resolved_target_token_ids": [["A", -1], ["B", 22]]}
        evidence = make_evidence(metadata=metadata)
        result, diagnostics = make_result_and_diagnostics(evidence, make_plan())
        with pytest.raises(InvalidDecisionError, match="non-negative int"):
            build_trace(evidence=evidence, result=result, diagnostics=diagnostics)


class TestCategoricalTraceDict:
    def test_to_dict_is_json_serializable(self) -> None:
        payload = build_trace().to_dict()
        encoded = json.dumps(payload)
        assert isinstance(encoded, str)

    def test_to_dict_carries_categorical_fields(self) -> None:
        payload = build_trace().to_dict()
        assert payload["scoring_strategy"] == "categorical_token_logits"
        assert payload["probability_true"] is None
        assert payload["positive_token_id"] == -1
        assert payload["negative_token_id"] == -1
        assert payload["candidate_mapping"] == [
            {
                "candidate_index": 0,
                "candidate_name": "billing",
                "candidate_description": "money",
                "scoring_label": "A",
            },
            {
                "candidate_index": 1,
                "candidate_name": "shipping",
                "candidate_description": "boxes",
                "scoring_label": "B",
            },
            {
                "candidate_index": 2,
                "candidate_name": "returns",
                "candidate_description": "warranty",
                "scoring_label": "C",
            },
        ]
        assert payload["resolved_target_token_ids"] == [["A", 11], ["B", 22], ["C", 33]]
        assert payload["scoring_diagnostics"]["kind"] == "choice"
        assert payload["scoring_diagnostics"]["candidate_mass"] == pytest.approx(5.0 / 6.0)


class TestCertaintyNotRebuilt:
    def test_result_certainty_carried_from_assembly(self) -> None:
        plan = make_plan()
        evidence = make_evidence(plan=plan)
        result, _ = make_result_and_diagnostics(evidence, plan)
        expected = Certainty.from_probabilities([0.2, 0.6, 0.2])
        assert result.certainty.entropy == pytest.approx(expected.entropy)
        assert result.certainty.margin == pytest.approx(expected.margin)


class TestDiagnoseChoiceEvidenceGuards:
    def test_binary_strategy_plan_rejected(self) -> None:
        from fuzzyai import BoolCompiler, BoolDecision

        binary_plan = BoolCompiler().compile(
            BoolDecision("Yes or no?"), BackendCapabilities(binary_token_logits=True)
        )
        evidence = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=("false", "true"),
            values=(0.0, 1.0),
            plan_fingerprint=binary_plan.fingerprint,
        )
        with pytest.raises(InvalidDecisionError, match="categorical_token_logits"):
            diagnose_choice_evidence(evidence, plan=binary_plan)

    def test_permuted_labels_rejected(self) -> None:
        plan = make_plan()
        evidence = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=("B", "A", "C"),
            values=(0.0, 0.0, 0.0),
            plan_fingerprint=plan.fingerprint,
            metadata=honest_metadata(),
        )
        with pytest.raises(InvalidDecisionError, match="exactly, in the same order"):
            diagnose_choice_evidence(evidence, plan=plan)
