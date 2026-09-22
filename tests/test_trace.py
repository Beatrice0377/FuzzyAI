"""Tests for DecisionTrace and build_decision_trace."""

import json
import math
from typing import Any

import pytest

from fuzzyai import (
    BINARY_EVIDENCE_LABELS,
    DecisionTrace,
    EvidenceKind,
    InferencePlan,
    InvalidDecisionError,
    RawEvidence,
    ScoringStrategy,
    assemble_bool_probability,
    build_decision_trace,
    fingerprint,
)

TIMESTAMP = "2026-01-01T00:00:00+00:00"

HONEST_METADATA: dict[str, Any] = {
    "positive_token_id": 9642,
    "negative_token_id": 3134,
    "model": "fake-model",
    "model_revision": "rev-1",
    "input_token_count": 42,
}


def make_plan(**overrides: Any) -> InferencePlan:
    kwargs: dict[str, Any] = {
        "decision_fingerprint": "a" * 64,
        "strategy": ScoringStrategy.BINARY_TOKEN_LOGITS,
        "prompt": "Question: Q?\nAnswer yes or no.",
        "system_prompt": "System instructions.",
        "positive_verbalizer": "yes",
        "negative_verbalizer": "no",
        "doctrine_id": "binary-semantic-judgment-v1",
    }
    kwargs.update(overrides)
    return InferencePlan(**kwargs)


def make_evidence(**overrides: Any) -> RawEvidence:
    kwargs: dict[str, Any] = {
        "kind": EvidenceKind.LOGITS,
        "labels": BINARY_EVIDENCE_LABELS,
        "values": (0.0, 1.0),
        "plan_fingerprint": "b" * 64,
        "metadata": dict(HONEST_METADATA),
    }
    kwargs.update(overrides)
    return RawEvidence(**kwargs)


def make_result(evidence: RawEvidence, trace_id: str | None = "trace-1"):
    return assemble_bool_probability(evidence, trace_id=trace_id)


def build_trace(**overrides: Any) -> DecisionTrace:
    kwargs: dict[str, Any] = {
        "trace_id": "trace-1",
        "timestamp": TIMESTAMP,
        "decision_fingerprint": "a" * 64,
        "plan": make_plan(),
        "evidence": make_evidence(),
        "backend_type": "FakeBackend",
        "latency_ms": 1.5,
    }
    if "result" not in kwargs:
        kwargs["result"] = make_result(kwargs["evidence"])
    if "evidence" in overrides:
        kwargs["evidence"] = overrides.pop("evidence")
        kwargs["result"] = make_result(kwargs["evidence"])
    kwargs.update(overrides)
    return build_decision_trace(**kwargs)


class TestDecisionTraceFields:
    def test_every_field(self) -> None:
        plan = make_plan()
        evidence = make_evidence(plan_fingerprint=plan.fingerprint)
        trace = build_trace(plan=plan, evidence=evidence)
        assert trace.trace_id == "trace-1"
        assert trace.timestamp == TIMESTAMP
        assert trace.decision_fingerprint == "a" * 64
        assert trace.plan_fingerprint == plan.fingerprint
        assert trace.plan_fingerprint == evidence.plan_fingerprint
        assert trace.scoring_strategy == "binary_token_logits"
        assert trace.doctrine_id == "binary-semantic-judgment-v1"
        assert trace.positive_verbalizer == "yes"
        assert trace.negative_verbalizer == "no"
        assert trace.positive_token_id == 9642
        assert trace.negative_token_id == 3134
        assert trace.evidence is evidence
        assert trace.probability_true == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
        assert trace.input_fingerprint == fingerprint(
            {"system_prompt": plan.system_prompt, "prompt": plan.prompt}
        )
        assert trace.backend_type == "FakeBackend"
        assert trace.latency_ms == 1.5
        assert trace.model == "fake-model"
        assert trace.model_revision == "rev-1"
        assert trace.input_token_count == 42
        assert trace.rendered_input is None

    def test_lineage_fields_come_from_plan_and_argument(self) -> None:
        plan = make_plan(decision_fingerprint="c" * 64)
        trace = build_trace(plan=plan, decision_fingerprint="d" * 64)
        assert trace.decision_fingerprint == "d" * 64
        assert trace.plan_fingerprint == plan.fingerprint

    def test_optional_metadata_absent_yields_none(self) -> None:
        metadata = {"positive_token_id": 1, "negative_token_id": 2}
        trace = build_trace(evidence=make_evidence(metadata=metadata))
        assert trace.model is None
        assert trace.model_revision is None
        assert trace.input_token_count is None
        assert trace.rendered_input is None

    def test_frozen(self) -> None:
        trace = build_trace()
        mutable: Any = trace
        with pytest.raises(AttributeError):
            mutable.trace_id = "other"


class TestInputFingerprint:
    def test_stable_across_builds(self) -> None:
        plan = make_plan()
        first = build_trace(plan=plan)
        second = build_trace(plan=plan)
        assert first.input_fingerprint == second.input_fingerprint

    def test_sensitive_to_prompt(self) -> None:
        assert build_trace(plan=make_plan(prompt="One prompt.")).input_fingerprint != (
            build_trace(plan=make_plan(prompt="Another prompt.")).input_fingerprint
        )

    def test_sensitive_to_system_prompt(self) -> None:
        assert build_trace(plan=make_plan(system_prompt="A.")).input_fingerprint != (
            build_trace(plan=make_plan(system_prompt="B.")).input_fingerprint
        )

    def test_is_64_hex(self) -> None:
        assert len(build_trace().input_fingerprint) == 64


class TestCaptureRenderedInput:
    def test_disabled_by_default(self) -> None:
        assert build_trace().rendered_input is None

    def test_enabled_captures_from_metadata(self) -> None:
        trace = build_trace(
            evidence=make_evidence(
                metadata={**HONEST_METADATA, "rendered_input": "RENDERED PROMPT"}
            ),
            capture_rendered_input=True,
        )
        assert trace.rendered_input == "RENDERED PROMPT"

    def test_enabled_without_metadata_is_none(self) -> None:
        metadata = {"positive_token_id": 1, "negative_token_id": 2}
        trace = build_trace(evidence=make_evidence(metadata=metadata), capture_rendered_input=True)
        assert trace.rendered_input is None


class TestInputFingerprintProvenance:
    def test_uses_rendered_input_when_evidence_carries_it(self) -> None:
        plan = make_plan()
        evidence = make_evidence(metadata={**HONEST_METADATA, "rendered_input": "RENDERED PROMPT"})
        trace = build_trace(plan=plan, evidence=evidence)
        assert trace.input_fingerprint == fingerprint({"rendered_input": "RENDERED PROMPT"})
        assert trace.input_fingerprint != fingerprint(
            {"system_prompt": plan.system_prompt, "prompt": plan.prompt}
        )

    def test_falls_back_to_plan_fields_without_rendered_input(self) -> None:
        plan = make_plan()
        trace = build_trace(plan=plan)
        assert trace.input_fingerprint == fingerprint(
            {"system_prompt": plan.system_prompt, "prompt": plan.prompt}
        )

    def test_capture_flag_does_not_change_fingerprint(self) -> None:
        evidence = make_evidence(metadata={**HONEST_METADATA, "rendered_input": "RENDERED PROMPT"})
        without_capture = build_trace(evidence=evidence, capture_rendered_input=False)
        with_capture = build_trace(evidence=evidence, capture_rendered_input=True)
        assert without_capture.rendered_input is None
        assert with_capture.rendered_input == "RENDERED PROMPT"
        assert without_capture.input_fingerprint == with_capture.input_fingerprint

    def test_different_renderings_give_different_fingerprints(self) -> None:
        plan = make_plan()
        first = build_trace(
            plan=plan,
            evidence=make_evidence(metadata={**HONEST_METADATA, "rendered_input": "RENDERING A"}),
        )
        second = build_trace(
            plan=plan,
            evidence=make_evidence(metadata={**HONEST_METADATA, "rendered_input": "RENDERING B"}),
        )
        assert first.input_fingerprint != second.input_fingerprint


class TestTraceMetadataValidation:
    @pytest.mark.parametrize("key", ["positive_token_id", "negative_token_id"])
    def test_missing_required_key_rejected(self, key: str) -> None:
        metadata = {k: v for k, v in HONEST_METADATA.items() if k != key}
        with pytest.raises(InvalidDecisionError, match=key):
            build_trace(evidence=make_evidence(metadata=metadata))

    @pytest.mark.parametrize("bad", ["9642", 9642.0, True])
    def test_wrong_type_token_id_rejected(self, bad: Any) -> None:
        metadata = dict(HONEST_METADATA)
        metadata["positive_token_id"] = bad
        with pytest.raises(InvalidDecisionError, match="positive_token_id"):
            build_trace(evidence=make_evidence(metadata=metadata))

    def test_wrong_type_model_rejected(self) -> None:
        metadata = dict(HONEST_METADATA)
        metadata["model"] = 7
        with pytest.raises(InvalidDecisionError, match="model"):
            build_trace(evidence=make_evidence(metadata=metadata))

    def test_wrong_type_input_token_count_rejected(self) -> None:
        metadata = dict(HONEST_METADATA)
        metadata["input_token_count"] = "42"
        with pytest.raises(InvalidDecisionError, match="input_token_count"):
            build_trace(evidence=make_evidence(metadata=metadata))


class TestToDict:
    def test_json_compatible(self) -> None:
        trace = build_trace(capture_rendered_input=True)
        encoded = json.dumps(trace.to_dict())
        assert isinstance(encoded, str)
        decoded = json.loads(encoded)
        assert decoded["trace_id"] == "trace-1"
        assert decoded["evidence"]["kind"] == "logits"
        assert decoded["evidence"]["labels"] == ["false", "true"]
        assert decoded["evidence"]["metadata"]["positive_token_id"] == 9642
        assert decoded["positive_token_id"] == 9642

    def test_to_dict_round_trips_every_scalar_field(self) -> None:
        trace = build_trace(capture_rendered_input=True)
        decoded = json.loads(json.dumps(trace.to_dict()))
        for key, value in trace.to_dict().items():
            assert decoded[key] == value

    def test_none_fields_survive_json(self) -> None:
        metadata = {"positive_token_id": 1, "negative_token_id": 2}
        trace = build_trace(evidence=make_evidence(metadata=metadata))
        decoded = json.loads(json.dumps(trace.to_dict()))
        assert decoded["model"] is None
        assert decoded["rendered_input"] is None


class TestTraceIdProvenance:
    def test_trace_id_is_not_any_fingerprint(self) -> None:
        trace = build_trace()
        fingerprints = {
            trace.decision_fingerprint,
            trace.plan_fingerprint,
            trace.input_fingerprint,
            trace.evidence.plan_fingerprint,
        }
        assert trace.trace_id not in fingerprints
        assert len(trace.trace_id) != 64
