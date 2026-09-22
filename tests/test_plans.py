"""Tests for InferencePlan, RawEvidence, and the Backend protocol."""

import typing
from typing import Any

import pytest

from fuzzyai import (
    Backend,
    BackendCapabilities,
    EvidenceKind,
    InferencePlan,
    InvalidDecisionError,
    InvalidProbabilityError,
    RawEvidence,
    ScoringStrategy,
)


def make_plan(**overrides: Any) -> InferencePlan:
    kwargs: dict[str, Any] = {
        "decision_fingerprint": "a" * 64,
        "strategy": ScoringStrategy.BINARY_TOKEN_LOGITS,
        "prompt": "Answer yes or no.",
    }
    kwargs.update(overrides)
    return InferencePlan(**kwargs)


class TestInferencePlan:
    def test_valid_construction(self) -> None:
        plan = make_plan()
        assert plan.decision_fingerprint == "a" * 64
        assert plan.strategy is ScoringStrategy.BINARY_TOKEN_LOGITS
        assert plan.prompt == "Answer yes or no."

    def test_defaults(self) -> None:
        plan = make_plan()
        assert plan.targets == ()
        assert plan.required_capabilities == BackendCapabilities.none()

    def test_empty_decision_fingerprint_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="decision_fingerprint"):
            make_plan(decision_fingerprint="")

    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="prompt"):
            make_plan(prompt="   ")

    def test_duplicate_targets_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="unique"):
            make_plan(targets=("a", "a"))

    def test_empty_target_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="targets"):
            make_plan(targets=("a", " "))

    def test_wrong_strategy_type_rejected(self) -> None:
        bad_strategy: Any = "binary_token_logits"
        with pytest.raises(InvalidDecisionError, match="ScoringStrategy"):
            make_plan(strategy=bad_strategy)

    def test_wrong_capabilities_type_rejected(self) -> None:
        bad_caps: Any = {"batching": True}
        with pytest.raises(InvalidDecisionError, match="BackendCapabilities"):
            make_plan(required_capabilities=bad_caps)

    def test_fingerprint_deterministic(self) -> None:
        assert make_plan().fingerprint == make_plan().fingerprint

    def test_fingerprint_distinguishes_plans(self) -> None:
        assert make_plan().fingerprint != make_plan(prompt="Different prompt.").fingerprint
        assert make_plan().fingerprint != make_plan(targets=("t",)).fingerprint

    def test_fingerprint_is_64_hex(self) -> None:
        assert len(make_plan().fingerprint) == 64

    def test_frozen(self) -> None:
        plan = make_plan()
        mutable: Any = plan
        with pytest.raises(AttributeError):
            mutable.prompt = "x"


class TestRawEvidence:
    def test_valid_construction(self) -> None:
        e = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=("yes", "no"),
            values=(2.0, -1.0),
            plan_fingerprint="b" * 64,
        )
        assert e.kind is EvidenceKind.LOGITS
        assert e.labels == ("yes", "no")
        assert e.values == (2.0, -1.0)

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="same length"):
            RawEvidence(kind=EvidenceKind.LOGITS, labels=("a", "b"), values=(1.0,))

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="unique"):
            RawEvidence(kind=EvidenceKind.LOGITS, labels=("a", "a"), values=(1.0, 2.0))

    def test_empty_labels_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="non-empty"):
            RawEvidence(kind=EvidenceKind.LOGITS, labels=(), values=())

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_rejected(self, bad: float) -> None:
        with pytest.raises(InvalidProbabilityError, match="finite"):
            RawEvidence(kind=EvidenceKind.LOGITS, labels=("a", "b"), values=(1.0, bad))

    def test_bool_value_rejected(self) -> None:
        bad_values: Any = (True, 1.0)
        with pytest.raises(InvalidProbabilityError):
            RawEvidence(kind=EvidenceKind.LOGITS, labels=("a", "b"), values=bad_values)

    def test_int_values_coerced_to_float(self) -> None:
        e = RawEvidence(kind=EvidenceKind.SAMPLE_COUNTS, labels=("a", "b"), values=(3, 4))
        assert e.values == (3.0, 4.0)
        assert all(isinstance(v, float) for v in e.values)

    def test_json_invalid_metadata_rejected(self) -> None:
        bad_metadata: Any = {"bad": object()}
        with pytest.raises(InvalidDecisionError, match="JSON-compatible"):
            RawEvidence(
                kind=EvidenceKind.LOGITS,
                labels=("a", "b"),
                values=(1.0, 2.0),
                metadata=bad_metadata,
            )

    def test_metadata_deep_copied(self) -> None:
        meta: dict[str, Any] = {"k": [1, 2]}
        e = RawEvidence(
            kind=EvidenceKind.LOGITS, labels=("a", "b"), values=(1.0, 2.0), metadata=meta
        )
        meta["k"].append(3)
        meta["new"] = "x"
        assert e.metadata == {"k": [1, 2]}

    def test_evidence_kind_round_trip(self) -> None:
        assert EvidenceKind("logits") is EvidenceKind.LOGITS
        assert EvidenceKind("logprobs") is EvidenceKind.LOGPROBS
        assert EvidenceKind("sample_counts") is EvidenceKind.SAMPLE_COUNTS
        assert EvidenceKind.LOGITS.value == "logits"

    # INV-18: evidence is not probability (raw logits need not sum to 1).
    def test_is_not_a_probability(self) -> None:
        # Raw logits are explicitly NOT probabilities: values may be negative
        # and need not sum to 1.
        e = RawEvidence(kind=EvidenceKind.LOGITS, labels=("a", "b"), values=(-5.0, 12.0))
        assert sum(e.values) != pytest.approx(1.0)

    def test_frozen(self) -> None:
        e = RawEvidence(kind=EvidenceKind.LOGITS, labels=("a", "b"), values=(1.0, 2.0))
        mutable: Any = e
        with pytest.raises(AttributeError):
            mutable.values = (3.0, 4.0)


class TestBackendProtocol:
    def test_fake_backend_satisfies_protocol(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.capabilities = BackendCapabilities(binary_token_logits=True)

            def execute(self, plan: InferencePlan) -> RawEvidence:
                return RawEvidence(
                    kind=EvidenceKind.LOGITS,
                    labels=("yes", "no"),
                    values=(1.0, 0.0),
                    plan_fingerprint=plan.fingerprint,
                )

        backend: Backend = FakeBackend()
        evidence = backend.execute(make_plan())
        assert evidence.labels == ("yes", "no")
        assert backend.capabilities.binary_token_logits is True

    # INV-16: backend ignorance (Backend never sees decisions or results).
    def test_backend_is_decision_agnostic(self) -> None:
        # The protocol's execute() only accepts an InferencePlan and returns
        # RawEvidence; decisions and results never appear in its signature.
        hints = typing.get_type_hints(Backend.execute)
        assert hints["plan"] is InferencePlan
        assert hints["return"] is RawEvidence
