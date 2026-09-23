"""Tests for InferencePlan, RawEvidence, and the Backend protocol."""

import typing
from typing import Any

import pytest

from fuzzyai import (
    PLAN_FINGERPRINT_VERSION,
    Backend,
    BackendCapabilities,
    CandidateLabelMapping,
    EvidenceKind,
    InferencePlan,
    InvalidDecisionError,
    InvalidProbabilityError,
    RawEvidence,
    ScoringStrategy,
)
from fuzzyai.assembler import (
    BINARY_ASSEMBLER_ID,
    BINARY_ASSEMBLER_VERSION,
    CATEGORICAL_ASSEMBLER_ID,
    CATEGORICAL_ASSEMBLER_VERSION,
)
from fuzzyai.compiler import (
    BINARY_COMPILER_ID,
    BINARY_COMPILER_VERSION,
    CATEGORICAL_COMPILER_ID,
    CATEGORICAL_COMPILER_VERSION,
)


def make_plan(**overrides: Any) -> InferencePlan:
    kwargs: dict[str, Any] = {
        "decision_fingerprint": "a" * 64,
        "strategy": ScoringStrategy.BINARY_TOKEN_LOGITS,
        "prompt": "Answer yes or no.",
        "positive_verbalizer": "yes",
        "negative_verbalizer": "no",
        "compiler_id": BINARY_COMPILER_ID,
        "compiler_version": BINARY_COMPILER_VERSION,
        "assembler_id": BINARY_ASSEMBLER_ID,
        "assembler_version": BINARY_ASSEMBLER_VERSION,
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


class TestInferencePlanVerbalizerValidation:
    def test_missing_positive_verbalizer_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="positive_verbalizer"):
            make_plan(positive_verbalizer=None)

    def test_missing_negative_verbalizer_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="negative_verbalizer"):
            make_plan(negative_verbalizer=None)

    def test_empty_positive_verbalizer_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="positive_verbalizer"):
            make_plan(positive_verbalizer="   ")

    def test_empty_negative_verbalizer_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="negative_verbalizer"):
            make_plan(negative_verbalizer="")

    def test_duplicate_verbalizers_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="must differ"):
            make_plan(positive_verbalizer="yes", negative_verbalizer="yes")

    @pytest.mark.parametrize(
        "strategy",
        [ScoringStrategy.CATEGORICAL_TOKEN_LOGITS, ScoringStrategy.TOKEN_LOGPROBS],
    )
    def test_verbalizers_rejected_for_non_binary_strategies(
        self, strategy: ScoringStrategy
    ) -> None:
        with pytest.raises(InvalidDecisionError, match="only meaningful"):
            make_plan(strategy=strategy, positive_verbalizer="yes", negative_verbalizer="no")

    def test_non_binary_strategy_without_verbalizers_allowed(self) -> None:
        plan = make_plan(
            strategy=ScoringStrategy.TOKEN_LOGPROBS,
            positive_verbalizer=None,
            negative_verbalizer=None,
            system_prompt=None,
            doctrine_id=None,
        )
        assert plan.positive_verbalizer is None
        assert plan.negative_verbalizer is None

    def test_empty_system_prompt_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="system_prompt"):
            make_plan(system_prompt="   ")

    def test_non_string_system_prompt_rejected(self) -> None:
        bad_prompt: Any = 7
        with pytest.raises(InvalidDecisionError, match="system_prompt"):
            make_plan(system_prompt=bad_prompt)

    def test_empty_doctrine_id_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="doctrine_id"):
            make_plan(doctrine_id="")

    def test_whitespace_doctrine_id_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="doctrine_id"):
            make_plan(doctrine_id="  ")

    def test_verbalizer_change_changes_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(positive_verbalizer="affirmative").fingerprint
        assert make_plan().fingerprint != make_plan(negative_verbalizer="negative").fingerprint

    def test_doctrine_id_change_changes_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(doctrine_id="other-doctrine").fingerprint

    def test_system_prompt_change_changes_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(system_prompt="Other instructions.").fingerprint

    def test_fingerprint_v4_commits_provenance(self) -> None:
        plan = make_plan(system_prompt="S", positive_verbalizer="yes", negative_verbalizer="no")
        assert len(plan.fingerprint) == 64
        assert plan.fingerprint != make_plan(system_prompt="S", compiler_version=2).fingerprint


class TestPlanFingerprintVersion:
    def test_fingerprint_version_is_the_schema_constant(self) -> None:
        assert make_plan().fingerprint_version == PLAN_FINGERPRINT_VERSION
        assert make_categorical_plan().fingerprint_version == PLAN_FINGERPRINT_VERSION

    def test_changing_the_schema_version_changes_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = make_plan().fingerprint
        monkeypatch.setattr("fuzzyai.plans.PLAN_FINGERPRINT_VERSION", PLAN_FINGERPRINT_VERSION + 1)
        assert make_plan().fingerprint != before


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


CATEGORICAL_MAPPING = (
    CandidateLabelMapping(0, "billing", "money", "A"),
    CandidateLabelMapping(1, "shipping", "boxes", "B"),
    CandidateLabelMapping(2, "returns", "warranty", "C"),
)


def make_categorical_plan(**overrides: Any) -> InferencePlan:
    kwargs: dict[str, Any] = {
        "decision_fingerprint": "b" * 64,
        "strategy": ScoringStrategy.CATEGORICAL_TOKEN_LOGITS,
        "prompt": "Answer with one label.",
        "targets": ("A", "B", "C"),
        "label_scheme_id": "categorical-labels-v1",
        "candidate_mapping": CATEGORICAL_MAPPING,
        "compiler_id": CATEGORICAL_COMPILER_ID,
        "compiler_version": CATEGORICAL_COMPILER_VERSION,
        "assembler_id": CATEGORICAL_ASSEMBLER_ID,
        "assembler_version": CATEGORICAL_ASSEMBLER_VERSION,
    }
    kwargs.update(overrides)
    return InferencePlan(**kwargs)


class TestPlanProvenance:
    def test_missing_compiler_provenance_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="compiler_id"):
            InferencePlan(
                decision_fingerprint="a" * 64,
                strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
                prompt="Answer.",
                positive_verbalizer="yes",
                negative_verbalizer="no",
            )

    def test_missing_assembler_provenance_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="assembler_id"):
            make_plan(assembler_id="")

    def test_non_positive_compiler_version_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="compiler_version"):
            make_plan(compiler_version=0)

    def test_non_positive_assembler_version_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="assembler_version"):
            make_plan(assembler_version=0)

    def test_compiler_id_changes_plan_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(compiler_id="other-compiler").fingerprint

    def test_compiler_version_changes_plan_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(compiler_version=2).fingerprint

    def test_assembler_id_changes_plan_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(assembler_id="other-assembler").fingerprint

    def test_assembler_version_changes_plan_fingerprint(self) -> None:
        assert make_plan().fingerprint != make_plan(assembler_version=2).fingerprint

    def test_categorical_provenance_committed(self) -> None:
        plan = make_categorical_plan()
        assert plan.compiler_id == CATEGORICAL_COMPILER_ID
        assert plan.assembler_id == CATEGORICAL_ASSEMBLER_ID
        assert plan.fingerprint != make_categorical_plan(assembler_version=2).fingerprint

    def test_provenance_absent_for_unsupported_strategy(self) -> None:
        plan = InferencePlan(
            decision_fingerprint="a" * 64,
            strategy=ScoringStrategy.TOKEN_LOGPROBS,
            prompt="Answer.",
        )
        assert plan.compiler_id == ""
        assert len(plan.fingerprint) == 64


class TestCategoricalPlanValidation:
    def test_valid_categorical_plan(self) -> None:
        plan = make_categorical_plan()
        assert plan.targets == ("A", "B", "C")
        assert plan.label_scheme_id == "categorical-labels-v1"
        assert plan.candidate_mapping == CATEGORICAL_MAPPING

    def test_single_target_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="at least 2 ordered targets"):
            make_categorical_plan(targets=("A",))

    def test_missing_label_scheme_id_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="label_scheme_id"):
            make_categorical_plan(label_scheme_id=None)

    def test_mapping_length_mismatch_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="one entry per target"):
            make_categorical_plan(candidate_mapping=CATEGORICAL_MAPPING[:2])

    def test_mapping_order_mismatch_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="indices must be 0"):
            make_categorical_plan(
                candidate_mapping=(
                    CandidateLabelMapping(1, "shipping", "boxes", "B"),
                    CandidateLabelMapping(0, "billing", "money", "A"),
                    CandidateLabelMapping(2, "returns", "warranty", "C"),
                )
            )

    def test_duplicate_candidate_names_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="names must be unique"):
            make_categorical_plan(
                candidate_mapping=(
                    CandidateLabelMapping(0, "billing", "money", "A"),
                    CandidateLabelMapping(1, "billing", "boxes", "B"),
                    CandidateLabelMapping(2, "returns", "warranty", "C"),
                )
            )

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="labels must be unique"):
            make_categorical_plan(
                candidate_mapping=(
                    CandidateLabelMapping(0, "billing", "money", "A"),
                    CandidateLabelMapping(1, "shipping", "boxes", "A"),
                    CandidateLabelMapping(2, "returns", "warranty", "C"),
                )
            )

    def test_label_target_mismatch_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="must follow target order"):
            make_categorical_plan(
                candidate_mapping=(
                    CandidateLabelMapping(0, "billing", "money", "A"),
                    CandidateLabelMapping(1, "shipping", "boxes", "C"),
                    CandidateLabelMapping(2, "returns", "warranty", "B"),
                )
            )

    def test_mapping_forbidden_for_binary_strategy(self) -> None:
        with pytest.raises(InvalidDecisionError, match="only meaningful"):
            make_plan(candidate_mapping=CATEGORICAL_MAPPING)

    def test_targets_included_in_fingerprint(self) -> None:
        four_labels = (
            CandidateLabelMapping(0, "billing", "money", "A"),
            CandidateLabelMapping(1, "shipping", "boxes", "B"),
            CandidateLabelMapping(2, "returns", "warranty", "C"),
            CandidateLabelMapping(3, "accounts", "login", "D"),
        )
        assert (
            make_categorical_plan().fingerprint
            != make_categorical_plan(
                targets=("A", "B", "C", "D"), candidate_mapping=four_labels
            ).fingerprint
        )

    def test_mapping_included_in_fingerprint(self) -> None:
        assert (
            make_categorical_plan().fingerprint
            != make_categorical_plan(
                candidate_mapping=(
                    CandidateLabelMapping(0, "shipping", "boxes", "A"),
                    CandidateLabelMapping(1, "billing", "money", "B"),
                    CandidateLabelMapping(2, "returns", "warranty", "C"),
                )
            ).fingerprint
        )

    def test_label_scheme_id_included_in_fingerprint(self) -> None:
        assert (
            make_categorical_plan().fingerprint
            != make_categorical_plan(label_scheme_id="categorical-labels-v2").fingerprint
        )

    def test_frozen(self) -> None:
        plan = make_categorical_plan()
        mutable: Any = plan
        with pytest.raises(AttributeError):
            mutable.targets = ("B", "A", "C")
