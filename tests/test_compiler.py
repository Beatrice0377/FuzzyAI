"""Tests for BoolCompiler: decision -> plan compilation and validation."""

from typing import Any

import pytest

from probvenance import (
    BINARY_DOCTRINE_ID,
    BINARY_SEMANTIC_JUDGMENT_V1,
    BackendCapabilities,
    BoolCompiler,
    BoolDecision,
    ChoiceDecision,
    InferencePlan,
    InvalidDecisionError,
    ScoringDoctrine,
    ScoringStrategy,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
)


def make_decision() -> BoolDecision:
    return BoolDecision(
        "Does the document state a delivery date?",
        context="The courier confirmed Thursday in writing.",
    )


def make_choice_decision() -> ChoiceDecision:
    return ChoiceDecision(
        "Which carrier delivered the parcel?",
        context="The courier confirmed Thursday in writing.",
        choices={"dhl": "DHL", "ups": "UPS"},
    )


class TestBoolCompilerInit:
    def test_defaults(self) -> None:
        compiler = BoolCompiler()
        assert compiler.doctrine is BINARY_SEMANTIC_JUDGMENT_V1
        assert compiler.positive_verbalizer == "yes"
        assert compiler.negative_verbalizer == "no"

    def test_custom_doctrine_and_verbalizers(self) -> None:
        doctrine = ScoringDoctrine(
            doctrine_id="custom",
            version=1,
            system_prompt="Fixed.",
            user_template=(
                "Q: {question}\nC: {context}\nA: {positive_verbalizer}/{negative_verbalizer}"
            ),
        )
        compiler = BoolCompiler(
            doctrine=doctrine, positive_verbalizer="affirm", negative_verbalizer="deny"
        )
        assert compiler.doctrine is doctrine
        assert compiler.positive_verbalizer == "affirm"
        assert compiler.negative_verbalizer == "deny"

    def test_non_doctrine_rejected(self) -> None:
        bad_doctrine: Any = "not a doctrine"
        with pytest.raises(InvalidDecisionError, match="ScoringDoctrine"):
            BoolCompiler(doctrine=bad_doctrine)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_verbalizers_rejected(self, bad: str) -> None:
        with pytest.raises(InvalidDecisionError, match="positive_verbalizer"):
            BoolCompiler(positive_verbalizer=bad)
        with pytest.raises(InvalidDecisionError, match="negative_verbalizer"):
            BoolCompiler(negative_verbalizer=bad)

    def test_identical_verbalizers_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="distinct"):
            BoolCompiler(positive_verbalizer="same", negative_verbalizer="same")


class TestBoolCompilerCompile:
    def test_plan_fields(self) -> None:
        compiler = BoolCompiler()
        decision = make_decision()
        plan = compiler.compile(decision, BackendCapabilities(binary_token_logits=True))
        assert isinstance(plan, InferencePlan)
        assert plan.decision_fingerprint == decision.fingerprint
        assert plan.strategy is ScoringStrategy.BINARY_TOKEN_LOGITS
        assert plan.targets == ()
        assert plan.system_prompt == BINARY_SEMANTIC_JUDGMENT_V1.system_prompt
        assert plan.positive_verbalizer == "yes"
        assert plan.negative_verbalizer == "no"
        assert plan.doctrine_id == BINARY_DOCTRINE_ID
        assert plan.required_capabilities == BackendCapabilities(binary_token_logits=True)

    def test_plan_prompt_contains_question_context_and_verbalizers(self) -> None:
        decision = make_decision()
        plan = BoolCompiler().compile(decision, BackendCapabilities(binary_token_logits=True))
        assert decision.question in plan.prompt
        assert "The courier confirmed Thursday in writing." in plan.prompt
        assert plan.system_prompt is not None and decision.question not in plan.system_prompt

    def test_compile_is_deterministic(self) -> None:
        compiler = BoolCompiler()
        decision = make_decision()
        caps = BackendCapabilities(binary_token_logits=True)
        first = compiler.compile(decision, caps)
        second = compiler.compile(decision, caps)
        assert first.fingerprint == second.fingerprint

    def test_compile_does_not_mutate_the_decision(self) -> None:
        compiler = BoolCompiler()
        decision = make_decision()
        before = decision.fingerprint
        compiler.compile(decision, BackendCapabilities(binary_token_logits=True))
        assert decision.fingerprint == before
        assert decision.question == "Does the document state a delivery date?"

    def test_custom_verbalizers_land_in_the_prompt(self) -> None:
        compiler = BoolCompiler(positive_verbalizer="affirm", negative_verbalizer="deny")
        plan = compiler.compile(make_decision(), BackendCapabilities(binary_token_logits=True))
        assert plan.prompt is not None
        assert "affirm" in plan.prompt
        assert "deny" in plan.prompt

    def test_missing_capability_rejected(self) -> None:
        compiler = BoolCompiler()
        with pytest.raises(UnsupportedCapabilityError, match="binary_token_logits"):
            compiler.compile(make_decision(), BackendCapabilities.none())
        with pytest.raises(UnsupportedCapabilityError, match="binary_token_logits"):
            compiler.compile(make_decision(), BackendCapabilities(binary_token_logits=False))

    def test_choice_decision_rejected(self) -> None:
        compiler = BoolCompiler()
        with pytest.raises(UnsupportedDecisionError, match="ChoiceDecision"):
            compiler.compile(make_choice_decision(), BackendCapabilities(binary_token_logits=True))

    def test_verbalizer_change_changes_plan_fingerprint(self) -> None:
        decision = make_decision()
        caps = BackendCapabilities(binary_token_logits=True)
        default = BoolCompiler().compile(decision, caps)
        custom = BoolCompiler(positive_verbalizer="affirm").compile(decision, caps)
        assert default.fingerprint != custom.fingerprint

    def test_doctrine_change_changes_plan_fingerprint(self) -> None:
        decision = make_decision()
        caps = BackendCapabilities(binary_token_logits=True)
        other_doctrine = ScoringDoctrine(
            doctrine_id="other-doctrine",
            version=2,
            system_prompt="Other fixed instructions.",
            user_template=(
                "Q: {question}\nC: {context}\nA: {positive_verbalizer}/{negative_verbalizer}"
            ),
        )
        default = BoolCompiler().compile(decision, caps)
        other = BoolCompiler(doctrine=other_doctrine).compile(decision, caps)
        assert default.fingerprint != other.fingerprint
        assert other.doctrine_id == "other-doctrine"
