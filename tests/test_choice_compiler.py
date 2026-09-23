"""Tests for ChoiceCompiler: ChoiceDecision -> categorical InferencePlan."""

from typing import Any

import pytest

from probvenance import (
    CATEGORICAL_COMPILER_VERSION,
    CATEGORICAL_DOCTRINE_ID,
    CATEGORICAL_DOCTRINE_VERSION,
    CATEGORICAL_LABEL_SCHEME_ID,
    CATEGORICAL_LABELS,
    CATEGORICAL_SEMANTIC_JUDGMENT_V1,
    BackendCapabilities,
    BoolDecision,
    CandidateLabelMapping,
    CategoricalScoringDoctrine,
    Choice,
    ChoiceCompiler,
    ChoiceDecision,
    InferencePlan,
    InvalidDecisionError,
    ScoringStrategy,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
)

CAPS = BackendCapabilities(categorical_token_logits=True)


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


def make_name_only_decision(count: int) -> ChoiceDecision:
    return ChoiceDecision(
        "Pick one.",
        choices=[Choice(name=f"c{i}") for i in range(count)],
    )


class TestLabelScheme:
    def test_scheme_id_and_version(self) -> None:
        assert CATEGORICAL_LABEL_SCHEME_ID == "categorical-labels-v1"
        assert CATEGORICAL_COMPILER_VERSION == 1
        assert CATEGORICAL_LABELS == ("A", "B", "C", "D", "E", "F", "G", "H")

    def test_labels_assigned_in_candidate_order(self) -> None:
        plan = ChoiceCompiler().compile(make_decision(), CAPS)
        assert plan.targets == ("A", "B", "C")
        assert [entry.candidate_name for entry in plan.candidate_mapping] == [
            "billing",
            "shipping",
            "returns",
        ]

    def test_candidate_descriptions_carried(self) -> None:
        plan = ChoiceCompiler().compile(make_decision(), CAPS)
        assert plan.candidate_mapping[0].candidate_description == "Payment, charges, invoices"
        assert plan.candidate_mapping[0].scoring_label == "A"
        assert plan.candidate_mapping[0].candidate_index == 0


class TestCompilePlan:
    def test_plan_shape(self) -> None:
        decision = make_decision()
        plan = ChoiceCompiler().compile(decision, CAPS)
        assert isinstance(plan, InferencePlan)
        assert plan.decision_fingerprint == decision.fingerprint
        assert plan.strategy is ScoringStrategy.CATEGORICAL_TOKEN_LOGITS
        assert plan.doctrine_id == CATEGORICAL_DOCTRINE_ID
        assert plan.doctrine_version == CATEGORICAL_DOCTRINE_VERSION
        assert plan.label_scheme_id == CATEGORICAL_LABEL_SCHEME_ID
        assert plan.required_capabilities == BackendCapabilities(categorical_token_logits=True)
        assert plan.positive_verbalizer is None
        assert plan.negative_verbalizer is None
        assert plan.system_prompt
        assert plan.prompt

    def test_compile_is_deterministic(self) -> None:
        first = ChoiceCompiler().compile(make_decision(), CAPS)
        second = ChoiceCompiler().compile(make_decision(), CAPS)
        assert first.fingerprint == second.fingerprint

    def test_two_choice_decision_gets_two_labels(self) -> None:
        plan = ChoiceCompiler().compile(make_name_only_decision(2), CAPS)
        assert plan.targets == ("A", "B")
        assert len(plan.candidate_mapping) == 2

    def test_max_label_count(self) -> None:
        plan = ChoiceCompiler().compile(make_name_only_decision(len(CATEGORICAL_LABELS)), CAPS)
        assert plan.targets == CATEGORICAL_LABELS

    def test_more_choices_than_labels_rejected(self) -> None:
        decision = make_name_only_decision(len(CATEGORICAL_LABELS) + 1)
        with pytest.raises(InvalidDecisionError, match="at most 8"):
            ChoiceCompiler().compile(decision, CAPS)

    def test_prompt_declares_mapping_lines(self) -> None:
        plan = ChoiceCompiler().compile(make_decision(), CAPS)
        assert "A = billing (Payment, charges, invoices)" in plan.prompt
        assert "B = shipping (Delivery, couriers, parcels)" in plan.prompt
        assert "C = returns (Refunds, exchanges, warranty)" in plan.prompt

    def test_prompt_without_description(self) -> None:
        plan = ChoiceCompiler().compile(make_name_only_decision(2), CAPS)
        assert "A = c0" in plan.prompt
        assert "B = c1" in plan.prompt

    def test_context_in_user_prompt_only(self) -> None:
        plan = ChoiceCompiler().compile(make_decision(), CAPS)
        assert "damaged parcel" in plan.prompt
        assert "damaged parcel" not in (plan.system_prompt or "")


class TestCompileRejections:
    def test_bool_decision_rejected(self) -> None:
        with pytest.raises(UnsupportedDecisionError, match="ChoiceDecision"):
            ChoiceCompiler().compile(BoolDecision("Yes or no?"), CAPS)

    def test_missing_capability_rejected(self) -> None:
        with pytest.raises(UnsupportedCapabilityError, match="categorical_token_logits"):
            ChoiceCompiler().compile(make_decision(), BackendCapabilities.none())


class TestCompilerConfiguration:
    def test_default_doctrine(self) -> None:
        compiler = ChoiceCompiler()
        assert compiler.doctrine is CATEGORICAL_SEMANTIC_JUDGMENT_V1

    def test_custom_doctrine(self) -> None:
        doctrine = CategoricalScoringDoctrine(
            doctrine_id="custom-doctrine",
            version=1,
            system_prompt="Custom system prompt.",
            user_template=(
                "Question:\n{question}\n\nContext:\n{context}\n\nMapping:\n{candidate_mapping}\n"
            ),
        )
        compiler = ChoiceCompiler(doctrine=doctrine)
        plan = compiler.compile(make_decision(), CAPS)
        assert plan.doctrine_id == "custom-doctrine"
        assert "A = billing" in plan.prompt

    def test_non_doctrine_rejected(self) -> None:
        bad_doctrine: Any = "not a doctrine"
        with pytest.raises(InvalidDecisionError, match="CategoricalScoringDoctrine"):
            ChoiceCompiler(doctrine=bad_doctrine)


class TestMappingIdentity:
    def test_candidate_mapping_entries_are_frozen(self) -> None:
        plan = ChoiceCompiler().compile(make_decision(), CAPS)
        entry = plan.candidate_mapping[0]
        assert isinstance(entry, CandidateLabelMapping)
        mutable: Any = entry
        with pytest.raises(AttributeError):
            mutable.scoring_label = "Z"

    def test_decision_fingerprint_ignores_mapping_plan_fingerprint_encodes_it(self) -> None:
        # The decision fingerprint never encodes the mapping (the mapping does
        # not exist on the decision); the plan fingerprint is deterministic
        # for the same decision and doctrine.
        decision = make_decision()
        plan = ChoiceCompiler().compile(decision, CAPS)
        assert plan.fingerprint == ChoiceCompiler().compile(decision, CAPS).fingerprint
        assert plan.decision_fingerprint == decision.fingerprint

    def _relabeled_plan(self, labels: tuple[str, ...]) -> InferencePlan:
        decision = make_decision()
        natural = ChoiceCompiler().compile(decision, CAPS)
        mapping_lines = tuple(
            f"{label} = {entry.candidate_name} ({entry.candidate_description})"
            for entry, label in zip(natural.candidate_mapping, labels, strict=True)
        )
        _, prompt = CATEGORICAL_SEMANTIC_JUDGMENT_V1.render(
            question=decision.question,
            context=decision.context,
            mapping_lines=mapping_lines,
        )
        return InferencePlan(
            decision_fingerprint=natural.decision_fingerprint,
            strategy=natural.strategy,
            prompt=prompt,
            decision_family=natural.decision_family,
            targets=labels,
            system_prompt=natural.system_prompt,
            doctrine_id=natural.doctrine_id,
            doctrine_version=natural.doctrine_version,
            label_scheme_id=natural.label_scheme_id,
            candidate_mapping=tuple(
                CandidateLabelMapping(
                    candidate_index=entry.candidate_index,
                    candidate_name=entry.candidate_name,
                    candidate_description=entry.candidate_description,
                    scoring_label=label,
                )
                for entry, label in zip(natural.candidate_mapping, labels, strict=True)
            ),
            required_capabilities=natural.required_capabilities,
            compiler_id=natural.compiler_id,
            compiler_version=natural.compiler_version,
            assembler_id=natural.assembler_id,
            assembler_version=natural.assembler_version,
        )

    def test_permuted_mapping_keeps_decision_fingerprint_and_changes_plan_fingerprint(
        self,
    ) -> None:
        natural = ChoiceCompiler().compile(make_decision(), CAPS)
        permuted = self._relabeled_plan(("C", "A", "B"))
        assert permuted.decision_fingerprint == natural.decision_fingerprint
        assert permuted.fingerprint != natural.fingerprint

    def test_prompt_keeps_semantic_candidate_order_under_permutation(self) -> None:
        natural = ChoiceCompiler().compile(make_decision(), CAPS)
        permuted = self._relabeled_plan(("C", "A", "B"))
        names = ["billing", "shipping", "returns"]
        natural_positions = [natural.prompt.index(name) for name in names]
        permuted_positions = [permuted.prompt.index(name) for name in names]
        assert natural_positions == sorted(natural_positions)
        assert permuted_positions == sorted(permuted_positions)
