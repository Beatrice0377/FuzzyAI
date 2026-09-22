"""Tests for assemble_choice_probability: categorical RawEvidence -> ChoiceResult."""

import math
from typing import Any

import pytest

from fuzzyai import (
    BINARY_EVIDENCE_LABELS,
    BackendCapabilities,
    BoolCompiler,
    BoolDecision,
    Certainty,
    Choice,
    ChoiceCompiler,
    ChoiceDecision,
    ChoiceResult,
    ChoiceScoringDiagnostics,
    EvidenceKind,
    InferencePlan,
    InvalidDecisionError,
    RawEvidence,
    ScoringStrategy,
    assemble_choice_probability,
    normalized_entropy,
    probability_margin,
    scoring_label_mass,
)

CAPS = BackendCapabilities(categorical_token_logits=True)


def _logsumexp(values: tuple[float, ...]) -> float:
    """Stable log-sum-exp, spelled out (the stdlib has no logaddexp)."""
    shift = max(values)
    return shift + math.log(sum(math.exp(value - shift) for value in values))


def _honest_metadata(values: tuple[float, ...]) -> dict[str, Any]:
    """Normalization statistics a truthful backend would report.

    The full vocabulary is the three candidate logits plus three dummy tokens
    at logit 0, so every derived probability is self-consistent.
    """
    return {
        "vocab_logsumexp": _logsumexp((*values, 0.0, 0.0, 0.0)),
        "top_token_id": 9642,
        "top_token_logit": max(values),
    }


def make_plan(**overrides: Any) -> InferencePlan:
    decision = ChoiceDecision(
        "Which department?",
        choices={"billing": "money", "shipping": "boxes", "returns": "warranty"},
    )
    plan = ChoiceCompiler().compile(decision, CAPS)
    assert plan.targets == ("A", "B", "C")
    return plan


def assemble(
    values: tuple[float, ...],
    *,
    labels: tuple[str, ...] | None = None,
    plan: InferencePlan | None = None,
    metadata: dict[str, Any] | None = None,
    **call_overrides: Any,
) -> tuple[ChoiceResult, Any]:
    active_plan = plan if plan is not None else make_plan()
    evidence = RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=labels if labels is not None else active_plan.targets,
        values=values,
        plan_fingerprint=active_plan.fingerprint,
        metadata=metadata if metadata is not None else _honest_metadata(values),
    )
    return assemble_choice_probability(evidence, plan=active_plan, **call_overrides)


class TestNWaySoftmaxMath:
    def test_equal_logits_are_uniform_thirds(self) -> None:
        result, _ = assemble((0.0, 0.0, 0.0))
        assert result.probabilities["billing"] == pytest.approx(1.0 / 3.0)
        assert result.probabilities["shipping"] == pytest.approx(1.0 / 3.0)
        assert result.probabilities["returns"] == pytest.approx(1.0 / 3.0)

    def test_ln2_gap_is_two_to_one_to_one(self) -> None:
        result, _ = assemble((math.log(2.0), 0.0, 0.0))
        assert result.probabilities["billing"] == pytest.approx(0.5)
        assert result.probabilities["shipping"] == pytest.approx(0.25)
        assert result.probabilities["returns"] == pytest.approx(0.25)

    def test_very_large_equal_logits_stay_uniform(self) -> None:
        result, _ = assemble((1000.0, 1000.0, 1000.0))
        for probability in result.probabilities.values():
            assert probability == pytest.approx(1.0 / 3.0)
            assert math.isfinite(probability)

    def test_one_dominant_logit_absorbs_mass(self) -> None:
        result, _ = assemble((1000.0, 0.0, 0.0))
        assert result.probabilities["billing"] == pytest.approx(1.0, abs=1e-12)
        assert result.probabilities["shipping"] == pytest.approx(0.0, abs=1e-12)
        assert result.probabilities["returns"] == pytest.approx(0.0, abs=1e-12)

    def test_five_way_uniform(self) -> None:
        decision = ChoiceDecision(
            "Pick one of five.",
            choices=[Choice(name=f"c{i}") for i in range(5)],
        )
        plan = ChoiceCompiler().compile(decision, CAPS)
        evidence = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=plan.targets,
            values=(0.0, 0.0, 0.0, 0.0, 0.0),
            plan_fingerprint=plan.fingerprint,
            metadata=_honest_metadata((0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        result, _ = assemble_choice_probability(evidence, plan=plan)
        assert len(result.probabilities) == 5
        for probability in result.probabilities.values():
            assert probability == pytest.approx(0.2)


class TestSemanticKeying:
    def test_probabilities_keyed_by_semantic_names_never_labels(self) -> None:
        result, _ = assemble((1.0, 2.0, 3.0))
        assert list(result.probabilities.keys()) == ["billing", "shipping", "returns"]
        assert "A" not in result.probabilities
        assert "B" not in result.probabilities
        assert "C" not in result.probabilities

    def test_semantic_mapping_follows_candidate_order_not_label_order(self) -> None:
        # The mapping, not the label, decides which semantic name gets which
        # probability: labels are execution-only.
        plan = make_plan()
        assert plan.candidate_mapping[0].candidate_name == "billing"
        result, _ = assemble((3.0, 1.0, 2.0), plan=plan)
        assert result.probabilities["billing"] > result.probabilities["shipping"]
        assert result.value == "billing"

    def test_result_is_choice_result(self) -> None:
        result, _ = assemble((0.0, 0.0, 0.0))
        assert isinstance(result, ChoiceResult)


class TestTieBreakFollowsSemanticOrder:
    def _relabeled_plan(self, labels: tuple[str, ...]) -> InferencePlan:
        base = make_plan()
        return InferencePlan(
            decision_fingerprint=base.decision_fingerprint,
            strategy=base.strategy,
            prompt=base.prompt,
            targets=labels,
            system_prompt=base.system_prompt,
            doctrine_id=base.doctrine_id,
            label_scheme_id=base.label_scheme_id,
            candidate_mapping=tuple(
                type(entry)(
                    candidate_index=entry.candidate_index,
                    candidate_name=entry.candidate_name,
                    candidate_description=entry.candidate_description,
                    scoring_label=new_label,
                )
                for entry, new_label in zip(base.candidate_mapping, labels, strict=True)
            ),
            required_capabilities=base.required_capabilities,
        )

    def test_tie_break_first_candidate_wins_under_natural_mapping(self) -> None:
        # billing -> A, shipping -> B: semantic order and label order agree.
        result, _ = assemble((0.0, 0.0, -1000.0))
        assert result.probabilities["billing"] == pytest.approx(result.probabilities["shipping"])
        assert result.probabilities["returns"] == pytest.approx(0.0)
        assert result.value == "billing"

    def test_tie_break_first_candidate_wins_under_permuted_mapping(self) -> None:
        # billing -> C, shipping -> A: label order disagrees with candidate
        # order, yet the tie-break must still follow candidate order.
        plan = self._relabeled_plan(("C", "A", "B"))
        result, _ = assemble((0.0, 0.0, -1000.0), labels=("C", "A", "B"), plan=plan)
        assert result.value == "billing"
        assert result.probabilities["billing"] == pytest.approx(result.probabilities["shipping"])
        assert result.choice_names == ("billing", "shipping", "returns")


class TestOrderedEvidenceEnforcement:
    def test_permuted_labels_rejected_even_with_legal_pairs(self) -> None:
        # ("B", "A", "C") with values (1.0, 2.0, 3.0) describes the same
        # (label, logit) pairs as the ordered version, but the order is wrong,
        # so it MUST raise.
        with pytest.raises(InvalidDecisionError, match="exactly, in the same order"):
            assemble((1.0, 2.0, 3.0), labels=("B", "A", "C"))

    def test_wrong_labels_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="exactly, in the same order"):
            assemble((1.0, 2.0, 3.0), labels=("X", "Y", "Z"))

    def test_wrong_label_count_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="exactly, in the same order"):
            assemble((1.0, 2.0), labels=("A", "B"))

    def test_wrong_kind_rejected(self) -> None:
        evidence = RawEvidence(
            kind=EvidenceKind.LOGPROBS,
            labels=make_plan().targets,
            values=(1.0, 2.0, 3.0),
        )
        with pytest.raises(InvalidDecisionError, match="logits"):
            assemble_choice_probability(evidence, plan=make_plan())

    def test_binary_strategy_plan_rejected(self) -> None:
        binary_plan = BoolCompiler().compile(
            BoolDecision("Yes or no?"), BackendCapabilities(binary_token_logits=True)
        )
        binary_evidence = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=BINARY_EVIDENCE_LABELS,
            values=(0.0, 1.0),
            plan_fingerprint=binary_plan.fingerprint,
        )
        with pytest.raises(InvalidDecisionError, match="categorical_token_logits"):
            assemble_choice_probability(binary_evidence, plan=binary_plan)


class TestDiagnosticsReturned:
    def test_uniform_vocab_diagnostics(self) -> None:
        # Uniform 6-token vocabulary: every candidate token has probability
        # 1/6 and the three candidates hold 3/6 = 0.5 of the mass.
        metadata = {"vocab_logsumexp": math.log(6.0), "top_token_id": 7, "top_token_logit": 0.0}
        result, diagnostics = assemble((0.0, 0.0, 0.0), metadata=metadata)
        assert diagnostics.scoring_label_mass == pytest.approx(0.5)
        for probability in diagnostics.scoring_label_token_probabilities:
            assert probability == pytest.approx(1.0 / 6.0)
        assert diagnostics.top_token_id == 7
        assert diagnostics.top_token_probability == pytest.approx(1.0 / 6.0)
        assert result.certainty.entropy == pytest.approx(1.0)
        assert result.certainty.margin == pytest.approx(0.0)

    def test_renamed_fields_present_old_candidate_fields_gone(self) -> None:
        # Phase 2B.1 renamed candidate_mass -> scoring_label_mass and
        # candidate_token_probabilities -> scoring_label_token_probabilities:
        # the old attribute names must not survive on the diagnostics object.
        diagnostics = ChoiceScoringDiagnostics(
            scoring_label_mass=0.75,
            scoring_label_token_probabilities=(0.25, 0.25, 0.25),
            top_token_id=3,
            top_token_probability=0.25,
        )
        assert not hasattr(diagnostics, "candidate_mass")
        assert not hasattr(diagnostics, "candidate_token_probabilities")
        assert diagnostics.scoring_label_mass == 0.75
        assert diagnostics.scoring_label_token_probabilities == (0.25, 0.25, 0.25)


class TestCertaintyAndDefaults:
    @pytest.mark.parametrize(
        "values",
        [(0.0, 0.0, 0.0), (math.log(2.0), 0.0, 0.0), (-1000.0, 1000.0, 0.0)],
    )
    def test_entropy_and_margin_match_definitions(self, values: tuple[float, ...]) -> None:
        result, _ = assemble(values)
        probabilities = list(result.probabilities.values())
        assert result.certainty.entropy == pytest.approx(normalized_entropy(probabilities))
        assert result.certainty.margin == pytest.approx(probability_margin(probabilities))
        assert isinstance(result.certainty, Certainty)

    def test_uncalibrated_defaults(self) -> None:
        result, _ = assemble((0.0, 0.0, 0.0))
        assert result.calibrated is False
        assert result.predicted_correctness is None

    def test_method_default_and_override(self) -> None:
        result, _ = assemble((0.0, 0.0, 0.0))
        assert result.method == ScoringStrategy.CATEGORICAL_TOKEN_LOGITS.value
        overridden, _ = assemble((0.0, 0.0, 0.0), method="categorical_token_logits:v2")
        assert overridden.method == "categorical_token_logits:v2"

    def test_trace_id_default_and_stamped(self) -> None:
        result, _ = assemble((0.0, 0.0, 0.0))
        assert result.trace_id is None
        stamped, _ = assemble((0.0, 0.0, 0.0), trace_id="trace-1")
        assert stamped.trace_id == "trace-1"


class TestScoringLabelMassMath:
    """scoring_label_mass is a full-vocabulary quantity, not a candidate-set one."""

    def test_three_candidates_of_four_uniform_tokens(self) -> None:
        assert scoring_label_mass((0.0, 0.0, 0.0), vocab_logsumexp=math.log(4.0)) == pytest.approx(
            0.75
        )

    def test_three_candidates_of_five_uniform_tokens(self) -> None:
        assert scoring_label_mass((0.0, 0.0, 0.0), vocab_logsumexp=math.log(5.0)) == pytest.approx(
            0.6
        )

    def test_mass_falls_as_the_vocabulary_grows(self) -> None:
        four = scoring_label_mass((0.0, 0.0, 0.0), vocab_logsumexp=math.log(4.0))
        five = scoring_label_mass((0.0, 0.0, 0.0), vocab_logsumexp=math.log(5.0))
        assert four > five

    def test_all_logits_equal_to_the_vocabulary_is_full_mass(self) -> None:
        assert scoring_label_mass((0.0, 0.0, 0.0), vocab_logsumexp=math.log(3.0)) == pytest.approx(
            1.0
        )
