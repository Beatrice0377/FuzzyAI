"""Tests for the ProbabilityAssembler: RawEvidence -> uncalibrated BoolResult."""

import math

import pytest

from probvenance import (
    BINARY_EVIDENCE_LABELS,
    EvidenceKind,
    InvalidDecisionError,
    RawEvidence,
    assemble_bool_probability,
    normalized_entropy,
    probability_margin,
)


def make_binary_evidence(
    values: tuple[float, ...], labels: tuple[str, ...] = BINARY_EVIDENCE_LABELS
) -> RawEvidence:
    return RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=labels,
        values=values,
        plan_fingerprint="b" * 64,
    )


class TestSoftmaxMatrix:
    def test_equal_logits_is_half(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, 0.0)))
        assert result.probability_true == pytest.approx(0.5)

    def test_ln3_gap_is_three_to_one(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, math.log(3.0))))
        assert result.probability_true == pytest.approx(0.75)

    def test_large_equal_logits_no_overflow(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((1000.0, 1000.0)))
        assert result.probability_true == pytest.approx(0.5)
        assert math.isfinite(result.probability_true)
        assert math.isfinite(result.certainty.entropy)
        assert math.isfinite(result.certainty.margin)

    def test_minus_thousand_true_gap(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((-1000.0, 1000.0)))
        assert result.probability_true == pytest.approx(1.0)

    def test_plus_thousand_false_gap(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((1000.0, -1000.0)))
        assert result.probability_true == pytest.approx(0.0)

    def test_label_order_true_false_maps_by_label(self) -> None:
        result = assemble_bool_probability(
            make_binary_evidence((0.0, math.log(3.0)), labels=("true", "false"))
        )
        assert result.probability_true == pytest.approx(0.25)


class TestCertaintyAgreement:
    @pytest.mark.parametrize(
        "values",
        [(0.0, 0.0), (0.0, math.log(3.0)), (-1000.0, 1000.0), (2.5, -1.25)],
    )
    def test_entropy_and_margin_match_definitions(self, values: tuple[float, float]) -> None:
        result = assemble_bool_probability(make_binary_evidence(values))
        p = result.probability_true
        assert result.certainty.entropy == pytest.approx(normalized_entropy([p, 1.0 - p]))
        assert result.certainty.margin == pytest.approx(probability_margin([p, 1.0 - p]))

    def test_uniform_evidence_has_max_entropy_zero_margin(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, 0.0)))
        assert result.certainty.entropy == pytest.approx(1.0)
        assert result.certainty.margin == pytest.approx(0.0)

    def test_extreme_evidence_has_min_entropy_unit_margin(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((-1000.0, 1000.0)))
        assert result.certainty.entropy == pytest.approx(0.0, abs=1e-9)
        assert result.certainty.margin == pytest.approx(1.0)


class TestAssemblerValidation:
    def test_wrong_kind_rejected(self) -> None:
        evidence = RawEvidence(
            kind=EvidenceKind.LOGPROBS, labels=("false", "true"), values=(0.0, 0.0)
        )
        with pytest.raises(InvalidDecisionError, match="logits"):
            assemble_bool_probability(evidence)

    def test_wrong_label_set_rejected(self) -> None:
        evidence = RawEvidence(kind=EvidenceKind.LOGITS, labels=("yes", "no"), values=(0.0, 0.0))
        with pytest.raises(InvalidDecisionError, match="labels"):
            assemble_bool_probability(evidence)

    def test_wrong_label_count_rejected(self) -> None:
        evidence = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=("false", "true", "maybe"),
            values=(0.0, 0.0, 1.0),
        )
        with pytest.raises(InvalidDecisionError, match="labels"):
            assemble_bool_probability(evidence)

    def test_reversed_binary_labels_accepted(self) -> None:
        result = assemble_bool_probability(
            make_binary_evidence((1.0, 1.0), labels=("true", "false"))
        )
        assert result.probability_true == pytest.approx(0.5)


class TestUncalibratedDefaults:
    def test_extreme_logits_stay_uncalibrated(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((-1000.0, 1000.0)))
        assert result.calibrated is False
        assert result.predicted_correctness is None

    def test_method_default(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, 0.0)))
        assert result.method == "binary_token_logits"

    def test_method_override(self) -> None:
        result = assemble_bool_probability(
            make_binary_evidence((0.0, 0.0)), method="binary_token_logits:v2"
        )
        assert result.method == "binary_token_logits:v2"

    def test_trace_id_default_none(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, 0.0)))
        assert result.trace_id is None

    def test_trace_id_stamped(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, 0.0)), trace_id="trace-1")
        assert result.trace_id == "trace-1"

    def test_probability_false_is_complement(self) -> None:
        result = assemble_bool_probability(make_binary_evidence((0.0, math.log(3.0))))
        assert result.probability_false == pytest.approx(0.25)
