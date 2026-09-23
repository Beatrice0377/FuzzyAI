"""Tests for the scoring diagnostics: full-vocabulary normalization math.

These pin the mathematical contract of ``verbalizer_mass`` against small,
hand-computable vocab logits, including the extreme-value cases where a naive
``exp`` would overflow or underflow.
"""

import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from probvenance import (
    BINARY_EVIDENCE_LABELS,
    EvidenceKind,
    InvalidDecisionError,
    InvalidProbabilityError,
    RawEvidence,
    ScoringDiagnostics,
    diagnose_bool_evidence,
)
from probvenance.diagnostics import (
    TOP_TOKEN_ID_KEY,
    TOP_TOKEN_LOGIT_KEY,
    TOP_TOKEN_TEXT_KEY,
    VOCAB_LOGSUMEXP_KEY,
    full_vocab_probability,
    log_verbalizer_mass,
    verbalizer_mass,
)

LOGIT_FALSE = 0.0
LOGIT_TRUE = math.log(3.0)


def logsumexp(*logits: float) -> float:
    """Reference full-vocabulary normalizer for small, safe logits."""
    largest = max(logits)
    return largest + math.log(sum(math.exp(logit - largest) for logit in logits))


def default_metadata() -> dict[str, Any]:
    return {
        "positive_token_id": 10,
        "negative_token_id": 11,
        VOCAB_LOGSUMEXP_KEY: logsumexp(LOGIT_FALSE, LOGIT_TRUE),
        TOP_TOKEN_ID_KEY: 10,
        TOP_TOKEN_LOGIT_KEY: LOGIT_TRUE,
    }


def make_evidence(**overrides: Any) -> RawEvidence:
    metadata = default_metadata()
    extra = overrides.pop("metadata", None)
    if isinstance(extra, dict):
        metadata.update(extra)
    kwargs: dict[str, Any] = {
        "kind": EvidenceKind.LOGITS,
        "labels": BINARY_EVIDENCE_LABELS,
        "values": (LOGIT_FALSE, LOGIT_TRUE),
        "metadata": metadata,
    }
    kwargs.update(overrides)
    return RawEvidence(**kwargs)


class TestVerbalizerMass:
    def test_uniform_four_token_vocab_gives_half(self) -> None:
        # Four equally likely tokens, two of which are the verbalizers.
        # INV-18: the mass is a share of the FULL vocabulary, not a renormalized
        # two-way split, so two of four candidates is exactly one half.
        assert verbalizer_mass(
            logit_true=0.0,
            logit_false=0.0,
            vocab_logsumexp=logsumexp(0.0, 0.0, 0.0, 0.0),
        ) == pytest.approx(0.5)

    def test_verbalizers_are_whole_vocabulary_gives_one(self) -> None:
        assert verbalizer_mass(
            logit_true=LOGIT_TRUE,
            logit_false=LOGIT_FALSE,
            vocab_logsumexp=logsumexp(LOGIT_FALSE, LOGIT_TRUE),
        ) == pytest.approx(1.0)

    def test_numerically_large_logits_do_not_overflow(self) -> None:
        assert verbalizer_mass(
            logit_true=1000.0,
            logit_false=-1000.0,
            vocab_logsumexp=1000.0,
        ) == pytest.approx(1.0)

    def test_both_verbalizers_in_the_tail_stay_tiny(self) -> None:
        mass = verbalizer_mass(logit_true=-100.0, logit_false=-100.0, vocab_logsumexp=0.0)
        assert 0.0 < mass < 1e-40

    def test_equal_verbalizers_share_the_mass_evenly(self) -> None:
        mass = verbalizer_mass(
            logit_true=0.0,
            logit_false=0.0,
            vocab_logsumexp=logsumexp(0.0, 0.0, 5.0),
        )
        assert mass == pytest.approx(math.exp(0.0) * 2 / (1 + 1 + math.exp(5.0)))

    def test_log_form_agrees_with_probability_form(self) -> None:
        # INV-18: the log form and the probability form are the same quantity.
        log_mass = log_verbalizer_mass(
            logit_true=LOGIT_TRUE,
            logit_false=LOGIT_FALSE,
            vocab_logsumexp=logsumexp(LOGIT_FALSE, LOGIT_TRUE, 2.0),
        )
        assert math.exp(log_mass) == pytest.approx(
            verbalizer_mass(
                logit_true=LOGIT_TRUE,
                logit_false=LOGIT_FALSE,
                vocab_logsumexp=logsumexp(LOGIT_FALSE, LOGIT_TRUE, 2.0),
            )
        )


class TestFullVocabProbability:
    def test_uniform_vocab_gives_reciprocal_of_size(self) -> None:
        assert full_vocab_probability(
            logit=0.0,
            vocab_logsumexp=logsumexp(0.0, 0.0, 0.0, 0.0),
        ) == pytest.approx(0.25)

    def test_rounding_level_overshoot_is_clamped(self) -> None:
        assert full_vocab_probability(logit=1e-12, vocab_logsumexp=0.0) == 1.0

    def test_measured_float32_overshoot_is_clamped(self) -> None:
        # A real bfloat16 run produced this exact overshoot; an absolute 1e-9
        # bound rejected it and lost 62 probe records across two models.
        assert (
            full_vocab_probability(
                logit=3.0,
                vocab_logsumexp=3.0 - 1.8553912184415822e-07,
            )
            == 1.0
        )

    def test_materially_inconsistent_normalizer_is_rejected(self) -> None:
        assert math.exp(1.0) > 1.0 + 1e-9
        with pytest.raises(InvalidProbabilityError, match="inconsistent"):
            full_vocab_probability(logit=1.0, vocab_logsumexp=0.0)

    def test_tail_token_is_effectively_zero(self) -> None:
        assert full_vocab_probability(logit=-1000.0, vocab_logsumexp=0.0) == pytest.approx(0.0)


class TestDiagnoseBoolEvidence:
    def test_matches_hand_computed_values(self) -> None:
        diagnostics = diagnose_bool_evidence(make_evidence())
        assert diagnostics.verbalizer_mass == pytest.approx(1.0)
        assert diagnostics.top_token_id == 10
        assert diagnostics.top_token_probability == pytest.approx(3 / 4)
        assert diagnostics.positive_token_probability == pytest.approx(3 / 4)
        assert diagnostics.negative_token_probability == pytest.approx(1 / 4)

    def test_top_token_text_is_optional(self) -> None:
        assert diagnose_bool_evidence(make_evidence()).top_token_text is None
        with_text = make_evidence(metadata={TOP_TOKEN_TEXT_KEY: "yes"})
        assert diagnose_bool_evidence(with_text).top_token_text == "yes"

    @pytest.mark.parametrize(
        "key",
        [VOCAB_LOGSUMEXP_KEY, TOP_TOKEN_ID_KEY, TOP_TOKEN_LOGIT_KEY],
    )
    def test_missing_required_metadata_key_rejected(self, key: str) -> None:
        metadata = {k: v for k, v in default_metadata().items() if k != key}
        evidence = RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=BINARY_EVIDENCE_LABELS,
            values=(LOGIT_FALSE, LOGIT_TRUE),
            metadata=metadata,
        )
        with pytest.raises(InvalidDecisionError, match=key):
            diagnose_bool_evidence(evidence)

    @pytest.mark.parametrize("bad", ["3.0", None, [3.0]])
    def test_non_numeric_normalizer_rejected(self, bad: object) -> None:
        with pytest.raises(InvalidDecisionError, match=VOCAB_LOGSUMEXP_KEY):
            diagnose_bool_evidence(make_evidence(metadata={VOCAB_LOGSUMEXP_KEY: bad}))

    def test_bool_normalizer_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match=VOCAB_LOGSUMEXP_KEY):
            diagnose_bool_evidence(make_evidence(metadata={VOCAB_LOGSUMEXP_KEY: True}))

    @pytest.mark.parametrize("bad", ["7", None, 7.5])
    def test_non_int_top_token_id_rejected(self, bad: object) -> None:
        with pytest.raises(InvalidDecisionError, match=TOP_TOKEN_ID_KEY):
            diagnose_bool_evidence(make_evidence(metadata={TOP_TOKEN_ID_KEY: bad}))

    def test_non_str_top_token_text_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match=TOP_TOKEN_TEXT_KEY):
            diagnose_bool_evidence(make_evidence(metadata={TOP_TOKEN_TEXT_KEY: 7}))

    def test_non_logits_evidence_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="logits"):
            diagnose_bool_evidence(make_evidence(kind=EvidenceKind.LOGPROBS))

    def test_wrong_label_set_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="labels"):
            diagnose_bool_evidence(make_evidence(labels=("low", "high"), values=(0.0, 1.0)))


class TestScoringDiagnosticsObject:
    def test_is_frozen(self) -> None:
        diagnostics = ScoringDiagnostics(
            verbalizer_mass=0.5,
            top_token_id=1,
            top_token_probability=0.5,
            positive_token_probability=0.25,
            negative_token_probability=0.25,
        )
        attribute = "verbalizer_mass"
        with pytest.raises(FrozenInstanceError):
            setattr(diagnostics, attribute, 0.9)

    def test_top_token_probability_must_be_the_largest(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="top_token_probability"):
            ScoringDiagnostics(
                verbalizer_mass=0.9,
                top_token_id=1,
                top_token_probability=0.1,
                positive_token_probability=0.8,
                negative_token_probability=0.1,
            )

    def test_mass_must_cover_the_verbalizers(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="verbalizer_mass"):
            ScoringDiagnostics(
                verbalizer_mass=0.1,
                top_token_id=1,
                top_token_probability=0.9,
                positive_token_probability=0.8,
                negative_token_probability=0.1,
            )

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_probability_out_of_range_rejected(self, bad: float) -> None:
        with pytest.raises(InvalidProbabilityError):
            ScoringDiagnostics(
                verbalizer_mass=bad,
                top_token_id=1,
                top_token_probability=0.5,
                positive_token_probability=0.25,
                negative_token_probability=0.25,
            )

    @pytest.mark.parametrize("bad", [True, -1, 1.5, "1"])
    def test_top_token_id_must_be_a_non_negative_int(self, bad: Any) -> None:
        with pytest.raises(InvalidDecisionError, match="top_token_id"):
            ScoringDiagnostics(
                verbalizer_mass=0.5,
                top_token_id=bad,
                top_token_probability=0.5,
                positive_token_probability=0.25,
                negative_token_probability=0.25,
            )
