"""Tests for Certainty, normalized_entropy, probability_margin, and result objects."""

import math
from typing import Any

import pytest

from probvenance import (
    BoolResult,
    Certainty,
    ChoiceResult,
    InvalidProbabilityError,
    normalized_entropy,
    probability_margin,
)


# INV-05: certainty math (normalized entropy in [0, 1], natural log).
class TestNormalizedEntropy:
    def test_uniform_binary(self) -> None:
        assert normalized_entropy([0.5, 0.5]) == 1.0

    def test_deterministic_binary(self) -> None:
        assert normalized_entropy([1.0, 0.0]) == 0.0

    def test_uniform_four_way(self) -> None:
        assert normalized_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1.0)

    def test_half_concentrated(self) -> None:
        # [0.5, 0.5, 0, 0] has entropy ln(2) / ln(4) = 0.5
        assert normalized_entropy([0.5, 0.5, 0.0, 0.0]) == pytest.approx(0.5)

    def test_single_outcome_is_zero(self) -> None:
        assert normalized_entropy([1.0]) == 0.0

    def test_result_clamped_into_unit_interval(self) -> None:
        value = normalized_entropy([0.5, 0.5])
        assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize(
        "bad",
        [
            [],
            [-0.1, 1.1],
            [0.5, 0.6],
            [0.5, 0.4],
            [1.5, -0.5],
            [float("nan"), 1.0],
            [float("inf"), 0.0],
        ],
    )
    def test_invalid_distributions_rejected(self, bad: list[float]) -> None:
        with pytest.raises(InvalidProbabilityError):
            normalized_entropy(bad)


# INV-05: certainty math (margin = top1 - top2; Bool is [p, 1 - p]).
class TestProbabilityMargin:
    def test_uniform_binary_margin_zero(self) -> None:
        assert probability_margin([0.5, 0.5]) == 0.0

    def test_deterministic_binary_margin_one(self) -> None:
        assert probability_margin([1.0, 0.0]) == 1.0

    def test_top_two_difference(self) -> None:
        assert probability_margin([0.6, 0.3, 0.1]) == pytest.approx(0.3)

    def test_single_outcome_margin_is_one(self) -> None:
        assert probability_margin([1.0]) == 1.0

    def test_bool_distribution_via_p_1_minus_p(self) -> None:
        p = 0.7
        assert probability_margin([p, 1.0 - p]) == pytest.approx(0.4)
        assert normalized_entropy([p, 1.0 - p]) == pytest.approx(
            -(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2)
        )

    def test_deterministic(self) -> None:
        dist = [0.5, 0.3, 0.2]
        assert probability_margin(dist) == probability_margin(dist)


# INV-05: certainty math (concentration = 1 - normalized entropy).
class TestCertainty:
    def test_uniform_binary(self) -> None:
        c = Certainty.from_probabilities([0.5, 0.5])
        assert c.entropy == pytest.approx(1.0)
        assert c.margin == pytest.approx(0.0)

    def test_deterministic(self) -> None:
        c = Certainty.from_probabilities([1.0, 0.0])
        assert c.entropy == 0.0
        assert c.margin == 1.0

    def test_uniform_four_way(self) -> None:
        c = Certainty.from_probabilities([0.25, 0.25, 0.25, 0.25])
        assert c.entropy == pytest.approx(1.0)

    def test_concentration_is_one_minus_entropy(self) -> None:
        c = Certainty.from_probabilities([0.7, 0.3])
        assert c.concentration == pytest.approx(1.0 - c.entropy)

    def test_direct_construction_valid(self) -> None:
        c = Certainty(entropy=0.5, margin=0.2)
        assert c.entropy == 0.5
        assert c.margin == 0.2

    @pytest.mark.parametrize("entropy", [-0.1, 1.1, float("nan"), float("inf")])
    def test_entropy_range_validated(self, entropy: float) -> None:
        with pytest.raises(InvalidProbabilityError):
            Certainty(entropy=entropy, margin=0.0)

    @pytest.mark.parametrize("margin", [-0.1, 1.1, float("nan"), float("inf")])
    def test_margin_range_validated(self, margin: float) -> None:
        with pytest.raises(InvalidProbabilityError):
            Certainty(entropy=0.0, margin=margin)

    def test_frozen(self) -> None:
        c = Certainty(0.5, 0.5)
        mutable: Any = c
        with pytest.raises(AttributeError):
            mutable.entropy = 0.9

    def test_hashable(self) -> None:
        assert hash(Certainty(0.5, 0.5)) == hash(Certainty(0.5, 0.5))


def make_result_certainty() -> Certainty:
    return Certainty(entropy=0.5, margin=0.2)


class TestBoolResult:
    def test_valid_construction(self) -> None:
        r = BoolResult(
            certainty=make_result_certainty(), method="token_logits", probability_true=0.8
        )
        assert r.probability_true == 0.8
        assert r.probability_false == pytest.approx(0.2)

    # INV-01: probability bounds.
    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
    def test_probability_bounds(self, bad: float) -> None:
        with pytest.raises(InvalidProbabilityError):
            BoolResult(certainty=make_result_certainty(), method="m", probability_true=bad)

    # INV-04: uncalibrated means None; predicted_correctness defaults to None.
    def test_defaults_uncalibrated(self) -> None:
        r = BoolResult(certainty=make_result_certainty(), method="m", probability_true=0.5)
        assert r.predicted_correctness is None
        assert r.calibrated is False

    # INV-04: predicted_correctness requires calibrated=True.
    def test_predicted_correctness_requires_calibration(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="calibrated"):
            BoolResult(
                certainty=make_result_certainty(),
                method="m",
                probability_true=0.5,
                predicted_correctness=0.9,
                calibrated=False,
            )

    # INV-04: the calibration gate is not bypassed by a truthy non-bool.
    def test_non_bool_calibrated_rejected(self) -> None:
        truthy: Any = "no"
        with pytest.raises(InvalidProbabilityError, match="calibrated must be a bool"):
            BoolResult(
                certainty=make_result_certainty(),
                method="m",
                probability_true=0.5,
                predicted_correctness=0.9,
                calibrated=truthy,
            )

    def test_calibrated_with_predicted_correctness_allowed(self) -> None:
        r = BoolResult(
            certainty=make_result_certainty(),
            method="m",
            probability_true=0.5,
            predicted_correctness=0.9,
            calibrated=True,
        )
        assert r.predicted_correctness == 0.9
        assert r.calibrated is True

    def test_empty_method_rejected(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="method"):
            BoolResult(certainty=make_result_certainty(), method="  ", probability_true=0.5)

    def test_non_certainty_rejected(self) -> None:
        bad_certainty: Any = "nope"
        with pytest.raises(InvalidProbabilityError, match="Certainty"):
            BoolResult(certainty=bad_certainty, method="m", probability_true=0.5)

    def test_frozen(self) -> None:
        r = BoolResult(certainty=make_result_certainty(), method="m", probability_true=0.5)
        mutable: Any = r
        with pytest.raises(AttributeError):
            mutable.probability_true = 0.9


class TestChoiceResult:
    def make_result(self, **overrides: Any) -> ChoiceResult:
        kwargs: dict[str, Any] = {
            "certainty": make_result_certainty(),
            "method": "m",
            "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2},
        }
        kwargs.update(overrides)
        return ChoiceResult(**kwargs)

    def test_valid_construction(self) -> None:
        r = self.make_result()
        assert r.value == "a"
        assert r.choice_names == ("a", "b", "c")

    def test_value_is_argmax(self) -> None:
        assert self.make_result(probabilities={"a": 0.2, "b": 0.7, "c": 0.1}).value == "b"

    # INV-03: deterministic tie-break; first entry in the original order wins.
    def test_tie_breaks_to_first_in_order(self) -> None:
        r = self.make_result(probabilities={"a": 0.4, "b": 0.4, "c": 0.2})
        assert r.value == "a"
        r2 = self.make_result(probabilities={"b": 0.4, "a": 0.4, "c": 0.2})
        assert r2.value == "b"

    def test_all_equal_first_key_wins(self) -> None:
        r = self.make_result(probabilities={"x": 0.5, "y": 0.5})
        assert r.value == "x"

    def test_value_always_one_of_keys(self) -> None:
        r = self.make_result()
        assert r.value in r.probabilities

    def test_key_order_preserved(self) -> None:
        r = self.make_result(probabilities={"z": 0.5, "a": 0.5})
        assert r.choice_names == ("z", "a")

    def test_probabilities_snapshot_isolated(self) -> None:
        caller: dict[str, float] = {"a": 0.6, "b": 0.4}
        r = self.make_result(probabilities=caller)
        caller["a"] = 0.0
        caller["c"] = 1.0
        assert r.probabilities == {"a": 0.6, "b": 0.4}
        assert r.value == "a"

    def test_fewer_than_two_entries_rejected(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="at least 2"):
            self.make_result(probabilities={"a": 1.0})
        with pytest.raises(InvalidProbabilityError, match="at least 2"):
            self.make_result(probabilities={})

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="non-empty str"):
            self.make_result(probabilities={"": 0.5, "b": 0.5})

    # INV-01/INV-02: probability bounds and normalization (sum to 1).
    @pytest.mark.parametrize(
        "bad",
        [
            {"a": -0.5, "b": 1.5},
            {"a": 0.5, "b": 0.6},
            {"a": 0.5, "b": 0.4},
            {"a": float("nan"), "b": 1.0},
        ],
    )
    def test_invalid_probability_mappings_rejected(self, bad: dict[str, float]) -> None:
        with pytest.raises(InvalidProbabilityError):
            self.make_result(probabilities=bad)

    # INV-04: predicted_correctness requires calibrated=True.
    def test_predicted_correctness_requires_calibration(self) -> None:
        with pytest.raises(InvalidProbabilityError, match="calibrated"):
            self.make_result(predicted_correctness=0.9, calibrated=False)

    def test_not_hashable(self) -> None:
        r = self.make_result()
        with pytest.raises(TypeError):
            hash(r)

    def test_equality_by_value(self) -> None:
        assert self.make_result() == self.make_result()

    def test_frozen(self) -> None:
        r = self.make_result()
        mutable: Any = r
        with pytest.raises(AttributeError):
            mutable.method = "other"
