"""Tests for BoolDecision and ChoiceDecision."""

from typing import Any

import pytest

from fuzzyai import BoolDecision, Choice, ChoiceDecision, InvalidDecisionError, JSONValue


class TestBoolDecision:
    def test_valid_construction_positional_context(self) -> None:
        d = BoolDecision("Is the invoice overdue?", {"id": 7})
        assert d.question == "Is the invoice overdue?"
        assert d.context == {"id": 7}

    def test_valid_construction_keyword_context(self) -> None:
        d = BoolDecision(question="q", context={"a": 1})
        assert d.context == {"a": 1}

    def test_default_context_is_none(self) -> None:
        d = BoolDecision("q")
        assert d.context is None

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_question_rejected(self, bad: str) -> None:
        with pytest.raises(InvalidDecisionError, match="non-empty"):
            BoolDecision(bad)

    @pytest.mark.parametrize("bad", [None, 42, True, ["q"]])
    def test_non_str_question_rejected(self, bad: Any) -> None:
        with pytest.raises(InvalidDecisionError, match="question must be str"):
            BoolDecision(bad)

    @pytest.mark.parametrize(
        "bad_context",
        [
            {"s": {1, 2}},
            {"t": (1, 2)},
            {"f": lambda: None},
            {"o": object()},
            float("nan"),
            [float("inf")],
            {"x": float("-inf")},
        ],
    )
    def test_invalid_context_rejected(self, bad_context: Any) -> None:
        with pytest.raises(InvalidDecisionError, match="JSON-compatible"):
            BoolDecision("q", bad_context)

    def test_context_isolation_from_caller_mutation(self) -> None:
        inner: list[JSONValue] = [1, 2]
        context: dict[str, JSONValue] = {"a": inner}
        d = BoolDecision("q", context)
        fp = d.fingerprint
        inner.append(3)
        context["b"] = "new"
        assert d.fingerprint == fp
        assert d.context == {"a": [1, 2]}

    def test_context_returns_fresh_value_each_access(self) -> None:
        d = BoolDecision("q", {"a": 1})
        first = d.context
        assert isinstance(first, dict)
        first["a"] = 999
        assert d.context == {"a": 1}

    def test_fingerprint_is_64_hex_and_stable(self) -> None:
        d1 = BoolDecision("q", {"a": 1, "b": 2})
        d2 = BoolDecision("q", {"b": 2, "a": 1})
        assert len(d1.fingerprint) == 64
        assert d1.fingerprint == d2.fingerprint

    def test_different_question_changes_fingerprint(self) -> None:
        assert BoolDecision("q1").fingerprint != BoolDecision("q2").fingerprint

    def test_different_context_changes_fingerprint(self) -> None:
        assert BoolDecision("q", {"a": 1}).fingerprint != BoolDecision("q", {"a": 2}).fingerprint

    def test_different_kind_changes_fingerprint(self) -> None:
        choice = ChoiceDecision("q", choices={"a": "A", "b": "B"})
        assert BoolDecision("q").fingerprint != choice.fingerprint

    def test_frozen(self) -> None:
        d = BoolDecision("q")
        mutable: Any = d
        with pytest.raises(AttributeError):
            mutable.question = "other"

    def test_repr_shows_live_context(self) -> None:
        d = BoolDecision("q", {"a": 1})
        assert "context={'a': 1}" in repr(d)
        assert "_context_json" not in repr(d)

    def test_hashable_and_equal(self) -> None:
        assert BoolDecision("q", {"a": 1}) == BoolDecision("q", {"a": 1})
        assert hash(BoolDecision("q", {"a": 1})) == hash(BoolDecision("q", {"a": 1}))


class TestChoiceDecision:
    def test_contract_example_call(self) -> None:
        d = ChoiceDecision(
            question="Which team owns this ticket?",
            context={"id": 1},
            choices={
                "billing": "Billing and payment issues",
                "shipping": "Shipping and logistics",
                "returns": "Returns and refunds",
            },
        )
        assert d.question == "Which team owns this ticket?"
        assert d.context == {"id": 1}
        assert d.choice_names == ("billing", "shipping", "returns")
        assert d.choices[0].description == "Billing and payment issues"

    def test_fewer_than_two_choices_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="at least 2"):
            ChoiceDecision("q", choices={"a": "A"})
        with pytest.raises(InvalidDecisionError, match="at least 2"):
            ChoiceDecision("q", choices={})

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="non-empty"):
            ChoiceDecision("q", choices={"": "A", "b": "B"})

    def test_duplicate_names_rejected(self) -> None:
        # A Mapping literal cannot carry duplicate keys (dict collapses them),
        # so duplicates are exercised through Sequence[Choice].
        with pytest.raises(InvalidDecisionError, match="unique"):
            ChoiceDecision("q", choices=[Choice("a", "A1"), Choice("a", "A2"), Choice("b", "B")])

    def test_non_str_description_in_mapping_rejected(self) -> None:
        bad_choices: Any = {"a": 1, "b": "B"}
        with pytest.raises(InvalidDecisionError, match="must be str"):
            ChoiceDecision("q", choices=bad_choices)

    def test_sequence_of_choices_accepted(self) -> None:
        d = ChoiceDecision("q", choices=[Choice("a", "A"), Choice("b")])
        assert d.choice_names == ("a", "b")
        assert d.choices[1].description is None

    def test_sequence_with_non_choice_rejected(self) -> None:
        bad_choices: Any = ["a", "b"]
        with pytest.raises(InvalidDecisionError, match="Choice"):
            ChoiceDecision("q", choices=bad_choices)

    def test_non_sequence_choices_rejected(self) -> None:
        bad_choices: Any = 5
        with pytest.raises(InvalidDecisionError, match="mapping or a sequence"):
            ChoiceDecision("q", choices=bad_choices)

    # INV-13: explicit choice order is semantics-bearing and preserved.
    def test_choice_order_preserved(self) -> None:
        d = ChoiceDecision("q", choices={"z": "Z", "a": "A", "m": "M"})
        assert d.choice_names == ("z", "a", "m")

    # INV-13: choice order enters the fingerprint.
    def test_fingerprint_order_sensitive_for_choices(self) -> None:
        d1 = ChoiceDecision("q", choices={"a": "A", "b": "B"})
        d2 = ChoiceDecision("q", choices={"b": "B", "a": "A"})
        assert d1.fingerprint != d2.fingerprint
        assert d1.choice_names == ("a", "b")
        assert d2.choice_names == ("b", "a")

    # INV-10: context dict insertion order does not affect the fingerprint.
    def test_fingerprint_dict_order_independent_for_context(self) -> None:
        d1 = ChoiceDecision("q", context={"a": 1, "b": 2}, choices={"x": "X", "y": "Y"})
        d2 = ChoiceDecision("q", context={"b": 2, "a": 1}, choices={"x": "X", "y": "Y"})
        assert d1.fingerprint == d2.fingerprint

    def test_fingerprint_stable_and_64_hex(self) -> None:
        d1 = ChoiceDecision("q", choices={"a": "A", "b": "B"})
        d2 = ChoiceDecision("q", choices={"a": "A", "b": "B"})
        assert d1.fingerprint == d2.fingerprint
        assert len(d1.fingerprint) == 64

    def test_invalid_question_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError, match="non-empty"):
            ChoiceDecision("  ", choices={"a": "A", "b": "B"})

    def test_invalid_context_rejected(self) -> None:
        bad_context: Any = (1, 2)
        with pytest.raises(InvalidDecisionError, match="JSON-compatible"):
            ChoiceDecision("q", context=bad_context, choices={"a": "A", "b": "B"})

    def test_frozen(self) -> None:
        d = ChoiceDecision("q", choices={"a": "A", "b": "B"})
        mutable: Any = d
        with pytest.raises(AttributeError):
            mutable.question = "other"

    def test_repr_shows_live_context(self) -> None:
        d = ChoiceDecision("q", context={"a": 1}, choices={"a": "A", "b": "B"})
        assert "context={'a': 1}" in repr(d)
        assert "_context_json" not in repr(d)
