"""Decision value objects: immutable, deeply isolated, fingerprint-stable.

A decision describes a question the runtime wants a model to answer, together
with the context in which it is asked. Two decisions are equal if and only if
their semantic content is equal, and equal decisions always produce equal
fingerprints.

Two concrete kinds exist:

- :class:`BoolDecision` — outcome space Ω = {False, True}.
- :class:`ChoiceDecision` — Ω = an ordered set of mutually exclusive
  categorical alternatives. Candidate order is semantics-bearing: it is
  preserved verbatim and influences the fingerprint (unlike dict key order
  inside ``context``, which is canonicalized away).

Both store their context as its canonical JSON string internally, so callers
can never mutate internal state through the value they passed in.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from probvenance.errors import FingerprintError, InvalidDecisionError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidDecisionError(
            f"{field_name} must be str, got {type(value).__name__} ({value!r})"
        )
    if not value.strip():
        raise InvalidDecisionError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _context_json(context: JSONValue) -> str:
    try:
        return canonical_json(context)
    except FingerprintError as exc:
        raise InvalidDecisionError(f"context must be JSON-compatible: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BoolDecision:
    """A yes/no decision. Outcome space Ω = {False, True}."""

    question: str
    _context_json: str = field(default="null", compare=True)
    _fingerprint: str = field(default="", compare=False)

    def __init__(self, question: str, context: JSONValue = None) -> None:
        question = _require_non_empty_str(question, "question")
        context_json = _context_json(context)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "_context_json", context_json)
        object.__setattr__(
            self,
            "_fingerprint",
            fingerprint({"v": 1, "kind": "bool", "question": question, "context": context}),
        )

    @property
    def context(self) -> JSONValue:
        """A fresh plain JSON value on every access (mutation-safe snapshot)."""
        context: JSONValue = json.loads(self._context_json)
        return context

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of this decision's semantic content."""
        return self._fingerprint

    def __repr__(self) -> str:
        return f"BoolDecision(question={self.question!r}, context={self.context!r})"


@dataclass(frozen=True, slots=True)
class Choice:
    """One mutually exclusive alternative of a :class:`ChoiceDecision`."""

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.name, "choice name")
        if self.description is not None and not isinstance(self.description, str):
            raise InvalidDecisionError(
                f"choice description must be None or str, got {type(self.description).__name__} "
                f"({self.description!r})"
            )


@dataclass(frozen=True, slots=True)
class ChoiceDecision:
    """A categorical decision over an ordered set of mutually exclusive choices.

    Candidate order is semantics-bearing: it is preserved verbatim and
    influences the fingerprint.
    """

    question: str
    _context_json: str = field(default="null", compare=True)
    _choices: tuple[Choice, ...] = field(default=(), compare=True)
    _fingerprint: str = field(default="", compare=False)

    def __init__(
        self,
        question: str,
        context: JSONValue = None,
        *,
        choices: Mapping[str, str] | Sequence[Choice],
    ) -> None:
        question = _require_non_empty_str(question, "question")
        context_json = _context_json(context)
        normalized = _normalize_choices(choices)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "_context_json", context_json)
        object.__setattr__(self, "_choices", normalized)
        object.__setattr__(
            self,
            "_fingerprint",
            fingerprint(
                {
                    "v": 1,
                    "kind": "choice",
                    "question": question,
                    "context": context,
                    "choices": [{"name": c.name, "description": c.description} for c in normalized],
                }
            ),
        )

    @property
    def context(self) -> JSONValue:
        """A fresh plain JSON value on every access (mutation-safe snapshot)."""
        context: JSONValue = json.loads(self._context_json)
        return context

    @property
    def choices(self) -> tuple[Choice, ...]:
        """The alternatives in candidate order (read-only tuple)."""
        return self._choices

    @property
    def choice_names(self) -> tuple[str, ...]:
        """The choice names in candidate order."""
        return tuple(c.name for c in self._choices)

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of this decision's semantic content."""
        return self._fingerprint

    def __repr__(self) -> str:
        return (
            f"ChoiceDecision(question={self.question!r}, context={self.context!r}, "
            f"choices={self.choice_names!r})"
        )


def _normalize_choices(choices: Mapping[str, str] | Sequence[Choice]) -> tuple[Choice, ...]:
    if isinstance(choices, Mapping):
        normalized: list[Choice] = []
        for name, description in choices.items():
            if not isinstance(description, str):
                raise InvalidDecisionError(
                    f"choice description for {name!r} must be str, "
                    f"got {type(description).__name__} ({description!r})"
                )
            normalized.append(Choice(name=name, description=description))
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        for item in choices:
            if not isinstance(item, Choice):
                raise InvalidDecisionError(
                    f"choices sequence items must be Choice, got {type(item).__name__} ({item!r})"
                )
        normalized = list(choices)
    else:
        raise InvalidDecisionError(
            f"choices must be a mapping or a sequence of Choice, got {type(choices).__name__}"
        )
    if len(normalized) < 2:
        raise InvalidDecisionError(
            f"a choice decision requires at least 2 choices, got {len(normalized)}"
        )
    names = [c.name for c in normalized]
    counts = Counter(names)
    if len(counts) != len(names):
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        raise InvalidDecisionError(f"choice names must be unique, duplicates: {duplicates}")
    return tuple(normalized)
