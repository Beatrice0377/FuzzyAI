"""Strict JSON parsing for identity-bearing artifacts.

Every persisted artifact in this project is fingerprint-bearing, so plain
``json.loads`` is too permissive in four ways that a stricter parser must close:

* a duplicated object key silently keeps the last value, which would let
  ``{"fingerprint": "A", "fingerprint": "B"}`` load as ``"B"``;
* the non-standard ``NaN`` / ``Infinity`` / ``-Infinity`` literals parse to
  non-finite floats, which must never enter an identity payload;
* a float literal such as ``1e9999`` is valid JSON text but parses to ``inf``;
* an enormous integer literal or a deeply nested document raises a low-level
  ``ValueError`` / ``RecursionError`` that must not escape as a raw parser
  exception;
* a corrupted artifact can carry an arbitrarily large field, or an integer with
  more digits than the interpreter will convert to text, so every value echoed
  into an error message is abbreviated to a bounded length.

This module is deliberately calibration-domain-neutral: only the noun used in
the error message varies between callers. That keeps a single implementation of
security-sensitive parsing instead of one copy per artifact kind.
"""

from __future__ import annotations

import json
import math
from typing import Any

from probvenance.errors import InvalidDecisionError

#: Deepest nesting accepted in a persisted artifact.
#:
#: ``json.loads`` converts its own ``RecursionError``, but a document can be
#: nested just deeply enough to parse and still overflow the recursive
#: canonicalization and freezing that every identity-bearing artifact performs
#: afterwards, which would leak a raw ``RecursionError`` from a public loader.
#: Identity payloads are shallow in practice (a few levels of method
#: configuration or rendering semantics), so this bound is generous while still
#: failing closed well before the interpreter recursion limit.
_MAX_ARTIFACT_NESTING_DEPTH = 64


def abbreviate_untrusted(value: Any, max_chars: int = 200) -> str:
    """Return a bounded repr of an untrusted value for an error message.

    A corrupted artifact can contain an arbitrarily large string, and it can
    contain an integer with more digits than the interpreter will convert to
    text (Python 3.11 raises a plain ``ValueError`` for that, which is exactly
    the low-level exception this module exists to keep out of public errors).
    Rendering must therefore never echo the value in full and must never raise.
    """
    try:
        text = repr(value)
    except ValueError:
        return f"<{type(value).__name__} with too many digits to display>"
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]} ...<{omitted} more chars>"


def parse_strict_json_object(serialized: str, *, subject: str) -> dict[str, Any]:
    """Parse ``serialized`` as a strict JSON object or raise a controlled error.

    ``subject`` names the artifact in error messages, for example
    ``"calibration profile"``. Every failure, including a non-finite value, a
    duplicated key, a huge integer literal, and a document nested too deeply to
    parse, is reported as :class:`~probvenance.errors.InvalidDecisionError`; no
    low-level parser exception escapes.
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidDecisionError(
                    f"the serialized {subject} contains a duplicate object key "
                    f"{abbreviate_untrusted(key)}"
                )
            result[key] = value
        return result

    def reject_constant(name: str) -> Any:
        raise InvalidDecisionError(
            f"the serialized {subject} contains a non-finite JSON number "
            f"{abbreviate_untrusted(name)}"
        )

    def strict_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise InvalidDecisionError(
                f"the serialized {subject} contains a non-finite JSON number "
                f"{abbreviate_untrusted(text)}"
            )
        return value

    try:
        document = json.loads(
            serialized,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=strict_float,
        )
    except json.JSONDecodeError as error:
        raise InvalidDecisionError(
            f"the serialized {subject} is not valid JSON: {error}"
        ) from error
    except ValueError as error:
        # Python 3.11+ raises a plain ValueError (not a JSONDecodeError) for an
        # integer literal exceeding the digit limit.
        raise InvalidDecisionError(
            f"the serialized {subject} is not valid JSON: {error}"
        ) from error
    except RecursionError as error:
        raise InvalidDecisionError(
            f"the serialized {subject} is nested too deeply to parse as JSON"
        ) from error

    if not isinstance(document, dict):
        raise InvalidDecisionError(
            f"the serialized {subject} must be a JSON object, got {type(document).__name__}"
        )
    require_bounded_nesting(document, subject=subject)
    return document


def require_bounded_nesting(
    document: Any,
    *,
    subject: str,
    limit: int = _MAX_ARTIFACT_NESTING_DEPTH,
) -> None:
    """Raise if ``document`` nests deeper than ``limit``.

    The walk is iterative so that checking the bound cannot itself overflow the
    interpreter stack on the hostile input it exists to reject.
    """

    pending: list[tuple[Any, int]] = [(document, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > limit:
            raise InvalidDecisionError(
                f"the serialized {subject} is nested more than {limit} levels deep"
            )
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
