"""Canonical JSON serialization and SHA-256 fingerprinting.

This module is the single source of deterministic identity for the whole
library: decisions, plans, and evidence all derive stable fingerprints from it.

Canonicalization rules:
- ``None`` -> ``null``; ``True``/``False`` -> ``true``/``false`` (``bool`` is
  checked before ``int`` so booleans never serialize as numbers).
- ``int`` -> decimal string.
- ``float`` -> must be finite; ``-0.0`` is normalized to ``0.0`` so both produce
  the same canonical form; serialized via :func:`json.dumps` (JSON-compliant).
- ``str`` -> emitted literally as UTF-8 (``ensure_ascii=False``), never as
  ``\\uXXXX`` escapes.
- ``list`` -> order preserved.
- ``dict`` -> keys must be ``str`` and are sorted lexicographically; key order
  in the input is therefore irrelevant.
- Anything else (``set``, ``tuple``, ``bytes``, callables, arbitrary objects)
  is rejected with :class:`FingerprintError`.

Unicode is deliberately NOT normalized (no NFC/NFD folding): different code
point sequences are different strings and produce different fingerprints.
"""

import hashlib
import json
import math

from fuzzyai.errors import FingerprintError

# PEP 604 unions evaluate fine at runtime here (``None | bool | ...`` works on
# 3.11), so the alias uses ``|`` to satisfy ruff's UP007 without a suppression.
# Forward references keep the alias recursive.
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

_ALLOWED_TYPE_NAMES = ("NoneType", "bool", "int", "float", "str", "list", "dict")


def canonical_json(value: JSONValue) -> str:
    """Return the canonical JSON string for ``value``.

    The output contains no insignificant whitespace, dict keys are sorted, and
    Unicode is emitted literally. Raises :class:`FingerprintError` for any
    value outside the allowed JSON types or for non-finite floats.
    """
    return _canonicalize(value, path="$")


def fingerprint(value: JSONValue) -> str:
    """Return the SHA-256 hex digest (64 lowercase hex chars) of the canonical JSON."""
    payload = canonical_json(value)
    try:
        encoded = payload.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FingerprintError(f"canonical JSON is not valid UTF-8: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: JSONValue, path: str) -> str:
    if value is None:
        return "null"
    # bool must be tested before int: bool is a subclass of int.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintError(
                f"float value at {path} must be finite, got {value!r} "
                "(nan, inf and -inf are not JSON-compatible)"
            )
        if value == 0.0:
            value = 0.0  # normalize -0.0 to 0.0 so both serialize identically
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return (
            "["
            + ",".join(_canonicalize(item, f"{path}[{i}]") for i, item in enumerate(value))
            + "]"
        )
    if isinstance(value, dict):
        # Keys are validated BEFORE sorting: sorting mixed, non-comparable keys
        # (e.g. {1: "a", "b": 2}) would otherwise raise TypeError here.
        for key in value:
            if not isinstance(key, str):
                raise FingerprintError(
                    f"dict key at {path} must be str, got {type(key).__name__} ({key!r})"
                )
        parts = []
        for key in sorted(value):
            parts.append(
                json.dumps(key, ensure_ascii=False)
                + ":"
                + _canonicalize(value[key], f"{path}.{key}")
            )
        return "{" + ",".join(parts) + "}"
    raise FingerprintError(
        f"value at {path} of type {type(value).__name__} is not JSON-compatible "
        f"(allowed: {', '.join(_ALLOWED_TYPE_NAMES)})"
    )
