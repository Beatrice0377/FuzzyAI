"""Tests for canonical JSON and SHA-256 fingerprinting."""

import datetime
from typing import Any

import pytest

from probvenance import FingerprintError, JSONValue, canonical_json, fingerprint


class TestCanonicalJson:
    def test_deterministic(self) -> None:
        value = {"b": [1, 2.5, "x"], "a": None, "c": True}
        assert canonical_json(value) == canonical_json(value)

    def test_scalars(self) -> None:
        assert canonical_json(None) == "null"
        assert canonical_json(True) == "true"
        assert canonical_json(False) == "false"
        assert canonical_json(42) == "42"
        assert canonical_json(-7) == "-7"
        assert canonical_json(2.5) == "2.5"
        assert canonical_json("") == '""'
        assert canonical_json("hi") == '"hi"'

    def test_no_whitespace(self) -> None:
        assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'
        assert canonical_json([1, 2, 3]) == "[1,2,3]"

    def test_nested_dict_list(self) -> None:
        value = {"outer": {"z": 1, "a": [True, None, "s"]}, "m": []}
        assert canonical_json(value) == '{"m":[],"outer":{"a":[true,null,"s"],"z":1}}'

    # INV-10: key-order independence (dict keys are sorted canonically).
    def test_dict_keys_sorted(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_list_order_preserved(self) -> None:
        assert canonical_json([1, 2]) != canonical_json([2, 1])

    # INV-11: literal Unicode (UTF-8 bytes, no \u escaping).
    def test_unicode_literal(self) -> None:
        assert canonical_json("决策") == '"决策"'
        assert canonical_json("🚀") == '"🚀"'
        assert "\\u" not in canonical_json("决策")

    def test_bool_before_int(self) -> None:
        assert canonical_json(True) == "true"
        assert canonical_json(True) != canonical_json(1)
        assert canonical_json(False) != canonical_json(0)

    def test_float_serialization(self) -> None:
        assert canonical_json(0.1) == "0.1"
        assert canonical_json(1e300) == canonical_json(1e300)
        assert canonical_json(1.0) == "1.0"

    def test_negative_zero_normalized(self) -> None:
        assert canonical_json(-0.0) == canonical_json(0.0)
        assert fingerprint(-0.0) == fingerprint(0.0)

    # INV-09: non-finite floats are rejected.
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_rejected(self, bad: float) -> None:
        with pytest.raises(FingerprintError, match="finite"):
            canonical_json(bad)
        with pytest.raises(FingerprintError, match="finite"):
            canonical_json([1.0, bad])

    # INV-08: canonicalization type whitelist.
    @pytest.mark.parametrize(
        "bad",
        [
            {1, 2},
            (1, 2),
            b"bytes",
            datetime.datetime(2024, 1, 1),
            lambda: None,
            object(),
        ],
    )
    def test_disallowed_types_rejected(self, bad: Any) -> None:
        with pytest.raises(FingerprintError, match="not JSON-compatible"):
            canonical_json(bad)

    def test_non_str_dict_keys_rejected(self) -> None:
        int_keys: Any = {1: "a"}
        none_keys: Any = {None: "a"}
        with pytest.raises(FingerprintError, match="must be str"):
            canonical_json(int_keys)
        with pytest.raises(FingerprintError, match="must be str"):
            canonical_json(none_keys)

    def test_mixed_type_dict_keys_rejected_without_typeerror(self) -> None:
        mixed: Any = {1: "a", "b": 2}
        with pytest.raises(FingerprintError, match="must be str"):
            canonical_json(mixed)

    def test_lone_surrogate_rejected(self) -> None:
        with pytest.raises(FingerprintError, match="UTF-8"):
            fingerprint("\ud800")

    def test_unicode_not_normalized(self) -> None:
        # NFC vs NFD: different code point sequences are different strings.
        nfc = "é"  # U+00E9
        nfd = "e\u0301"  # e + combining acute
        assert nfc != nfd
        assert fingerprint(nfc) != fingerprint(nfd)


# INV-07: semantic determinism (same semantic input, same fingerprint).
class TestFingerprint:
    def test_is_64_lowercase_hex(self) -> None:
        fp = fingerprint({"a": 1})
        assert len(fp) == 64
        assert fp == fp.lower()
        assert all(c in "0123456789abcdef" for c in fp)

    def test_stable_across_calls(self) -> None:
        value: JSONValue = {"k": [1, "two", 3.5, None, True]}
        assert fingerprint(value) == fingerprint(value) == fingerprint(value)

    def test_stable_across_equal_semantics(self) -> None:
        assert fingerprint({"a": 1, "b": [2, 3]}) == fingerprint({"b": [2, 3], "a": 1})

    def test_different_values_differ(self) -> None:
        assert fingerprint({"a": 1}) != fingerprint({"a": 2})
        assert fingerprint([1]) != fingerprint([1, 1])

    def test_mutation_isolation(self) -> None:
        inner: list[JSONValue] = [1, 2]
        context: dict[str, JSONValue] = {"a": inner}
        fp = fingerprint(context)
        inner.append(3)
        context["b"] = "new"
        assert fingerprint({"a": [1, 2]}) == fp
        assert fingerprint(context) != fp

    # INV-12: a fingerprint is the SHA-256 hex digest of the canonical JSON.
    def test_matches_sha256_of_canonical(self) -> None:
        import hashlib

        value = {"x": [1, 2], "y": "z"}
        assert (
            fingerprint(value) == hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
        )

    # INV-09: non-finite floats are rejected at any nesting depth.
    def test_nan_in_nested_structure_rejected(self) -> None:
        with pytest.raises(FingerprintError, match="finite"):
            fingerprint({"deep": {"list": [float("nan")]}})
