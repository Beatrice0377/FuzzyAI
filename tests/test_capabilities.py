"""Tests for BackendCapabilities."""

import dataclasses
from typing import Any

import pytest

from probvenance import BackendCapabilities


# INV-14: capabilities are explicit declared data, never hasattr probes.
class TestBackendCapabilities:
    def test_defaults_all_false(self) -> None:
        c = BackendCapabilities()
        for f in dataclasses.fields(c):
            assert getattr(c, f.name) is False

    def test_none_returns_all_false(self) -> None:
        assert BackendCapabilities.none() == BackendCapabilities()

    def test_explicit_construction(self) -> None:
        c = BackendCapabilities(binary_token_logits=True, batching=True)
        assert c.binary_token_logits is True
        assert c.batching is True
        assert c.token_logprobs is False

    def test_equality_of_equal_capabilities(self) -> None:
        assert BackendCapabilities(batching=True) == BackendCapabilities(batching=True)

    def test_inequality(self) -> None:
        assert BackendCapabilities(batching=True) != BackendCapabilities(prefix_cache=True)
        assert BackendCapabilities() != BackendCapabilities(batching=True)

    def test_hashable(self) -> None:
        assert hash(BackendCapabilities(batching=True)) == hash(BackendCapabilities(batching=True))
        assert len({BackendCapabilities(), BackendCapabilities(batching=True)}) == 2

    def test_frozen(self) -> None:
        c = BackendCapabilities()
        mutable: Any = c
        with pytest.raises(dataclasses.FrozenInstanceError):
            mutable.batching = True
