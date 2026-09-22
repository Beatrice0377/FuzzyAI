"""Backend capability data.

Capabilities are explicit, declared facts about a backend — never inferred and
never consulted through ``hasattr``. Phase 1 contains no capability inference
and no fallback logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared capabilities of a backend. Frozen, hashable, structurally comparable."""

    binary_token_logits: bool = False
    categorical_token_logits: bool = False
    token_logprobs: bool = False
    batching: bool = False
    prefix_cache: bool = False
    constrained_decoding: bool = False

    @classmethod
    def none(cls) -> "BackendCapabilities":
        """All capabilities disabled (the default for an unknown backend)."""
        return cls()
