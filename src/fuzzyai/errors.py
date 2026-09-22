"""Stable exception hierarchy for FuzzyAI.

Every error is part of the public SDK surface: messages are written for ordinary
users (they include the offending value's repr/type where useful) and never leak
stack-trace internals.
"""


class FuzzyAIError(Exception):
    """Base class for all FuzzyAI errors."""


class InvalidDecisionError(FuzzyAIError):
    """A decision spec, plan, or evidence structure is invalid."""


class InvalidProbabilityError(FuzzyAIError):
    """A probability, certainty, or evidence numeric value is invalid."""


class UnsupportedCapabilityError(FuzzyAIError):
    """A backend cannot satisfy a required capability."""


class FingerprintError(FuzzyAIError):
    """A value is not deterministically serializable (JSON-incompatible)."""


class UnsupportedDecisionError(FuzzyAIError):
    """A compiler or runtime does not support this decision type."""


class VerbalizerError(FuzzyAIError):
    """A verbalizer cannot be resolved to exactly one distinct scoring token."""
