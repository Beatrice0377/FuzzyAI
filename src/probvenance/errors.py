"""Stable exception hierarchy for Probvenance.

Every error is part of the public SDK surface: messages are written for ordinary
users (they include the offending value's repr/type where useful) and never leak
stack-trace internals.
"""


class ProbvenanceError(Exception):
    """Base class for all Probvenance errors."""


class InvalidDecisionError(ProbvenanceError):
    """A decision spec, plan, or evidence structure is invalid."""


class InvalidProbabilityError(ProbvenanceError):
    """A probability, certainty, or evidence numeric value is invalid."""


class UnsupportedCapabilityError(ProbvenanceError):
    """A backend cannot satisfy a required capability."""


class FingerprintError(ProbvenanceError):
    """A value is not deterministically serializable (JSON-incompatible)."""


class UnsupportedDecisionError(ProbvenanceError):
    """A compiler or runtime does not support this decision type."""


class UnsupportedAssemblerError(ProbvenanceError):
    """No probability assembler matches a plan's declared assembler identity.

    The runtime executes only an implementation whose strategy, assembler id,
    and assembler version match the plan declaration exactly.
    """


class ScoringLabelError(ProbvenanceError):
    """A scoring label cannot be resolved to exactly one distinct scoring token."""


class VerbalizerError(ScoringLabelError):
    """A verbalizer cannot be resolved to exactly one distinct scoring token."""
