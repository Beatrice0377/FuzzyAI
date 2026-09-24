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


class CalibrationProfileStoreError(ProbvenanceError):
    """A calibration profile store could not complete an operation.

    This covers operational failures (the root path is not a directory, a
    managed read or write failed) as well as the two more specific store
    conditions below.
    """


class CalibrationProfileNotFoundError(CalibrationProfileStoreError):
    """No artifact exists at the requested exact profile identity.

    Absence is distinct from corruption: this is raised only when the exact
    requested identity has no stored artifact, never when a stored artifact
    exists but fails validation.
    """


class CalibrationProfileStoreIntegrityError(CalibrationProfileStoreError):
    """A stored artifact exists at the requested identity but is invalid.

    The caller supplied a valid store key, so the managed stored artifact
    violated the store invariant: it is unreadable, malformed, not the
    canonical serialization, or does not match the requested profile identity.
    """
