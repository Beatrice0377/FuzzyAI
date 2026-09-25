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


class CalibrationProfileCatalogStoreError(ProbvenanceError):
    """A calibration profile catalog store could not complete an operation.

    This covers operational failures (the root path is not a directory, a
    managed read or write failed) as well as the two more specific store
    conditions below. It is deliberately separate from the profile store
    hierarchy: one store's operational failure is not the other's.
    """


class CalibrationProfileCatalogNotFoundError(CalibrationProfileCatalogStoreError):
    """No artifact exists at the requested exact catalog snapshot identity.

    Absence is distinct from corruption: this is raised only when the exact
    requested identity has no stored artifact, never when a stored artifact
    exists but fails validation.
    """


class CalibrationProfileCatalogStoreIntegrityError(CalibrationProfileCatalogStoreError):
    """A stored artifact exists at the requested identity but is invalid.

    The caller supplied a valid store key, so the managed stored artifact
    violated the store invariant: it is unreadable, malformed, not the
    canonical serialization, or does not restore the requested catalog snapshot
    identity.
    """


class CalibrationProfileSelectionError(ProbvenanceError):
    """An explicit candidate set could not yield exactly one eligible profile.

    Selection is authorization, not recommendation: it establishes that a
    profile is semantically eligible for a declared runtime population and
    never that one profile is statistically better, newer, or preferable.
    """


class NoEligibleCalibrationProfileError(CalibrationProfileSelectionError):
    """No candidate profile is eligible for the declared runtime population.

    The caller decides what an absent eligible profile means operationally.
    The selector never substitutes ``None``, a nearest candidate, or a silent
    fallback to uncalibrated behaviour.
    """


class AmbiguousCalibrationProfileSelectionError(CalibrationProfileSelectionError):
    """More than one distinct candidate profile is eligible.

    The v1 policy has no tie-break: it does not prefer a method, a training
    dataset, fitted parameters, a quality metric, recency, or candidate order.
    The caller must disambiguate by supplying a narrower candidate set.
    """
