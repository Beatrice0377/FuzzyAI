"""Backend protocol.

A backend knows NOTHING about decisions, results, certainty, or calibration.
It only executes an already-compiled :class:`~fuzzyai.plans.InferencePlan` and
returns :class:`~fuzzyai.plans.RawEvidence`. All decision-level semantics live
above this protocol.

Phase 1 defines no concrete backend, no fallback, and no capability checking.
"""

from typing import Protocol

from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.plans import InferencePlan, RawEvidence


class Backend(Protocol):
    """Anything that can execute an :class:`InferencePlan` into :class:`RawEvidence`."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def execute(self, plan: InferencePlan) -> RawEvidence: ...
