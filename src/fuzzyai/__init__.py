"""FuzzyAI: provider-agnostic probabilistic decision runtime for language models.

Phase 1 contains only the deterministic core: canonical fingerprinting, decision
value objects, capability data, inference plans, raw evidence, and result value
objects. No model loading, no I/O, no provider integrations.
"""

from fuzzyai.backends import Backend
from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.decisions import BoolDecision, Choice, ChoiceDecision
from fuzzyai.errors import (
    FingerprintError,
    FuzzyAIError,
    InvalidDecisionError,
    InvalidProbabilityError,
    UnsupportedCapabilityError,
)
from fuzzyai.fingerprint import JSONValue, canonical_json, fingerprint
from fuzzyai.plans import EvidenceKind, InferencePlan, RawEvidence, ScoringStrategy
from fuzzyai.results import (
    BoolResult,
    Certainty,
    ChoiceResult,
    DecisionResult,
    normalized_entropy,
    probability_margin,
)

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BoolDecision",
    "BoolResult",
    "Certainty",
    "Choice",
    "ChoiceDecision",
    "ChoiceResult",
    "DecisionResult",
    "EvidenceKind",
    "FingerprintError",
    "FuzzyAIError",
    "InferencePlan",
    "InvalidDecisionError",
    "InvalidProbabilityError",
    "JSONValue",
    "RawEvidence",
    "ScoringStrategy",
    "UnsupportedCapabilityError",
    "canonical_json",
    "fingerprint",
    "normalized_entropy",
    "probability_margin",
]
