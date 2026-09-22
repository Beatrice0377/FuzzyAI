"""FuzzyAI: provider-agnostic probabilistic decision runtime for language models.

Phase 2A adds the deterministic decision pipeline: scoring doctrines, the Bool
compiler, the probability assembler, decision traces, and the runtime facade
that ties a backend to the pipeline end to end.
"""

from fuzzyai.assembler import BINARY_EVIDENCE_LABELS, assemble_bool_probability
from fuzzyai.backends import Backend
from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.compiler import BINARY_COMPILER_VERSION, BoolCompiler
from fuzzyai.decisions import BoolDecision, Choice, ChoiceDecision
from fuzzyai.diagnostics import ScoringDiagnostics, diagnose_bool_evidence
from fuzzyai.doctrine import (
    BINARY_DOCTRINE_ID,
    BINARY_DOCTRINE_VERSION,
    BINARY_SEMANTIC_JUDGMENT_V1,
    ScoringDoctrine,
)
from fuzzyai.errors import (
    FingerprintError,
    FuzzyAIError,
    InvalidDecisionError,
    InvalidProbabilityError,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
    VerbalizerError,
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
from fuzzyai.runtime import Evaluation, FuzzyAI
from fuzzyai.trace import DecisionTrace, build_decision_trace

__all__ = [
    "BINARY_COMPILER_VERSION",
    "BINARY_DOCTRINE_ID",
    "BINARY_DOCTRINE_VERSION",
    "BINARY_EVIDENCE_LABELS",
    "BINARY_SEMANTIC_JUDGMENT_V1",
    "Backend",
    "BackendCapabilities",
    "BoolCompiler",
    "BoolDecision",
    "BoolResult",
    "Certainty",
    "Choice",
    "ChoiceDecision",
    "ChoiceResult",
    "DecisionResult",
    "DecisionTrace",
    "Evaluation",
    "EvidenceKind",
    "FingerprintError",
    "FuzzyAI",
    "FuzzyAIError",
    "InferencePlan",
    "InvalidDecisionError",
    "InvalidProbabilityError",
    "JSONValue",
    "RawEvidence",
    "ScoringDiagnostics",
    "ScoringDoctrine",
    "ScoringStrategy",
    "UnsupportedCapabilityError",
    "UnsupportedDecisionError",
    "VerbalizerError",
    "assemble_bool_probability",
    "build_decision_trace",
    "canonical_json",
    "diagnose_bool_evidence",
    "fingerprint",
    "normalized_entropy",
    "probability_margin",
]
