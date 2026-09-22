"""FuzzyAI: provider-agnostic probabilistic decision runtime for language models.

Phase 2A adds the deterministic decision pipeline: scoring doctrines, the Bool
compiler, the probability assembler, decision traces, and the runtime facade
that ties a backend to the pipeline end to end. Phase 2B adds direct
categorical Choice: the Choice compiler with a versioned scoring-label scheme,
the categorical doctrine, choice scoring diagnostics, the choice assembler,
and runtime dispatch on the decision type.
"""

from fuzzyai.assembler import assemble_bool_probability, assemble_choice_probability
from fuzzyai.backends import Backend
from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.compiler import (
    BINARY_COMPILER_VERSION,
    CATEGORICAL_COMPILER_VERSION,
    CATEGORICAL_LABEL_SCHEME_ID,
    CATEGORICAL_LABELS,
    BoolCompiler,
    ChoiceCompiler,
)
from fuzzyai.decisions import BoolDecision, Choice, ChoiceDecision
from fuzzyai.diagnostics import (
    BINARY_EVIDENCE_LABELS,
    ChoiceScoringDiagnostics,
    ScoringDiagnostics,
    candidate_mass,
    diagnose_bool_evidence,
    diagnose_choice_evidence,
)
from fuzzyai.doctrine import (
    BINARY_DOCTRINE_ID,
    BINARY_DOCTRINE_VERSION,
    BINARY_SEMANTIC_JUDGMENT_V1,
    CATEGORICAL_DOCTRINE_ID,
    CATEGORICAL_DOCTRINE_VERSION,
    CATEGORICAL_SEMANTIC_JUDGMENT_V1,
    CategoricalScoringDoctrine,
    ScoringDoctrine,
)
from fuzzyai.errors import (
    FingerprintError,
    FuzzyAIError,
    InvalidDecisionError,
    InvalidProbabilityError,
    ScoringLabelError,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
    VerbalizerError,
)
from fuzzyai.fingerprint import JSONValue, canonical_json, fingerprint
from fuzzyai.plans import (
    CandidateLabelMapping,
    EvidenceKind,
    InferencePlan,
    RawEvidence,
    ScoringStrategy,
)
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
    "CATEGORICAL_COMPILER_VERSION",
    "CATEGORICAL_DOCTRINE_ID",
    "CATEGORICAL_DOCTRINE_VERSION",
    "CATEGORICAL_LABELS",
    "CATEGORICAL_LABEL_SCHEME_ID",
    "CATEGORICAL_SEMANTIC_JUDGMENT_V1",
    "Backend",
    "BackendCapabilities",
    "BoolCompiler",
    "BoolDecision",
    "BoolResult",
    "CandidateLabelMapping",
    "CategoricalScoringDoctrine",
    "Certainty",
    "Choice",
    "ChoiceCompiler",
    "ChoiceDecision",
    "ChoiceResult",
    "ChoiceScoringDiagnostics",
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
    "ScoringLabelError",
    "ScoringStrategy",
    "UnsupportedCapabilityError",
    "UnsupportedDecisionError",
    "VerbalizerError",
    "assemble_bool_probability",
    "assemble_choice_probability",
    "build_decision_trace",
    "candidate_mass",
    "canonical_json",
    "diagnose_bool_evidence",
    "diagnose_choice_evidence",
    "fingerprint",
    "normalized_entropy",
    "probability_margin",
]
