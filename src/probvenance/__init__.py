"""Probvenance: provider-agnostic probabilistic decision runtime for language models.

Phase 2A adds the deterministic decision pipeline: scoring doctrines, the Bool
compiler, the probability assembler, decision traces, and the runtime facade
that ties a backend to the pipeline end to end. Phase 2B adds direct
categorical Choice: the Choice compiler with a versioned scoring-label scheme,
the categorical doctrine, choice scoring diagnostics, the choice assembler,
and runtime dispatch on the decision type.
"""

from probvenance.assembler import (
    assemble_bool_probability,
    assemble_choice_probability,
    assemble_probability,
    resolve_probability_assembler,
)
from probvenance.backends import Backend
from probvenance.capabilities import BackendCapabilities
from probvenance.compiler import (
    BINARY_COMPILER_VERSION,
    CATEGORICAL_COMPILER_VERSION,
    CATEGORICAL_LABEL_SCHEME_ID,
    CATEGORICAL_LABELS,
    BoolCompiler,
    ChoiceCompiler,
)
from probvenance.decisions import BoolDecision, Choice, ChoiceDecision
from probvenance.diagnostics import (
    BINARY_EVIDENCE_LABELS,
    ChoiceScoringDiagnostics,
    ScoringDiagnostics,
    diagnose_bool_evidence,
    diagnose_choice_evidence,
    scoring_label_mass,
)
from probvenance.doctrine import (
    BINARY_DOCTRINE_ID,
    BINARY_DOCTRINE_VERSION,
    BINARY_SEMANTIC_JUDGMENT_V1,
    CATEGORICAL_DOCTRINE_ID,
    CATEGORICAL_DOCTRINE_VERSION,
    CATEGORICAL_SEMANTIC_JUDGMENT_V1,
    CategoricalScoringDoctrine,
    ScoringDoctrine,
)
from probvenance.errors import (
    AmbiguousCalibrationProfileSelectionError,
    CalibrationProfileNotFoundError,
    CalibrationProfileSelectionError,
    CalibrationProfileStoreError,
    CalibrationProfileStoreIntegrityError,
    FingerprintError,
    InvalidDecisionError,
    InvalidProbabilityError,
    NoEligibleCalibrationProfileError,
    ProbvenanceError,
    ScoringLabelError,
    UnsupportedAssemblerError,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
    VerbalizerError,
)
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint
from probvenance.plans import (
    PLAN_FINGERPRINT_VERSION,
    CandidateLabelMapping,
    EvidenceKind,
    InferencePlan,
    RawEvidence,
    ScoringStrategy,
)
from probvenance.results import (
    BoolResult,
    Certainty,
    ChoiceResult,
    DecisionResult,
    normalized_entropy,
    probability_margin,
)
from probvenance.runtime import Evaluation, Probvenance
from probvenance.trace import DecisionTrace, build_decision_trace

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
    "PLAN_FINGERPRINT_VERSION",
    "AmbiguousCalibrationProfileSelectionError",
    "Backend",
    "BackendCapabilities",
    "BoolCompiler",
    "BoolDecision",
    "BoolResult",
    "CalibrationProfileNotFoundError",
    "CalibrationProfileSelectionError",
    "CalibrationProfileStoreError",
    "CalibrationProfileStoreIntegrityError",
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
    "InferencePlan",
    "InvalidDecisionError",
    "InvalidProbabilityError",
    "JSONValue",
    "NoEligibleCalibrationProfileError",
    "Probvenance",
    "ProbvenanceError",
    "RawEvidence",
    "ScoringDiagnostics",
    "ScoringDoctrine",
    "ScoringLabelError",
    "ScoringStrategy",
    "UnsupportedAssemblerError",
    "UnsupportedCapabilityError",
    "UnsupportedDecisionError",
    "VerbalizerError",
    "assemble_bool_probability",
    "assemble_choice_probability",
    "assemble_probability",
    "build_decision_trace",
    "canonical_json",
    "diagnose_bool_evidence",
    "diagnose_choice_evidence",
    "fingerprint",
    "normalized_entropy",
    "probability_margin",
    "resolve_probability_assembler",
    "scoring_label_mass",
]
