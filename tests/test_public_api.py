"""Tests for the public API surface of the fuzzyai package."""

import fuzzyai


def test_import_works() -> None:
    assert fuzzyai.__name__ == "fuzzyai"


def test_all_names_exist_on_module() -> None:
    for name in fuzzyai.__all__:
        assert hasattr(fuzzyai, name), f"missing public name: {name}"


def test_all_is_sorted_and_unique() -> None:
    # Ruff's RUF022 enforces isort-style ordering for ``__all__``:
    # SCREAMING_SNAKE constants, then CamelCase classes, then snake_case
    # functions, each group lexicographic. Plain ``sorted()`` disagrees once
    # constant and class names share no prefix, so assert the RUF022 order.
    def ruf022_key(name: str) -> tuple[bool, bool, str]:
        return (not name.isupper(), name.islower(), name)

    assert fuzzyai.__all__ == sorted(fuzzyai.__all__, key=ruf022_key)
    assert len(fuzzyai.__all__) == len(set(fuzzyai.__all__))


def test_representative_names_present() -> None:
    expected = {
        "BoolDecision",
        "Choice",
        "ChoiceDecision",
        "BoolResult",
        "ChoiceResult",
        "DecisionResult",
        "Certainty",
        "normalized_entropy",
        "probability_margin",
        "BackendCapabilities",
        "Backend",
        "InferencePlan",
        "ScoringStrategy",
        "RawEvidence",
        "EvidenceKind",
        "JSONValue",
        "canonical_json",
        "fingerprint",
        "FuzzyAIError",
        "InvalidDecisionError",
        "InvalidProbabilityError",
        "UnsupportedCapabilityError",
        "FingerprintError",
        "UnsupportedDecisionError",
        "VerbalizerError",
        "ScoringDoctrine",
        "BINARY_DOCTRINE_ID",
        "BINARY_DOCTRINE_VERSION",
        "BINARY_SEMANTIC_JUDGMENT_V1",
        "BoolCompiler",
        "BINARY_COMPILER_VERSION",
        "BINARY_EVIDENCE_LABELS",
        "assemble_bool_probability",
        "DecisionTrace",
        "build_decision_trace",
        "Evaluation",
        "FuzzyAI",
        "ScoringDiagnostics",
        "diagnose_bool_evidence",
        "ChoiceCompiler",
        "ChoiceScoringDiagnostics",
        "CATEGORICAL_COMPILER_VERSION",
        "CATEGORICAL_DOCTRINE_ID",
        "CATEGORICAL_DOCTRINE_VERSION",
        "CATEGORICAL_LABELS",
        "CATEGORICAL_LABEL_SCHEME_ID",
        "CATEGORICAL_SEMANTIC_JUDGMENT_V1",
        "CandidateLabelMapping",
        "CategoricalScoringDoctrine",
        "ScoringLabelError",
        "assemble_choice_probability",
        "scoring_label_mass",
        "diagnose_choice_evidence",
    }
    assert expected <= set(fuzzyai.__all__)


def test_renamed_scoring_label_names_present() -> None:
    assert "scoring_label_mass" in fuzzyai.__all__
    assert callable(fuzzyai.scoring_label_mass)


def test_old_candidate_mass_names_gone() -> None:
    # Phase 2B.1 renamed candidate_mass -> scoring_label_mass and
    # candidate_token_probabilities -> scoring_label_token_probabilities:
    # the old names measure protocol adherence, not candidate coverage, and
    # must not linger on the public surface in any form.
    assert "candidate_mass" not in fuzzyai.__all__
    assert "candidate_token_probabilities" not in fuzzyai.__all__
    assert not hasattr(fuzzyai, "candidate_mass")
    assert not hasattr(fuzzyai, "candidate_token_probabilities")


def test_no_internal_helpers_exported() -> None:
    for name in fuzzyai.__all__:
        assert not name.startswith("_")
    # Internal helpers must not be part of the public surface.
    for internal in ("_canonicalize", "_validate_distribution", "_normalize_choices"):
        assert internal not in fuzzyai.__all__


def test_internal_diagnostic_helpers_not_exported() -> None:
    for internal in (
        "log_verbalizer_mass",
        "verbalizer_mass",
        "full_vocab_probability",
        "_logaddexp",
    ):
        assert internal not in fuzzyai.__all__


def test_exception_hierarchy() -> None:
    assert issubclass(fuzzyai.InvalidDecisionError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.InvalidProbabilityError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.UnsupportedCapabilityError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.FingerprintError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.UnsupportedDecisionError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.ScoringLabelError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.VerbalizerError, fuzzyai.ScoringLabelError)
    assert issubclass(fuzzyai.FuzzyAIError, Exception)


def test_scoring_label_error_is_verbalizer_compatible() -> None:
    # Existing `except VerbalizerError` handlers keep working after the
    # re-parenting, and the new categorical error is catchable through it.
    assert isinstance(fuzzyai.VerbalizerError("v"), fuzzyai.ScoringLabelError)
    assert isinstance(fuzzyai.ScoringLabelError("s"), fuzzyai.FuzzyAIError)
