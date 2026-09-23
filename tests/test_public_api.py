"""Tests for the public API surface of the probvenance package."""

import probvenance


def test_import_works() -> None:
    assert probvenance.__name__ == "probvenance"


def test_all_names_exist_on_module() -> None:
    for name in probvenance.__all__:
        assert hasattr(probvenance, name), f"missing public name: {name}"


def test_all_is_sorted_and_unique() -> None:
    # Ruff's RUF022 enforces isort-style ordering for ``__all__``:
    # SCREAMING_SNAKE constants, then CamelCase classes, then snake_case
    # functions, each group lexicographic. Plain ``sorted()`` disagrees once
    # constant and class names share no prefix, so assert the RUF022 order.
    def ruf022_key(name: str) -> tuple[bool, bool, str]:
        return (not name.isupper(), name.islower(), name)

    assert probvenance.__all__ == sorted(probvenance.__all__, key=ruf022_key)
    assert len(probvenance.__all__) == len(set(probvenance.__all__))


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
        "ProbvenanceError",
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
        "Probvenance",
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
    assert expected <= set(probvenance.__all__)


def test_renamed_scoring_label_names_present() -> None:
    assert "scoring_label_mass" in probvenance.__all__
    assert callable(probvenance.scoring_label_mass)


def test_old_candidate_mass_names_gone() -> None:
    # Phase 2B.1 renamed candidate_mass -> scoring_label_mass and
    # candidate_token_probabilities -> scoring_label_token_probabilities:
    # the old names measure protocol adherence, not candidate coverage, and
    # must not linger on the public surface in any form.
    assert "candidate_mass" not in probvenance.__all__
    assert "candidate_token_probabilities" not in probvenance.__all__
    assert not hasattr(probvenance, "candidate_mass")
    assert not hasattr(probvenance, "candidate_token_probabilities")


def test_no_internal_helpers_exported() -> None:
    for name in probvenance.__all__:
        assert not name.startswith("_")
    # Internal helpers must not be part of the public surface.
    for internal in ("_canonicalize", "_validate_distribution", "_normalize_choices"):
        assert internal not in probvenance.__all__


def test_internal_diagnostic_helpers_not_exported() -> None:
    for internal in (
        "log_verbalizer_mass",
        "verbalizer_mass",
        "full_vocab_probability",
        "_logaddexp",
    ):
        assert internal not in probvenance.__all__


def test_exception_hierarchy() -> None:
    assert issubclass(probvenance.InvalidDecisionError, probvenance.ProbvenanceError)
    assert issubclass(probvenance.InvalidProbabilityError, probvenance.ProbvenanceError)
    assert issubclass(probvenance.UnsupportedCapabilityError, probvenance.ProbvenanceError)
    assert issubclass(probvenance.FingerprintError, probvenance.ProbvenanceError)
    assert issubclass(probvenance.UnsupportedDecisionError, probvenance.ProbvenanceError)
    assert issubclass(probvenance.ScoringLabelError, probvenance.ProbvenanceError)
    assert issubclass(probvenance.VerbalizerError, probvenance.ScoringLabelError)
    assert issubclass(probvenance.ProbvenanceError, Exception)


def test_scoring_label_error_is_verbalizer_compatible() -> None:
    # Existing `except VerbalizerError` handlers keep working after the
    # re-parenting, and the new categorical error is catchable through it.
    assert isinstance(probvenance.VerbalizerError("v"), probvenance.ScoringLabelError)
    assert isinstance(probvenance.ScoringLabelError("s"), probvenance.ProbvenanceError)
