"""Tests for the public API surface of the fuzzyai package."""

import fuzzyai


def test_import_works() -> None:
    assert fuzzyai.__name__ == "fuzzyai"


def test_all_names_exist_on_module() -> None:
    for name in fuzzyai.__all__:
        assert hasattr(fuzzyai, name), f"missing public name: {name}"


def test_all_is_sorted_and_unique() -> None:
    assert fuzzyai.__all__ == sorted(fuzzyai.__all__)
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
    }
    assert expected <= set(fuzzyai.__all__)


def test_no_internal_helpers_exported() -> None:
    for name in fuzzyai.__all__:
        assert not name.startswith("_")
    # Internal helpers must not be part of the public surface.
    for internal in ("_canonicalize", "_validate_distribution", "_normalize_choices"):
        assert internal not in fuzzyai.__all__


def test_exception_hierarchy() -> None:
    assert issubclass(fuzzyai.InvalidDecisionError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.InvalidProbabilityError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.UnsupportedCapabilityError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.FingerprintError, fuzzyai.FuzzyAIError)
    assert issubclass(fuzzyai.FuzzyAIError, Exception)
