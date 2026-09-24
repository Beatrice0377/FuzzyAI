"""Contract tests for explicit runtime calibration profile selection.

Selection is authorization, not recommendation. These tests pin the v1 policy:
exact-binding plus target/input eligibility, uniqueness-or-fail-closed
outcomes, order independence, and the deliberate non-use of method, training
dataset, ground-truth semantics, metrics, recency, and candidate order as
tie-breaks.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from itertools import permutations
from typing import Any

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    choice_observation,
    make_bool_decision,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
)
from test_profile_application import forge_profile

from probvenance import (
    AmbiguousCalibrationProfileSelectionError,
    InvalidDecisionError,
    NoEligibleCalibrationProfileError,
)
from probvenance.calibration import (
    UNCALIBRATED_SELECTED_PROBABILITY_ID,
    UNCALIBRATED_SELECTED_PROBABILITY_VERSION,
    WINNER_CORRECTNESS_TARGET_ID,
    WINNER_CORRECTNESS_TARGET_VERSION,
    CalibrationDataset,
    CalibrationObservation,
    CalibrationProfile,
    _binding_payloads_equal,
    _runtime_calibration_binding,
    apply_profile_to_runtime_evaluation,
    fit_l2_logistic_selected_probability,
)
from probvenance.calibration_selection import (
    select_calibration_profile_for_runtime,
)
from probvenance.errors import CalibrationProfileSelectionError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def runtime_evaluation(decision: Any = None) -> Any:
    runtime, _ = make_choice_runtime()
    return runtime.evaluate_with_trace(decision if decision is not None else make_choice_decision())


def fit(observations: list[Any], l2_strength: float = 1.0) -> CalibrationProfile:
    return fit_l2_logistic_selected_probability(
        CalibrationDataset.create(observations), l2_strength=l2_strength
    )


def eligible_profile() -> CalibrationProfile:
    return fit([choice_observation()])


def other_binding_profile() -> CalibrationProfile:
    return fit([choice_observation(backend=FakeCategoricalBackend(model="other-model"))])


def other_semantics_profile() -> CalibrationProfile:
    return fit([choice_observation(resolved_truth("shipping", labeling_rule="another rule"))])


def other_training_profile() -> CalibrationProfile:
    return fit([choice_observation(), choice_observation(resolved_truth("billing"))])


def other_strength_profile() -> CalibrationProfile:
    return fit([choice_observation()], l2_strength=0.5)


def runtime_binding_of(evaluation: Any, **declarations: Any) -> Any:
    return _runtime_calibration_binding(
        evaluation.trace,
        task_id=declarations.get("task_id"),
        domain_id=declarations.get("domain_id"),
        taxonomy_id=declarations.get("taxonomy_id"),
        taxonomy_version=declarations.get("taxonomy_version"),
    )


def module_code_without_docstrings(module: Any) -> str:
    """Return a module's source with every docstring removed.

    Docstrings legitimately describe forbidden behaviour (for example naming a
    quality metric or a registry as something the module does not do), so a
    scope check must inspect executable code only.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, holders) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# Eligibility and outcomes (PART 45, PART 46, PART 47, PART 48)
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_one_eligible_profile_is_returned(self) -> None:
        evaluation = runtime_evaluation()
        profile = eligible_profile()
        selected = select_calibration_profile_for_runtime(evaluation, (profile,))
        assert selected is profile

    def test_eligible_profile_is_selected_among_ineligible_candidates(self) -> None:
        evaluation = runtime_evaluation()
        profile = eligible_profile()
        ineligible = (other_binding_profile(), other_binding_profile())
        for candidates in permutations((profile, ineligible[0])):
            assert select_calibration_profile_for_runtime(evaluation, candidates) is profile

    def test_different_binding_is_ineligible(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (other_binding_profile(),))

    def test_all_ineligible_candidates_raise_no_eligible(self) -> None:
        evaluation = runtime_evaluation()
        ineligible_a = other_binding_profile()
        ineligible_b = forge_profile(eligible_profile(), target_id="some-other-target")
        assert ineligible_a.fingerprint != ineligible_b.fingerprint
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (ineligible_a, ineligible_b))

    def test_empty_candidate_tuple_is_no_eligible(self) -> None:
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(runtime_evaluation(), ())

    def test_wrong_target_is_ineligible(self) -> None:
        evaluation = runtime_evaluation()
        profile = forge_profile(eligible_profile(), target_id="some-other-target")
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (profile,))

    def test_wrong_input_score_is_ineligible(self) -> None:
        evaluation = runtime_evaluation()
        profile = forge_profile(eligible_profile(), input_score_id="some-other-input-score")
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (profile,))

    def test_wrong_target_version_is_ineligible(self) -> None:
        evaluation = runtime_evaluation()
        profile = forge_profile(eligible_profile(), target_version=2)
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (profile,))

    def test_current_target_and_input_constants_are_pinned(self) -> None:
        profile = eligible_profile()
        assert profile.target_id == WINNER_CORRECTNESS_TARGET_ID
        assert profile.target_version == WINNER_CORRECTNESS_TARGET_VERSION
        assert profile.input_score_id == UNCALIBRATED_SELECTED_PROBABILITY_ID
        assert profile.input_score_version == UNCALIBRATED_SELECTED_PROBABILITY_VERSION


class TestAmbiguity:
    def test_two_distinct_eligible_profiles_are_ambiguous(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(
                evaluation, (eligible_profile(), other_training_profile())
            )

    def test_ambiguity_is_order_independent(self) -> None:
        evaluation = runtime_evaluation()
        a = eligible_profile()
        b = other_training_profile()
        assert a.fingerprint != b.fingerprint
        for candidates in permutations((a, b)):
            with pytest.raises(AmbiguousCalibrationProfileSelectionError):
                select_calibration_profile_for_runtime(evaluation, candidates)

    def test_ground_truth_semantics_is_not_an_eligibility_filter(self) -> None:
        evaluation = runtime_evaluation()
        a = eligible_profile()
        b = other_semantics_profile()
        assert a.fingerprint != b.fingerprint
        assert a.ground_truth_semantics.fingerprint != b.ground_truth_semantics.fingerprint
        for candidates in permutations((a, b)):
            with pytest.raises(AmbiguousCalibrationProfileSelectionError):
                select_calibration_profile_for_runtime(evaluation, candidates)

    def test_training_dataset_is_not_a_tie_break(self) -> None:
        evaluation = runtime_evaluation()
        a = eligible_profile()
        b = other_training_profile()
        assert a.training_dataset_fingerprint != b.training_dataset_fingerprint
        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(evaluation, (a, b))

    def test_method_configuration_is_not_a_tie_break(self) -> None:
        evaluation = runtime_evaluation()
        a = eligible_profile()
        b = other_strength_profile()
        assert a.method_configuration != b.method_configuration
        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(evaluation, (a, b))

    def test_ambiguity_message_sorts_identities_and_hides_payloads(self) -> None:
        evaluation = runtime_evaluation()
        a = eligible_profile()
        b = other_training_profile()
        with pytest.raises(AmbiguousCalibrationProfileSelectionError) as caught:
            select_calibration_profile_for_runtime(evaluation, (a, b))
        text = str(caught.value)
        assert text.index(min(a.fingerprint, b.fingerprint)) < text.index(
            max(a.fingerprint, b.fingerprint)
        )
        assert "fitted_parameters" not in text
        assert "slope" not in text


# ---------------------------------------------------------------------------
# Method neutrality (PART 18, PART 19, PART 25, PART 62)
# ---------------------------------------------------------------------------


class TestMethodNeutrality:
    def test_unsupported_method_is_still_eligible(self) -> None:
        evaluation = runtime_evaluation()
        profile = forge_profile(eligible_profile(), method_id="not-a-supported-method")
        assert select_calibration_profile_for_runtime(evaluation, (profile,)) is profile

    def test_unsupported_method_selection_then_application_fails_closed(self) -> None:
        evaluation = runtime_evaluation()
        profile = forge_profile(eligible_profile(), method_id="not-a-supported-method")
        selected = select_calibration_profile_for_runtime(evaluation, (profile,))
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_runtime_evaluation(evaluation, selected)

    def test_two_distinct_methods_are_ambiguous(self) -> None:
        evaluation = runtime_evaluation()
        a = eligible_profile()
        b = forge_profile(eligible_profile(), method_id="another-method")
        assert a.fingerprint != b.fingerprint
        with pytest.raises(AmbiguousCalibrationProfileSelectionError):
            select_calibration_profile_for_runtime(evaluation, (a, b))

    def test_selection_does_not_read_the_selected_probability(self) -> None:
        import probvenance.calibration_selection as module

        code = module_code_without_docstrings(module)
        for accessor in ("probability_true", "probabilities", "probability_false"):
            assert accessor not in code


# ---------------------------------------------------------------------------
# Candidate set shape and validity (PART 5, PART 6, PART 52, PART 53, PART 54)
# ---------------------------------------------------------------------------


class TestCandidateSet:
    def test_duplicate_exact_identity_is_invalid_input_not_ambiguity(self) -> None:
        evaluation = runtime_evaluation()
        profile = eligible_profile()
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(evaluation, (profile, profile))

    @pytest.mark.parametrize(
        "bad",
        [None, object(), "serialized", 7],
        ids=["none", "object", "string", "int"],
    )
    def test_malformed_candidate_is_rejected(self, bad: Any) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(evaluation, (bad,))  # type: ignore[arg-type]

    def test_binding_is_not_a_candidate(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(
                evaluation,
                (eligible_profile().binding,),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "container",
        [
            lambda p: [p],
            lambda p: iter([p]),
        ],
        ids=["list", "generator"],
    )
    def test_non_tuple_container_is_rejected(self, container: Any) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(
                evaluation,
                container(eligible_profile()),  # type: ignore[arg-type]
            )

    def test_a_set_container_cannot_even_be_built(self) -> None:
        profile = eligible_profile()
        with pytest.raises(TypeError):
            _ = {profile}
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(
                runtime_evaluation(),
                frozenset(),  # type: ignore[arg-type]
            )

    def test_malformed_candidate_message_is_abbreviated(self) -> None:
        evaluation = runtime_evaluation()
        huge = "X" * 1_000_000
        with pytest.raises(InvalidDecisionError) as caught:
            select_calibration_profile_for_runtime(evaluation, (huge,))  # type: ignore[arg-type]
        assert len(str(caught.value)) < 2000


# ---------------------------------------------------------------------------
# Runtime input contract (PART 9, PART 10, PART 55, PART 56, PART 57)
# ---------------------------------------------------------------------------


class TestRuntimeInputContract:
    def test_mismatched_result_and_trace_cannot_select(self) -> None:
        first = runtime_evaluation()
        second = runtime_evaluation()
        broken = replace(first, result=replace(first.result, trace_id="other-trace-id"))
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(broken, (eligible_profile(),))
        assert second.result.trace_id is not None

    def test_choice_result_with_bool_trace_cannot_select(self) -> None:
        evaluation = runtime_evaluation(make_bool_decision())
        assert isinstance(evaluation.result, CalibrationProfile.__mro__[0].__class__) is False
        broken = replace(
            evaluation,
            trace=replace(evaluation.trace, decision_family="choice"),
        )
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(broken, (eligible_profile(),))

    def test_bool_result_with_choice_trace_cannot_select(self) -> None:
        evaluation = runtime_evaluation()
        broken = replace(
            evaluation,
            trace=replace(evaluation.trace, decision_family="bool"),
        )
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(broken, (eligible_profile(),))

    def test_unknown_decision_family_cannot_select(self) -> None:
        evaluation = runtime_evaluation()
        broken = replace(
            evaluation,
            trace=replace(evaluation.trace, decision_family="unknown-family"),
        )
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(broken, (eligible_profile(),))

    def test_already_calibrated_evaluation_cannot_select(self) -> None:
        evaluation = runtime_evaluation()
        calibrated = apply_profile_to_runtime_evaluation(evaluation, eligible_profile())
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime(calibrated, (eligible_profile(),))

    def test_non_evaluation_input_is_rejected(self) -> None:
        with pytest.raises(InvalidDecisionError):
            select_calibration_profile_for_runtime("not-an-evaluation", ())  # type: ignore[arg-type]

    def test_selection_does_not_mutate_the_evaluation(self) -> None:
        evaluation = runtime_evaluation()
        before = evaluation.result
        select_calibration_profile_for_runtime(evaluation, (eligible_profile(),))
        assert evaluation.result is before
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None
        assert evaluation.result.calibration_profile_fingerprint is None
        assert evaluation.trace.calibration_profile_fingerprint is None


# ---------------------------------------------------------------------------
# Caller declarations (PART 13, PART 14, PART 58, PART 59, PART 60)
# ---------------------------------------------------------------------------


class TestDeclarations:
    def declared_profile(self) -> CalibrationProfile:
        observation = CalibrationObservation.from_evaluation(
            runtime_evaluation(),
            resolved_truth("shipping"),
            task_id="support-routing",
            taxonomy_id="support",
            taxonomy_version=3,
        )
        return fit([observation])

    def test_omitted_declarations_make_a_declared_profile_ineligible(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(evaluation, (self.declared_profile(),))

    def test_matching_declarations_make_it_eligible(self) -> None:
        evaluation = runtime_evaluation()
        profile = self.declared_profile()
        selected = select_calibration_profile_for_runtime(
            evaluation,
            (profile,),
            task_id="support-routing",
            taxonomy_id="support",
            taxonomy_version=3,
        )
        assert selected is profile

    def test_contradicting_declaration_makes_it_ineligible(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(
                evaluation,
                (self.declared_profile(),),
                task_id="support-routing",
                taxonomy_id="other-taxonomy",
                taxonomy_version=3,
            )

    def test_none_is_not_a_wildcard(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(NoEligibleCalibrationProfileError):
            select_calibration_profile_for_runtime(
                evaluation,
                (self.declared_profile(),),
                task_id=None,
                taxonomy_id="support",
                taxonomy_version=3,
            )

    def test_declarations_are_not_copied_from_the_profile(self) -> None:
        evaluation = runtime_evaluation()
        profile = self.declared_profile()
        selected = select_calibration_profile_for_runtime(
            evaluation,
            (profile,),
            task_id="support-routing",
            taxonomy_id="support",
            taxonomy_version=3,
        )
        supplied = runtime_binding_of(
            evaluation,
            task_id="support-routing",
            taxonomy_id="support",
            taxonomy_version=3,
        )
        assert _binding_payloads_equal(selected.binding, supplied)
        absent = runtime_binding_of(evaluation)
        assert not _binding_payloads_equal(selected.binding, absent)


# ---------------------------------------------------------------------------
# Selector / application consistency (PART 41, PART 42)
# ---------------------------------------------------------------------------


class TestSelectorApplicationConsistency:
    def test_eligibility_agrees_with_the_application_binding_gate(self) -> None:
        evaluation = runtime_evaluation()
        binding = runtime_binding_of(evaluation)
        candidates = (
            eligible_profile(),
            other_binding_profile(),
            other_semantics_profile(),
            other_training_profile(),
        )
        for candidate in candidates:
            try:
                candidate.require_binding_match(binding)
                application_accepts = True
            except InvalidDecisionError:
                application_accepts = False
            if application_accepts:
                assert select_calibration_profile_for_runtime(evaluation, (candidate,)) is candidate
            else:
                with pytest.raises(NoEligibleCalibrationProfileError):
                    select_calibration_profile_for_runtime(evaluation, (candidate,))

    def test_application_keeps_its_own_binding_gate(self) -> None:
        evaluation = runtime_evaluation()
        with pytest.raises(InvalidDecisionError):
            apply_profile_to_runtime_evaluation(evaluation, other_binding_profile())


# ---------------------------------------------------------------------------
# Error hierarchy and scope boundaries (PART 8, PART 32, PART 34, PART 37-40)
# ---------------------------------------------------------------------------


class TestHierarchyAndScope:
    def test_selection_errors_share_one_base(self) -> None:
        assert issubclass(NoEligibleCalibrationProfileError, CalibrationProfileSelectionError)
        assert issubclass(
            AmbiguousCalibrationProfileSelectionError, CalibrationProfileSelectionError
        )

    def test_no_match_and_ambiguity_are_distinct_types(self) -> None:
        assert not issubclass(
            NoEligibleCalibrationProfileError,
            AmbiguousCalibrationProfileSelectionError,
        )

    def test_no_selection_provenance_artifact_is_introduced(self) -> None:
        import probvenance.calibration_selection as module

        forbidden = (
            "SelectionResult",
            "SelectionDecision",
            "SelectionTrace",
            "SelectionFingerprint",
            "SelectionPolicyFingerprint",
            "select_and_apply",
            "compatible_with",
            "matches_runtime",
            "binding_distance",
        )
        for name in forbidden:
            assert not hasattr(module, name)

    def test_selector_is_storage_agnostic(self) -> None:
        import probvenance.calibration_selection as module

        code = module_code_without_docstrings(module)
        for forbidden in (
            "calibration_store",
            "DirectoryCalibrationProfileStore",
            "environ",
            "getenv",
            "expanduser",
            "glob",
            "listdir",
            "iterdir",
            "registry",
        ):
            assert forbidden not in code

    def test_selector_does_not_import_quality_metrics(self) -> None:
        import probvenance.calibration_selection as module

        code = module_code_without_docstrings(module).lower()
        for forbidden in (
            "brier",
            "log_loss",
            "reliability",
            "binned_absolute_gap",
            "diagnostics",
            "evaluate_post_calibration",
        ):
            assert forbidden not in code
