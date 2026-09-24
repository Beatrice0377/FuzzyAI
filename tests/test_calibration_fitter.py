"""Phase 4C.2: the first supported scalar fitter.

Covers the mathematical regressions of Part 26 to Part 33, the method
configuration exactness of Part 41, and the adversarial list of Part 47.

Every dataset here is built through the supported runtime path, so the selected
probability is always the probability of the recorded argmax selection. That
restriction is real and is asserted explicitly: ``p = 0`` is unreachable for a
SELECTED probability under the supported construction path, so the endpoint
property at ``p = 0`` is verified at the objective level instead.
"""

import inspect
import math
import subprocess
import sys
from dataclasses import fields, replace
from typing import Any

import pytest
from test_calibration import (
    FakeCategoricalBackend,
    _honest_choice_metadata,
    adjudicated_provenance,
    bool_observation,
    make_choice_decision,
    make_choice_runtime,
    resolved_truth,
)

from probvenance import EvidenceKind, RawEvidence
from probvenance.calibration import (
    _L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE,
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID,
    L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION,
    CalibrationBinding,
    CalibrationDataset,
    CalibrationProfile,
    _binary_logistic_terms,
    _l2_logistic_certificate_threshold,
    _l2_logistic_gradient,
    _l2_logistic_objective,
    _ordered_fitting_rows,
    _selected_probability,
    _solve_l2_logistic,
    fit_l2_logistic_selected_probability,
)
from probvenance.errors import InvalidDecisionError

LOG_UNDERFLOW = -800.0


class TunedBackend(FakeCategoricalBackend):
    """A fake backend whose scoring logits the test chooses explicitly."""

    def __init__(
        self,
        *,
        bool_logits: tuple[float, float] | None = None,
        choice_logits: tuple[float, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bool_logits = bool_logits
        self._choice_logits = choice_logits

    def execute(self, plan: Any) -> RawEvidence:
        if plan.strategy.value == "binary_token_logits" and self._bool_logits is not None:
            values = self._bool_logits
            return RawEvidence(
                kind=EvidenceKind.LOGITS,
                labels=("false", "true"),
                values=values,
                plan_fingerprint=plan.fingerprint,
                metadata={
                    "vocab_logsumexp": max(values) + math.log(4.0),
                    "top_token_id": 9642,
                    "top_token_logit": max(values),
                    "positive_token_id": 9642,
                    "negative_token_id": 3134,
                    "model": self.model,
                    "model_revision": self.model_revision,
                    "tokenizer": self.tokenizer,
                    "tokenizer_revision": self.tokenizer_revision,
                    "rendering_config": self.rendering_config,
                },
            )
        if self._choice_logits is not None:
            values = self._choice_logits
            return RawEvidence(
                kind=EvidenceKind.LOGITS,
                labels=plan.targets,
                values=values,
                plan_fingerprint=plan.fingerprint,
                metadata={
                    **_honest_choice_metadata(plan),
                    "vocab_logsumexp": max(values) + math.log(6.0),
                    "top_token_logit": max(values),
                    "model": self.model,
                    "model_revision": self.model_revision,
                    "tokenizer": self.tokenizer,
                    "tokenizer_revision": self.tokenizer_revision,
                    "rendering_config": self.rendering_config,
                },
            )
        return super().execute(plan)


def bool_observation_at(p_true: float, truth: bool) -> Any:
    """A Bool observation whose uncalibrated P(True) is exactly ``p_true``.

    The backend returns logits whose sigmoid difference is ``p_true``; the
    runtime then records the argmax selection, so the SELECTED probability is
    ``max(p_true, 1 - p_true)``.
    """
    if p_true <= 0.0:
        logits = (0.0, LOG_UNDERFLOW)
    elif p_true >= 1.0:
        logits = (LOG_UNDERFLOW, 0.0)
    else:
        z = math.log(p_true / (1.0 - p_true))
        logits = (0.0, z)
    backend = TunedBackend(bool_logits=logits)
    return bool_observation(resolved_truth(truth), backend=backend)


def observation_at(p_selected: float, correct: bool) -> Any:
    """An observation with a chosen SELECTED probability and winner correctness.

    The selected probability is the probability of the recorded argmax
    selection, so for a binary decision it is at least 0.5:
    ``p_selected = max(p_true, 1 - p_true)``. A value of exactly 0.5 resolves to
    the ``False`` outcome through the frozen first-in-order tie-break, so the
    truth is inverted to keep the requested correctness.
    """
    if p_selected > 0.5:
        return bool_observation_at(p_selected, correct)
    if p_selected != 0.5:
        raise AssertionError(
            "a binary selected probability below 0.5 is not constructible: "
            "selection is the argmax, so the selected probability is >= 0.5"
        )
    return bool_observation_at(0.5, not correct)


def fitting_dataset(rows: list[tuple[float, bool]]) -> CalibrationDataset:
    """A fitting dataset from ``(selected probability, winner correctness)`` rows."""
    observations = [observation_at(p_selected, correct) for p_selected, correct in rows]
    return CalibrationDataset.create(observations)


def fit(rows: list[tuple[float, bool]], l2_strength: float = 1.0) -> CalibrationProfile:
    return fit_l2_logistic_selected_probability(fitting_dataset(rows), l2_strength=l2_strength)


def slope_of(profile: CalibrationProfile) -> float:
    return float(profile.fitted_parameters["slope"])  # type: ignore[arg-type]


def intercept_of(profile: CalibrationProfile) -> float:
    return float(profile.fitted_parameters["intercept"])  # type: ignore[arg-type]


def rows_of(rows: list[tuple[float, bool]]) -> list[tuple[float, float]]:
    dataset = fitting_dataset(rows)
    return sorted(
        (1.0 if observation.correct else 0.0, _selected_probability(observation))
        for observation in dataset.observations
    )


def objective_at(profile: CalibrationProfile, rows: list[tuple[float, bool]], lam: float) -> float:
    pairs = [
        (_selected_probability(o), 1.0 if o.correct else 0.0)
        for o in fitting_dataset(rows).observations
    ]
    return _l2_logistic_objective(pairs, slope_of(profile), intercept_of(profile), lam)


# ---------------------------------------------------------------------------
# Part 26 / 27 / 28: mathematical regressions
# ---------------------------------------------------------------------------


class TestMathematicalRegressions:
    def test_balanced_no_signal_predicts_half(self):
        # p_selected is identical for every row and the labels are balanced, so
        # the unique regularized optimum is (0, 0) and q(p) = 0.5.
        profile = fit([(0.5, False), (0.5, True)])
        assert slope_of(profile) == pytest.approx(0.0, abs=1e-9)
        assert intercept_of(profile) == pytest.approx(0.0, abs=1e-9)

    def test_balanced_no_signal_at_high_probability(self):
        profile = fit([(1.0, False), (1.0, True)])
        assert slope_of(profile) == pytest.approx(0.0, abs=1e-9)
        assert intercept_of(profile) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_separable_keeps_parameters_finite(self):
        # The reachable analogue of the symmetric separable example: the low
        # selected probability carries y=0 and the high one carries y=1.
        profile = fit([(0.5, False), (1.0, True)])
        slope = slope_of(profile)
        intercept = intercept_of(profile)
        assert math.isfinite(slope) and math.isfinite(intercept)
        assert slope > 0.0
        assert intercept < 0.0

    def test_symmetric_separable_mapping_is_ordered(self):
        # A weaker penalty is used because on a two-row dataset the L2 term
        # dominates and pulls the optimum towards (0, 0); this test is about the
        # ordering the mapping induces, not about the strength of the penalty.
        profile = fit([(0.5, False), (1.0, True)], l2_strength=0.05)
        slope = slope_of(profile)
        intercept = intercept_of(profile)
        low = 1.0 / (1.0 + math.exp(-(slope * 0.5 + intercept)))
        high = 1.0 / (1.0 + math.exp(-(slope * 1.0 + intercept)))
        assert low < 0.5 < high

    def test_all_correct_labels_fit_finitely(self):
        profile = fit([(0.5, True), (0.6, True), (0.9, True)])
        assert math.isfinite(slope_of(profile))
        assert math.isfinite(intercept_of(profile))
        assert profile.fingerprint

    def test_all_wrong_labels_fit_finitely(self):
        profile = fit([(0.5, False), (0.6, False), (0.9, False)])
        assert math.isfinite(slope_of(profile))
        assert math.isfinite(intercept_of(profile))
        assert profile.fingerprint

    def test_completely_separated_labels_fit_finitely(self):
        profile = fit([(0.5, False), (0.6, False), (0.9, True), (1.0, True)])
        assert math.isfinite(slope_of(profile))
        assert math.isfinite(intercept_of(profile))
        assert slope_of(profile) > 0.0

    def test_large_l2_strength_does_not_overflow_the_hessian_solve(self):
        # det(lambda*I + M) overflows to infinity above roughly 1e154 when the
        # penalty is added to a diagonal entry before the determinant is formed.
        for l2_strength in (1e154, 1e155, 1e300):
            profile = fit_l2_logistic_selected_probability(
                fitting_dataset([(0.5, False), (0.9, True)]),
                l2_strength=l2_strength,
            )
            assert math.isfinite(slope_of(profile))
            assert math.isfinite(intercept_of(profile))

    def test_constant_probability_with_tiny_l2_strength_still_fits(self):
        # A constant selected probability makes the data Hessian rank
        # deficient, so adding a tiny penalty to a diagonal entry rounds it away
        # and makes a well-posed fit look singular.
        profile = fit_l2_logistic_selected_probability(
            fitting_dataset([(0.9, True), (0.9, True)]),
            l2_strength=1e-17,
        )
        assert math.isfinite(slope_of(profile))
        assert math.isfinite(intercept_of(profile))

    def test_selected_probability_one_endpoint_fits(self):
        # p_selected = 1.0 exactly, reached by an underflowed opposite logit.
        profile = fit([(1.0, True), (0.5, False)])
        assert math.isfinite(slope_of(profile))
        assert math.isfinite(intercept_of(profile))
        assert slope_of(profile) > 0.0

    def test_objective_is_finite_at_the_p_zero_endpoint(self):
        # p = 0 cannot be a SELECTED probability under the supported runtime
        # (selection is the argmax), so the endpoint is exercised directly on
        # the objective, which is where an epsilon or clipping would live.
        objective = _l2_logistic_objective([(0.0, 0.0), (1.0, 1.0)], 1.0, -1.0, 1.0)
        assert math.isfinite(objective)

    def test_objective_is_finite_at_both_endpoints_for_fitted_parameters(self):
        profile = fit([(0.5, False), (1.0, True)])
        slope, intercept = slope_of(profile), intercept_of(profile)
        assert math.isfinite(_l2_logistic_objective([(0.0, 0.0)], slope, intercept, 1.0))
        assert math.isfinite(_l2_logistic_objective([(1.0, 1.0)], slope, intercept, 1.0))

    def test_first_order_optimality_against_independent_gradient(self):
        rows: list[tuple[float, bool]] = [
            (0.5, False),
            (0.6, True),
            (0.8, False),
            (1.0, True),
            (0.7, True),
        ]
        lam = 0.5
        profile = fit(rows, l2_strength=lam)
        slope, intercept = slope_of(profile), intercept_of(profile)
        pairs = [
            (_selected_probability(o), 1.0 if o.correct else 0.0)
            for o in fitting_dataset(rows).observations
        ]
        # Independently written gradient, not the module's helper.
        grad = [0.0, 0.0]
        for p, y in pairs:
            q = 1.0 / (1.0 + math.exp(-(slope * p + intercept)))
            grad[0] += (q - y) * p
            grad[1] += q - y
        grad[0] = grad[0] / len(pairs) + lam * slope
        grad[1] = grad[1] / len(pairs) + lam * intercept
        assert max(abs(grad[0]), abs(grad[1])) <= 1e-8

    def test_objective_decreases_strictly_on_a_learnable_problem(self):
        rows: list[tuple[float, bool]] = [
            (0.5, False),
            (0.55, False),
            (0.9, True),
            (1.0, True),
        ]
        lam = 0.1
        profile = fit(rows, l2_strength=lam)
        pairs = [
            (_selected_probability(o), 1.0 if o.correct else 0.0)
            for o in fitting_dataset(rows).observations
        ]
        initial = _l2_logistic_objective(pairs, 0.0, 0.0, lam)
        final = _l2_logistic_objective(pairs, slope_of(profile), intercept_of(profile), lam)
        assert final <= initial
        assert final < initial


# ---------------------------------------------------------------------------
# Part 6: input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize("bad", [None, {}, [], "dataset", 3, object()])
    def test_non_dataset_is_rejected(self, bad: Any):
        with pytest.raises(InvalidDecisionError, match="CalibrationDataset"):
            fit_l2_logistic_selected_probability(bad, l2_strength=1.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad", [True, False, 0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf"), "1", None]
    )
    def test_invalid_l2_strength_is_rejected(self, bad: Any):
        dataset = fitting_dataset([(0.5, True)])
        with pytest.raises(InvalidDecisionError, match="l2_strength"):
            fit_l2_logistic_selected_probability(dataset, l2_strength=bad)

    def test_integer_l2_strength_is_normalized_to_float(self):
        profile = fit_l2_logistic_selected_probability(
            fitting_dataset([(0.5, True)]), l2_strength=2
        )
        assert profile.method_configuration["l2_strength"] == 2.0
        assert isinstance(profile.method_configuration["l2_strength"], float)

    def test_unrepresentable_integer_l2_strength_is_rejected_cleanly(self):
        with pytest.raises(InvalidDecisionError, match="l2_strength"):
            fit_l2_logistic_selected_probability(
                fitting_dataset([(0.5, True)]), l2_strength=10**400
            )


# ---------------------------------------------------------------------------
# Part 12 / 22 / 41: method configuration and fitted parameters
# ---------------------------------------------------------------------------


class TestMethodConfiguration:
    def test_method_identity_is_the_l2_logistic_method(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        assert profile.method_id == L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID
        assert profile.method_version == L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_VERSION

    def test_configuration_records_every_semantics_bearing_policy(self):
        config = fit([(0.5, True)], l2_strength=0.25).method_configuration
        assert set(config) == {
            "objective",
            "l2_strength",
            "regularized_parameters",
            "input_transform",
            "endpoint_policy",
            "solver",
        }
        assert config["objective"] == {"id": "mean-bernoulli-nll-plus-l2", "version": 1}
        assert config["l2_strength"] == 0.25
        assert config["regularized_parameters"] == ("slope", "intercept")
        assert config["input_transform"] == {
            "id": "identity-selected-probability",
            "version": 1,
        }
        assert config["endpoint_policy"] == {
            "id": "exact-raw-selected-probability",
            "version": 1,
            "epsilon": None,
            "clipping": False,
            "label_smoothing": False,
        }
        solver = config["solver"]
        assert solver["id"] == "newton-backtracking"
        assert solver["version"] == 2
        assert solver["initial_slope"] == 0.0
        assert solver["initial_intercept"] == 0.0
        assert solver["max_iterations"] == 100
        assert solver["backtracking_factor"] == 0.5
        assert solver["armijo_coefficient"] == 1e-4
        assert solver["max_backtracking_steps"] == 60
        assert solver["convergence"] == {
            "id": "strong-convexity-objective-gap",
            "version": 1,
            "objective_suboptimality_tolerance": (_L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE),
        }
        assert "gradient_tolerance" not in solver

    def test_fitted_parameters_are_only_slope_and_intercept(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        assert set(profile.fitted_parameters) == {"slope", "intercept"}
        assert isinstance(profile.fitted_parameters["slope"], float)
        assert isinstance(profile.fitted_parameters["intercept"], float)

    def test_fitted_parameters_carry_no_telemetry(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        for forbidden in ("iterations", "iteration", "loss", "duration", "timestamp", "hostname"):
            assert forbidden not in profile.fitted_parameters

    def test_profile_identity_commits_target_and_input_score(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        assert profile.target_id == "winner_correctness"
        assert profile.target_version == 1
        assert profile.input_score_id == "uncalibrated-selected-probability"
        assert profile.input_score_version == 1

    def test_profile_fingerprint_version_is_still_one(self):
        assert CALIBRATION_PROFILE_FINGERPRINT_VERSION == 1
        assert fit([(0.5, True)], l2_strength=1.0).canonical_payload()["v"] == 1

    def test_profile_fingerprint_is_computable(self):
        assert len(fit([(0.5, True)], l2_strength=1.0).fingerprint) == 64


# ---------------------------------------------------------------------------
# Part 18 / 19 / 30 / 31: determinism and provenance
# ---------------------------------------------------------------------------


class TestDeterminismAndProvenance:
    def test_repeated_fit_is_identical(self):
        rows = [(0.5, False), (0.9, True)]
        first = fit(rows, l2_strength=0.5)
        second = fit(rows, l2_strength=0.5)
        assert first.fitted_parameters == second.fitted_parameters
        assert first.method_configuration == second.method_configuration
        assert first.fingerprint == second.fingerprint

    def test_row_permutation_gives_the_same_profile(self):
        forward = fit([(0.5, False), (0.6, True), (0.9, True)], l2_strength=0.3)
        backward = fit([(0.9, True), (0.6, True), (0.5, False)], l2_strength=0.3)
        assert forward.training_dataset_fingerprint == backward.training_dataset_fingerprint
        assert forward.fitted_parameters == backward.fitted_parameters
        assert forward.fingerprint == backward.fingerprint

    def test_multiplicity_changes_the_profile_fingerprint(self):
        single = fit([(0.5, False), (0.9, True)], l2_strength=1.0)
        doubled = fit([(0.5, False), (0.9, True), (0.9, True)], l2_strength=1.0)
        assert single.training_dataset_fingerprint != doubled.training_dataset_fingerprint
        assert single.fingerprint != doubled.fingerprint

    def test_duplicate_rows_are_not_deduplicated(self):
        single = fit([(0.9, True)], l2_strength=0.01)
        doubled = fit([(0.9, True), (0.9, True)], l2_strength=0.01)
        assert single.training_dataset_fingerprint != doubled.training_dataset_fingerprint
        assert single.fingerprint != doubled.fingerprint

    def test_different_training_dataset_changes_the_profile_fingerprint(self):
        first = fit([(0.5, False), (0.9, True)], l2_strength=1.0)
        second = fit([(0.5, False), (1.0, True)], l2_strength=1.0)
        assert first.fingerprint != second.fingerprint

    def test_different_l2_strength_changes_the_profile_fingerprint(self):
        dataset = fitting_dataset([(0.5, False), (0.9, True)])
        first = fit_l2_logistic_selected_probability(dataset, l2_strength=0.5)
        second = fit_l2_logistic_selected_probability(dataset, l2_strength=2.0)
        assert first.method_configuration != second.method_configuration
        assert first.fingerprint != second.fingerprint

    def test_same_l2_strength_value_via_int_and_float_collapses(self):
        dataset = fitting_dataset([(0.5, False), (0.9, True)])
        from_int = fit_l2_logistic_selected_probability(dataset, l2_strength=1)
        from_float = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        assert from_int.fingerprint == from_float.fingerprint

    def test_fitting_rows_are_row_order_independent(self):
        rows = [(0.5, False), (0.6, True), (0.9, True)]
        assert rows_of(rows) == rows_of(list(reversed(rows)))


# ---------------------------------------------------------------------------
# Part 35 / 36: taxonomy and binding inheritance
# ---------------------------------------------------------------------------


class TestInheritedProfileContract:
    def test_binding_is_the_dataset_binding_exactly(self):
        dataset = fitting_dataset([(0.5, True)])
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        assert profile.binding is dataset.binding
        profile.require_binding_match(dataset.binding)

    def test_different_binding_is_rejected_by_the_inherited_matcher(self):
        dataset = fitting_dataset([(0.5, True)])
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        other = CalibrationBinding(
            probability_formulation_fingerprint="f" * 64,
            probability_formulation_fingerprint_version=2,
            model="other-model",
            model_revision="rev-1",
            tokenizer="fake-tokenizer",
            tokenizer_revision="tok-rev-1",
            rendering_semantics={"v": 1, "enable_thinking": False},
            task_id=None,
            taxonomy_id=None,
            taxonomy_version=None,
        )
        with pytest.raises(InvalidDecisionError, match="does not match"):
            profile.require_binding_match(other)

    def test_taxonomy_mismatch_still_fails_closed(self):
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        from probvenance.calibration import CalibrationObservation

        observation = CalibrationObservation.from_evaluation(
            evaluation,
            resolved_truth("shipping", taxonomy_id="legacy-support", taxonomy_version=1),
            taxonomy_id="support",
            taxonomy_version=3,
        )
        with pytest.raises(InvalidDecisionError, match="taxonomy"):
            fit_l2_logistic_selected_probability(
                CalibrationDataset.create([observation]), l2_strength=1.0
            )

    def test_taxonomy_version_mismatch_still_fails_closed(self):
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        from probvenance.calibration import CalibrationObservation

        observation = CalibrationObservation.from_evaluation(
            evaluation,
            resolved_truth("shipping", taxonomy_id="support", taxonomy_version=1),
            taxonomy_id="support",
            taxonomy_version=3,
        )
        with pytest.raises(InvalidDecisionError, match="version"):
            fit_l2_logistic_selected_probability(
                CalibrationDataset.create([observation]), l2_strength=1.0
            )

    def test_taxonomy_validation_is_not_duplicated_in_the_fitter(self):
        source = inspect.getsource(
            sys.modules["probvenance.calibration"].__dict__["fit_l2_logistic_selected_probability"]
        )
        assert "taxonomy_id" not in source
        assert "GroundTruthSemantics" not in source


# ---------------------------------------------------------------------------
# Part 17 / 37 / 38 / 39 / 40: boundary and absence checks
# ---------------------------------------------------------------------------


class TestScopeBoundaries:
    def test_fitter_uses_the_internal_builder_and_adds_no_identity_assembly(self):
        import probvenance.calibration as calibration_module

        source = inspect.getsource(
            calibration_module.__dict__["fit_l2_logistic_selected_probability"]
        )
        assert "_from_fitted_state" in source
        assert "training_dataset_fingerprint" not in source
        assert "canonical_payload" not in source
        dataset = fitting_dataset([(0.5, True), (0.9, False)])
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        assert profile.training_dataset_fingerprint == dataset.fingerprint
        assert profile.binding is dataset.binding
        assert profile.canonical_payload()["method_configuration"] == (
            calibration_module._l2_logistic_method_configuration(1.0)
        )

    def test_no_public_application_api_exists(self):
        import probvenance.calibration as calibration_module

        for forbidden in ("profile_predict", "apply_calibration", "calibrate_result"):
            assert not hasattr(calibration_module, forbidden)
        assert not hasattr(CalibrationProfile, "predict")
        assert not hasattr(CalibrationProfile, "apply")

    def test_no_post_calibration_metric_exists(self):
        import probvenance.calibration_evaluation as evaluation_module

        source = inspect.getsource(evaluation_module)
        for forbidden in ("post_calibration", "post-calibration", "calibrated_brier"):
            assert forbidden not in source

    def test_fitter_performs_no_hyperparameter_search(self):
        source = inspect.getsource(
            sys.modules["probvenance.calibration"].__dict__["fit_l2_logistic_selected_probability"]
        )
        for forbidden in ("for candidate", "grid", "cross_val", "search"):
            assert forbidden not in source

    def test_no_sample_or_class_weighting_exists(self):
        import probvenance.calibration as calibration_module

        source = inspect.getsource(calibration_module)
        for forbidden in ("sample_weight", "class_weight", "oversample", "undersample"):
            assert forbidden not in source

    def test_fitter_is_not_exported_at_top_level(self):
        import probvenance

        assert not hasattr(probvenance, "fit_l2_logistic_selected_probability")
        assert not hasattr(probvenance, "CalibrationProfile")

    def test_runtime_result_remains_uncalibrated(self):
        runtime, _ = make_choice_runtime()
        evaluation = runtime.evaluate_with_trace(make_choice_decision())
        assert evaluation.result.calibrated is False
        assert evaluation.result.predicted_correctness is None

    def test_no_new_dependency_is_imported_by_the_fitter(self):
        import probvenance.calibration as calibration_module

        source = inspect.getsource(calibration_module)
        for forbidden in ("scipy", "numpy", "sklearn", "torch", "transformers"):
            assert forbidden not in source

    def test_no_endpoint_manipulation_tokens_in_the_fitter_code(self):
        import probvenance.calibration as calibration_module

        source = inspect.getsource(calibration_module)
        for forbidden in ("epsilon =", "clip(", "min(max(", "logit("):
            assert forbidden not in source

    def test_selected_probability_extraction_has_one_owner(self):
        import probvenance.calibration as calibration_module
        import probvenance.calibration_evaluation as evaluation_module

        assert hasattr(calibration_module, "_selected_probability")
        assert "def _selected_probability" not in inspect.getsource(evaluation_module)
        assert evaluation_module._selected_probability is calibration_module._selected_probability

    def test_fitter_does_not_recompute_the_winner(self):
        source = inspect.getsource(
            sys.modules["probvenance.calibration"].__dict__["fit_l2_logistic_selected_probability"]
        )
        for forbidden in ("argmax", "max(", "tie", "outcome_order"):
            assert forbidden not in source


# ---------------------------------------------------------------------------
# Part 21: forged correctness never coerced
# ---------------------------------------------------------------------------


class TestForgedInputsFailClosed:
    def test_non_bool_correctness_is_rejected_without_coercion(self):
        dataset = fitting_dataset([(0.5, True)])
        object.__setattr__(dataset.observations[0], "correct", None)
        with pytest.raises(InvalidDecisionError, match="winner-correctness"):
            fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)

    def test_forged_zero_selected_probability_still_fits(self):
        # p_selected = 0 is unreachable through the supported runtime, because
        # the recorded selection is the argmax. A forged observation must still
        # be an ordinary finite feature rather than trip an epsilon or logit
        # path, so the whole fitter is exercised on it and not just the
        # objective.
        observation = bool_observation_at(1.0, True)
        object.__setattr__(observation, "selected_value", False)
        assert _selected_probability(observation) == 0.0
        dataset = CalibrationDataset.create([observation])
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        assert math.isfinite(slope_of(profile))
        assert math.isfinite(intercept_of(profile))

    def test_unconverged_fit_never_produces_a_profile(self):
        # A frozen solver that cannot iterate must fail loudly, not return a
        # partially converged parameter pair.
        import probvenance.calibration as calibration_module

        dataset = fitting_dataset([(0.5, True)])
        original = calibration_module._L2_LOGISTIC_MAX_ITERATIONS
        try:
            object.__setattr__(calibration_module, "_L2_LOGISTIC_MAX_ITERATIONS", 0)
            with pytest.raises(InvalidDecisionError, match="uncertified"):
                fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        finally:
            object.__setattr__(calibration_module, "_L2_LOGISTIC_MAX_ITERATIONS", original)


# ---------------------------------------------------------------------------
# Part 4 / 44: import graph and Phase 4B immutability
# ---------------------------------------------------------------------------


class TestImportGraphAndFrozenIdentities:
    def test_calibration_does_not_import_evaluation(self):
        source = inspect.getsource(sys.modules["probvenance.calibration"])
        assert "calibration_evaluation" not in source

    def test_import_calibration_first(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import probvenance.calibration"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    def test_import_evaluation_first(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import probvenance.calibration_evaluation"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    def test_importing_calibration_alone_does_not_load_evaluation(self):
        script = (
            "import sys, probvenance.calibration; "
            "print('probvenance.calibration_evaluation' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "False"

    def test_phase_4b_artifact_fingerprints_are_unchanged(self):
        from test_calibration_evaluation import (
            _reliability_observation,
            evaluation_dataset,
        )

        from probvenance.calibration_evaluation import (
            evaluate_uncalibrated_winner_brier,
            evaluate_uncalibrated_winner_diagnostics,
            evaluate_uncalibrated_winner_log_loss,
            evaluate_uncalibrated_winner_reliability,
            evaluate_winner_binned_absolute_gap,
        )

        dataset = evaluation_dataset(
            [
                _reliability_observation(0.10, False),
                _reliability_observation(0.40, True),
                _reliability_observation(0.60, True),
                _reliability_observation(0.90, False),
            ]
        )
        reliability = evaluate_uncalibrated_winner_reliability(dataset, bin_count=2)
        assert evaluate_uncalibrated_winner_brier(dataset).fingerprint == (
            "9b1841c3cd3749d8ef8f65bec2b70312803371b9eebcbd59ada5e149823ffd3c"
        )
        assert evaluate_uncalibrated_winner_log_loss(dataset).fingerprint == (
            "db9b7e2435e9a8d74c3fc6c064d2e6bc2f7dedff4c6c42201433f4f6ae1c9db5"
        )
        assert evaluate_uncalibrated_winner_diagnostics(dataset).fingerprint == (
            "4405acc66efbaf97e48fb2783e301a98c67fa143d7351e4e40deb6cbfb055279"
        )
        assert reliability.fingerprint == (
            "0c555f03397268e03e7093556ba228b4288c597be564ea51ee853a044eb006d0"
        )
        assert evaluate_winner_binned_absolute_gap(reliability).fingerprint == (
            "7bb2e5ad24b6f05e4885aa756d93f032cc20994dcb010236b85d24d7303f9368"
        )


# ---------------------------------------------------------------------------
# Part 25 / 42: the 4C.1 artifact foundation still holds
# ---------------------------------------------------------------------------


class TestProfileFoundationStillHolds:
    def test_direct_construction_still_rejected(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        with pytest.raises(InvalidDecisionError, match="constructed directly"):
            CalibrationProfile(
                binding=profile.binding,
                ground_truth_semantics=profile.ground_truth_semantics,
                target_id=profile.target_id,
                target_version=profile.target_version,
                input_score_id=profile.input_score_id,
                input_score_version=profile.input_score_version,
                method_id=profile.method_id,
                method_version=profile.method_version,
                method_configuration=profile.method_configuration,
                fitted_parameters=profile.fitted_parameters,
                training_dataset_fingerprint=profile.training_dataset_fingerprint,
                training_dataset_fingerprint_version=(profile.training_dataset_fingerprint_version),
            )

    def test_replace_still_rejected(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        with pytest.raises(InvalidDecisionError):
            replace(profile)

    def test_caller_mutation_of_the_dataset_does_not_change_the_fitted_profile(self):
        dataset = fitting_dataset([(0.5, False), (0.9, True)])
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=1.0)
        before = profile.fingerprint
        object.__setattr__(dataset, "observations", ())
        assert profile.fingerprint == before

    def test_payload_mutation_does_not_change_the_fitted_profile(self):
        profile = fit([(0.5, True)], l2_strength=1.0)
        before = profile.fingerprint
        payload = profile.canonical_payload()
        payload["method_id"] = "forged"
        configuration = payload["method_configuration"]
        assert isinstance(configuration, dict)
        configuration["l2_strength"] = 99.0
        assert profile.fingerprint == before
        assert profile.canonical_payload()["method_id"] == (
            L2_LOGISTIC_SELECTED_PROBABILITY_METHOD_ID
        )

    def test_profile_declares_exactly_the_declared_identity_fields(self):
        names = {field.name for field in fields(CalibrationProfile)}
        assert names == {
            "binding",
            "ground_truth_semantics",
            "target_id",
            "target_version",
            "input_score_id",
            "input_score_version",
            "method_id",
            "method_version",
            "method_configuration",
            "fitted_parameters",
            "training_dataset_fingerprint",
            "training_dataset_fingerprint_version",
        }

    def test_provenance_is_adjudicated_in_every_used_dataset(self):
        dataset = fitting_dataset([(0.5, True)])
        assert dataset.observations[0].ground_truth.provenance == adjudicated_provenance()
        assert all(observation.fit_eligible for observation in dataset.observations)


# ---------------------------------------------------------------------------
# Phase 4C.2a: numerical convergence hardening
# ---------------------------------------------------------------------------


def _constant_rows(probability: float, correct: bool) -> list[tuple[float, bool]]:
    return [(probability, correct), (probability, correct)]


def _oracle_constant_optimum(
    probability: float, correct: bool, l2_strength: float
) -> tuple[float, float]:
    """Independent scalar oracle for an all-identical-score fitting problem.

    With every row sharing the score ``p`` and the label, the first-order
    conditions force ``slope = p * intercept``, so the optimum reduces to one
    scalar equation in ``z = intercept * (1 + p**2)``. It is found by bisection
    so the oracle never consults the fitter.
    """
    scale = 1.0 + probability * probability
    if correct:

        def residual(z: float) -> float:
            return 1.0 / (1.0 + math.exp(z)) - l2_strength * z / scale

        low, high = 0.0, 1.0
        while residual(high) > 0.0:
            high *= 2.0
    else:

        def residual(z: float) -> float:
            return -(1.0 / (1.0 + math.exp(-z)) + l2_strength * z / scale)

        high, low = 0.0, -1.0
        while residual(low) <= 0.0:
            low *= 2.0
    for _ in range(300):
        middle = (low + high) / 2.0
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    intercept = (low + high) / 2.0 / scale
    return probability * intercept, intercept


class TestNumericalConvergenceHardening:
    """Part 1 to Part 21 and the Part 35 attacks of the 4C.2a repair."""

    def test_terms_match_independent_central_differences(self):
        step = 1e-4

        def stable_nll(value: float, correct: bool) -> float:
            exponent = -value if correct else value
            return math.log1p(math.exp(exponent))

        for z in (-3.0, -1.0, 0.0, 1.0, 3.0):
            for correct in (True, False):
                _loss, residual, curvature = _binary_logistic_terms(z, correct)
                numeric_residual = (
                    stable_nll(z + step, correct) - stable_nll(z - step, correct)
                ) / (2.0 * step)
                numeric_curvature = (
                    stable_nll(z + step, correct)
                    - 2.0 * stable_nll(z, correct)
                    + stable_nll(z - step, correct)
                ) / (step * step)
                assert residual == pytest.approx(numeric_residual, rel=1e-5, abs=1e-9)
                assert curvature == pytest.approx(numeric_curvature, rel=1e-3, abs=1e-6)

    def test_objective_and_gradient_agree_by_finite_difference(self):
        rows = [(0.2, False), (0.55, True), (0.9, True)]
        l2_strength = 0.7
        slope, intercept = 0.4, -0.3
        step = 1e-6
        numeric_slope = (
            _l2_logistic_objective(rows, slope + step, intercept, l2_strength)
            - _l2_logistic_objective(rows, slope - step, intercept, l2_strength)
        ) / (2.0 * step)
        numeric_intercept = (
            _l2_logistic_objective(rows, slope, intercept + step, l2_strength)
            - _l2_logistic_objective(rows, slope, intercept - step, l2_strength)
        ) / (2.0 * step)
        gradient_slope, gradient_intercept = _l2_logistic_gradient(
            rows, slope, intercept, l2_strength
        )
        assert gradient_slope == pytest.approx(numeric_slope, abs=1e-8)
        assert gradient_intercept == pytest.approx(numeric_intercept, abs=1e-8)

    def test_large_positive_z_terms_do_not_cancel(self):
        for z in (20.0, 40.0):
            expected = math.exp(-z) / (1.0 + math.exp(-z))
            loss, residual, curvature = _binary_logistic_terms(z, True)
            assert loss > 0.0
            assert residual < 0.0
            assert curvature > 0.0
            assert loss == pytest.approx(expected, rel=1e-12)
            assert residual == pytest.approx(-expected, rel=1e-12)
            assert curvature == pytest.approx(expected, rel=1e-12)

    def test_large_negative_z_terms_do_not_cancel(self):
        for z in (-20.0, -40.0):
            expected = math.exp(z) / (1.0 + math.exp(z))
            loss, residual, curvature = _binary_logistic_terms(z, False)
            assert loss > 0.0
            assert residual > 0.0
            assert curvature > 0.0
            assert loss == pytest.approx(expected, rel=1e-12)
            assert residual == pytest.approx(expected, rel=1e-12)
            assert curvature == pytest.approx(expected, rel=1e-12)

    def test_terms_stay_representable_at_extreme_z(self):
        for z, correct, sign in ((100.0, True, -1.0), (-100.0, False, 1.0)):
            loss, residual, curvature = _binary_logistic_terms(z, correct)
            assert loss > 0.0
            assert curvature > 0.0
            assert residual != 0.0
            assert residual * sign > 0.0

    def test_the_cancelling_objective_form_is_gone_from_the_source(self):
        import probvenance.calibration as calibration_module

        source = inspect.getsource(calibration_module)
        assert "softplus(z) - y * z" not in source
        assert "softplus(z) - y*z" not in source

    def test_the_certificate_threshold_is_the_strong_convexity_bound(self):
        assert _l2_logistic_certificate_threshold(1.0) == pytest.approx(
            math.sqrt(2.0 * _L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE), rel=1e-12
        )
        assert _l2_logistic_certificate_threshold(4.0) == pytest.approx(
            2.0 * _l2_logistic_certificate_threshold(1.0), rel=1e-12
        )

    def test_the_certificate_threshold_never_underflows_or_overflows(self):
        assert 2.0 * 1e-320 * _L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE == 0.0
        for l2_strength in (5e-324, 1e-320, 1e-20, 1.0, 1e154, 1e300):
            threshold = _l2_logistic_certificate_threshold(l2_strength)
            assert math.isfinite(threshold)
            assert threshold > 0.0

    def test_the_previously_accepted_point_fails_the_certificate(self):
        rows = _constant_rows(0.9, True)
        for l2_strength in (1e-16, 1e-18, 1e-20):
            slope, intercept = 11.5305, 12.8117
            norm = math.hypot(*_l2_logistic_gradient(rows, slope, intercept, l2_strength))
            assert norm > _l2_logistic_certificate_threshold(l2_strength)
            assert norm * norm / (2.0 * l2_strength) > (
                _L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE
            )

    def test_the_fitter_never_returns_the_old_false_success(self):
        dataset = fitting_dataset([(0.9, True), (0.9, True)])
        rows = _constant_rows(0.9, True)
        for l2_strength in (1e-18, 1e-20):
            try:
                profile = fit_l2_logistic_selected_probability(dataset, l2_strength=l2_strength)
            except InvalidDecisionError:
                continue
            gradient = _l2_logistic_gradient(
                rows, slope_of(profile), intercept_of(profile), l2_strength
            )
            assert math.hypot(*gradient) <= _l2_logistic_certificate_threshold(l2_strength)
            assert abs(0.9 * slope_of(profile) + intercept_of(profile) - 23.1891) > 1.0

    def test_a_tiny_strength_either_certifies_or_fails_closed(self):
        # Part 12 accepts either outcome; what is forbidden is a success that
        # the certificate does not cover, or a stale underflow diagnosis.
        dataset = fitting_dataset([(0.9, True), (0.9, True)])
        rows = _ordered_fitting_rows(dataset)
        for l2_strength in (1e-19, 1e-20):
            try:
                profile = fit_l2_logistic_selected_probability(dataset, l2_strength=l2_strength)
            except InvalidDecisionError as error:
                message = str(error)
                assert "certify" in message
                assert "underflow" not in message
                assert "gradient tolerance" not in message
                continue
            gradient = _l2_logistic_gradient(
                rows, slope_of(profile), intercept_of(profile), l2_strength
            )
            assert math.hypot(*gradient) <= _l2_logistic_certificate_threshold(l2_strength)

    def test_constant_positive_labels_match_the_independent_scalar_oracle(self):
        l2_strength = 1e-18
        dataset = fitting_dataset([(0.9, True), (0.9, True)])
        rows = _ordered_fitting_rows(dataset)
        probability = rows[0][0]
        profile = fit_l2_logistic_selected_probability(dataset, l2_strength=l2_strength)
        slope, intercept = slope_of(profile), intercept_of(profile)
        assert slope == pytest.approx(probability * intercept, rel=1e-9)
        oracle_slope, oracle_intercept = _oracle_constant_optimum(probability, True, l2_strength)
        fitted_objective = _l2_logistic_objective(rows, slope, intercept, l2_strength)
        oracle_objective = _l2_logistic_objective(rows, oracle_slope, oracle_intercept, l2_strength)
        assert fitted_objective >= oracle_objective
        assert fitted_objective - oracle_objective <= (
            _L2_LOGISTIC_OBJECTIVE_SUBOPTIMALITY_TOLERANCE
        )

    def test_ordinary_strength_parameters_reach_the_independent_oracle(self):
        # Where the strong-convexity bound is tight, the fitted parameters must
        # land on the independently derived optimum, not merely certify.
        rows = [(0.9, 1.0), (0.9, 1.0)]
        for l2_strength in (0.05, 1.0):
            oracle_slope, oracle_intercept = _oracle_constant_optimum(0.9, True, l2_strength)
            slope, intercept = _solve_l2_logistic(rows, l2_strength)
            assert slope == pytest.approx(oracle_slope, rel=1e-9)
            assert intercept == pytest.approx(oracle_intercept, rel=1e-9)

    def test_tiny_strength_looseness_is_bounded_and_certified(self):
        # A tiny strong-convexity modulus makes the objective-gap certificate
        # loose in PARAMETER terms: the returned point can sit a few percent
        # away from the true optimum while still certifying. That looseness is
        # recorded rather than hidden, and the certificate still holds.
        rows = [(0.9, 1.0), (0.9, 1.0)]
        l2_strength = 1e-18
        oracle_slope, oracle_intercept = _oracle_constant_optimum(0.9, True, l2_strength)
        slope, intercept = _solve_l2_logistic(rows, l2_strength)
        relative_gap = max(
            abs(slope - oracle_slope) / abs(oracle_slope),
            abs(intercept - oracle_intercept) / abs(oracle_intercept),
        )
        assert 1e-6 < relative_gap < 0.5
        gradient = _l2_logistic_gradient(rows, slope, intercept, l2_strength)
        assert math.hypot(*gradient) <= _l2_logistic_certificate_threshold(l2_strength)

    def test_constant_negative_labels_are_the_exact_mirror(self):
        probability, l2_strength = 0.9, 1e-18
        positive = fit([(probability, True), (probability, True)], l2_strength=l2_strength)
        negative = fit([(probability, False), (probability, False)], l2_strength=l2_strength)
        assert slope_of(negative) == pytest.approx(-slope_of(positive), rel=1e-9)
        assert intercept_of(negative) == pytest.approx(-intercept_of(positive), rel=1e-9)

    def test_balanced_no_signal_is_the_exact_zero_optimum(self):
        for l2_strength in (1e-20, 1.0):
            profile = fit([(0.5, False), (0.5, True)], l2_strength=l2_strength)
            assert slope_of(profile) == 0.0
            assert intercept_of(profile) == 0.0

    def test_every_produced_profile_satisfies_the_certificate(self):
        datasets = [
            [(0.5, False), (0.5, True)],
            [(0.5, False), (1.0, True)],
            [(0.6, False), (0.9, True)],
            [(0.9, True), (0.9, True)],
            [(0.9, False), (0.9, False)],
            [(1.0, True)],
        ]
        for rows in datasets:
            for l2_strength in (0.05, 0.5, 1.0, 2.0, 1e-12, 1e-16):
                try:
                    profile = fit(rows, l2_strength=l2_strength)
                except InvalidDecisionError:
                    continue
                gradient = _l2_logistic_gradient(
                    rows, slope_of(profile), intercept_of(profile), l2_strength
                )
                assert math.hypot(*gradient) <= (_l2_logistic_certificate_threshold(l2_strength))

    def test_ordinary_strength_optima_are_certified_and_stable(self):
        # Pinned certified baselines for the ordinary strength range. A solver
        # contract change re-measures them; the values match the pre-hardening
        # solver to within one float64 ulp, so a change beyond that is a
        # regression rather than a re-measurement.
        expected = {
            0.01: (4.223427207949436, -3.010267396178393),
            0.05: (1.4100660705060897, -0.8763321265979911),
            0.1: (0.8039038041256639, -0.42911826388273655),
            0.5: (0.20517543661252272, -0.0512414386090694),
            1.0: (0.1108100666536127, -0.016614064906515198),
            2.0: (0.058394873785144914, -0.004865462890014107),
        }
        rows = [(0.5, False), (1.0, True)]
        for l2_strength, (slope, intercept) in expected.items():
            profile = fit(rows, l2_strength=l2_strength)
            assert slope_of(profile) == slope
            assert intercept_of(profile) == intercept
            gradient = _l2_logistic_gradient(rows, slope, intercept, l2_strength)
            assert math.hypot(*gradient) <= _l2_logistic_certificate_threshold(l2_strength)

    def test_subnormal_strengths_are_valid_inputs(self):
        import probvenance.calibration as calibration_module

        assert calibration_module._require_l2_strength(1e-300) == 1e-300
        assert calibration_module._require_l2_strength(5e-324) == 5e-324
