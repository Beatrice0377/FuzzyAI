"""Offline tests for the semantic signal experiment harness.

No model, no GPU, no network: deterministic and fast. The dry-run path uses
the fake backend from run.py; the metrics are checked against hand-built
synthetic record sets, including deliberately violated orderings.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

HARNESS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HARNESS_DIR.parents[1]
SRC = REPO_ROOT / "src"

for path in (str(SRC), str(HARNESS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HARNESS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


formulations = _load("formulations")
run_module = _load("run")
metrics = _load("metrics")

from probvenance import ScoringDoctrine, VerbalizerError  # noqa: E402
from probvenance.compiler import BoolCompiler  # noqa: E402
from probvenance.doctrine import BINARY_SEMANTIC_JUDGMENT_V1  # noqa: E402
from probvenance.fingerprint import fingerprint  # noqa: E402


class TestCaseSet:
    def test_parses_and_structure_is_complete(self) -> None:
        case_set = metrics.load_case_set()
        assert case_set["case_set_version"] == "semantic-signal-v1"
        assert len(case_set["rung_order"]) == 5
        assert len(case_set["ladders"]) == 8
        for ladder in case_set["ladders"]:
            assert set(ladder["rungs"]) == set(case_set["rung_order"])
        assert len(case_set["polarity_pairs"]) == 12
        assert len(case_set["contrast_groups"]) == 4
        assert len(case_set["injection_probes"]) == 4
        probes = run_module.expand_probes(case_set)
        assert len(probes) == 80
        assert {p.probe_kind for p in probes} == {"ladder", "polarity", "contrast", "injection"}

    def test_case_set_fingerprint_is_stable_across_loads(self) -> None:
        first = fingerprint(metrics.load_case_set())
        second = fingerprint(metrics.load_case_set())
        assert first == second
        assert len(first) == 64


class TestFormulations:
    def test_every_doctrine_constructs_against_real_validation(self) -> None:
        for formulation in formulations.all_formulations(True, True):
            doctrine = formulation.doctrine
            assert isinstance(doctrine, ScoringDoctrine)
            system, user = doctrine.render(
                question="q",
                context="c",
                positive_verbalizer=formulation.positive_verbalizer,
                negative_verbalizer=formulation.negative_verbalizer,
            )
            assert system
            assert formulation.positive_verbalizer in user
            compiler = formulation.make_compiler()
            assert isinstance(compiler, BoolCompiler)

    def test_matrix_counts_and_distinct_ids(self) -> None:
        assert len(formulations.all_formulations(False, False)) == 9
        full = formulations.all_formulations(True, True)
        assert len(full) == 27
        ids = [f.formulation_id for f in full]
        assert len(set(ids)) == 27

    def test_label_families_orders_mappings_metadata(self) -> None:
        by_id = {f.formulation_id: f for f in formulations.all_formulations(True, True)}
        assert len(by_id) == 27
        canonical = by_id["D1_ab_positive_first_canonical"]
        assert canonical.positive_verbalizer == "A"
        assert canonical.negative_verbalizer == "B"
        assert canonical.label_mapping == "canonical"
        swapped = by_id["D1_ab_positive_first_swapped"]
        assert swapped.positive_verbalizer == "B"
        assert swapped.negative_verbalizer == "A"
        assert swapped.label_mapping == "swapped"
        neg_first = by_id["D2_true_false_negative_first_canonical"]
        assert neg_first.label_order == "negative_first"
        assert neg_first.positive_verbalizer == "true"
        template = neg_first.doctrine.user_template
        assert template.index("{negative_verbalizer}") < template.index("{positive_verbalizer}")
        ab_doctrine = by_id["D3_ab_positive_first_canonical"].doctrine
        assert "supported" in ab_doctrine.user_template

    def test_d1_base_is_the_existing_doctrine_object(self) -> None:
        base = formulations.all_formulations(False, False)
        d1 = next(f for f in base if f.formulation_id == "D1_yes_no_positive_first_canonical")
        assert d1.doctrine is BINARY_SEMANTIC_JUDGMENT_V1


REQUIRED_KEYS = set(run_module.REQUIRED_RECORD_KEYS)


class TestDryRun:
    def test_dry_run_produces_well_formed_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "dry.jsonl"
        exit_code = run_module.main(
            [
                "--dry-run",
                "--out",
                str(out),
                "--limit",
                "6",
                "--label-families",
                "yes_no,true_false",
            ]
        )
        assert exit_code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 2
        header = json.loads(lines[0])
        assert header["record_type"] == "run_header"
        assert header["case_set_version"] == "semantic-signal-v1"
        assert len(header["case_set_fingerprint"]) == 64
        assert header["started_at"] and header["finished_at"]
        records = [json.loads(line) for line in lines[1:]]
        assert records
        for record in records:
            assert set(record) >= REQUIRED_KEYS
            assert record["status"] == "ok"
            assert 0.0 <= record["probability_true"] <= 1.0
            assert 0.0 <= record["verbalizer_mass"] <= 1.0
            assert record["question"] and record["context"]
            assert record["execution_fingerprint"]
            assert record["trace_id"]

    def test_dry_run_records_unsupported_label_family(self, tmp_path: Path) -> None:
        out = tmp_path / "dry_ab.jsonl"
        exit_code = run_module.main(["--dry-run", "--out", str(out), "--limit", "3"])
        assert exit_code == 0
        records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()[1:]]
        unsupported = [r for r in records if r["status"] == "unsupported"]
        assert unsupported, "the ab family should be unsupported in the fake backend"
        for record in unsupported:
            assert record["error_type"] == "VerbalizerError"
            assert record["probability_true"] is None
        for record in (r for r in records if r["status"] == "ok"):
            assert record["label_family"] != "ab"


def _record(
    case_id: str,
    probe_kind: str,
    probability: float,
    *,
    expected_relation: str = "",
    doctrine: str = "d",
    family: str = "yes_no",
    mapping: str = "canonical",
    order: str = "positive_first",
    mass: float = 0.9,
) -> dict[str, Any]:
    return {
        "model": "m",
        "doctrine": doctrine,
        "label_family": family,
        "label_mapping": mapping,
        "label_order": order,
        "case_id": case_id,
        "probe_kind": probe_kind,
        "expected_relation": expected_relation,
        "probability_true": probability,
        "verbalizer_mass": mass,
        "status": "ok",
    }


RUNG_NAMES = (
    "strong_positive",
    "weak_positive",
    "insufficient",
    "weak_negative",
    "strong_negative",
)


class TestMetrics:
    def test_directional_accuracy_detects_violation(self) -> None:
        good = [
            _record("l1::strong_positive", "ladder", 0.9, expected_relation="strong_positive"),
            _record("l1::strong_negative", "ladder", 0.1, expected_relation="strong_negative"),
        ]
        assert metrics.directional_accuracy(good)["directional_accuracy"] == 1.0
        bad = [
            _record("l1::strong_positive", "ladder", 0.2, expected_relation="strong_positive"),
            _record("l1::strong_negative", "ladder", 0.8, expected_relation="strong_negative"),
        ]
        assert metrics.directional_accuracy(bad)["directional_accuracy"] == 0.0

    def test_ladder_monotonicity_detects_violation(self) -> None:
        values = [0.9, 0.7, 0.5, 0.3, 0.1]
        good = [
            _record(f"l1::{name}", "ladder", p, expected_relation=name)
            for name, p in zip(RUNG_NAMES, values, strict=True)
        ]
        result = metrics.ladder_monotonicity(good)
        assert result["fully_monotonic_rate"] == 1.0
        assert result["mean_adjacent_concordance"] == 1.0
        violated = [dict(r) for r in good]
        violated[2]["probability_true"] = 0.95
        result = metrics.ladder_monotonicity(violated)
        assert result["fully_monotonic_rate"] == 0.0
        assert result["mean_adjacent_concordance"] == pytest.approx(3 / 4)

    def test_polarity_consistency_detects_violation(self) -> None:
        good = [
            _record("p1::supporting", "polarity", 0.9, expected_relation="supporting"),
            _record("p1::opposite", "polarity", 0.2, expected_relation="opposite"),
        ]
        assert metrics.polarity_consistency(good)["polarity_consistency"] == 1.0
        bad = [
            _record("p1::supporting", "polarity", 0.2, expected_relation="supporting"),
            _record("p1::opposite", "polarity", 0.9, expected_relation="opposite"),
        ]
        assert metrics.polarity_consistency(bad)["polarity_consistency"] == 0.0

    def test_contrast_ordering_detects_violation(self) -> None:
        base = {
            "expected_order_json": json.dumps(
                ["relevant_positive", "irrelevant", "relevant_negative"]
            ),
            "required_relation_json": json.dumps(["relevant_positive", "relevant_negative"]),
        }
        good = [
            {**_record(f"g1::{role}", "contrast", p, expected_relation=role), **base}
            for role, p in (
                ("relevant_positive", 0.9),
                ("irrelevant", 0.5),
                ("relevant_negative", 0.1),
            )
        ]
        result = metrics.contrast_ordering(good)
        assert result["full_order_rate"] == 1.0
        assert result["required_relation_rate"] == 1.0
        violated = [dict(r) for r in good]
        violated[1]["probability_true"] = 0.95
        result = metrics.contrast_ordering(violated)
        assert result["full_order_rate"] == 0.0
        assert result["required_relation_rate"] == 1.0

    def test_label_mapping_consistency_uses_signs_not_means(self) -> None:
        records: list[dict[str, Any]] = []
        signs = [0.9, 0.2, 0.8]
        for index, value in enumerate(signs):
            records.append(_record(f"c{index}", "ladder", value, doctrine="d", family="yes_no"))
            records.append(
                _record(f"c{index}", "ladder", value + 0.05, doctrine="d", family="true_false")
            )
        result = metrics.label_mapping_consistency(records)
        assert result["cross_family_sign_agreement_mean"] == 1.0
        flipped = [dict(r) for r in records]
        flipped[3]["probability_true"] = 0.8
        result = metrics.label_mapping_consistency(flipped)
        assert result["cross_family_sign_agreement_mean"] == pytest.approx(2 / 3)

    def test_collapse_rate_detects_always_yes(self) -> None:
        healthy = [
            _record("l1::strong_positive", "ladder", 0.9, expected_relation="strong_positive"),
            _record("l1::strong_negative", "ladder", 0.1, expected_relation="strong_negative"),
        ]
        assert metrics.collapse_rate(healthy)["collapse_rate"] == 0.0
        collapsed = [
            _record("l1::strong_positive", "ladder", 0.9, expected_relation="strong_positive"),
            _record("l1::strong_negative", "ladder", 0.8, expected_relation="strong_negative"),
        ]
        assert metrics.collapse_rate(collapsed)["collapse_rate"] == 1.0

    def test_prior_bias_and_mass_bands(self) -> None:
        records = [
            _record("l1::insufficient", "ladder", 0.7, expected_relation="insufficient"),
            _record("injection-02", "injection", 0.3),
        ]
        prior = metrics.prior_bias(records)
        assert prior["probes_total"] == 2
        family_report = prior["by_label_family"]["yes_no"]
        assert family_report["count"] == 2
        assert family_report["mean"] == pytest.approx(0.5)
        bands = metrics.verbalizer_mass_bands(
            [
                _record("a", "ladder", 0.5, mass=0.001),
                _record("b", "ladder", 0.5, mass=0.1),
                _record("c", "ladder", 0.5, mass=0.9),
            ]
        )
        assert bands["bands"]["below_0.01"] == pytest.approx(1 / 3)
        assert bands["bands"]["0.01_to_0.5"] == pytest.approx(1 / 3)
        assert bands["bands"]["above_0.5"] == pytest.approx(1 / 3)

    def test_spearman_by_hand(self) -> None:
        assert metrics.spearman_rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(
            1.0
        )
        assert metrics.spearman_rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(
            -1.0
        )
        assert metrics.spearman_rank_correlation([1, 2], [1, 1]) is None
        value = metrics.spearman_rank_correlation([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
        assert value is not None and math.isclose(value, 0.8, abs_tol=1e-9)

    def test_summarize_groups_by_formulation(self) -> None:
        records = [
            _record("l1::strong_positive", "ladder", 0.9, expected_relation="strong_positive"),
            _record("l1::strong_negative", "ladder", 0.1, expected_relation="strong_negative"),
            _record(
                "l1::strong_positive",
                "ladder",
                0.9,
                expected_relation="strong_positive",
                family="true_false",
            ),
        ]
        summary = metrics.summarize(records)
        assert len(summary["formulations"]) == 2
        assert summary["formulations"]["m|d|yes_no|canonical|positive_first"]["records_ok"] == 2


class RaisingBackend:
    """Backend whose execute always raises VerbalizerError."""

    def __init__(self) -> None:
        self.capabilities = run_module.BackendCapabilities(binary_token_logits=True)

    def execute(self, plan: Any) -> Any:
        raise VerbalizerError("label not single-token")


class TestUnsupportedHandling:
    def test_runner_records_unsupported_and_never_substitutes(self, tmp_path: Path) -> None:
        out = tmp_path / "unsupported.jsonl"
        case_set = metrics.load_case_set()
        run_context: dict[str, Any] = {
            "model": "m",
            "revision": None,
            "dtype": "float32",
            "device": "cpu",
            "case_set_version": case_set["case_set_version"],
            "case_set_fingerprint": "0" * 64,
        }
        written = run_module.run_sweep(
            backend=RaisingBackend(),
            formulations=formulations.all_formulations(False, False),
            probes=run_module.expand_probes(case_set),
            run_context=run_context,
            out_path=out,
            limit=3,
        )
        assert written == 9
        records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()[1:]]
        assert len(records) == 9
        for record in records:
            assert record["status"] == "unsupported"
            assert record["error_type"] == "VerbalizerError"
            assert record["probability_true"] is None
            assert record["skipped_probes"] == 2
