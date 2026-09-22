"""Offline tests for the Choice experiment metrics.

No model, no GPU, no network. Every expected value is hand-computed so the
metrics cannot quietly change meaning.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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
    spec.loader.exec_module(module)
    return module


metrics = _load("metrics")


def _record(
    case_id: str,
    probabilities: dict[str, float],
    *,
    mapping: dict[str, str] | None = None,
    candidate_mass: float = 0.5,
    candidate_set: str = "three_way",
    expected_category: str | None = None,
) -> dict[str, Any]:
    order = list(probabilities)
    winner = metrics.semantic_argmax(probabilities, order)
    record: dict[str, Any] = {
        "model": "fake",
        "revision": None,
        "dtype": "float32",
        "gpu": "none",
        "case_id": case_id,
        "candidate_set": candidate_set,
        "candidate_order": order,
        "candidate_descriptions": {name: f"{name} desc" for name in order},
        "label_scheme": "categorical-labels-v1",
        "mapping": mapping or {name: label for name, label in zip(order, "ABC", strict=False)},
        "resolved_token_ids": [
            [label, 100 + index] for index, label in enumerate("ABC"[: len(order)])
        ],
        "probabilities": probabilities,
        "candidate_mass": candidate_mass,
        "top_token": "x",
        "top_token_probability": 0.2,
        "certainty": {"entropy": 0.1, "margin": 0.2},
        "argmax": winner,
        "plan_fingerprint": "p",
        "execution_fingerprint": "e",
        "latency_ms": 1.0,
    }
    if expected_category is not None:
        record["expected_category"] = expected_category
    return record


# --- total variation ---------------------------------------------------------


def test_tv_distance_between_identical_distributions_is_zero() -> None:
    probs = {"a": 0.5, "b": 0.5}
    assert metrics.tv_distance(probs, dict(probs)) == 0.0


def test_tv_distance_between_disjoint_distributions_is_one() -> None:
    left = {"a": 1.0, "b": 0.0}
    right = {"a": 0.0, "b": 1.0}
    assert metrics.tv_distance(left, right) == 1.0


def test_tv_distance_hand_computed() -> None:
    left = {"a": 0.7, "b": 0.3}
    right = {"a": 0.4, "b": 0.6}
    # 0.5 * (|0.7-0.4| + |0.3-0.6|) = 0.5 * 0.6 = 0.3
    assert abs(metrics.tv_distance(left, right) - 0.3) < 1e-12


def test_tv_distance_treats_missing_keys_as_zero() -> None:
    left = {"a": 0.6, "b": 0.4}
    right = {"a": 0.6, "b": 0.3, "c": 0.1}
    # 0.5 * (0 + 0.1 + 0.1) = 0.1
    assert abs(metrics.tv_distance(left, right) - 0.1) < 1e-12


def test_max_abs_drift_hand_computed() -> None:
    left = {"a": 0.7, "b": 0.3}
    right = {"a": 0.4, "b": 0.6}
    assert abs(metrics.max_abs_drift(left, right) - 0.3) < 1e-12


# --- argmax ------------------------------------------------------------------


def test_semantic_argmax_uses_order_for_ties() -> None:
    probs = {"billing": 0.5, "shipping": 0.5, "returns": 0.0}
    assert metrics.semantic_argmax(probs, ["billing", "shipping", "returns"]) == "billing"
    assert metrics.semantic_argmax(probs, ["shipping", "billing", "returns"]) == "shipping"


def test_semantic_argmax_ignores_dictionary_insertion_order() -> None:
    left = {"billing": 0.5, "shipping": 0.5}
    right = {"shipping": 0.5, "billing": 0.5}
    order = ["billing", "shipping"]
    assert metrics.semantic_argmax(left, order) == metrics.semantic_argmax(right, order)


# --- permutation metrics -----------------------------------------------------


def test_permutation_metrics_identity_only_is_perfectly_preserved() -> None:
    records = [
        _record(
            "c1",
            {"billing": 0.7, "shipping": 0.2, "technical": 0.1},
            mapping={"billing": "A", "shipping": "B", "technical": "C"},
        ),
        _record(
            "c1",
            {"billing": 0.7, "shipping": 0.2, "technical": 0.1},
            mapping={"billing": "A", "shipping": "B", "technical": "C"},
        ),
    ]
    summary = metrics.permutation_metrics(records)
    assert summary["records"] == 2
    assert summary["argmax_preservation_rate"] == 1.0
    assert summary["max_tv"] == 0.0


def test_permutation_metrics_detects_a_changed_winner() -> None:
    reference = _record(
        "c1",
        {"billing": 0.7, "shipping": 0.2, "technical": 0.1},
        mapping={"billing": "A", "shipping": "B", "technical": "C"},
    )
    flipped = _record(
        "c1",
        {"billing": 0.2, "shipping": 0.7, "technical": 0.1},
        mapping={"billing": "C", "shipping": "A", "technical": "B"},
    )
    summary = metrics.permutation_metrics(
        [reference, flipped], reference_mapping={"billing": "A", "shipping": "B", "technical": "C"}
    )
    assert summary["records"] == 2
    assert summary["argmax_preserved"] == 1
    assert summary["argmax_preservation_rate"] == 0.5
    assert abs(summary["max_tv"] - 0.5) < 1e-12


def test_permutation_metrics_explicit_reference_mapping_selects_the_reference() -> None:
    identity = _record(
        "c1",
        {"billing": 0.7, "shipping": 0.2, "technical": 0.1},
        mapping={"billing": "A", "shipping": "B", "technical": "C"},
    )
    permuted = _record(
        "c1",
        {"billing": 0.7, "shipping": 0.2, "technical": 0.1},
        mapping={"billing": "B", "shipping": "C", "technical": "A"},
    )
    # The permuted record is listed first, so without an explicit reference it
    # would be chosen as the reference by position.
    summary = metrics.permutation_metrics(
        [permuted, identity],
        reference_mapping={"billing": "A", "shipping": "B", "technical": "C"},
    )
    assert summary["argmax_preservation_rate"] == 1.0
    assert summary["max_tv"] == 0.0


# --- winner changes ----------------------------------------------------------


def test_winner_changes_reports_per_case_comparison() -> None:
    base = [_record("c1", {"billing": 0.7, "shipping": 0.2, "technical": 0.1})]
    perturbed = [_record("c1", {"billing": 0.1, "shipping": 0.8, "technical": 0.1})]
    rows = metrics.winner_changes(base, perturbed)
    assert len(rows) == 1
    assert rows[0]["base_winner"] == "billing"
    assert rows[0]["perturbed_winner"] == "shipping"
    assert rows[0]["winner_preserved"] is False


# --- mass --------------------------------------------------------------------


def test_mass_summary_hand_computed() -> None:
    records = [
        _record("c1", {"a": 1.0}, candidate_mass=0.75),
        _record("c2", {"a": 1.0}, candidate_mass=0.25),
    ]
    summary = metrics.mass_summary(records)
    assert summary["count"] == 2
    assert abs(summary["mean"] - 0.5) < 1e-12
    assert summary["min"] == 0.25
    assert summary["max"] == 0.75


def test_mass_by_candidate_set_separates_n_three_from_n_five() -> None:
    records = [
        _record("c1", {"a": 1.0}, candidate_mass=0.75, candidate_set="three_way"),
        _record("c2", {"a": 1.0}, candidate_mass=0.60, candidate_set="five_way"),
    ]
    grouped = metrics.mass_by_candidate_set(records)
    assert set(grouped) == {"five_way", "three_way"}
    assert grouped["three_way"]["count"] == 1
    assert abs(float(grouped["five_way"]["mean"]) - 0.60) < 1e-12


# --- descriptive counting ----------------------------------------------------


def test_accuracy_like_skips_records_without_an_expected_category() -> None:
    records = [
        _record("c1", {"billing": 0.9, "shipping": 0.1}, expected_category="billing"),
        _record("c2", {"billing": 0.9, "shipping": 0.1}, expected_category="shipping"),
        _record("c3", {"billing": 0.9, "shipping": 0.1}),
    ]
    summary = metrics.accuracy_like(records)
    assert summary["scored"] == 2
    assert summary["matched"] == 1
    assert summary["match_rate"] == 0.5


# --- io ---------------------------------------------------------------------


def test_load_records_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"case_id": "c1"}) + "\n\n" + json.dumps({"case_id": "c2"}) + "\n",
        encoding="utf-8",
    )
    records = metrics.load_records(path)
    assert [record["case_id"] for record in records] == ["c1", "c2"]


def test_format_table_renders_floats_and_headers() -> None:
    table = metrics.format_table([{"name": "a", "value": 0.5}], ["name", "value"])
    lines = table.splitlines()
    assert lines[0] == "| name | value |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| a | 0.500000 |"


# --- required record schema --------------------------------------------------


def test_required_fields_are_all_present_in_a_synthetic_record() -> None:
    record = _record("c1", {"billing": 1.0})
    missing = [field for field in metrics.REQUIRED_FIELDS if field not in record]
    assert missing == []
