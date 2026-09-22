"""Metrics for the Phase 2B direct categorical Choice experiment.

Pure functions over recorded evaluation records. A record is a mapping with the
fields listed in ``REQUIRED_FIELDS``. Nothing here loads a model or touches a
GPU, so the module is importable in the default test environment.

The metrics deliberately carry no pass threshold. They describe how a Choice
distribution moves when the scoring representation, the candidate set, or the
candidate description changes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "model",
    "revision",
    "dtype",
    "gpu",
    "rendering_config",
    "case_id",
    "candidate_set",
    "candidate_order",
    "candidate_descriptions",
    "label_scheme",
    "mapping",
    "resolved_token_ids",
    "probabilities",
    "scoring_label_mass",
    "top_token",
    "top_token_probability",
    "certainty",
    "argmax",
    "plan_fingerprint",
    "execution_fingerprint",
    "latency_ms",
)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records, skipping blank lines."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"each JSONL line must be an object, got {type(value).__name__}")
            records.append(value)
    return records


def _as_probabilities(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def tv_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Total variation distance ``0.5 * sum_i |P_i - Q_i|`` over the union of keys.

    Missing keys are treated as zero so a shrunken candidate set can be compared
    against a superset.
    """
    if not left and not right:
        return 0.0
    keys = set(left) | set(right)
    total = 0.0
    for key in keys:
        total += abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
    return 0.5 * total


def max_abs_drift(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Largest per-candidate absolute probability difference over the union of keys."""
    if not left and not right:
        return 0.0
    keys = set(left) | set(right)
    return max((abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) for k in keys), default=0.0)


def semantic_argmax(probabilities: Mapping[str, float], order: Sequence[str]) -> str:
    """Argmax in semantic candidate order, with ties going to the first candidate.

    This mirrors the runtime tie-break rule so the experiment and the runtime
    cannot disagree about what the winner is.
    """
    if not order:
        raise ValueError("order must contain at least one candidate")
    best = order[0]
    best_value = float(probabilities.get(best, 0.0))
    for name in order[1:]:
        value = float(probabilities.get(name, 0.0))
        if value > best_value:
            best = name
            best_value = value
    return best


def _group_by_case(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["case_id"]), []).append(record)
    return grouped


def permutation_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    reference_mapping: Mapping[str, str] | None = None,
) -> dict[str, float | int]:
    """Compare every label permutation against a reference mapping per case.

    ``reference_mapping`` maps candidate name to the reference scoring label. When
    omitted, the reference is the first record seen for each case.
    """
    grouped = _group_by_case(records)
    total = 0
    preserved = 0
    tv_values: list[float] = []
    drift_values: list[float] = []
    for case_id in sorted(grouped):
        entries = grouped[case_id]
        if not entries:
            continue
        reference: Mapping[str, Any] | None = None
        if reference_mapping is not None:
            for entry in entries:
                if _mapping_as_dict(entry) == dict(reference_mapping):
                    reference = entry
                    break
        if reference is None:
            reference = entries[0]
        reference_probs = _as_probabilities(reference["probabilities"])
        reference_tuple = entry_argmax(reference)
        for entry in entries:
            total += 1
            if entry_argmax(entry) == reference_tuple:
                preserved += 1
            probs = _as_probabilities(entry["probabilities"])
            tv_values.append(tv_distance(reference_probs, probs))
            drift_values.append(max_abs_drift(reference_probs, probs))
    return {
        "cases": len(grouped),
        "records": total,
        "argmax_preserved": preserved,
        "argmax_preservation_rate": (preserved / total) if total else 0.0,
        "mean_tv": (sum(tv_values) / len(tv_values)) if tv_values else 0.0,
        "max_tv": max(tv_values) if tv_values else 0.0,
        "max_abs_drift": max(drift_values) if drift_values else 0.0,
    }


def _mapping_as_dict(record: Mapping[str, Any]) -> dict[str, str]:
    raw = record.get("mapping")
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def entry_argmax(record: Mapping[str, Any]) -> str:
    """Winner of a record, preferring the recorded argmax and falling back to recomputation."""
    recorded = record.get("argmax")
    if isinstance(recorded, str) and recorded:
        return recorded
    order = record.get("candidate_order")
    if not isinstance(order, Sequence) or isinstance(order, str) or not order:
        raise ValueError("record has neither a usable argmax nor a candidate_order")
    return semantic_argmax(_as_probabilities(record["probabilities"]), [str(x) for x in order])


def mass_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summary of the candidate-space mass observations."""
    values = [float(record["scoring_label_mass"]) for record in records]
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def mass_by_candidate_set(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Candidate-space mass grouped by candidate-set name, so N=3 and N=5 can be compared."""
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record["candidate_set"]), []).append(
            float(record["scoring_label_mass"])
        )
    out: dict[str, dict[str, float | int]] = {}
    for name in sorted(grouped):
        values = grouped[name]
        out[name] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    return out


def accuracy_like(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Share of records whose winner equals the fixture's obvious expected category.

    Only meaningful for records that carry ``expected_category``. This is a
    descriptive count of an experiment fixture, not an accuracy claim about the
    model.
    """
    total = 0
    matched = 0
    for record in records:
        expected = record.get("expected_category")
        if not isinstance(expected, str) or not expected:
            continue
        total += 1
        if entry_argmax(record) == expected:
            matched += 1
    return {
        "scored": total,
        "matched": matched,
        "match_rate": (matched / total) if total else 0.0,
    }


def winner_changes(
    base: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Per-case winner comparison between two runs of the same cases."""
    base_by_case = {str(r["case_id"]): r for r in base}
    out: list[dict[str, Any]] = []
    for record in perturbed:
        case_id = str(record["case_id"])
        reference = base_by_case.get(case_id)
        if reference is None:
            continue
        out.append(
            {
                "case_id": case_id,
                "base_winner": entry_argmax(reference),
                "perturbed_winner": entry_argmax(record),
                "winner_preserved": entry_argmax(reference) == entry_argmax(record),
                "tv": tv_distance(
                    _as_probabilities(reference["probabilities"]),
                    _as_probabilities(record["probabilities"]),
                ),
                "base_scoring_label_mass": float(reference["scoring_label_mass"]),
                "perturbed_scoring_label_mass": float(record["scoring_label_mass"]),
            }
        )
    return out


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Render rows as a GitHub markdown table."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
