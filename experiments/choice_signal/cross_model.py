"""Cross-model tables for the Choice scoring-representation experiments.

Post-processing only: reads the per-model JSONL artifacts under ``results/``,
recomputes every number used in ``REPORT.md`` with :mod:`metrics`, and prints
them. No model, no GPU, no network. Every table in the report is produced from
the raw records by this script rather than typed by hand.
"""

from __future__ import annotations

import collections
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS = (
    ("Qwen3.5-2B", "choice-signal-qwen35-2b.jsonl"),
    ("MiniCPM5-2B", "choice-signal-minicpm5-2b.jsonl"),
)
PERMUTATION_GROUPS = ("three_way", "five_way")


def evaluations(path: Path) -> list[dict[str, Any]]:
    return [r for r in metrics.load_records(path) if r.get("record_type") == "evaluation"]


def header(path: Path) -> dict[str, Any]:
    for record in metrics.load_records(path):
        if record.get("record_type") == "header":
            return record
    raise ValueError(f"no header record in {path}")


def in_group(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("group") == name]


def probabilities(record: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in record["probabilities"].items()}


def per_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    return grouped


def worst_cases(
    records: list[dict[str, Any]], limit: int = 3
) -> list[tuple[float, str, str, dict, dict]]:
    out = []
    for case_id, entries in sorted(per_case(records).items()):
        reference = entries[0]
        ref_probs = probabilities(reference)
        best_tv = 0.0
        best_entry = entries[0]
        for entry in entries[1:]:
            tv = metrics.tv_distance(ref_probs, probabilities(entry))
            if tv > best_tv:
                best_tv, best_entry = tv, entry
        out.append(
            (
                best_tv,
                case_id,
                str(best_entry.get("permutation")),
                ref_probs,
                probabilities(best_entry),
            )
        )
    out.sort(reverse=True)
    return out[:limit]


def identity_agreement(records: list[dict[str, Any]]) -> tuple[int, int]:
    identity = [r for r in records if r.get("permutation") in ("ABC", "ABCD", "ABCDE")]
    correct = sum(1 for r in identity if r.get("argmax") == r.get("expected_category"))
    return correct, len(identity)


def pair_preservation(
    base: list[dict[str, Any]], perturbed: list[dict[str, Any]]
) -> tuple[int, int, list[float]]:
    base_by_case = {str(r["case_id"]): r for r in base}
    tvs: list[float] = []
    preserved = 0
    total = 0
    for record in perturbed:
        case_id = str(record["case_id"])
        if case_id not in base_by_case:
            continue
        total += 1
        if metrics.entry_argmax(record) == metrics.entry_argmax(base_by_case[case_id]):
            preserved += 1
        tvs.append(metrics.tv_distance(probabilities(base_by_case[case_id]), probabilities(record)))
    return preserved, total, tvs


def resolved_tokens(records: list[dict[str, Any]]) -> dict[str, int]:
    seen: dict[str, int] = {}
    for record in records:
        raw = record.get("resolved_token_ids") or {}
        items = raw.items() if isinstance(raw, dict) else raw
        for label, token_id in items:
            seen.setdefault(str(label), int(token_id))
    return dict(sorted(seen.items()))


def latency_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(r["latency_ms"]) for r in records)
    return {
        "min": values[0],
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": values[-1],
    }


def entropy_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(r["certainty"]["entropy"]) for r in records]
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def fingerprint_uniqueness(records: list[dict[str, Any]]) -> dict[str, int]:
    case_id = "3w-billing-01"
    entries = per_case(records).get(case_id, [])
    return {
        "records": len(entries),
        "decision": len({str(r["decision_fingerprint"]) for r in entries}),
        "plan": len({str(r["plan_fingerprint"]) for r in entries}),
        "execution": len({str(r["execution_fingerprint"]) for r in entries}),
    }


def report_model(label: str, filename: str) -> None:
    path = RESULTS_DIR / filename
    head = header(path)
    records = evaluations(path)

    print(f"\n{'=' * 72}\n{label}  ({filename})\n{'=' * 72}")
    print(
        f"header: records={head['records']} duration_s={head['duration_s']} dtype={head['dtype']} "
        f"gpu={head['gpu']!r} model={head['model']} backend={head['backend']}"
    )
    print(
        f"        case_set_version={head['case_set_version']} "
        f"case_set_fingerprint={head['case_set_fingerprint']}"
    )
    print(f"resolved scoring tokens: {resolved_tokens(records)}")
    print(f"latency_ms: { {k: round(v, 1) for k, v in latency_stats(records).items()} }")
    print(f"entropy:    { {k: round(v, 4) for k, v in entropy_stats(records).items()} }")
    print(f"fingerprint uniqueness for 3w-billing-01: {fingerprint_uniqueness(records)}")

    for name in PERMUTATION_GROUPS:
        subset = in_group(records, name)
        perm = metrics.permutation_metrics(subset)
        correct, total = identity_agreement(subset)
        print(
            f"\n[{name}] cases={perm['cases']} records={perm['records']} "
            f"argmax_preserved={perm['argmax_preserved']}/{perm['records']} "
            f"rate={perm['argmax_preservation_rate']:.4f} "
            f"mean_TV={perm['mean_tv']:.4f} max_TV={perm['max_tv']:.4f} "
            f"max_abs_drift={perm['max_abs_drift']:.4f}"
        )
        print(f"    identity-only agreement with expected_category: {correct}/{total}")
        summary = {k: round(float(v), 4) for k, v in metrics.mass_summary(subset).items()}
        print(f"    scoring_label_mass: {summary}")
        for tv, case_id, perm_name, ref_probs, drift_probs in worst_cases(subset):
            ref_s = ", ".join(f"{k}={v:.3f}" for k, v in ref_probs.items())
            dr_s = ", ".join(f"{k}={v:.3f}" for k, v in drift_probs.items())
            print(f"    worst: {case_id} -> {perm_name} TV={tv:.4f}")
            print(f"           ref ABC: {ref_s}")
            print(f"           drift:   {dr_s}")

    base = in_group(records, "irrelevant_addition_base")
    extended = in_group(records, "irrelevant_addition_extended")
    preserved, total, tvs = pair_preservation(base, extended)
    print(
        f"\n[irrelevant_addition] winner_preserved={preserved}/{total} "
        f"TV_min={min(tvs):.4f} TV_max={max(tvs):.4f} "
        f"mass_base_mean={metrics.mass_summary(base)['mean']:.4f} "
        f"mass_extended_mean={metrics.mass_summary(extended)['mean']:.4f}"
    )
    for case_id, entries in sorted(per_case(extended).items()):
        ref = next((r for r in base if str(r["case_id"]) == case_id), None)
        if ref is None:
            continue
        tv = metrics.tv_distance(probabilities(ref), probabilities(entries[0]))
        winner = metrics.entry_argmax(entries[0])
        mass_base = float(ref["scoring_label_mass"])
        mass_extended = float(entries[0]["scoring_label_mass"])
        print(
            f"    {case_id}: winner={winner} TV={tv:.4f} "
            f"mass {mass_base:.4f} -> {mass_extended:.4f}"
        )

    original = in_group(records, "description_paraphrase_original")
    variant = in_group(records, "description_paraphrase_variant")
    preserved, total, tvs = pair_preservation(original, variant)
    print(
        f"\n[description_paraphrase] winner_preserved={preserved}/{total} "
        f"TV_min={min(tvs):.4f} TV_max={max(tvs):.4f}"
    )
    for case_id, entries in sorted(per_case(variant).items()):
        ref = next((r for r in original if str(r["case_id"]) == case_id), None)
        if ref is None:
            continue
        tv = metrics.tv_distance(probabilities(ref), probabilities(entries[0]))
        print(f"    {case_id}: winner={metrics.entry_argmax(entries[0])} TV={tv:.4f}")

    print("\n[taxonomy_overlap]")
    for record in sorted(in_group(records, "taxonomy_overlap"), key=lambda r: str(r["case_id"])):
        probs = ", ".join(f"{k}={v:.3f}" for k, v in probabilities(record).items())
        mass = float(record["scoring_label_mass"])
        print(f"    {record['case_id']}: winner={record['argmax']} mass={mass:.4f} | {probs}")

    print("\n[out_of_set]")
    for record in sorted(in_group(records, "out_of_set"), key=lambda r: str(r["case_id"])):
        probs = ", ".join(f"{k}={v:.3f}" for k, v in probabilities(record).items())
        mass = float(record["scoring_label_mass"])
        top_p = float(record.get("top_token_probability", 0.0))
        print(
            f"    {record['case_id']}: winner={record['argmax']} mass={mass:.4f} "
            f"top={record.get('top_token')!r} top_p={top_p:.4f}"
        )
        print(f"        {probs}")

    print("\n[scoring_label_mass by candidate set]")
    for name, stats in metrics.mass_by_candidate_set(records).items():
        print(
            f"    {name}: n={stats['count']} min={stats['min']:.4f} "
            f"mean={stats['mean']:.4f} max={stats['max']:.4f}"
        )


COMPARED_FIELDS = (
    "probabilities",
    "scoring_label_mass",
    "resolved_token_ids",
    "plan_fingerprint",
)
RENAMED_KEYS = (
    ("candidate_mass", "scoring_label_mass"),
    ("candidate_token_probabilities", "scoring_label_token_probabilities"),
)


def migrate_keys(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for old_key, new_key in RENAMED_KEYS:
        if old_key in out:
            out[new_key] = out.pop(old_key)
    return out


def compare_pre_rename(ref: str = "HEAD", filename: str = MODELS[0][1]) -> None:
    """Show that the 2B.1 rename changed keys only.

    Reads the artifact as committed at ``ref``, renames the two diagnostic keys
    the way the migration did, and compares the semantic fields REPORT.md relies
    on against the working copy. Fields that legitimately differ across runs
    (``execution_fingerprint``, ``trace_id``, latency) are deliberately excluded.
    """
    repo_root = Path(__file__).resolve().parents[2]
    git_path = (RESULTS_DIR / filename).relative_to(repo_root).as_posix()
    raw = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{git_path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    old: dict[tuple[str, str], dict[str, Any]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        record = migrate_keys(json.loads(line))
        if record.get("record_type") == "evaluation":
            old[(str(record["case_id"]), str(record["permutation"]))] = record

    new = {
        (str(r["case_id"]), str(r["permutation"])): r for r in evaluations(RESULTS_DIR / filename)
    }

    diffs = {field: 0 for field in COMPARED_FIELDS}
    compared = 0
    for key, old_record in old.items():
        new_record = new.get(key)
        if new_record is None:
            continue
        compared += 1
        for field in COMPARED_FIELDS:
            if old_record.get(field) != new_record.get(field):
                diffs[field] += 1
    print(f"\n[rename value-preservation] ref={ref} pairs={compared} diffs={diffs}")


def main() -> None:
    for label, filename in MODELS:
        report_model(label, filename)
    compare_pre_rename()


if __name__ == "__main__":
    main()
