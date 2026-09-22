"""Pure metrics over semantic signal JSONL records.

Every function here is a pure function over parsed JSONL record dicts (as
written by ``run.py``). Stdlib only: no numpy, no scipy, no pandas. Nothing
rejects a probe and nothing applies a validity threshold: all outputs are
descriptive reports.

Grouping key everywhere: ``(model, doctrine, label_family, label_mapping,
label_order)``.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

CASE_SET_PATH = Path(__file__).with_name("cases.json")

#: Canonical rung order of every ladder, from the frozen case set.
RUNG_ORDER: tuple[str, ...] = (
    "strong_positive",
    "weak_positive",
    "insufficient",
    "weak_negative",
    "strong_negative",
)

#: Report-only bands for verbalizer mass. Nothing is rejected on these.
MASS_BANDS: tuple[tuple[str, float, float], ...] = (
    ("below_0.01", 0.0, 0.01),
    ("0.01_to_0.5", 0.01, 0.5),
    ("above_0.5", 0.5, float("inf")),
)


def load_case_set(path: Path | None = None) -> dict[str, Any]:
    """Load and return the frozen case set (parsed cases.json)."""
    with (path or CASE_SET_PATH).open(encoding="utf-8") as handle:
        case_set: dict[str, Any] = json.load(handle)
    return case_set


def group_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """The (model, doctrine, label_family, label_mapping, label_order) key."""
    return (
        str(record.get("model", "")),
        str(record.get("doctrine", "")),
        str(record.get("label_family", "")),
        str(record.get("label_mapping", "")),
        str(record.get("label_order", "")),
    )


def group_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]]:
    """Group records by the standard formulation key."""
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(group_key(record), []).append(record)
    return grouped


def _ok(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in records if r.get("status") == "ok"]


def _case_group(record: Mapping[str, Any]) -> str:
    """The case a record belongs to: the part of case_id before '::'.

    Ladder, polarity and contrast case ids are ``<case>::<slot>``; grouping by
    the prefix puts all slots of one case together. Injection probe ids have
    no separator and pass through unchanged.
    """
    return str(record.get("case_id", "")).split("::", 1)[0]


def _by_case(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        index.setdefault(_case_group(record), []).append(record)
    return index


def _probability(record: Mapping[str, Any]) -> float:
    return float(record["probability_true"])


def directional_accuracy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fraction of ladders where P(True | strong_positive) > P(True | strong_negative)."""
    ok = _ok(records)
    ladders = _by_case(r for r in ok if str(r.get("probe_kind")) == "ladder")
    total = 0
    correct = 0
    for _group_id, entries in ladders.items():
        strong_pos = [r for r in entries if r.get("expected_relation") == "strong_positive"]
        strong_neg = [r for r in entries if r.get("expected_relation") == "strong_negative"]
        if not strong_pos or not strong_neg:
            continue
        total += 1
        if _probability(strong_pos[0]) > _probability(strong_neg[0]):
            correct += 1
    return {
        "ladders_total": total,
        "ladders_correct": correct,
        "directional_accuracy": (correct / total) if total else None,
    }


def ladder_monotonicity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Adjacent-rung ordering along the canonical rung order, per ladder.

    Reports the fraction of fully monotonic ladders (all 4 adjacent
    comparisons strictly decreasing) and the mean adjacent-pair concordance.
    """
    ok = _ok(records)
    ladders = _by_case(r for r in ok if str(r.get("probe_kind")) == "ladder")
    fully_monotonic = 0
    total_ladders = 0
    pairs_correct = 0
    pairs_total = 0
    for entries in ladders.values():
        by_rung = {str(r.get("expected_relation")): _probability(r) for r in entries}
        if any(rung not in by_rung for rung in RUNG_ORDER):
            continue
        total_ladders += 1
        all_correct = True
        for first, second in pairwise(RUNG_ORDER):
            pairs_total += 1
            if by_rung[first] > by_rung[second]:
                pairs_correct += 1
            else:
                all_correct = False
        if all_correct:
            fully_monotonic += 1
    return {
        "ladders_total": total_ladders,
        "fully_monotonic_ladders": fully_monotonic,
        "fully_monotonic_rate": (fully_monotonic / total_ladders) if total_ladders else None,
        "adjacent_pairs_total": pairs_total,
        "adjacent_pairs_correct": pairs_correct,
        "mean_adjacent_concordance": (pairs_correct / pairs_total) if pairs_total else None,
    }


def polarity_consistency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fraction of polarity pairs where P(supporting=True) > P(opposite=True)."""
    ok = _ok(records)
    pairs = _by_case(r for r in ok if str(r.get("probe_kind")) == "polarity")
    total = 0
    correct = 0
    for entries in pairs.values():
        supporting = [r for r in entries if r.get("expected_relation") == "supporting"]
        opposite = [r for r in entries if r.get("expected_relation") == "opposite"]
        if not supporting or not opposite:
            continue
        total += 1
        if _probability(supporting[0]) > _probability(opposite[0]):
            correct += 1
    return {
        "pairs_total": total,
        "pairs_correct": correct,
        "polarity_consistency": (correct / total) if total else None,
    }


def contrast_ordering(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per contrast group: does the full expected order and the required relation hold?"""
    ok = _ok(records)
    groups = _by_case(r for r in ok if str(r.get("probe_kind")) == "contrast")
    results: list[dict[str, Any]] = []
    for group_id, entries in groups.items():
        by_role = {str(r.get("expected_relation")): _probability(r) for r in entries}
        expected_order = [
            str(x) for x in json.loads(str(entries[0].get("expected_order_json", "[]")))
        ]
        required_relation = [
            str(x) for x in json.loads(str(entries[0].get("required_relation_json", "[]")))
        ]
        full_order_holds: bool | None = None
        if expected_order and all(role in by_role for role in expected_order):
            full_order_holds = all(by_role[a] > by_role[b] for a, b in pairwise(expected_order))
        required_holds: bool | None = None
        if len(required_relation) == 2 and all(role in by_role for role in required_relation):
            required_holds = by_role[required_relation[0]] > by_role[required_relation[1]]
        results.append(
            {
                "group_id": group_id,
                "full_expected_order_holds": full_order_holds,
                "required_relation_holds": required_holds,
            }
        )
    evaluated = [r for r in results if r["full_expected_order_holds"] is not None]
    return {
        "groups": results,
        "groups_total": len(evaluated),
        "full_order_rate": (
            sum(1 for r in evaluated if r["full_expected_order_holds"]) / len(evaluated)
        )
        if evaluated
        else None,
        "required_relation_rate": (
            sum(1 for r in evaluated if r["required_relation_holds"]) / len(evaluated)
        )
        if evaluated
        else None,
    }


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _spearman_from_pairs(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Spearman rank correlation over paired observations (hand-rolled)."""
    if len(pairs) < 2:
        return None
    return spearman_rank_correlation([a for a, _ in pairs], [b for _, b in pairs])


def label_mapping_consistency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sign and ranking agreement of P(True) across label families and swaps.

    Compares formulations that differ only in ``label_family`` (canonical
    mapping) and formulations that differ only in ``label_mapping``. Agreement
    is measured on the SIGN of (P(True) - 0.5) per case and on the ranking of
    cases (Spearman), never on means alone.
    """
    ok = _ok(records)
    by_formulation: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for record in ok:
        key = group_key(record)
        case_id = str(record.get("case_id", ""))
        by_formulation.setdefault(key, {})[case_id] = _probability(record)
    family_agreements: list[float] = []
    swap_agreements: list[float] = []
    swap_rankings: list[float] = []
    keys = sorted(by_formulation)
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            if key_a[0] != key_b[0] or key_a[1] != key_b[1] or key_a[4] != key_b[4]:
                continue
            cases_a = by_formulation[key_a]
            cases_b = by_formulation[key_b]
            shared = sorted(set(cases_a) & set(cases_b))
            if not shared:
                continue
            signs_agree = sum(
                1
                for case_id in shared
                if _sign(cases_a[case_id] - 0.5) == _sign(cases_b[case_id] - 0.5)
            )
            rate = signs_agree / len(shared)
            if key_a[2] != key_b[2] and key_a[3] == "canonical" and key_b[3] == "canonical":
                family_agreements.append(rate)
            elif key_a[3] != key_b[3] and key_a[2] == key_b[2]:
                swap_agreements.append(rate)
                correlation = _spearman_from_pairs([(cases_a[c], cases_b[c]) for c in shared])
                if correlation is not None:
                    swap_rankings.append(correlation)
    return {
        "cross_family_sign_agreement_rates": family_agreements,
        "cross_family_sign_agreement_mean": (
            sum(family_agreements) / len(family_agreements) if family_agreements else None
        ),
        "swap_sign_agreement_rates": swap_agreements,
        "swap_sign_agreement_mean": (
            sum(swap_agreements) / len(swap_agreements) if swap_agreements else None
        ),
        "swap_rank_spearman_mean": (
            sum(swap_rankings) / len(swap_rankings) if swap_rankings else None
        ),
    }


def collapse_rate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fraction of strong_positive / strong_negative probes on the same side of 0.5.

    Detects always-yes / always-first-label collapse: a model that ignores the
    context pushes both extreme rungs to the same semantic side.
    """
    ok = _ok(records)
    ladders = _by_case(
        r
        for r in ok
        if str(r.get("probe_kind")) == "ladder"
        and str(r.get("expected_relation")) in ("strong_positive", "strong_negative")
    )
    collapsed = 0
    total = 0
    for entries in ladders.values():
        strong_pos = [r for r in entries if r.get("expected_relation") == "strong_positive"]
        strong_neg = [r for r in entries if r.get("expected_relation") == "strong_negative"]
        if not strong_pos or not strong_neg:
            continue
        total += 1
        p_pos = _probability(strong_pos[0])
        p_neg = _probability(strong_neg[0])
        if (p_pos > 0.5) == (p_neg > 0.5):
            collapsed += 1
    return {
        "ladders_total": total,
        "collapsed_ladders": collapsed,
        "collapse_rate": (collapsed / total) if total else None,
    }


def prior_bias(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Distribution of P(True) on prior-only probes (insufficient rung, injection-02).

    Exposes label priors: on probes with no evidence, P(True) should reflect
    the model's prior over the surface labels, not the proposition.
    """
    ok = _ok(records)
    prior_probes = [
        r
        for r in ok
        if (
            str(r.get("probe_kind")) == "ladder"
            and str(r.get("expected_relation")) == "insufficient"
        )
        or str(r.get("case_id")) == "injection-02"
    ]
    by_family: dict[str, list[float]] = {}
    for record in prior_probes:
        by_family.setdefault(str(record.get("label_family")), []).append(_probability(record))
    report: dict[str, Any] = {"probes_total": len(prior_probes), "by_label_family": {}}
    for family, values in sorted(by_family.items()):
        report["by_label_family"][family] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "fraction_above_0.5": sum(1 for v in values if v > 0.5) / len(values),
        }
    return report


def verbalizer_mass_bands(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report-only bands of verbalizer_mass, to separate scoring from semantics."""
    ok = _ok(records)
    counts = {name: 0 for name, _, _ in MASS_BANDS}
    total = 0
    for record in ok:
        mass = float(record["verbalizer_mass"])
        total += 1
        for name, low, high in MASS_BANDS:
            if low <= mass < high:
                counts[name] += 1
                break
    return {
        "probes_total": total,
        "bands": {name: (count / total if total else None) for name, count in counts.items()},
    }


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (ties get the mean of their positions, 1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman_rank_correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation by hand: rank, then Pearson on the ranks.

    Returns None when the inputs are too short or have zero rank variance.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    ranks_a = _ranks(list(a))
    ranks_b = _ranks(list(b))
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b, strict=True))
    variance_a = sum((x - mean_a) ** 2 for x in ranks_a)
    variance_b = sum((y - mean_b) ** 2 for y in ranks_b)
    if variance_a == 0.0 or variance_b == 0.0:
        return None
    return covariance / math.sqrt(variance_a * variance_b)


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute every metric, grouped by formulation key."""
    summary: dict[str, Any] = {"formulations": {}}
    for key, group in sorted(group_records(records).items()):
        model, doctrine, family, mapping, order = key
        summary["formulations"]["|".join(key)] = {
            "model": model,
            "doctrine": doctrine,
            "label_family": family,
            "label_mapping": mapping,
            "label_order": order,
            "records_total": len(group),
            "records_ok": len(_ok(group)),
            "directional_accuracy": directional_accuracy(group),
            "ladder_monotonicity": ladder_monotonicity(group),
            "polarity_consistency": polarity_consistency(group),
            "contrast_ordering": contrast_ordering(group),
            "label_mapping_consistency": label_mapping_consistency(records),
            "collapse_rate": collapse_rate(group),
            "prior_bias": prior_bias(group),
            "verbalizer_mass_bands": verbalizer_mass_bands(group),
        }
    return summary


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a JSONL file, skipping blank lines."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Print and return the structured summary for one or more JSONL files."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python experiments/semantic_signal/metrics.py RESULTS.jsonl [...]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    records: list[dict[str, Any]] = []
    for arg in args:
        records.extend(
            record for record in read_jsonl(Path(arg)) if record.get("record_type") != "run_header"
        )
    summary = summarize(records)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
