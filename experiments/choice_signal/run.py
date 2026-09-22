"""Run the Phase 2B direct categorical Choice experiment against a local model.

One evaluation is one ChoiceDecision compiled into one InferencePlan and executed
with exactly one model forward pass. The harness perturbs the scoring
representation (the candidate-to-label assignment), the candidate set, and the
candidate descriptions, and records the resulting distributions.

Serial by design: a single backend is constructed once and reused, so only one
model is ever resident on the GPU.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fuzzyai import (
    CATEGORICAL_LABEL_SCHEME_ID,
    CATEGORICAL_LABELS,
    CATEGORICAL_SEMANTIC_JUDGMENT_V1,
    CandidateLabelMapping,
    ChoiceCompiler,
    ChoiceDecision,
    ChoiceScoringDiagnostics,
    FuzzyAI,
    InferencePlan,
)
from fuzzyai.fingerprint import fingerprint

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases.json"
DEFAULT_OUT_DIR = HERE / "results"


class PermutedChoiceCompiler(ChoiceCompiler):
    """ChoiceCompiler that assigns an explicit scoring-label permutation.

    Experiment-side only. The runtime scheme assigns labels in candidate order;
    here the representation is perturbed deliberately so the experiment can
    measure how much the restricted distribution depends on it. Compiler duties
    are unchanged: it still consults no tokenizer, model, or token id.
    """

    def __init__(self, labels: tuple[str, ...] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._labels = labels

    @property
    def labels(self) -> tuple[str, ...] | None:
        return self._labels

    def compile(self, decision: Any, capabilities: Any) -> InferencePlan:
        natural = super().compile(decision, capabilities)
        if self._labels is None:
            return natural
        entries = natural.candidate_mapping
        lines = [
            (
                f"{label} = {entry.candidate_name}"
                if entry.candidate_description is None
                else f"{label} = {entry.candidate_name} ({entry.candidate_description})"
            )
            for entry, label in zip(entries, self._labels, strict=True)
        ]
        _, prompt = self.doctrine.render(
            question=decision.question,
            context=decision.context,
            mapping_lines=lines,
        )
        return InferencePlan(
            decision_fingerprint=natural.decision_fingerprint,
            strategy=natural.strategy,
            prompt=prompt,
            targets=self._labels,
            system_prompt=natural.system_prompt,
            doctrine_id=natural.doctrine_id,
            label_scheme_id=natural.label_scheme_id,
            candidate_mapping=tuple(
                CandidateLabelMapping(
                    candidate_index=entry.candidate_index,
                    candidate_name=entry.candidate_name,
                    candidate_description=entry.candidate_description,
                    scoring_label=label,
                )
                for entry, label in zip(entries, self._labels, strict=True)
            ),
            required_capabilities=natural.required_capabilities,
        )


def permutation_labels(count: int) -> list[tuple[str, ...]]:
    """The label permutations to run for a candidate count."""
    base = CATEGORICAL_LABELS[:count]
    if count == 3:
        return [tuple(p) for p in itertools.permutations(base)]
    identity = tuple(base)
    reverse = tuple(reversed(base))
    cyclic = base[1:] + base[:1]
    swap_first_two = (base[1], base[0], *base[2:])
    rotate_two = base[2:] + base[:2]
    return [identity, reverse, cyclic, swap_first_two, rotate_two]


def permutation_name(labels: Sequence[str]) -> str:
    return "".join(labels)


def choice_mapping(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for candidate in candidates:
        name = str(candidate["name"])
        description = candidate.get("description")
        mapping[name] = name if description is None else str(description)
    return mapping


def candidate_order(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(c["name"]) for c in candidates]


def candidate_descriptions(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(c["name"]): c.get("description") for c in candidates}


def build_record(
    evaluation: Any,
    *,
    case_id: str,
    candidate_set: str,
    candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    run_meta: Mapping[str, Any],
    expected_category: str | None,
    group: str,
    note: str | None,
) -> dict[str, Any]:
    result = evaluation.result
    trace = evaluation.trace
    diagnostics = trace.scoring_diagnostics
    if not isinstance(diagnostics, ChoiceScoringDiagnostics):
        raise RuntimeError(
            f"expected ChoiceScoringDiagnostics for a Choice evaluation, got "
            f"{type(diagnostics).__name__}"
        )
    mapping = {entry.candidate_name: entry.scoring_label for entry in trace.candidate_mapping}
    return {
        "record_type": "evaluation",
        "group": group,
        "note": note,
        "expected_category": expected_category,
        "case_id": case_id,
        "model": run_meta["model"],
        "revision": run_meta["revision"],
        "dtype": run_meta["dtype"],
        "gpu": run_meta["gpu"],
        "candidate_set": candidate_set,
        "candidate_order": candidate_order(candidates),
        "candidate_descriptions": candidate_descriptions(candidates),
        "label_scheme": CATEGORICAL_LABEL_SCHEME_ID,
        "labels": list(labels),
        "permutation": permutation_name(labels),
        "mapping": mapping,
        "resolved_token_ids": {
            label: token_id for label, token_id in trace.resolved_target_token_ids
        },
        "probabilities": {name: float(value) for name, value in result.probabilities.items()},
        "candidate_mass": diagnostics.candidate_mass,
        "candidate_token_probabilities": list(diagnostics.candidate_token_probabilities),
        "top_token": diagnostics.top_token_text,
        "top_token_id": diagnostics.top_token_id,
        "top_token_probability": diagnostics.top_token_probability,
        "certainty": {
            "entropy": result.certainty.entropy,
            "margin": result.certainty.margin,
            "concentration": result.certainty.concentration,
        },
        "argmax": result.value,
        "decision_fingerprint": trace.decision_fingerprint,
        "plan_fingerprint": trace.plan_fingerprint,
        "execution_fingerprint": trace.execution_fingerprint,
        "trace_id": trace.trace_id,
        "latency_ms": trace.latency_ms,
        "method": result.method,
    }


def evaluate(
    backend: Any,
    decision: ChoiceDecision,
    labels: tuple[str, ...] | None,
) -> Any:
    ai = FuzzyAI(
        backend=backend,
        choice_compiler=PermutedChoiceCompiler(labels),
        capture_rendered_input=True,
    )
    return ai.evaluate_with_trace(decision)


def run_group(
    backend: Any,
    cases: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    group: str,
    candidate_set: str,
    run_meta: Mapping[str, Any],
    permutations: Sequence[tuple[str, ...] | None],
    expect: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    question = str(run_meta.get("question", ""))
    choices = choice_mapping(candidates)
    for case in cases:
        case_decision = ChoiceDecision(
            question,
            context=case["context"],
            choices=choices,
        )
        for labels in permutations:
            evaluation = evaluate(backend, case_decision, labels)
            effective = labels if labels is not None else CATEGORICAL_LABELS[: len(candidates)]
            records.append(
                build_record(
                    evaluation,
                    case_id=str(case["id"]),
                    candidate_set=candidate_set,
                    candidates=candidates,
                    labels=effective,
                    run_meta=run_meta,
                    expected_category=case.get("expected_category") if expect else None,
                    group=group,
                    note=case.get("note"),
                )
            )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--backend-factory", default=None, metavar="MODULE:ATTRIBUTE")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tag", default="choice-signal")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a single case per group to verify the harness end to end",
    )
    return parser.parse_args(argv)


def load_backend_factory(name: str | None) -> Any:
    if name is None:
        from fuzzyai.backends.transformers import TransformersBackend

        return TransformersBackend
    module_name, separator, attribute = name.partition(":")
    if not separator:
        raise ValueError(f"--backend-factory must be MODULE:ATTRIBUTE, got {name!r}")
    return getattr(importlib.import_module(module_name), attribute)


def gpu_name() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return str(torch.cuda.get_device_name(0))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases_path = Path(args.cases)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    case_set_fingerprint = fingerprint(payload)
    sets = payload["candidate_sets"]

    three = list(payload["three_way_cases"])
    five = list(payload["five_way_cases"])
    paraphrase = payload["description_paraphrase"]
    irrelevant = payload["irrelevant_addition"]
    three_by_id = {str(c["id"]): c for c in three}
    if args.smoke:
        three = three[:1]
        five = five[:1]
        paraphrase = {**paraphrase, "case_ids": paraphrase["case_ids"][:1]}
        irrelevant = {**irrelevant, "case_ids": irrelevant["case_ids"][:1]}

    factory = load_backend_factory(args.backend_factory)
    backend = factory(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    run_meta = {
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "gpu": gpu_name(),
        "question": str(payload["question"]),
    }
    records: list[dict[str, Any]] = []
    started = time.time()

    records += run_group(
        backend,
        three,
        sets["three_way"],
        group="three_way",
        candidate_set="three_way",
        run_meta=run_meta,
        permutations=permutation_labels(3),
        expect=True,
    )
    records += run_group(
        backend,
        five,
        sets["five_way"],
        group="five_way",
        candidate_set="five_way",
        run_meta=run_meta,
        permutations=permutation_labels(5),
        expect=True,
    )
    base_cases = [three_by_id[cid] for cid in irrelevant["case_ids"]]
    records += run_group(
        backend,
        base_cases,
        sets[irrelevant["base_set"]],
        group="irrelevant_addition_base",
        candidate_set=irrelevant["base_set"],
        run_meta=run_meta,
        permutations=[None],
        expect=True,
    )
    records += run_group(
        backend,
        base_cases,
        sets[irrelevant["extended_set"]],
        group="irrelevant_addition_extended",
        candidate_set=irrelevant["extended_set"],
        run_meta=run_meta,
        permutations=[None],
        expect=True,
    )
    para_cases = [three_by_id[cid] for cid in paraphrase["case_ids"]]
    records += run_group(
        backend,
        para_cases,
        sets[paraphrase["original_set"]],
        group="description_paraphrase_original",
        candidate_set=paraphrase["original_set"],
        run_meta=run_meta,
        permutations=[None],
        expect=True,
    )
    records += run_group(
        backend,
        para_cases,
        sets[paraphrase["variant_set"]],
        group="description_paraphrase_variant",
        candidate_set=paraphrase["variant_set"],
        run_meta=run_meta,
        permutations=[None],
        expect=True,
    )
    records += run_group(
        backend,
        payload["taxonomy_overlap_cases"],
        sets["taxonomy_overlap"],
        group="taxonomy_overlap",
        candidate_set="taxonomy_overlap",
        run_meta=run_meta,
        permutations=[None],
        expect=False,
    )
    records += run_group(
        backend,
        payload["out_of_set_cases"],
        sets["three_way"],
        group="out_of_set",
        candidate_set="three_way",
        run_meta=run_meta,
        permutations=[None],
        expect=False,
    )

    header = {
        "record_type": "header",
        "tag": args.tag,
        "case_set_version": payload["case_set_version"],
        "case_set_fingerprint": case_set_fingerprint,
        "label_scheme": CATEGORICAL_LABEL_SCHEME_ID,
        "doctrine_id": CATEGORICAL_SEMANTIC_JUDGMENT_V1.doctrine_id,
        "backend": f"{type(backend).__module__}.{type(backend).__name__}",
        "records": len(records),
        "duration_s": round(time.time() - started, 3),
        **run_meta,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {len(records)} evaluation records to {out_path}", file=sys.stderr)
    print(json.dumps(header, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
