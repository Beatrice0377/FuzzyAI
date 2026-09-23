"""CLI runner for the semantic signal validation experiment.

Runs one formulation matrix against exactly ONE model per process, batch size
1, and writes one JSONL record per probe. Real runs use TransformersBackend;
``--dry-run`` uses a deterministic fake backend (no torch, no model load).

Usage from the repo root:

    python experiments/semantic_signal/run.py --model MODEL_ID [options]
    python experiments/semantic_signal/run.py --dry-run [options]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bootstrap: make src/ importable when the package is not installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from formulations import Formulation, all_formulations  # noqa: E402
from metrics import load_case_set  # noqa: E402

from probvenance import (  # noqa: E402
    BINARY_EVIDENCE_LABELS,
    BackendCapabilities,
    BoolDecision,
    BoolResult,
    EvidenceKind,
    InferencePlan,
    Probvenance,
    RawEvidence,
)
from probvenance.errors import VerbalizerError  # noqa: E402
from probvenance.fingerprint import fingerprint  # noqa: E402
from probvenance.results import Certainty  # noqa: E402
from probvenance.trace import DecisionTrace  # noqa: E402

CASE_SET_PATH = Path(__file__).with_name("cases.json")
RESULTS_DIR = Path(__file__).with_name("results")

#: Single-token labels the dry-run fake backend accepts. Anything else raises
#: VerbalizerError, mirroring what a real tokenizer would reject.
_FAKE_SINGLE_TOKEN_LABELS = frozenset({"yes", "no", "true", "false"})

_FAKE_POSITIVE_TOKEN_ID = 9642
_FAKE_NEGATIVE_TOKEN_ID = 3134
_FAKE_VOCAB_SIZE_LOG = math.log(100.0)  # mass ~0.01 beyond the two verbalizers

REQUIRED_RECORD_KEYS: tuple[str, ...] = (
    "model",
    "revision",
    "tokenizer_revision",
    "dtype",
    "execution_fingerprint",
    "case_id",
    "probe_kind",
    "theme",
    "expected_relation",
    "doctrine",
    "label_family",
    "label_mapping",
    "label_order",
    "question",
    "question_fingerprint",
    "context_fingerprint",
    "probability_true",
    "verbalizer_mass",
    "top_token_text",
    "top_token_id",
    "top_token_probability",
    "positive_token_probability",
    "negative_token_probability",
    "entropy",
    "margin",
    "latency_ms",
    "input_tokens",
    "plan_fingerprint",
    "input_fingerprint",
    "trace_id",
    "status",
    "context",
)


@dataclass(frozen=True)
class ProbeSpec:
    """One probe expanded from the frozen case set."""

    case_id: str
    probe_kind: str
    theme: str
    question: str
    context: str
    expected_relation: str
    expected_order: tuple[str, ...] = ()
    required_relation: tuple[str, ...] = ()


def expand_probes(case_set: Mapping[str, Any]) -> list[ProbeSpec]:
    """Expand the case set into the 80 probes per formulation."""
    probes: list[ProbeSpec] = []
    for ladder in case_set["ladders"]:
        for rung in case_set["rung_order"]:
            probes.append(
                ProbeSpec(
                    case_id=f"{ladder['ladder_id']}::{rung}",
                    probe_kind="ladder",
                    theme=str(ladder["theme"]),
                    question=str(ladder["question"]),
                    context=str(ladder["rungs"][rung]),
                    expected_relation=str(rung),
                )
            )
    for pair in case_set["polarity_pairs"]:
        for kind, question in (
            ("supporting", pair["supporting_question"]),
            ("opposite", pair["opposite_question"]),
        ):
            probes.append(
                ProbeSpec(
                    case_id=f"{pair['pair_id']}::{kind}",
                    probe_kind="polarity",
                    theme=str(pair["theme"]),
                    question=str(question),
                    context=str(pair["context"]),
                    expected_relation=kind,
                )
            )
    for group in case_set["contrast_groups"]:
        for role, context in group["contexts"].items():
            probes.append(
                ProbeSpec(
                    case_id=f"{group['group_id']}::{role}",
                    probe_kind="contrast",
                    theme=str(group["theme"]),
                    question=str(group["question"]),
                    context=str(context),
                    expected_relation=str(role),
                    expected_order=tuple(group["expected_order"]),
                    required_relation=tuple(group["required_relation"]),
                )
            )
    for probe in case_set["injection_probes"]:
        probes.append(
            ProbeSpec(
                case_id=str(probe["probe_id"]),
                probe_kind="injection",
                theme=str(probe["theme"]),
                question=str(probe["question"]),
                context=str(probe["context"]),
                expected_relation=str(probe["semantic_expectation"]),
            )
        )
    return probes


class FakeDryRunBackend:
    """Deterministic fake backend for ``--dry-run``: no torch, no model.

    Returns fixed-shape fake logits derived deterministically from the plan
    prompt, so the dry-run path exercises the full record pipeline without any
    model. It validates verbalizers like a real tokenizer would: labels outside
    the single-token allowlist raise :class:`VerbalizerError`.
    """

    def __init__(self) -> None:
        self.capabilities = BackendCapabilities(binary_token_logits=True)

    def execute(self, plan: InferencePlan) -> RawEvidence:
        """Return deterministic fake binary logits for the plan."""
        if plan.system_prompt is None or plan.positive_verbalizer is None:
            raise VerbalizerError("plan lacks system prompt or verbalizers")
        negative = plan.negative_verbalizer
        if negative is None:
            raise VerbalizerError("plan lacks a negative_verbalizer")
        for label in (plan.positive_verbalizer, negative):
            if label not in _FAKE_SINGLE_TOKEN_LABELS:
                raise VerbalizerError(
                    f"fake tokenizer: verbalizer {label!r} is not a single known token"
                )
        digest = hashlib.sha256(plan.prompt.encode("utf-8")).digest()
        x = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        logit_true = x * 4.0
        logit_false = (1.0 - x) * 4.0
        top_logit = max(logit_true, logit_false)
        top_token_id = (
            _FAKE_POSITIVE_TOKEN_ID if logit_true >= logit_false else _FAKE_NEGATIVE_TOKEN_ID
        )
        vocab_logsumexp = _logaddexp(logit_true, logit_false) + _FAKE_VOCAB_SIZE_LOG
        return RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=BINARY_EVIDENCE_LABELS,
            values=(logit_false, logit_true),
            plan_fingerprint=plan.fingerprint,
            metadata={
                "positive_token_id": _FAKE_POSITIVE_TOKEN_ID,
                "negative_token_id": _FAKE_NEGATIVE_TOKEN_ID,
                "model": "fake-dry-run-model",
                "model_revision": "fake-revision",
                "input_token_count": 16 + len(plan.prompt) % 32,
                "vocab_logsumexp": vocab_logsumexp,
                "top_token_id": top_token_id,
                "top_token_logit": top_logit,
                "top_token_text": "fake",
                "tokenizer": "fake-tokenizer",
                "tokenizer_revision": "fake-revision",
                "dtype": "float32",
            },
        )


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) without overflow (same trick as probvenance.diagnostics)."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _sanitize_model_id(model: str) -> str:
    return model.replace("/", "__").replace("\\", "__")


def _git_commit() -> str | None:
    """Best-effort current git commit hash; None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _cuda_info() -> dict[str, Any]:
    """GPU facts, only when torch and CUDA are actually available."""
    info: dict[str, Any] = {
        "cuda_version": None,
        "gpu_name": None,
        "torch_version": None,
        "transformers_version": None,
    }
    try:
        import torch
    except ImportError:
        return info
    info["torch_version"] = str(torch.__version__)
    try:
        import transformers

        info["transformers_version"] = str(transformers.__version__)
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["cuda_version"] = str(getattr(torch.version, "cuda", None))
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def _peak_vram_mib() -> tuple[float | None, float | None]:
    """Peak allocated / reserved CUDA memory in MiB, or (None, None)."""
    try:
        import torch
    except ImportError:
        return (None, None)
    if not torch.cuda.is_available():
        return (None, None)
    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return (allocated, reserved)


def _record_from_evaluation(
    *,
    probe: ProbeSpec,
    evaluation_result: BoolResult,
    trace: DecisionTrace,
    formulation: Formulation,
    run_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one JSONL record from an evaluation and its trace."""
    diagnostics = trace.scoring_diagnostics
    certainty: Certainty = evaluation_result.certainty
    record: dict[str, Any] = {
        "record_type": "probe",
        "model": run_context["model"],
        "revision": run_context["revision"],
        "tokenizer_revision": trace.tokenizer_revision,
        "dtype": trace.dtype,
        "execution_fingerprint": trace.execution_fingerprint,
        "case_id": probe.case_id,
        "probe_kind": probe.probe_kind,
        "theme": probe.theme,
        "expected_relation": probe.expected_relation,
        "doctrine": formulation.doctrine_id,
        "label_family": formulation.label_family,
        "label_mapping": formulation.label_mapping,
        "label_order": formulation.label_order,
        "question": probe.question,
        "question_fingerprint": fingerprint(probe.question),
        "context_fingerprint": fingerprint(probe.context),
        "probability_true": evaluation_result.probability_true,
        "verbalizer_mass": diagnostics.verbalizer_mass,
        "top_token_text": diagnostics.top_token_text,
        "top_token_id": diagnostics.top_token_id,
        "top_token_probability": diagnostics.top_token_probability,
        "positive_token_probability": diagnostics.positive_token_probability,
        "negative_token_probability": diagnostics.negative_token_probability,
        "entropy": certainty.entropy,
        "margin": certainty.margin,
        "latency_ms": trace.latency_ms,
        "input_tokens": trace.input_token_count,
        "plan_fingerprint": trace.plan_fingerprint,
        "input_fingerprint": trace.input_fingerprint,
        "trace_id": trace.trace_id,
        "status": "ok",
        "context": probe.context,
        "expected_order_json": json.dumps(list(probe.expected_order)),
        "required_relation_json": json.dumps(list(probe.required_relation)),
    }
    return record


def _error_record(
    *,
    probe: ProbeSpec,
    formulation: Formulation,
    run_context: Mapping[str, Any],
    status: str,
    error: BaseException,
    skipped_probes: int = 0,
) -> dict[str, Any]:
    """Build a record for a failed or unsupported probe."""
    record: dict[str, Any] = {
        "record_type": "probe",
        "model": run_context["model"],
        "revision": run_context["revision"],
        "tokenizer_revision": None,
        "dtype": run_context["dtype"],
        "execution_fingerprint": None,
        "case_id": probe.case_id,
        "probe_kind": probe.probe_kind,
        "theme": probe.theme,
        "expected_relation": probe.expected_relation,
        "doctrine": formulation.doctrine_id,
        "label_family": formulation.label_family,
        "label_mapping": formulation.label_mapping,
        "label_order": formulation.label_order,
        "question": probe.question,
        "question_fingerprint": fingerprint(probe.question),
        "context_fingerprint": fingerprint(probe.context),
        "probability_true": None,
        "verbalizer_mass": None,
        "top_token_text": None,
        "top_token_id": None,
        "top_token_probability": None,
        "positive_token_probability": None,
        "negative_token_probability": None,
        "entropy": None,
        "margin": None,
        "latency_ms": None,
        "input_tokens": None,
        "plan_fingerprint": None,
        "input_fingerprint": None,
        "trace_id": None,
        "status": status,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": probe.context,
        "skipped_probes": skipped_probes,
    }
    return record


def run_sweep(
    *,
    backend: Any,
    formulations: Sequence[Formulation],
    probes: Sequence[ProbeSpec],
    run_context: Mapping[str, Any],
    out_path: Path,
    limit: int | None = None,
) -> int:
    """Run every formulation over the probes, writing JSONL records.

    Returns the number of probe records written (excluding the header).
    A single probe failure is recorded with status "error"; a formulation
    whose verbalizers the backend rejects is recorded once with status
    "unsupported" and the rest of its probes are skipped.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {
        "record_type": "run_header",
        "git_commit": _git_commit(),
        "model": run_context["model"],
        "revision": run_context["revision"],
        "dtype": run_context["dtype"],
        "device": run_context["device"],
        "doctrine_versions": {
            f.formulation_id: {"doctrine_id": f.doctrine_id, "version": f.doctrine.version}
            for f in formulations
        },
        "case_set_version": run_context["case_set_version"],
        "case_set_fingerprint": run_context["case_set_fingerprint"],
        "peak_vram_allocated_mib": None,
        "peak_vram_reserved_mib": None,
        "started_at": _now_iso(),
        "transformers_version": run_context.get("transformers_version"),
        "torch_version": run_context.get("torch_version"),
        "cuda_version": run_context.get("cuda_version"),
        "gpu_name": run_context.get("gpu_name"),
    }
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for formulation in formulations:
            compiler = formulation.make_compiler()
            runtime = Probvenance(backend=backend, compiler=compiler, capture_rendered_input=True)
            formulation_probes = list(probes)[:limit] if limit is not None else list(probes)
            skipped = False
            skipped_count = 0
            for probe in formulation_probes:
                if skipped:
                    skipped_count += 1
                    continue
                decision = BoolDecision(question=probe.question, context=probe.context)
                try:
                    evaluation = runtime.evaluate_with_trace(decision)
                except VerbalizerError as error:
                    # The label family is not representable for this tokenizer.
                    # Record it once, never substitute a label, move on.
                    skipped = True
                    record = _error_record(
                        probe=probe,
                        formulation=formulation,
                        run_context=run_context,
                        status="unsupported",
                        error=error,
                        skipped_probes=len(formulation_probes) - 1,
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
                    continue
                except Exception as error:
                    record = _error_record(
                        probe=probe,
                        formulation=formulation,
                        run_context=run_context,
                        status="error",
                        error=error,
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
                    continue
                record = _record_from_evaluation(
                    probe=probe,
                    evaluation_result=evaluation.result,
                    trace=evaluation.trace,
                    formulation=formulation,
                    run_context=run_context,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                written += 1
            if skipped:
                print(
                    f"[formulation] {formulation.formulation_id}: unsupported "
                    f"({skipped_count} probes skipped)"
                )
            else:
                print(f"[formulation] {formulation.formulation_id}: ok ({written} records so far)")
    allocated, reserved = _peak_vram_mib()
    _rewrite_header_fields(
        out_path,
        {
            "peak_vram_allocated_mib": allocated,
            "peak_vram_reserved_mib": reserved,
            "finished_at": _now_iso(),
        },
    )
    return written


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _rewrite_header_fields(path: Path, fields: Mapping[str, Any]) -> None:
    """Patch fields into the first (header) line of a JSONL file."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return
    header = json.loads(lines[0])
    header.update(fields)
    lines[0] = json.dumps(header, sort_keys=True) + "\n"
    path.write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Semantic signal validation sweep (one model per process)."
    )
    parser.add_argument(
        "--model", default=None, help="Hugging Face model id (required unless --dry-run)"
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--revision", default=None, help="model revision / commit hash")
    parser.add_argument(
        "--label-families", default=None, help="comma subset of yes_no,true_false,ab"
    )
    parser.add_argument("--doctrines", default=None, help="comma subset of D1,D2,D3")
    parser.add_argument(
        "--order-ablation", action="store_true", help="include negative-first variants"
    )
    parser.add_argument(
        "--mapping-swap", action="store_true", help="include swapped-mapping variants"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max probes per formulation (smoke tests)"
    )
    parser.add_argument(
        "--out", default=None, help="JSONL output path (default: results/<model>.jsonl)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fake backend, no torch, no model load"
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict real runs to local files (default: true)",
    )
    parser.add_argument(
        "--backend-factory",
        default=None,
        metavar="MODULE:ATTRIBUTE",
        help=(
            "callable with the TransformersBackend signature to construct the backend "
            "(default: probvenance.backends.transformers:TransformersBackend). Used to score "
            "checkpoints whose text tower needs a harness-side loader adaptation."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI runner."""
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.model:
        print("error: --model is required unless --dry-run", file=sys.stderr)
        return 2
    case_set = load_case_set(CASE_SET_PATH)
    probes = expand_probes(case_set)
    formulations = all_formulations(
        include_order_ablation=args.order_ablation,
        include_mapping_swap=args.mapping_swap,
    )
    if args.label_families:
        allowed = {f.strip() for f in args.label_families.split(",")}
        formulations = [f for f in formulations if f.label_family in allowed]
    if args.doctrines:
        allowed = {d.strip() for d in args.doctrines.split(",")}
        formulations = [f for f in formulations if f.formulation_id.split("_")[0] in allowed]
    if not formulations:
        print("error: formulation filters matched nothing", file=sys.stderr)
        return 2

    model = "fake-dry-run-model" if args.dry_run else str(args.model)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{_sanitize_model_id(model)}.jsonl"
    run_context: dict[str, Any] = {
        "model": model,
        "revision": args.revision,
        "dtype": None if args.dry_run else args.dtype,
        "device": "fake" if args.dry_run else (args.device or "auto"),
        "case_set_version": case_set["case_set_version"],
        "case_set_fingerprint": fingerprint(case_set),
    }

    if args.dry_run:
        backend: Any = FakeDryRunBackend()
    else:
        # Real run: import lazily so --dry-run never touches torch.
        try:
            if args.backend_factory is None:
                from probvenance.backends.transformers import TransformersBackend

                factory: Any = TransformersBackend
            else:
                module_name, separator, attribute = args.backend_factory.partition(":")
                if not separator:
                    print(
                        f"error: --backend-factory must be MODULE:ATTRIBUTE, got "
                        f"{args.backend_factory!r}",
                        file=sys.stderr,
                    )
                    return 2
                factory = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        try:
            backend = factory(
                model=model,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                local_files_only=args.local_files_only,
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception as error:
            print(
                f"error: backend initialization failed (the thinking-disable flag "
                f"and tokenizer are part of the frozen protocol): "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 3
        run_context.update(_cuda_info())

    _reset_peak_vram_guarded()
    written = run_sweep(
        backend=backend,
        formulations=formulations,
        probes=probes,
        run_context=run_context,
        out_path=out_path,
        limit=args.limit,
    )
    print(f"[done] {written} probe records -> {out_path}")
    return 0


def _reset_peak_vram_guarded() -> None:
    """Reset CUDA peak stats before the sweep, if torch and CUDA exist."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    raise SystemExit(main())
