"""Local boolean demo against a Hugging Face causal LM (Phase 2A slice).

Runs a few low-risk, local-only demo decisions through the real FuzzyAI
vertical slice:

    BoolDecision -> BoolCompiler -> InferencePlan -> TransformersBackend
        -> RawEvidence -> assemble_bool_probability -> BoolResult -> DecisionTrace

The backend never generates text. It scores the ``yes`` / ``no`` verbalizer
tokens at the last input position and reads those two logits directly, so
``probability_true`` is a numerically stable two-way softmax over exactly
those two logits.

Nothing here is calibrated: ``calibrated`` is False and
``predicted_correctness`` is None in every run. The printed numbers are one
implementation's behaviour on one model, not a quality claim.

Usage:

    uv sync --extra transformers
    uv run --extra transformers python examples/bool_local.py
    FUZZYAI_MODEL=Qwen/Qwen3-0.6B uv run --extra transformers python examples/bool_local.py
"""

from __future__ import annotations

import argparse
import os

from fuzzyai.decisions import BoolDecision
from fuzzyai.runtime import FuzzyAI

try:
    from fuzzyai.backends.transformers import TransformersBackend
except ImportError as exc:  # pragma: no cover - only hit without the extra
    raise SystemExit(
        "This demo needs the optional transformers extra: uv sync --extra transformers"
    ) from exc

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

DEMO_DECISIONS: tuple[BoolDecision, ...] = (
    BoolDecision(
        question="Did the customer's package fail to arrive?",
        context=(
            "The tracking page says the parcel was delivered, but the customer "
            "reports that nothing ever showed up."
        ),
    ),
    BoolDecision(
        question="Does the customer want to cancel their subscription?",
        context="Please stop my plan before the next billing cycle starts.",
    ),
    BoolDecision(
        question="Is the customer asking for a refund?",
        context="I would like my money back for an order that never arrived.",
    ),
)


def _device_report() -> str:
    """Best-effort device/dtype line; empty when torch is unavailable."""
    try:
        import torch
    except ImportError:  # pragma: no cover - only hit without the extra
        return ""
    if not torch.cuda.is_available():
        return "device: cpu"
    index = torch.cuda.current_device()
    total_mib = torch.cuda.get_device_properties(index).total_memory / (1024 * 1024)
    return (
        f"device: {torch.cuda.get_device_name(index)}"
        f" ({total_mib:.0f} MiB, capability {torch.cuda.get_device_capability(index)})"
    )


def _run_decision(ai: FuzzyAI, decision: BoolDecision) -> None:
    evaluation = ai.evaluate_with_trace(decision)
    result = evaluation.result
    trace = evaluation.trace

    print(f"question:              {decision.question}")
    print(f"P(True):               {result.probability_true:.4f}")
    print(f"certainty entropy:     {result.certainty.entropy:.4f}")
    print(f"certainty margin:      {result.certainty.margin:.4f}")
    print(f"method:                {result.method}")
    print(f"calibrated:            {result.calibrated}")
    print(f"predicted_correctness: {result.predicted_correctness}")
    print(f"verbalizers:           {trace.positive_verbalizer}/{trace.negative_verbalizer}")
    print(f"scoring token ids:     {trace.positive_token_id}/{trace.negative_token_id}")
    print(f"doctrine:              {trace.doctrine_id}")
    print(f"model:                 {trace.model} @ {trace.model_revision}")
    print(f"input tokens:          {trace.input_token_count}")
    print(f"latency_ms:            {trace.latency_ms:.1f}")
    print(f"trace_id:              {trace.trace_id}")
    print(f"decision fingerprint:  {trace.decision_fingerprint}")
    print(f"plan fingerprint:      {trace.plan_fingerprint}")
    if trace.rendered_input is not None:
        print(f"rendered input:        {trace.rendered_input!r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("FUZZYAI_MODEL", DEFAULT_MODEL),
        help="Hugging Face model id (env: FUZZYAI_MODEL, default: %(default)s)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "leave the model's default thinking mode on "
            "(env: FUZZYAI_ENABLE_THINKING=1); the default disables it"
        ),
    )
    args = parser.parse_args()

    # Thinking-mode models (e.g. Qwen3) emit their reasoning opener as the next
    # token after the generation prompt, pushing both verbalizer logits into the
    # vanishing tail, so the two-way P(True) becomes tail noise. Disabling the
    # thinking block pre-fills an empty one and the model answers directly.
    enable_thinking = args.thinking or os.environ.get("FUZZYAI_ENABLE_THINKING", "") in (
        "1",
        "true",
        "yes",
    )

    print(f"model: {args.model}")
    report = _device_report()
    if report:
        print(report)
    print()

    backend = TransformersBackend(
        args.model,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    ai = FuzzyAI(backend=backend, capture_rendered_input=True)
    for decision in DEMO_DECISIONS:
        _run_decision(ai, decision)


if __name__ == "__main__":
    main()
