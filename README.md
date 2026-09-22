**English** | [简体中文](README.zh-CN.md)

# FuzzyAI

**Status: early development.** The deterministic core and a Bool vertical
slice with a real local Hugging Face backend are implemented. Choice
inference, calibration, and abstention are not.

FuzzyAI is a provider-agnostic probabilistic decision runtime. It turns language
models into evaluable, calibratable, trackable semantic-probability decision
components.

The problem: LLMs get wired into program logic as if a raw text blob or an
unexamined score were a trustworthy probability. Systems end up unable to say
what a score means, unable to reproduce a past decision, and unable to separate
"the model was unsure" from "the program should decline to act". FuzzyAI puts a
narrow, deterministic core under that space.

It is not a chat framework, not an agent framework, and not merely a
structured-output wrapper.

## Core principle

> **The LLM handles semantic uncertainty. The program handles deterministic policy.**

## Currently working: the Bool vertical slice

The one end-to-end path that exists today is Bool-only. A thin `FuzzyAI` facade
orchestrates the pipeline in a fixed order and adds no fallbacks of its own:

```
BoolDecision -> BoolCompiler -> InferencePlan -> TransformersBackend
   -> RawEvidence -> assemble_bool_probability -> BoolResult -> DecisionTrace
```

The backend is an optional extra. The base package has zero runtime
dependencies; installing the extra is what pulls in `torch` and `transformers`:

```bash
uv sync --extra transformers    # group: torch>=2.7, transformers>=4.53
```

```python
from fuzzyai import BoolDecision, FuzzyAI
from fuzzyai.backends.transformers import TransformersBackend

backend = TransformersBackend(
    "Qwen/Qwen3-0.6B",  # a local HF causal LM
    # Qwen3 starts a reasoning block first otherwise; see the note below.
    chat_template_kwargs={"enable_thinking": False},
)
ai = FuzzyAI(backend=backend)

result = ai.evaluate(
    BoolDecision(
        question="Does the evidence support the claim?",
        context={"claim": "the cache was warm", "evidence": "hit ratio rose"},
    )
)

result.probability_true        # two-way softmax over the two verbalizer logits
result.certainty               # entropy + margin of that distribution
result.predicted_correctness   # None: calibration does not exist yet
result.trace_id                # correlates the result with its DecisionTrace
```

`ai.evaluate_with_trace(decision)` returns an `Evaluation` (the result plus its
`DecisionTrace`). The trace records the decision and plan fingerprints, the
scoring strategy, the doctrine id, the resolved verbalizer token ids, the raw
evidence, the input fingerprint, the backend type, and the latency.

What `probability_true` is: a numerically stable two-way softmax over exactly
two verbalizer-token logits, equal to `sigmoid(l_true - l_false)`. What it is
not: a full-vocabulary probability, a real-world event probability, a
prediction-accuracy figure, or a calibrated number. `calibrated` stays `False`
and `predicted_correctness` stays `None`.

The backend loads a local causal LM, runs it under `eval()` and
`torch.inference_mode()`, prefers CUDA and falls back to CPU, and reads only
the final-position logits. It never calls `generate()` and never parses
generated text. Verbalizers default to `yes`/`no` and must each resolve to
exactly one distinct scoring token after the real chat-templated prefix;
otherwise a `VerbalizerError` is raised. There is no silent fallback, no
truncation, no multi-token summing, and no sampling fallback.

`probability_true` is only meaningful if the model is genuinely at a decision
position, meaning its next token really is one of the two verbalizers. Some
models (thinking or reasoning models) start a reasoning block instead: with the
wrong chat-template mode the model's top token is its reasoning opener and both
verbalizer tokens sit in the far tail, so the two-way softmax renormalizes tail
noise into a number that looks plausible. The demo above therefore renders with
`chat_template_kwargs={"enable_thinking": False}`. The measured difference is
recorded in [docs/claims.md](docs/claims.md): the same plan and the same model
gave `P(True) = 0.3479` in thinking mode and `P(True) = 0.9951` with thinking
disabled. Phase 2A does not detect this automatically, and refusing to answer
would be a policy action that does not exist yet.

Decision context is rendered into the user prompt as evidence only and never
into the system prompt. That is a structural placement rule, not a claim of
prompt-injection safety.

`ChoiceDecision` has no runtime path: it remains a Phase 1 data model, and
compiling one raises `UnsupportedDecisionError`.

## Planned API (not implemented)

The convenience surface below is still the **planned API**. There is no
`ai.bool` or `ai.choice` entry point; the working call is
`ai.evaluate(BoolDecision(...))` as shown above.

```python
risk = ai.bool("Is this transaction suspicious?", context=transaction)

route = ai.choice(
    "Which team should handle this ticket?",
    choices={
        "billing": "Payment and billing issues",
        "shipping": "Delivery and logistics",
        "returns": "Returns and refunds",
    },
    context=ticket,
)
```

## Probability, certainty, predicted correctness: not the same thing

- **Probability**: the decision distribution produced by some scoring strategy.
  It is not automatically a real-world correctness rate. Today's Bool
  probability comes from binary token logits and is not a full-vocabulary
  distribution.
- **Certainty**: how concentrated that distribution is (entropy, margin). A
  mathematical property, not a correctness probability. We never call it
  "confidence".
- **Predicted correctness**: only expressible after valid calibration. Every
  result produced today has `predicted_correctness = None` and
  `calibrated = False`. A max softmax, a logit, an entropy, or an LLM saying
  "I'm 95% sure" is none of these things.

The full normative definitions live in the
[design constitution](docs/design-constitution.md).

## What exists today

The Phase 1 deterministic core:

- `BoolDecision` and `ChoiceDecision` decision specs
- The result model: `BoolResult`, `ChoiceResult`, `Certainty`
- `BackendCapabilities` (explicit data) and the narrow `Backend` Protocol
- `InferencePlan` and `RawEvidence` abstractions
- The deterministic fingerprint system (canonical JSON + SHA-256)
- The error taxonomy rooted at `FuzzyAIError`

The Phase 2A Bool path:

- `ScoringDoctrine` and the default `BINARY_SEMANTIC_JUDGMENT_V1` doctrine
- `BoolCompiler` (a pure planner: it never computes probabilities)
- `assemble_bool_probability` (two-way softmax over the binary token logits)
- `TransformersBackend` (optional `transformers` extra, logits only)
- `DecisionTrace` / `build_decision_trace` (its `trace_id` is never derived
  from a fingerprint)
- The thin `FuzzyAI` / `Evaluation` facade

```
BoolDecision --(BoolCompiler)--> InferencePlan --(TransformersBackend)--> RawEvidence
                                                                             |
                              assemble_bool_probability     (calibration: still future)
                                                                             v
                                                BoolResult + DecisionTrace (via FuzzyAI)
```

Choice inference, calibration, and abstention remain future work.

## Evidence and claims

Performance, quality, calibration, and provider-support claims are tracked by
evidence status in [docs/claims.md](docs/claims.md). The project makes no
benchmark, performance, or model-support claims today: the Bool path runs
against a local Hugging Face causal LM, but no model has been evaluated.
Hypotheses and roadmap items are labelled as such there, rather than presented
as capabilities.

## Development

The project's own dev commands (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

The `transformers` backend tests skip automatically unless
`uv sync --extra transformers` has been run; they never download models and
never need a GPU.

## Documentation

- [docs/design-constitution.md](docs/design-constitution.md): binding design
  constraints, glossary, and normative invariants. Read this before proposing
  anything.
- [docs/roadmap.md](docs/roadmap.md): intended phases. Not a schedule, not a
  commitment.
- [docs/claims.md](docs/claims.md): the evidence status of every capability claim ([V] verified, [E] experimental, [H] hypothesis, [R] roadmap).

## Non-goals

There is still no HTTP, no OpenAI/Anthropic/vLLM/SGLang integration, no cloud
backend, no automatic routing, no decision graph, no calibration algorithm, no
abstention policy, and no dashboard, server, agent, RAG, database, telemetry,
or web UI. This project makes no benchmark or performance claims, and claims no
model quality or support.

## License

MIT. See [LICENSE](LICENSE).
