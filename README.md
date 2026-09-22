**English** | [简体中文](README.zh-CN.md)

# FuzzyAI

**Status: early development.** Phase 1 only. No real model backend exists yet.

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

## Planned API

The surface below is the **planned API, not yet implemented**. There is no
`ai.bool` or `ai.choice` entry point in Phase 1.

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
  It is not automatically a real-world correctness rate.
- **Certainty**: how concentrated that distribution is (entropy, margin). A
  mathematical property, not a correctness probability. We never call it
  "confidence".
- **Predicted correctness**: only expressible after valid calibration. Every
  Phase 1 result has `predicted_correctness = None`. A max softmax, a logit, an
  entropy, or an LLM saying "I'm 95% sure" is none of these things.

The full normative definitions live in the
[design constitution](docs/design-constitution.md).

## What exists today (Phase 1)

Phase 1 ships the contracts, not the inference:

- `BoolDecision` and `ChoiceDecision` decision specs (the only Phase 1 primitives)
- The result model: `BoolResult`, `ChoiceResult`, `Certainty`
- `BackendCapabilities` (explicit data) and the narrow `Backend` Protocol
- `InferencePlan` and `RawEvidence` abstractions
- The deterministic fingerprint system (canonical JSON + SHA-256)
- The error taxonomy rooted at `FuzzyAIError`

The compiler and all real backends are future work. Architecture is proven on
paper, not yet wired end-to-end.

```
DecisionSpec  --(Compiler, future)-->  InferencePlan  -->  Backend (Protocol)  -->  RawEvidence
                                                                                        |
                                                          (future scoring/calibration)  v
                                                                                  DecisionResult
```

## Development

The project's own dev commands (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Documentation

- [docs/design-constitution.md](docs/design-constitution.md): binding design
  constraints, glossary, and normative invariants. Read this before proposing
  anything.
- [docs/roadmap.md](docs/roadmap.md): intended phases. Not a schedule, not a
  commitment.

## Non-goals

Phase 1 has no model loading, no HTTP, no transformers/OpenAI/Anthropic/vLLM/
SGLang integration, no automatic routing, no decision graph, no calibration
algorithm, and no dashboard, server, agent, RAG, database, telemetry, or web
UI. This project makes no benchmark or performance claims, and claims no model
support.

## License

MIT. See [LICENSE](LICENSE).
