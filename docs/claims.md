# Claims Register

Purpose: this document exists to stop an AI project from gradually turning
hypotheses into stated facts as the README evolves. Every capability,
performance, quality, calibration, or provider-support claim made about this
project must carry one of the four evidence statuses defined below. A claim
without a status (and, for `[V]`, without a pointer to evidence) does not
belong in any document in this repository.

## How to read this document

- `[V] VERIFIED`: verified by tests in this repository, by a reproducible
  experiment, or by explicit evidence that currently exists. A verified claim
  must point at tests, a code invariant, or a reproducible command.
- `[E] EXPERIMENTAL`: an experimental result observed under specific
  conditions (model, dataset, version, configuration). State the conditions.
  Never present a single experimental result as a general truth.
- `[H] HYPOTHESIS`: plausible and worth verifying, but not established.
  Wording must stay hedged ("may", "hypothesis", "to be evaluated").
- `[R] ROADMAP`: not implemented. A roadmap item is never a capability claim.

This register is referenced by the Architecture Principles of the
[design constitution](./design-constitution.md), and roadmap entries link to
[the roadmap](./roadmap.md) rather than duplicating it here.

## Verified claims

This section contains NO model-quality, performance, or accuracy claims. The
Bool path runs against a real local Hugging Face backend and has been executed
on real hardware; that single session is recorded under "Experimental records"
below and demonstrates nothing about model quality. Everything else is a
statement about the deterministic core, the Bool vertical slice, and their
tests, reproducible with `uv run pytest`. The `transformers` backend tests skip
automatically unless the optional extra is installed
(`uv sync --extra transformers`); they use injected fakes, never download
models, and never need a GPU.

### Deterministic core (Phase 1)

- `[V]` Decision fingerprints are deterministic under the documented
  canonicalization rules. Evidence: `tests/test_fingerprint.py`
  (`test_deterministic`, `test_stable_across_calls`), INV-07 and INV-12 in
  `docs/design-constitution.md`.
- `[V]` Context dict insertion order does not affect fingerprints. Evidence:
  `test_fingerprint_dict_order_independent_for_context` in
  `tests/test_decisions.py`, `test_dict_keys_sorted` in
  `tests/test_fingerprint.py`, INV-10.
- `[V]` Choice ordering affects `ChoiceDecision` semantics and its
  fingerprint. Evidence: `test_choice_order_preserved` and
  `test_fingerprint_order_sensitive_for_choices` in `tests/test_decisions.py`,
  INV-13.
- `[V]` Uncalibrated Phase 1 results cannot expose `predicted_correctness`.
  Evidence: `test_defaults_uncalibrated` and
  `test_predicted_correctness_requires_calibration` in `tests/test_results.py`,
  INV-04.
- `[V]` `NaN` and `Infinity` (in any sign, at any nesting depth) are rejected
  by fingerprinting. Evidence: `test_non_finite_floats_rejected` and
  `test_nan_in_nested_structure_rejected` in `tests/test_fingerprint.py`,
  INV-09.
- `[V]` `ChoiceResult.value` uses a deterministic first-in-order tie-break.
  Evidence: `test_tie_breaks_to_first_in_order` in `tests/test_results.py`,
  INV-03.
- `[V]` `BackendCapabilities` is explicit data and is never probed with
  `hasattr`. Evidence: `tests/test_capabilities.py`, INV-14.
- `[V]` The `Backend` protocol is unaware of decisions and results. Evidence:
  `test_backend_is_decision_agnostic` in `tests/test_plans.py`, INV-16.

### Bool vertical slice (Phase 2A)

- `[V]` `assemble_bool_probability` computes a numerically stable two-way
  softmax over exactly two labeled logits, equal to `sigmoid(l_true - l_false)`,
  with no overflow at extreme magnitudes. Evidence:
  `tests/test_assembler.py::TestSoftmaxMatrix::test_equal_logits_is_half`,
  `::test_ln3_gap_is_three_to_one`, `::test_large_equal_logits_no_overflow`,
  `::test_minus_thousand_true_gap`, `::test_plus_thousand_false_gap`,
  `::test_label_order_true_false_maps_by_label`, INV-01.
- `[V]` Assembled Bool results stay uncalibrated even at extreme logits:
  `calibrated = False` and `predicted_correctness = None`. Evidence:
  `tests/test_assembler.py::TestUncalibratedDefaults::test_extreme_logits_stay_uncalibrated`,
  INV-04.
- `[V]` A verbalizer that does not resolve to exactly one distinct scoring
  token raises `VerbalizerError`: no silent fallback, no truncation, no
  multi-token logit summing. Evidence:
  `tests/test_verbalizers.py::test_multi_token_positive_raises`,
  `::test_multi_token_negative_raises`, `::test_identical_token_for_both_raises`,
  `::test_zero_added_tokens_raises`, and
  `tests/test_backends_transformers.py::test_execute_rejects_multi_token_verbalizer_before_forward`
  (the error is raised before the model runs any forward pass).
- `[V]` `TransformersBackend` declares exactly one capability,
  `binary_token_logits=True`, and every other capability field False.
  Evidence: `tests/test_backends_transformers.py::test_capabilities_are_honest`,
  INV-14.
- `[V]` The backend reads only the final-position logits row. Evidence:
  `tests/test_backends_transformers.py::test_execute_reads_last_position_only`;
  the fake logits object in that test asserts the only permitted read is
  `logits[0, -1, :]`.
- `[V]` The compiler selects a strategy only from declared capabilities;
  insufficient capability raises `UnsupportedCapabilityError` before
  execution. Evidence:
  `tests/test_compiler.py::TestBoolCompilerCompile::test_missing_capability_rejected`,
  `tests/test_runtime.py::TestRuntimeRejections::test_missing_capability_rejected_before_execution`,
  INV-15.
- `[V]` `ChoiceDecision` has no runtime path: compiling or evaluating one
  raises `UnsupportedDecisionError` before execution. Evidence:
  `tests/test_compiler.py::TestBoolCompilerCompile::test_choice_decision_rejected`,
  `tests/test_runtime.py::TestRuntimeRejections::test_choice_decision_rejected_before_execution`.
- `[V]` `DecisionTrace.trace_id` is never derived from any fingerprint.
  Evidence:
  `tests/test_trace.py::TestTraceIdProvenance::test_trace_id_is_not_any_fingerprint`.
- `[V]` The `FuzzyAI` facade composes compile, execute, assemble, and trace in
  a fixed order and rejects evidence whose plan fingerprint does not match the
  plan it came from. Evidence:
  `tests/test_runtime.py::TestEndToEnd::test_evaluate_with_trace_full_pipeline`,
  `::test_trace_lineage_matches_decision_plan_and_evidence`,
  `tests/test_runtime.py::TestRuntimeRejections::test_foreign_plan_fingerprint_rejected`.
- `[V]` Rendered decision context never enters the system prompt; it is placed
  into the user prompt as evidence only. This is a structural placement rule,
  not a prompt-injection-safety claim. Evidence:
  `tests/test_doctrine.py::TestScoringDoctrineRender::test_malicious_context_never_enters_system_prompt`.
- `[V]` The trace's `input_fingerprint` describes the text actually rendered
  and fed to the model rather than the plan fields, and does not depend on
  whether the full rendered text was captured. Two different renderings of the
  same plan produce different fingerprints. Evidence:
  `TestInputFingerprintProvenance` in `tests/test_trace.py`
  (`test_uses_rendered_input_when_evidence_carries_it`,
  `test_falls_back_to_plan_fields_without_rendered_input`,
  `test_capture_flag_does_not_change_fingerprint`,
  `test_different_renderings_give_different_fingerprints`).
- `[V]` The runtime rejects evidence that carries no plan fingerprint, so
  lineage cannot be skipped. Evidence:
  `tests/test_runtime.py::test_evidence_without_lineage_is_rejected`.
- `[V]` The transformers backend performs read-only scoring and never calls
  `model.generate()`. Evidence:
  `tests/test_backends_transformers.py::test_execute_never_calls_generate`
  (the fake model records the call and raises, so a regression fails loudly).

### Scoring validity and execution provenance (Phase 2A.1)

- `[V]` Restricted-probability diagnostics are computed from full-vocabulary
  normalization, not from the two candidate logits alone. Evidence:
  `tests/test_diagnostics.py` (33 tests: verbalizer-mass math, full-vocab
  probability, `diagnose_bool_evidence` field validation) and
  `tests/test_backends_transformers.py::test_metadata_carries_full_vocabulary_statistics`.
- `[V]` A `DecisionTrace` always carries `ScoringDiagnostics`; it cannot be
  constructed without them. Evidence:
  `tests/test_runtime.py::TestEndToEnd::test_trace_carries_scoring_diagnostics`.
- `[V]` The execution fingerprint is deterministic for the same configuration
  and changes when the rendered input, the rendering config, the model
  revision, or either resolved token id changes. Evidence:
  `tests/test_trace.py::TestExecutionFingerprint`
  (`test_same_config_same_fingerprint`,
  `test_different_rendered_input_different_fingerprint`,
  `test_different_rendering_config_different_fingerprint`,
  `test_different_model_revision_different_fingerprint`,
  `test_different_positive_token_id_different_fingerprint`,
  `test_different_negative_token_id_different_fingerprint`).
- `[V]` Trace id is per-execution identity, distinct from every fingerprint:
  the same execution fingerprint can carry different trace ids. Evidence:
  `tests/test_trace.py::TestExecutionFingerprint::test_same_execution_fingerprint_different_trace_id`
  and
  `tests/test_trace.py::TestExecutionFingerprint::test_distinct_from_plan_decision_and_input_fingerprints`.
- `[V]` `capture_rendered_input` does not change the input or execution
  fingerprint. Evidence:
  `tests/test_trace.py::TestExecutionFingerprint::test_capture_flag_does_not_change_either_fingerprint`.
- `[V]` One Bool evaluation performs exactly one model forward pass (no extra
  pass for diagnostics). Evidence:
  `tests/test_backends_transformers.py::test_single_forward_per_evaluation`.
- `[V]` Decoding the top token text is best-effort and never fails an
  inference. Evidence:
  `tests/test_backends_transformers.py::test_top_token_text_decode_failure_is_not_fatal`.

## Experimental records

Single-session observations, each reported with the conditions under which it
was obtained. They are not quality, performance, or support claims, and none of
them generalises to other models, revisions, prompts, or tasks.

- `[E]` Rendering mode changes the decision distribution of the same
  `InferencePlan`. Model `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, dtype `float32`, on an NVIDIA
  GeForce RTX 5060 Laptop GPU (8123 MiB), run offline against the local cache
  through `examples/bool_local.py`. With the tokenizer's default chat-template
  mode the assistant turn starts in thinking mode, the model's next token is
  its reasoning opener at probability 0.9997, and both resolved verbalizer
  tokens (`yes` = 9693, `no` = 2152) sit at probability ~0, so the two-way
  softmax renormalizes tail noise: `P(True) = 0.3479` for "Did the customer's
  package fail to arrive?". With `chat_template_kwargs={"enable_thinking":
  False}` the verbalizers become the actual decision candidates and the same
  plan gives `P(True) = 0.9951`. Both values are uncalibrated. Neither is
  evidence of correctness; the difference records how much the probability
  depended on an unrecorded rendering choice.
- `[E]` One session of three Bool decisions on the same setup, thinking
  disabled: `P(True) = 0.9951` (123 input tokens, first call 1880.3 ms
  including model warm-up), `0.9926` (115 tokens, 51.5 ms), `0.9964` (116
  tokens, 51.9 ms). `calibrated` was `False` and `predicted_correctness` was
  `None` in all three. Each question is one a human would answer "yes", but
  this is not a correctness measurement: no ground truth was collected and
  three samples cannot support a quality claim.
- `[E]` Restricted probability versus verbalizer mass at a decision point.
  Model `Qwen/Qwen3-0.6B`, full precision (float32), `NVIDIA GeForce RTX 5060
  Laptop GPU` with 8123 MiB capacity, offline (`HF_HUB_OFFLINE=1`), scoring
  verbalizers `yes` / `no`. The restricted probability alone could not reveal
  that the model was not at a decision point, while `verbalizer_mass` could.
  Same plan, same question, only the chat-template rendering mode differed:

  | rendering | P(True) | verbalizer_mass | top token | top token probability |
  |---|---|---|---|---|
  | thinking disabled | 0.9988 | 0.954228 | 'yes' (id 9693) | 0.953119 |
  | thinking enabled | 0.5116 | 0.000000 | '<think>' (id 151667) | 0.999699 |

  This is a single-case observation about scoring position, not a quality
  or accuracy measurement. These probabilities are uncalibrated and were not
  evaluated against ground truth.
- `[E]` Six-case exploratory session (thinking disabled) on the same model and
  hardware. Recorded as observations only. All six cases resolved to top token
  'yes'; this is NOT an accuracy claim (one 0.6B model, six hand-written
  examples, no ground truth, no calibration):

  | case | P(True) | verbalizer_mass | top token prob | entropy | margin | input tokens | latency ms |
  |---|---|---|---|---|---|---|---|
  | obvious true | 0.9988 | 0.954228 | 0.953119 | 0.0130 | 0.9977 | 106 | 17781.4 (includes warm-up) |
  | obvious false | 0.9928 | 0.858544 | 0.852394 | 0.0613 | 0.9857 | 112 | 185.5 |
  | ambiguous | 0.9956 | 0.965834 | 0.961592 | 0.0407 | 0.9912 | 111 | 62.4 |
  | insufficient evidence | 0.9931 | 0.730371 | 0.725351 | 0.0593 | 0.9863 | 108 | 53.7 |
  | contradictory | 0.9949 | 0.900218 | 0.895594 | 0.0464 | 0.9897 | 121 | 52.6 |
  | prompt-injection-like | 0.9999 | 0.999018 | 0.998910 | 0.0016 | 0.9998 | 115 | 54.0 |

  In every case `calibrated=False`, `predicted_correctness=None`, and
  `method=binary_token_logits`. No conclusion is drawn about the model being
  right or wrong on any case.

## Current hypotheses

- `[H]` An explicit scoring doctrine may improve cross-model semantic
  stability of the same decision. The doctrine itself now exists as
  `BINARY_SEMANTIC_JUDGMENT_V1`; confirming or refuting the stability claim
  requires evaluation work that does not exist yet.
- `[H]` Small local models combined with task-specific calibration may be
  sufficient for useful narrow semantic decisions. Confirming or refuting
  this requires calibration and evaluation work that does not exist yet.
- `[H]` Cache-preserving compilation may materially reduce local inference
  cost when many decisions share a long context. Confirming or refuting this
  requires benchmark work on real backends that does not exist yet.

Hypotheses are recorded here so they can be tested, and must not be quoted as
features.

## Roadmap claims

Planned and unimplemented directions are tracked in
[the roadmap](./roadmap.md) and must not be restated here as capabilities.
They may carry `[R]` status in this register only once they have a named entry
in that roadmap.
