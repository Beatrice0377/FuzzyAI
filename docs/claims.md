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

## Verified claims (Phase 1)

This section contains NO model-quality, performance, or accuracy claims,
because no real model backend exists yet. Everything below is a statement
about the deterministic core and its tests, reproducible with `uv run pytest`.

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

## Current hypotheses

- `[H]` An explicit scoring doctrine may improve cross-model semantic
  stability of the same decision. Confirming or refuting this requires
  evaluation work that does not exist yet.
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
