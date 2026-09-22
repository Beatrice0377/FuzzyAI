# Semantic Signal Validation (Phase 2A.2)

This is an experiment harness, not product runtime code and not a benchmark.
It measures whether the binary scoring position of a causal language model
carries a semantic signal at all, and which parts of that signal come from the
model, the doctrine, the label family, the label order, and the label mapping.

## What this tests

Mechanism validation. Each probe asks one binary question over one context and
records the full next-token scoring picture: `probability_true`, the raw
verbalizer probabilities, the verbalizer mass over the full vocabulary, the
top token, entropy, margin, latency, and the full fingerprint lineage. The
results are NOT calibrated and NOT a correctness measure. Nothing here decides
whether a scoring position is valid; that requires a threshold policy that
does not exist yet.

## Research questions

- RQ1 semantic sensitivity: does P(True) move in the right direction as the
  evidence strengthens (ladder rungs, contrast groups)?
- RQ2 polarity sensitivity: does P(True) separate a supporting question from
  its opposite, asked over the same context (polarity pairs, injection probes)?
- RQ3 label invariance: is the semantic signal invariant across label families
  (yes/no, true/false, A/B), across label order (positive first vs negative
  first), and across the mapping swap (semantic TRUE bound to the other
  surface label)?
- RQ4 model dependence: which of the above hold for one model but not another
  (one model per process, one process per model)?

## Case set

`cases.json` is frozen. `case_set_version` is `semantic-signal-v1`. It contains:

- 8 ladders x 5 rungs (strong_positive, weak_positive, insufficient,
  weak_negative, strong_negative)
- 12 polarity pairs (supporting question vs opposite question, same context)
- 4 contrast groups (relevant_positive, irrelevant, relevant_negative
  contexts, with an expected order and a required relation)
- 4 injection probes (embedded instructions against the evidence)

That expands to exactly 80 probes per formulation. The run header records
`case_set_fingerprint`, the SHA-256 fingerprint of the parsed case set over
canonical JSON, so any later edit of the case set is detectable.

## Formulation matrix

Three doctrines: D1 is the existing `BINARY_SEMANTIC_JUDGMENT_V1` baseline,
D2 is a minimal neutral variant (`minimal-neutral-v1`), D3 is an
evidence-oriented variant (`evidence-oriented-v1`) that frames the task as
support evaluation. Three label families: `yes_no`, `true_false`, `ab`. For
`ab` the doctrine template explicitly declares which surface label means
supported and which means not supported, so the mapping is carried by the
doctrine variant, never by a runtime change.

Two ablations, both opt-in:

- `--order-ablation`: the prompt lists the negative label first. The semantic
  mapping is unchanged; only the presentation order changes.
- `--mapping-swap`: the semantic mapping is inverted (for `ab`, the doctrine
  declares the inverted mapping and the compiler binds the other surface label
  to semantic TRUE). Every record carries `label_mapping` (`canonical` or
  `swapped`) so the analysis can tell whether the model followed the declared
  mapping or the surface token.

With both flags on the matrix is 27 formulations: the base 9 (3 doctrines x 3
families, positive first, canonical), plus 9 swapped and 9 negative-first.

## How to run

From the repo root:

    python experiments/semantic_signal/run.py --dry-run --out /tmp/smoke.jsonl

    python experiments/semantic_signal/run.py --model Qwen/Qwen3-1.7B \
        --order-ablation --mapping-swap

    python experiments/semantic_signal/run.py --model Qwen/Qwen3-1.7B \
        --label-families yes_no,true_false --doctrines D1,D3 --limit 10

Summarize results (stdlib only):

    python experiments/semantic_signal/metrics.py results/<model>.jsonl

## GPU discipline

- Exactly ONE model resident at a time. One process per model. The runner
  loads the model once and never a second one; batch size is always 1.
- Run `nvidia-smi` before and after each run to confirm nothing else is
  resident and nothing leaked.
- No quantization: dtype is bfloat16 (default), float16, or float32. No 4 bit
  or 8 bit loading, no offload tricks.
- The runner records peak VRAM (allocated and reserved, in MiB) in the run
  header.
- `chat_template_kwargs={"enable_thinking": False}` is part of the frozen
  protocol. If the backend or tokenizer rejects it, that is a hard failure to
  record, never something to silently drop.

## Unsupported label families

Label families are validated by the real tokenizer only at execution time. If
a label family raises (for example a multi-token label, or a VerbalizerError),
the runner writes one record with `"status": "unsupported"` plus the error
type and message, skips the remaining probes of that formulation, and
continues with the remaining formulations. It never substitutes a different
label and never retries with a modified label.

## Output

One JSONL file per run. The first line is a `run_header` record (git commit,
model, revision, dtype, device, library versions, CUDA and GPU info, doctrine
versions, case set version and fingerprint, peak VRAM, timestamps). Every
following line is one probe record with the full diagnostics, the full
question and context text, and a `status` of `ok`, `error`, or `unsupported`.

## What the results are not

The numbers are not calibrated probabilities and not a correctness measure.
No threshold is applied anywhere in this harness, no probe is auto-rejected,
and no record carries a validity verdict. The verbalizer mass bands in
`metrics.py` are report only: they exist to separate scoring position failures
from semantic failures, and nothing is rejected on them.
