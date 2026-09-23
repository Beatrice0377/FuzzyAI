"""Pure continuation and verbalizer helpers for token-logit backends.

These helpers are deliberately free of any torch/transformers import so they
are unit-testable with a fake tokenizer, no network, and no GPU. They serve both
scoring paths: binary verbalizers and categorical scoring labels.

- A scoring label (verbalizer or categorical label) is valid only if it
  preserves the REAL rendered prefix tokenization exactly and appends exactly
  one token: ``tokenizer(prefix + label) == tokenizer(prefix) + [label_token]``.
  A net delta of one token is NOT sufficient; a retokenized prefix is rejected
  even when the count happens to work out. The check is performed against the
  concatenated text (``tokenizer(prefix + label)``), never as
  ``tokenizer(prefix) + tokenizer(label)``, because BPE boundary effects are
  exactly what is being validated.
- The positive and negative verbalizers must resolve to DISTINCT token ids.
- Any violation raises a scoring-label error (``VerbalizerError`` for
  verbalizers, ``ScoringLabelError`` for categorical labels). There is no
  silent fallback, no truncation, and no multi-token logit summing.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from fuzzyai.errors import ScoringLabelError, VerbalizerError
from fuzzyai.fingerprint import JSONValue


class TokenizedText(Protocol):
    """Result of tokenizing a text: anything list()-able of ints (e.g. a tensor)."""

    input_ids: object


class SupportsChatTemplate(Protocol):
    """The minimal tokenizer surface the verbalizer helpers rely on."""

    chat_template: str | None

    def __call__(self, text: str, *, add_special_tokens: bool = ...) -> TokenizedText: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        **kwargs: JSONValue,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class VerbalizerTokens:
    """Resolved single scoring tokens for the positive and negative verbalizers."""

    positive_verbalizer: str
    negative_verbalizer: str
    positive_token_id: int
    negative_token_id: int


# Keys that would collide with the arguments render_input_text controls itself
# when forwarding template_kwargs into apply_chat_template. These are rejected
# up front with a readable error; anything else (template-specific flags like
# Qwen3's enable_thinking) is the template's business and passes through
# unchanged.
_RESERVED_TEMPLATE_KWARGS: frozenset[str] = frozenset(
    {
        "tokenize",
        "add_generation_prompt",
        "messages",
        "chat_template",
        "return_dict",
        "return_tensors",
        "tokenizer_kwargs",
    }
)


def _validate_template_kwargs(
    template_kwargs: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    """Copy template_kwargs, rejecting reserved keys with a readable error."""
    if not template_kwargs:
        return {}
    kwargs = dict(template_kwargs)
    for key in sorted(kwargs):
        if key in _RESERVED_TEMPLATE_KWARGS:
            raise VerbalizerError(
                f"template_kwargs contains reserved key {key!r}, which would conflict "
                f"with the chat-template call FuzzyAI controls; remove it. "
                f"Reserved keys: {', '.join(sorted(_RESERVED_TEMPLATE_KWARGS))}"
            )
    return kwargs


def render_input_text(
    tokenizer: SupportsChatTemplate,
    *,
    system_prompt: str | None,
    user_prompt: str,
    template_kwargs: Mapping[str, JSONValue] | None = None,
) -> str:
    """Chat-template the (system, user) pair when the tokenizer has a template.

    ``template_kwargs`` are forwarded verbatim into ``apply_chat_template`` so a
    caller can control template-specific rendering modes (e.g. a thinking
    model's reasoning-mode flag). Unknown keys are NOT validated: they are the
    template's business. Reserved keys that would collide with this helper's
    own arguments raise :class:`~fuzzyai.errors.VerbalizerError`. On the
    plain-concatenation fallback (no chat template) ``template_kwargs`` is
    ignored.

    Falls back to plain concatenation otherwise: ``"{system}\\n\\n{user}"`` when
    a system prompt exists, else just the user prompt. Deterministic.
    """
    if getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **_validate_template_kwargs(template_kwargs),
        )
    if system_prompt is not None:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt


def _as_int_ids(
    input_ids: object,
    *,
    label_descriptor: str,
    error_type: type[ScoringLabelError],
) -> list[int]:
    """Narrow an ``input_ids`` field to a list of plain ints, or raise.

    ``error_type`` is one of the scoring-label error classes so every tokenizer
    misbehaviour surfaces as the contract's error type rather than ``TypeError``.
    """
    if not isinstance(input_ids, Iterable):
        raise error_type(
            f"{label_descriptor} produced a non-iterable token id container "
            f"({type(input_ids).__name__}); a single scoring token is required"
        )
    validated: list[int] = []
    for token_id in input_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise error_type(
                f"{label_descriptor} tokenized to a non-int token id "
                f"({type(token_id).__name__}); a single scoring token is required"
            )
        validated.append(token_id)
    return validated


def resolve_exact_single_token_continuation(
    tokenizer: SupportsChatTemplate,
    *,
    prefix_text: str,
    label_text: str,
    label_descriptor: str,
    prefix_ids: list[int] | None = None,
    error_type: type[ScoringLabelError] = ScoringLabelError,
) -> int:
    """Resolve one scoring label to the single token it appends to the prefix.

    A label is a valid single-token continuation only when tokenizing
    ``prefix_text + label_text`` reproduces the prefix token sequence exactly
    and appends exactly one token::

        full_ids == prefix_ids + [full_ids[-1]]

    A net delta of one token is NOT sufficient. A tokenizer can retokenize the
    tail of the prefix while still landing on
    ``len(full_ids) == len(prefix_ids) + 1``; the final id is then not a
    next-token continuation of the original prefix and must be rejected.

    Execution-only: this knows nothing about decisions, strategies, doctrines,
    or assemblers. ``label_descriptor`` is used only in error messages, for
    example ``"scoring label 'A'"`` or ``"verbalizer 'yes' (positive)"``.

    Raises:
        error_type: on empty ids, non-int ids, a delta other than one, or a
            continuation that does not preserve the rendered prefix
            tokenization. Nothing is truncated, skipped, or re-encoded.
    """
    if prefix_ids is None:
        prefix_ids = _as_int_ids(
            tokenizer(prefix_text, add_special_tokens=False).input_ids,
            label_descriptor="rendered prefix",
            error_type=error_type,
        )
    if prefix_text and not prefix_ids:
        raise error_type(
            "the rendered prefix tokenized to an empty id sequence although the "
            "prefix text is non-empty; an exact single-token continuation cannot "
            "be verified"
        )
    full_ids = _as_int_ids(
        tokenizer(prefix_text + label_text, add_special_tokens=False).input_ids,
        label_descriptor=label_descriptor,
        error_type=error_type,
    )
    if len(full_ids) == 0:
        raise error_type(
            f"{label_descriptor} tokenized to an empty id sequence for the "
            f"concatenated prefix; a single scoring token is required"
        )
    added = len(full_ids) - len(prefix_ids)
    if added != 1:
        raise error_type(
            f"{label_descriptor} produced {added} additional token(s) after the rendered "
            f"prefix (prefix has {len(prefix_ids)} tokens, concatenated text has "
            f"{len(full_ids)}); a single scoring token is required"
        )
    if full_ids[:-1] != prefix_ids:
        raise error_type(
            f"{label_descriptor} is not an exact single-token continuation: the "
            f"concatenated text has {len(full_ids)} tokens against a {len(prefix_ids)}-token "
            f"prefix, but its first {len(prefix_ids)} token ids do not reproduce the rendered "
            f"prefix tokenization; a single scoring token is required"
        )
    return full_ids[-1]


def resolve_verbalizers(
    tokenizer: SupportsChatTemplate,
    *,
    prefix_text: str,
    positive_verbalizer: str,
    negative_verbalizer: str,
) -> VerbalizerTokens:
    """Resolve each verbalizer to exactly one continuation token after the prefix.

    Both verbalizers are resolved through
    :func:`resolve_exact_single_token_continuation`, so each must preserve the
    rendered prefix tokenization exactly and append exactly one token.

    Raises :class:`~fuzzyai.errors.VerbalizerError` for any violation: zero or
    multiple added tokens, a retokenized prefix, empty ids, non-int ids, or
    identical positive and negative token ids.
    """
    prefix_ids = _as_int_ids(
        tokenizer(prefix_text, add_special_tokens=False).input_ids,
        label_descriptor="rendered prefix",
        error_type=VerbalizerError,
    )
    positive_token_id = resolve_exact_single_token_continuation(
        tokenizer,
        prefix_text=prefix_text,
        label_text=positive_verbalizer,
        label_descriptor=f"verbalizer {positive_verbalizer!r} (positive)",
        prefix_ids=prefix_ids,
        error_type=VerbalizerError,
    )
    negative_token_id = resolve_exact_single_token_continuation(
        tokenizer,
        prefix_text=prefix_text,
        label_text=negative_verbalizer,
        label_descriptor=f"verbalizer {negative_verbalizer!r} (negative)",
        prefix_ids=prefix_ids,
        error_type=VerbalizerError,
    )
    if positive_token_id == negative_token_id:
        raise VerbalizerError(
            f"positive verbalizer {positive_verbalizer!r} and negative verbalizer "
            f"{negative_verbalizer!r} both resolve to token id {positive_token_id}; "
            f"Phase 2A requires a single DISTINCT scoring token per verbalizer"
        )
    return VerbalizerTokens(
        positive_verbalizer=positive_verbalizer,
        negative_verbalizer=negative_verbalizer,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
