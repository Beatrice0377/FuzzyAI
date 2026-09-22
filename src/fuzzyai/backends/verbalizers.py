"""Pure verbalizer helpers for token-logit backends.

These helpers are deliberately free of any torch/transformers import so they
are unit-testable with a fake tokenizer, no network, and no GPU. They encode
the Phase 2A verbalizer contract:

- A verbalizer is valid only if appending it to the REAL rendered prefix adds
  EXACTLY ONE token. The check is performed against the concatenated text
  (``tokenizer(prefix + verbalizer)``), never as
  ``tokenizer(prefix) + tokenizer(verbalizer)``, because BPE boundary effects
  are exactly what is being validated.
- The positive and negative verbalizers must resolve to DISTINCT token ids.
- Any violation raises :class:`~fuzzyai.errors.VerbalizerError`. There is no
  silent fallback, no truncation, and no multi-token logit summing.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from fuzzyai.errors import VerbalizerError
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


def _as_id_list(input_ids: object) -> list[object]:
    """Narrow the protocol's ``object`` field to an iterable for ``list()``.

    Raises :class:`VerbalizerError` (instead of ``TypeError``) when the
    tokenizer produced something that is not iterable of ints, so every
    tokenizer misbehaviour surfaces as the contract's error type.
    """
    if not isinstance(input_ids, Iterable):
        raise VerbalizerError(
            f"tokenizer input_ids is not iterable, got {type(input_ids).__name__}; "
            f"Phase 2A requires a single scoring token per verbalizer"
        )
    return list(input_ids)


def _resolve_single_token(
    tokenizer: SupportsChatTemplate,
    *,
    prefix_text: str,
    verbalizer: str,
    role: str,
    prefix_ids: list[object],
) -> int:
    """Resolve one verbalizer to exactly one continuation token after ``prefix_text``.

    The token count delta is computed against the CONCATENATED text
    (``prefix_text + verbalizer``), never as separate encodings, so BPE
    boundary effects are part of the validation.
    """
    full_ids = _as_id_list(tokenizer(prefix_text + verbalizer, add_special_tokens=False).input_ids)
    if len(full_ids) == 0:
        raise VerbalizerError(
            f"verbalizer {verbalizer!r} ({role}) tokenized to an empty id sequence for the "
            f"concatenated prefix; Phase 2A requires a single scoring token"
        )
    validated_ids: list[int] = []
    for token_id in full_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise VerbalizerError(
                f"verbalizer {verbalizer!r} ({role}) tokenized to a non-int token id "
                f"({type(token_id).__name__}); Phase 2A requires a single scoring token"
            )
        validated_ids.append(token_id)
    added = len(validated_ids) - len(prefix_ids)
    if added != 1:
        raise VerbalizerError(
            f"verbalizer {verbalizer!r} ({role}) produced {added} additional token(s) after the "
            f"rendered prefix (prefix has {len(prefix_ids)} tokens, concatenated text has "
            f"{len(validated_ids)}); Phase 2A requires a single scoring token"
        )
    return validated_ids[-1]


def resolve_verbalizers(
    tokenizer: SupportsChatTemplate,
    *,
    prefix_text: str,
    positive_verbalizer: str,
    negative_verbalizer: str,
) -> VerbalizerTokens:
    """Resolve each verbalizer to exactly one continuation token after the prefix.

    Raises :class:`~fuzzyai.errors.VerbalizerError` for any violation: zero or
    multiple added tokens, empty ids, non-int ids, or identical positive and
    negative token ids.
    """
    prefix_ids = _as_id_list(tokenizer(prefix_text, add_special_tokens=False).input_ids)
    positive_token_id = _resolve_single_token(
        tokenizer,
        prefix_text=prefix_text,
        verbalizer=positive_verbalizer,
        role="positive",
        prefix_ids=prefix_ids,
    )
    negative_token_id = _resolve_single_token(
        tokenizer,
        prefix_text=prefix_text,
        verbalizer=negative_verbalizer,
        role="negative",
        prefix_ids=prefix_ids,
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
