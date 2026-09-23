"""Tests for the pure verbalizer helpers (no torch, no network, no GPU)."""

import pytest

from probvenance.backends.verbalizers import (
    SupportsChatTemplate,
    TokenizedText,
    VerbalizerTokens,
    render_input_text,
    resolve_exact_single_token_continuation,
    resolve_verbalizers,
)
from probvenance.errors import ScoringLabelError, VerbalizerError

# Independent copy of the contract's reserved keys: the test must fail if the
# implementation's reserved set silently shrinks.
RESERVED_KEYS = (
    "tokenize",
    "add_generation_prompt",
    "messages",
    "chat_template",
    "return_dict",
    "return_tensors",
    "tokenizer_kwargs",
)


class FakeTokenized:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids: object = ids


class FakeTokenizer:
    """Word-map tokenizer: each whitespace-separated word is one token.

    Unknown words get deterministic ids via a growing vocabulary. This makes
    single-word verbalizers single-token and multi-word verbalizers
    multi-token, and it exhibits a BPE-like boundary effect: a verbalizer that
    merges with the prefix's last word (no whitespace) changes tokenization.
    """

    def __init__(self, chat_template: str | None = None) -> None:
        self.chat_template = chat_template
        self._vocab: dict[str, int] = {}
        self.calls: list[tuple[str, bool]] = []
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def _tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split(" "):
            if not word:
                continue  # real tokenizers never emit a token for pure whitespace
            if word not in self._vocab:
                self._vocab[word] = 1000 + len(self._vocab)
            ids.append(self._vocab[word])
        return ids

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> FakeTokenized:
        self.calls.append((text, add_special_tokens))
        return FakeTokenized(self._tokenize(text))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        assert tokenize is False
        self.template_calls.append((messages, dict(kwargs)))
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)


def make_tokenizer() -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    # Warm the vocabulary so ids are stable across tests that share nothing.
    tokenizer("Answer yes or no.")
    return tokenizer


def test_single_token_positive_and_negative_resolve() -> None:
    tokenizer = make_tokenizer()
    prefix = "Answer with one word: "
    resolved = resolve_verbalizers(
        tokenizer,
        prefix_text=prefix,
        positive_verbalizer="yes",
        negative_verbalizer="no",
    )
    assert isinstance(resolved, VerbalizerTokens)
    assert resolved.positive_verbalizer == "yes"
    assert resolved.negative_verbalizer == "no"
    assert resolved.positive_token_id != resolved.negative_token_id
    # The scoring token is the LAST id of the concatenated encoding.
    expected_pos = tokenizer(prefix + "yes").input_ids
    assert isinstance(expected_pos, list)
    assert resolved.positive_token_id == expected_pos[-1]
    expected_neg = tokenizer(prefix + "no").input_ids
    assert isinstance(expected_neg, list)
    assert resolved.negative_token_id == expected_neg[-1]


def test_multi_token_positive_raises() -> None:
    tokenizer = make_tokenizer()
    with pytest.raises(VerbalizerError, match="positive"):
        resolve_verbalizers(
            tokenizer,
            prefix_text="Answer: ",
            positive_verbalizer="absolutely yes",
            negative_verbalizer="no",
        )


def test_multi_token_negative_raises() -> None:
    tokenizer = make_tokenizer()
    with pytest.raises(VerbalizerError, match="negative"):
        resolve_verbalizers(
            tokenizer,
            prefix_text="Answer: ",
            positive_verbalizer="yes",
            negative_verbalizer="definitely not",
        )


def test_identical_token_for_both_raises() -> None:
    tokenizer = make_tokenizer()
    # Distinct strings that tokenize to the same single token after the prefix:
    # the word map is case-insensitive-ish here via identical words.
    with pytest.raises(VerbalizerError, match="DISTINCT"):
        resolve_verbalizers(
            tokenizer,
            prefix_text="Answer: ",
            positive_verbalizer="yes",
            negative_verbalizer="yes",
        )


def test_error_message_names_verbalizer_and_token_count() -> None:
    tokenizer = make_tokenizer()
    with pytest.raises(VerbalizerError) as exc_info:
        resolve_verbalizers(
            tokenizer,
            prefix_text="Answer: ",
            positive_verbalizer="absolutely yes",
            negative_verbalizer="no",
        )
    message = str(exc_info.value)
    assert "absolutely yes" in message
    assert "2" in message  # how many tokens it produced
    assert "single scoring token" in message


def test_render_input_text_uses_chat_template_when_set() -> None:
    tokenizer = FakeTokenizer(chat_template="{{ messages }}")
    rendered = render_input_text(tokenizer, system_prompt="Be terse.", user_prompt="Is it so?")
    assert rendered == "<|system|>Be terse.<|user|>Is it so?<|assistant|>"


def test_render_input_text_omits_system_message_when_none() -> None:
    tokenizer = FakeTokenizer(chat_template="tpl")
    rendered = render_input_text(tokenizer, system_prompt=None, user_prompt="Is it so?")
    assert rendered == "<|user|>Is it so?<|assistant|>"


def test_render_input_text_concatenated_fallback_without_template() -> None:
    tokenizer = FakeTokenizer(chat_template=None)
    rendered = render_input_text(tokenizer, system_prompt="Be terse.", user_prompt="Is it so?")
    assert rendered == "Be terse.\n\nIs it so?"
    assert render_input_text(tokenizer, system_prompt=None, user_prompt="Is it so?") == (
        "Is it so?"
    )


def test_render_input_text_forwards_template_kwargs() -> None:
    tokenizer = FakeTokenizer(chat_template="{{ messages }}")
    rendered = render_input_text(
        tokenizer,
        system_prompt="Be terse.",
        user_prompt="Is it so?",
        template_kwargs={"enable_thinking": False, "depth": 2},
    )
    assert rendered == "<|system|>Be terse.<|user|>Is it so?<|assistant|>"
    assert tokenizer.template_calls[-1][1] == {"enable_thinking": False, "depth": 2}


@pytest.mark.parametrize("key", RESERVED_KEYS)
def test_render_input_text_rejects_reserved_template_kwargs(key: str) -> None:
    tokenizer = FakeTokenizer(chat_template="{{ messages }}")
    with pytest.raises(VerbalizerError, match=key) as exc_info:
        render_input_text(
            tokenizer,
            system_prompt="s",
            user_prompt="u",
            template_kwargs={key: "x"},
        )
    assert "template_kwargs" in str(exc_info.value)


def test_render_input_text_passes_through_unknown_template_kwargs() -> None:
    tokenizer = FakeTokenizer(chat_template="{{ messages }}")
    rendered = render_input_text(
        tokenizer,
        system_prompt="s",
        user_prompt="u",
        template_kwargs={"some_qwen_specific_flag": True},
    )
    assert rendered == "<|system|>s<|user|>u<|assistant|>"
    assert tokenizer.template_calls[-1][1] == {"some_qwen_specific_flag": True}


def test_render_input_text_ignores_template_kwargs_on_fallback_path() -> None:
    tokenizer = FakeTokenizer(chat_template=None)
    rendered = render_input_text(
        tokenizer,
        system_prompt="Be terse.",
        user_prompt="Is it so?",
        template_kwargs={"enable_thinking": False},
    )
    assert rendered == "Be terse.\n\nIs it so?"
    assert tokenizer.template_calls == []


def test_boundary_effect_delta_from_concatenated_prefix() -> None:
    """The delta must come from tokenizer(prefix + verbalizer), not separate encodes.

    With a word-map tokenizer, prefix "Answer: yes" (no trailing space) makes
    the verbalizer "yes" MERGE into the prefix's last word, changing the token
    count by 0 - which must raise. A separate-encode implementation would
    wrongly see exactly 1 added token.
    """
    tokenizer = make_tokenizer()
    with pytest.raises(VerbalizerError):
        resolve_verbalizers(
            tokenizer,
            prefix_text="Answer: yes",
            positive_verbalizer="yes",
            negative_verbalizer="no",
        )
    # And the recorded calls prove the concatenated text was encoded, never
    # the bare verbalizer alone.
    encoded_texts = [text for text, _ in tokenizer.calls]
    assert "Answer: yesyes" in encoded_texts
    assert "yes" not in encoded_texts
    # A separate-encode implementation would see "yes" alone as exactly one
    # token and wrongly accept it; only the concatenated check catches this.


def test_zero_added_tokens_raises() -> None:
    tokenizer = make_tokenizer()
    with pytest.raises(VerbalizerError, match="0 additional"):
        resolve_verbalizers(
            tokenizer,
            prefix_text="Answer: yes",
            positive_verbalizer="yes",
            negative_verbalizer="no",
        )


def test_non_int_token_id_raises() -> None:
    class BadIds:
        input_ids: object = ["not", "ints"]

    class BadTokenizer:
        chat_template: str | None = None

        def __call__(self, text: str, *, add_special_tokens: bool = True) -> BadIds:
            return BadIds()

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = True,
            add_generation_prompt: bool = False,
            **kwargs: object,
        ) -> str:
            return ""

    with pytest.raises(VerbalizerError, match="non-int"):
        resolve_verbalizers(
            BadTokenizer(),
            prefix_text="Answer: ",
            positive_verbalizer="yes",
            negative_verbalizer="no",
        )


def test_empty_ids_raises() -> None:
    class EmptyIds:
        input_ids: object = []

    class EmptyTokenizer:
        chat_template: str | None = None

        def __call__(self, text: str, *, add_special_tokens: bool = True) -> EmptyIds:
            return EmptyIds()

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = True,
            add_generation_prompt: bool = False,
            **kwargs: object,
        ) -> str:
            return ""

    with pytest.raises(VerbalizerError, match="empty"):
        resolve_verbalizers(
            EmptyTokenizer(),
            prefix_text="Answer: ",
            positive_verbalizer="yes",
            negative_verbalizer="no",
        )


def test_protocols_are_structural() -> None:
    """The fakes satisfy the protocols without inheritance (duck typing)."""
    tokenizer: SupportsChatTemplate = make_tokenizer()
    tokenized: TokenizedText = tokenizer("x")
    assert tokenized.input_ids is not None


class ScriptedTokenizer:
    chat_template: str | None = None

    def __init__(self, sequences: dict[str, list[int]]) -> None:
        self._sequences = sequences
        self.calls: list[str] = []

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> FakeTokenized:
        self.calls.append(text)
        return FakeTokenized(list(self._sequences.get(text, [])))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        return ""


RETOKENIZED_PREFIX = [10, 20, 30]
RETOKENIZED_FULL = [10, 99, 40, 50]


def test_exact_continuation_accepts_preserved_prefix() -> None:
    tokenizer = ScriptedTokenizer({"P": [1, 2, 3], "Px": [1, 2, 3, 4]})
    token_id = resolve_exact_single_token_continuation(
        tokenizer, prefix_text="P", label_text="x", label_descriptor="scoring label 'x'"
    )
    assert token_id == 4


def test_exact_continuation_rejects_zero_delta() -> None:
    tokenizer = ScriptedTokenizer({"P": [1, 2, 3], "Px": [1, 2, 3]})
    with pytest.raises(ScoringLabelError, match="0 additional"):
        resolve_exact_single_token_continuation(
            tokenizer, prefix_text="P", label_text="x", label_descriptor="scoring label 'x'"
        )


def test_exact_continuation_rejects_empty_prefix_for_non_empty_text() -> None:
    class EmptyPrefixTokenizer:
        chat_template: str | None = None

        def __call__(self, text: str, *, add_special_tokens: bool = True) -> FakeTokenized:
            return FakeTokenized([])

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = True,
            add_generation_prompt: bool = False,
            **kwargs: object,
        ) -> str:
            return ""

    with pytest.raises(ScoringLabelError, match="empty id sequence"):
        resolve_exact_single_token_continuation(
            EmptyPrefixTokenizer(),
            prefix_text="P",
            label_text="x",
            label_descriptor="scoring label 'x'",
        )


def test_exact_continuation_rejects_multi_token_delta() -> None:
    tokenizer = ScriptedTokenizer({"P": [1, 2, 3], "Px": [1, 2, 3, 4, 5]})
    with pytest.raises(ScoringLabelError, match="2 additional"):
        resolve_exact_single_token_continuation(
            tokenizer, prefix_text="P", label_text="x", label_descriptor="scoring label 'x'"
        )


def test_exact_continuation_rejects_retokenized_prefix_when_net_delta_is_one() -> None:
    tokenizer = ScriptedTokenizer({"P": RETOKENIZED_PREFIX, "Px": RETOKENIZED_FULL})
    full_ids = tokenizer("Px").input_ids
    prefix_ids = tokenizer("P").input_ids
    assert isinstance(full_ids, list)
    assert isinstance(prefix_ids, list)
    # The trap this test exists for: the net delta is exactly +1, which the
    # previous length-only check accepted, yet the prefix was retokenized.
    assert len(full_ids) - len(prefix_ids) == 1
    assert full_ids[:-1] != prefix_ids
    with pytest.raises(ScoringLabelError, match="exact single-token continuation"):
        resolve_exact_single_token_continuation(
            tokenizer, prefix_text="P", label_text="x", label_descriptor="scoring label 'x'"
        )


def test_exact_continuation_uses_the_requested_error_type() -> None:
    tokenizer = ScriptedTokenizer({"P": RETOKENIZED_PREFIX, "Pyes": RETOKENIZED_FULL})
    with pytest.raises(VerbalizerError, match="exact single-token continuation"):
        resolve_exact_single_token_continuation(
            tokenizer,
            prefix_text="P",
            label_text="yes",
            label_descriptor="verbalizer 'yes' (positive)",
            error_type=VerbalizerError,
        )


def test_resolve_verbalizers_rejects_a_retokenized_prefix() -> None:
    tokenizer = ScriptedTokenizer(
        {"P": RETOKENIZED_PREFIX, "Pyes": RETOKENIZED_FULL, "Pno": [10, 20, 30, 7]}
    )
    with pytest.raises(VerbalizerError, match="exact single-token continuation"):
        resolve_verbalizers(
            tokenizer, prefix_text="P", positive_verbalizer="yes", negative_verbalizer="no"
        )
