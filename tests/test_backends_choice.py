"""Tests for the categorical Choice path of TransformersBackend.

These tests never download anything: the model/tokenizer objects are fakes
injected by monkeypatching the module's ``AutoTokenizer``/``AutoModelForCausalLM``
loaders. In the default environment (no torch) the whole module SKIPs, which is
expected and acceptable.
"""

import re

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from fuzzyai.backends.transformers import TransformersBackend
from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.errors import ScoringLabelError, UnsupportedCapabilityError
from fuzzyai.plans import (
    CandidateLabelMapping,
    EvidenceKind,
    InferencePlan,
    ScoringStrategy,
)


class FakeTokenized:
    """Mimics a real tokenizer output: ``input_ids`` is a plain list of ints."""

    def __init__(self, ids: list[int]) -> None:
        self.input_ids: object = ids


_PIECE_PATTERN = re.compile(r"[A-Za-z0-9]+|\s+|[^\sA-Za-z0-9]")


class FakeTokenizer:
    """Sub-word-ish tokenizer: words, whitespace runs, and single punctuation
    marks are separate tokens.

    Like the fake in ``tests/test_backends_transformers.py`` it reproduces the
    real BPE boundary behaviour the label contract validates: appending a single
    letter after a prefix ending in punctuation adds exactly one new token,
    while appending letters directly onto a word merges into the word piece.
    """

    def __init__(self, chat_template: str | None = None) -> None:
        self.chat_template = chat_template
        self.name_or_path = "fake/model"
        self.init_kwargs: dict[str, object] = {"_commit_hash": "def456"}
        self._vocab: dict[str, int] = {}
        self.calls: list[tuple[str, bool]] = []

    def decode(self, ids: list[int], **kwargs: object) -> str:
        pieces = []
        for token_id in ids:
            for piece, piece_id in self._vocab.items():
                if piece_id == token_id:
                    pieces.append(piece)
                    break
        return "".join(pieces)

    def _piece_key(self, piece: str) -> str:
        return piece

    def _tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _PIECE_PATTERN.findall(text):
            key = self._piece_key(piece)
            if key not in self._vocab:
                self._vocab[key] = 1000 + len(self._vocab)
            ids.append(self._vocab[key])
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
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|end|>")
        return "".join(parts)


class CollidingTokenizer(FakeTokenizer):
    """FakeTokenizer variant where lowercase ``a`` shares the token id of ``A``.

    Models a real-vocabulary alias (two surface forms, one token id) so two
    distinct labels can resolve to the SAME token id and exercise the
    pairwise-distinctness rejection.
    """

    def _piece_key(self, piece: str) -> str:
        return "A" if piece == "a" else piece


class FakeLogits:
    """Deterministic fake ``logits`` tensor that ONLY allows last-position reads.

    The final-position read returns a REAL 1-D float32 ``torch.Tensor`` so the
    backend's ``.to()``, ``torch.logsumexp`` and ``torch.argmax`` calls run for
    real; any other index still raises.
    """

    def __init__(self, row: list[float], seq_len: int) -> None:
        self._row = row
        self._seq_len = seq_len

    def __getitem__(self, index: tuple[int, int, slice]) -> torch.Tensor:
        i, j, k = index
        assert (i, j, k) == (0, -1, slice(None)), "backend must read ONLY logits[0, -1, :]"
        return torch.tensor(self._row, dtype=torch.float32)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, self._seq_len, len(self._row))


class FakeOutput:
    def __init__(self, row: list[float], seq_len: int) -> None:
        self.logits = FakeLogits(row, seq_len)


class FakeModel:
    """Deterministic fake causal LM: every vocab piece with id n gets logit -n.

    The head covers the tokenizer's vocabulary (shared dict, so label ids
    assigned during resolution are present too), like a real LM. The row is
    VOCAB-SIZED and wider than any id the tokenizer can assign.
    """

    _VOCAB_SIZE = 4096

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.config = type("Config", (), {"_commit_hash": "abc123"})()
        self._vocab = vocab if vocab is not None else {}
        self.eval_called = False
        self.forward_calls = 0
        self.generate_calls = 0
        self.last_batch_shape: tuple[int, ...] | None = None

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def to(self, device: str) -> "FakeModel":
        return self

    def generate(self, *args: object, **kwargs: object) -> FakeOutput:
        self.generate_calls += 1
        raise AssertionError(
            "TransformersBackend.execute() called model.generate(); the backend "
            "must score with a single forward pass and never generate text"
        )

    def __call__(self, *, input_ids: torch.Tensor) -> FakeOutput:
        self.forward_calls += 1
        self.last_batch_shape = tuple(input_ids.shape)
        row = [0.0] * self._VOCAB_SIZE
        for token_id in self._vocab.values():
            row[token_id] = -float(token_id)
        return FakeOutput(row, seq_len=int(input_ids.shape[1]))


def make_choice_plan(targets: tuple[str, ...]) -> InferencePlan:
    return InferencePlan(
        decision_fingerprint="dfp",
        strategy=ScoringStrategy.CATEGORICAL_TOKEN_LOGITS,
        prompt="Which bucket?",
        system_prompt="Be terse.",
        targets=targets,
        label_scheme_id="categorical-labels-v1",
        candidate_mapping=tuple(
            CandidateLabelMapping(
                candidate_index=i,
                candidate_name=f"choice-{label}",
                candidate_description=None,
                scoring_label=label,
            )
            for i, label in enumerate(targets)
        ),
    )


def make_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: FakeTokenizer | None = None,
) -> TransformersBackend:
    tokenizer = tokenizer or FakeTokenizer()
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "fuzzyai.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "fuzzyai.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    return TransformersBackend("fake/model", device="cpu")


def test_capabilities_declare_exactly_binary_and_categorical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = make_backend(monkeypatch)
    caps = backend.capabilities
    assert caps == BackendCapabilities(binary_token_logits=True, categorical_token_logits=True)
    assert not caps.token_logprobs
    assert not caps.batching
    assert not caps.prefix_cache
    assert not caps.constrained_decoding


def test_three_labels_execute_in_plan_order(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(monkeypatch)
    plan = make_choice_plan(("A", "B", "C"))
    evidence = backend.execute(plan)

    assert evidence.kind == EvidenceKind.LOGITS
    assert evidence.labels == ("A", "B", "C")
    assert evidence.plan_fingerprint == plan.fingerprint

    resolved = evidence.metadata["resolved_target_token_ids"]
    assert resolved == [["A", resolved[0][1]], ["B", resolved[1][1]], ["C", resolved[2][1]]]
    ids = [token_id for _, token_id in resolved]
    assert len(set(ids)) == 3

    # FakeModel assigns logit -token_id: values must be exactly those cells,
    # in plan.targets order (never sorted by id or by logit).
    assert evidence.values == tuple(-float(token_id) for token_id in ids)

    model = backend._model
    assert model.forward_calls == 1
    assert model.generate_calls == 0
    assert model.last_batch_shape is not None
    assert model.last_batch_shape[0] == 1


def test_five_labels_execute_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(monkeypatch)
    plan = make_choice_plan(("A", "B", "C", "D", "E"))
    evidence = backend.execute(plan)

    assert evidence.labels == ("A", "B", "C", "D", "E")
    resolved = evidence.metadata["resolved_target_token_ids"]
    assert [label for label, _ in resolved] == ["A", "B", "C", "D", "E"]
    ids = [token_id for _, token_id in resolved]
    assert len(set(ids)) == 5
    assert evidence.values == tuple(-float(token_id) for token_id in ids)
    assert backend._model.forward_calls == 1


def test_multi_token_label_raises_before_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(monkeypatch)
    # "A B" tokenizes as three added pieces (space, A, space, B) after the
    # prefix, so it can never be a single scoring token.
    plan = make_choice_plan(("A", "B C", "D"))
    with pytest.raises(ScoringLabelError, match="B C"):
        backend.execute(plan)
    assert backend._model.forward_calls == 0


def test_colliding_labels_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = CollidingTokenizer()
    backend = make_backend(monkeypatch, tokenizer=tokenizer)
    # "a" shares the token id of "A" in this fake vocabulary.
    plan = make_choice_plan(("A", "a", "B"))
    with pytest.raises(ScoringLabelError, match="token id"):
        backend.execute(plan)
    assert backend._model.forward_calls == 0


def test_wrong_strategy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(monkeypatch)
    plan = InferencePlan(
        decision_fingerprint="dfp",
        strategy=ScoringStrategy.TOKEN_LOGPROBS,
        prompt="Is it so?",
    )
    with pytest.raises(UnsupportedCapabilityError):
        backend.execute(plan)
    assert backend._model.forward_calls == 0


def test_metadata_carries_full_vocabulary_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = make_backend(monkeypatch)
    plan = make_choice_plan(("A", "B", "C"))
    evidence = backend.execute(plan)
    metadata = evidence.metadata

    resolved = metadata["resolved_target_token_ids"]
    assert [label for label, _ in resolved] == ["A", "B", "C"]
    assert all(isinstance(token_id, int) for _, token_id in resolved)

    # Reconstruct the fake row exactly as FakeModel builds it.
    vocab = backend._tokenizer._vocab
    row = [0.0] * backend._model._VOCAB_SIZE
    for token_id in vocab.values():
        row[token_id] = -float(token_id)
    row_tensor = torch.tensor(row, dtype=torch.float32)

    assert isinstance(metadata["vocab_logsumexp"], float)
    assert metadata["vocab_logsumexp"] == pytest.approx(torch.logsumexp(row_tensor, dim=-1).item())
    assert isinstance(metadata["top_token_id"], int)
    assert metadata["top_token_id"] == int(torch.argmax(row_tensor).item())
    assert isinstance(metadata["top_token_logit"], float)
    assert metadata["top_token_logit"] == pytest.approx(max(row))
    top_probability = metadata["top_token_probability"]
    assert isinstance(top_probability, float)
    assert 0.0 < top_probability <= 1.0
    assert top_probability == pytest.approx(
        float(torch.softmax(row_tensor, dim=-1)[metadata["top_token_id"]].item()), rel=1e-6
    )
    assert metadata["top_token_text"] is None or isinstance(metadata["top_token_text"], str)
    assert metadata["rendered_input"] == "Be terse.\n\nWhich bucket?"
    assert metadata["rendering_config"] == {}
    assert metadata["backend_version"] == "1"
