"""Tests for TransformersBackend (skipped unless torch+transformers installed).

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

from probvenance.assembler import BINARY_ASSEMBLER_ID, BINARY_ASSEMBLER_VERSION
from probvenance.backends.transformers import TRANSFORMERS_BACKEND_VERSION, TransformersBackend
from probvenance.capabilities import BackendCapabilities
from probvenance.compiler import BINARY_COMPILER_ID, BINARY_COMPILER_VERSION
from probvenance.diagnostics import diagnose_bool_evidence
from probvenance.doctrine import BINARY_DOCTRINE_ID, BINARY_DOCTRINE_VERSION
from probvenance.errors import UnsupportedCapabilityError, VerbalizerError
from probvenance.fingerprint import JSONValue
from probvenance.plans import (
    DECISION_FAMILY_BOOL,
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

    Like the fixed fake in ``tests/test_verbalizers.py`` it never emits a token
    for pure whitespace on its own, and unlike a naive whole-word map it shows
    the real BPE boundary behaviour the verbalizer contract validates: appending
    ``"yes"`` after a prefix ending in ``"?"`` adds exactly one new token (the
    punctuation does not merge with the following letters), while appending
    letters directly onto a word keeps the piece count stable only when the
    word genuinely merges.
    """

    def __init__(
        self,
        chat_template: str | None = None,
        *,
        decode_raises: bool = False,
    ) -> None:
        self.chat_template = chat_template
        self.decode_raises = decode_raises
        self.name_or_path = "fake/model"
        self.init_kwargs: dict[str, object] = {"_commit_hash": "def456"}
        self._vocab: dict[str, int] = {}
        self.calls: list[tuple[str, bool]] = []
        self.template_calls: list[dict[str, object]] = []

    def decode(self, ids: list[int], **kwargs: object) -> str:
        if self.decode_raises:
            raise RuntimeError("decode is broken in this fake")
        pieces = []
        for token_id in ids:
            for piece, piece_id in self._vocab.items():
                if piece_id == token_id:
                    pieces.append(piece)
                    break
        return "".join(pieces)

    def _tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _PIECE_PATTERN.findall(text):
            if piece not in self._vocab:
                self._vocab[piece] = 1000 + len(self._vocab)
            ids.append(self._vocab[piece])
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
        self.template_calls.append(dict(kwargs))
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)


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

    The head covers the tokenizer's vocabulary (shared dict, so verbalizer ids
    assigned during resolution are present too), like a real LM.
    """

    _VOCAB_SIZE = 4096

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.config = type("Config", (), {"_commit_hash": "abc123"})()
        self._vocab = vocab if vocab is not None else {}
        self.to_calls: list[str] = []
        self.eval_called = False
        self.forward_calls = 0
        self.generate_calls = 0
        self.last_batch_shape: tuple[int, ...] | None = None

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def to(self, device: str) -> "FakeModel":
        self.to_calls.append(device)
        return self

    def generate(self, *args: object, **kwargs: object) -> "FakeOutput":
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


class UniformLogitsModel(FakeModel):
    """FakeModel variant: EVERY vocabulary position gets logit 0.0.

    The next-token distribution is then exactly uniform over the 4096-wide
    vocabulary, so the two verbalizer tokens hold a hand-computable share of
    the full-vocabulary mass: 2 / 4096.
    """

    def __call__(self, *, input_ids: torch.Tensor) -> FakeOutput:
        self.forward_calls += 1
        self.last_batch_shape = tuple(input_ids.shape)
        return FakeOutput([0.0] * self._VOCAB_SIZE, seq_len=int(input_ids.shape[1]))


def make_plan() -> InferencePlan:
    return InferencePlan(
        decision_fingerprint="dfp",
        strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
        prompt="Is it so?",
        decision_family=DECISION_FAMILY_BOOL,
        system_prompt="Be terse.",
        positive_verbalizer="yes",
        negative_verbalizer="no",
        doctrine_id=BINARY_DOCTRINE_ID,
        doctrine_version=BINARY_DOCTRINE_VERSION,
        compiler_id=BINARY_COMPILER_ID,
        compiler_version=BINARY_COMPILER_VERSION,
        assembler_id=BINARY_ASSEMBLER_ID,
        assembler_version=BINARY_ASSEMBLER_VERSION,
    )


@pytest.fixture()
def backend(monkeypatch: pytest.MonkeyPatch) -> TransformersBackend:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    return TransformersBackend("fake/model", device="cpu")


def test_capabilities_are_honest(backend: TransformersBackend) -> None:
    assert backend.capabilities == BackendCapabilities(
        binary_token_logits=True, categorical_token_logits=True
    )
    caps = backend.capabilities
    assert not caps.token_logprobs
    assert not caps.batching
    assert not caps.prefix_cache
    assert not caps.constrained_decoding


def test_execute_reads_last_position_only(backend: TransformersBackend) -> None:
    evidence = backend.execute(make_plan())
    assert evidence.kind == EvidenceKind.LOGITS
    assert evidence.labels == ("false", "true")
    assert evidence.plan_fingerprint == make_plan().fingerprint

    positive_id = evidence.metadata["positive_token_id"]
    negative_id = evidence.metadata["negative_token_id"]
    assert positive_id != negative_id
    # FakeModel assigns logit -token_id: values must be exactly those two cells.
    assert evidence.values == (-float(negative_id), -float(positive_id))

    assert evidence.metadata["model"] == "fake/model"
    assert evidence.metadata["model_revision"] == "abc123"
    assert evidence.metadata["device"] == "cpu"
    assert evidence.metadata["rendered_input"] == "Be terse.\n\nIs it so?"
    reencoded = backend._tokenizer(evidence.metadata["rendered_input"], add_special_tokens=False)
    assert evidence.metadata["input_token_count"] == len(reencoded.input_ids)


def test_execute_sends_single_batch_to_model(backend: TransformersBackend) -> None:
    backend.execute(make_plan())
    model = backend._model
    assert model.eval_called
    assert model.forward_calls == 1
    assert model.last_batch_shape is not None
    assert model.last_batch_shape[0] == 1  # batch of exactly one sequence


def test_execute_never_calls_generate(backend: TransformersBackend) -> None:
    evidence = backend.execute(make_plan())
    assert evidence.kind == EvidenceKind.LOGITS
    model = backend._model
    assert model.generate_calls == 0, (
        "TransformersBackend.execute() must never call model.generate(); "
        f"it was called {model.generate_calls} time(s)"
    )


def test_execute_moves_model_to_requested_device(backend: TransformersBackend) -> None:
    backend.execute(make_plan())
    # The model was moved to the requested device at construction.
    assert backend._model.to_calls == ["cpu"]


def test_single_forward_per_evaluation(backend: TransformersBackend) -> None:
    backend.execute(make_plan())
    model = backend._model
    assert model.forward_calls == 1, (
        "the full-vocabulary statistics must come from the SAME single forward pass"
    )
    assert model.generate_calls == 0


def test_metadata_carries_full_vocabulary_statistics(backend: TransformersBackend) -> None:
    evidence = backend.execute(make_plan())
    metadata = evidence.metadata

    # Reconstruct the fake row exactly as FakeModel builds it: every vocab piece
    # with id n gets logit -n, everything else 0.0.
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
    assert metadata["top_token_text"] is None or isinstance(metadata["top_token_text"], str)
    assert metadata["backend_version"] == str(TRANSFORMERS_BACKEND_VERSION)
    assert metadata["backend_version"] == "1"
    assert metadata["tokenizer"] == "fake/model"
    assert metadata["tokenizer_revision"] == "def456"
    assert isinstance(metadata["runtime_version"], str)
    assert metadata["dtype"] == "float32"
    assert metadata["rendering_config"] == {}


def test_top_token_text_decode_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer(decode_raises=True)
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    backend = TransformersBackend("fake/model", device="cpu")
    evidence = backend.execute(make_plan())
    assert evidence.metadata["top_token_text"] is None
    assert evidence.kind == EvidenceKind.LOGITS
    assert model.forward_calls == 1


def test_rendering_config_reflects_constructor_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _ = make_patched_backend(
        monkeypatch,
        chat_template="yes",
        chat_template_kwargs={"enable_thinking": False},
    )
    evidence = backend.execute(make_plan())
    assert evidence.metadata["rendering_config"] == {"enable_thinking": False}


def test_diagnostics_metadata_drives_verbalizer_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Uniform logits over the whole vocabulary: every one of the 4096 tokens is
    # equally likely, so the two verbalizer tokens hold exactly 2/4096 of the
    # full-vocabulary probability mass. Hand-computed, no diagnostics code involved.
    tokenizer = FakeTokenizer()
    model = UniformLogitsModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    backend = TransformersBackend("fake/model", device="cpu")
    evidence = backend.execute(make_plan())

    diagnostics = diagnose_bool_evidence(evidence)
    assert diagnostics.verbalizer_mass == pytest.approx(2 / 4096, rel=1e-6)
    assert diagnostics.top_token_probability == pytest.approx(1 / 4096, rel=1e-6)
    assert diagnostics.positive_token_probability == pytest.approx(1 / 4096, rel=1e-6)
    assert diagnostics.negative_token_probability == pytest.approx(1 / 4096, rel=1e-6)
    assert diagnostics.top_token_id == evidence.metadata["top_token_id"]


def test_tokenizer_revision_falls_back_to_requested_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs = {}
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    backend = TransformersBackend("fake/model", device="cpu", revision="v2.0")
    evidence = backend.execute(make_plan())
    assert evidence.metadata["tokenizer_revision"] == "v2.0"

    plain_backend = TransformersBackend("fake/model", device="cpu")
    plain_evidence = plain_backend.execute(make_plan())
    assert plain_evidence.metadata["tokenizer_revision"] is None


def test_execute_uses_chat_template_when_tokenizer_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer(chat_template="yes")
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    backend = TransformersBackend("fake/model", device="cpu")
    evidence = backend.execute(make_plan())
    assert evidence.metadata["rendered_input"].startswith("<|system|>")
    # Chat-template text must NOT get special tokens added again.
    assert tokenizer.calls[-1] == (evidence.metadata["rendered_input"], False)


def test_execute_rejects_non_binary_strategy(backend: TransformersBackend) -> None:
    plan = InferencePlan(
        decision_fingerprint="dfp",
        strategy=ScoringStrategy.TOKEN_LOGPROBS,
        prompt="Is it so?",
        decision_family=DECISION_FAMILY_BOOL,
    )
    with pytest.raises(UnsupportedCapabilityError):
        backend.execute(plan)


def test_execute_rejects_multi_token_verbalizer_before_forward(
    backend: TransformersBackend,
) -> None:
    plan = InferencePlan(
        decision_fingerprint="dfp",
        strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
        prompt="Is it so?",
        decision_family=DECISION_FAMILY_BOOL,
        system_prompt="Be terse.",
        positive_verbalizer="absolutely yes",
        negative_verbalizer="no",
        doctrine_id=BINARY_DOCTRINE_ID,
        doctrine_version=BINARY_DOCTRINE_VERSION,
        compiler_id=BINARY_COMPILER_ID,
        compiler_version=BINARY_COMPILER_VERSION,
        assembler_id=BINARY_ASSEMBLER_ID,
        assembler_version=BINARY_ASSEMBLER_VERSION,
    )
    with pytest.raises(VerbalizerError):
        backend.execute(plan)
    assert backend._model.forward_calls == 0


class RetokenizingTokenizer(FakeTokenizer):
    """Appending a verbalizer retokenizes the prefix tail, net delta still +1."""

    def _tokenize(self, text: str) -> list[int]:
        if text.endswith(("yes", "no")):
            return [11, 99, 88, 44]
        return [11, 22, 33]


def test_retokenized_prefix_verbalizer_raises_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = RetokenizingTokenizer()
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    backend = TransformersBackend("fake/model", device="cpu")
    with pytest.raises(VerbalizerError, match="exact single-token continuation"):
        backend.execute(make_plan())
    assert model.forward_calls == 0


def test_verbalizer_resolution_is_cached(backend: TransformersBackend) -> None:
    plan = make_plan()
    backend.execute(plan)
    cache_size = len(backend._verbalizer_cache)
    tokenizer = backend._tokenizer
    # Resolution always encodes without special tokens, so a cache hit re-encodes nothing.
    resolution_calls = sum(1 for _, add_special in tokenizer.calls if not add_special)
    backend.execute(plan)
    assert len(backend._verbalizer_cache) == cache_size
    assert sum(1 for _, add_special in tokenizer.calls if not add_special) == resolution_calls


def make_patched_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, JSONValue] | None = None,
) -> tuple[TransformersBackend, FakeTokenizer]:
    tokenizer = FakeTokenizer(chat_template=chat_template)
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    backend = TransformersBackend(
        "fake/model",
        device="cpu",
        chat_template_kwargs=chat_template_kwargs,
    )
    return backend, tokenizer


def test_execute_forwards_chat_template_kwargs_to_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, tokenizer = make_patched_backend(
        monkeypatch,
        chat_template="yes",
        chat_template_kwargs={"enable_thinking": False, "depth": 2},
    )
    evidence = backend.execute(make_plan())
    assert tokenizer.template_calls[-1] == {"enable_thinking": False, "depth": 2}
    assert evidence.metadata["rendered_input"].startswith("<|system|>")


def test_verbalizer_cache_key_tracks_chat_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan()
    kwargs_backend, _ = make_patched_backend(
        monkeypatch,
        chat_template="yes",
        chat_template_kwargs={"enable_thinking": False, "depth": 2},
    )
    reordered_backend, _ = make_patched_backend(
        monkeypatch,
        chat_template="yes",
        chat_template_kwargs={"depth": 2, "enable_thinking": False},
    )
    extra_key_backend, _ = make_patched_backend(
        monkeypatch,
        chat_template="yes",
        chat_template_kwargs={"enable_thinking": False},
    )
    plain_backend, _ = make_patched_backend(monkeypatch, chat_template="yes")

    kwargs_backend.execute(plan)
    reordered_backend.execute(plan)
    extra_key_backend.execute(plan)
    plain_backend.execute(plan)

    (kwargs_key,) = tuple(kwargs_backend._verbalizer_cache)
    (reordered_key,) = tuple(reordered_backend._verbalizer_cache)
    (extra_key_key,) = tuple(extra_key_backend._verbalizer_cache)
    (plain_key,) = tuple(plain_backend._verbalizer_cache)
    # Same kwargs in a different insertion order must produce the SAME key
    # (deterministic and order-insensitive), different kwargs a DIFFERENT key.
    assert kwargs_key == reordered_key
    assert kwargs_key != extra_key_key
    assert kwargs_key != plain_key


def test_dtype_validation() -> None:
    with pytest.raises(ValueError, match="dtype"):
        TransformersBackend("fake/model", device="cpu", dtype="float64")


def test_default_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer._vocab)
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "probvenance.backends.transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: model,
    )
    monkeypatch.setattr("probvenance.backends.transformers.torch.cuda.is_available", lambda: True)
    backend = TransformersBackend("fake/model")
    assert backend._device == "cuda"
    assert backend._model.to_calls == ["cuda"]
