"""Score the TEXT tower of a Qwen3.5 vision-language checkpoint.

``Qwen/Qwen3.5-*`` is published as a vision-language checkpoint
(``Qwen3_5ForConditionalGeneration``): the text tower is stored under
``model.language_model.*`` next to a ``model.visual.*`` tower and an ``mtp.*``
multi-token-prediction head. The text-only ``Qwen3_5ForCausalLM`` expects that
same tower under ``model.*``.

This adapter rebuilds the text-only model from ``text_config``, rewrites the
text tower key prefix mechanically, and drops the non-text towers. Rendering,
verbalizer resolution, scoring, diagnostics and provenance all run through the
real ``TransformersBackend`` code path; only checkpoint loading differs.

This is a checkpoint adaptation and therefore belongs to the experiment
harness, not to ``src/probvenance``.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from probvenance.backends.transformers import _ALLOWED_DTYPES, TransformersBackend
from probvenance.backends.verbalizers import VerbalizerTokens
from probvenance.fingerprint import JSONValue, canonical_json

TEXT_TOWER_PREFIX = "model.language_model."
NON_TEXT_PREFIXES = ("model.visual.", "mtp.")

#: Must match the pattern set the harness downloader used, otherwise
#: ``snapshot_download(local_files_only=True)`` reports the slimmed local
#: snapshot as incomplete because of files (README, LICENSE, .gitattributes)
#: that are irrelevant to scoring and were deliberately never fetched.
SNAPSHOT_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "tokenizer*",
    "*.py",
    "*.jinja",
]


class Qwen35TextBackend(TransformersBackend):
    """``TransformersBackend`` that scores the text tower of a Qwen3.5 VL checkpoint."""

    #: Filled in by ``_load_text_tower``: what the mechanical rewrite did not cover.
    load_report: dict[str, list[str]]

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        chat_template_kwargs: Mapping[str, JSONValue] | None = None,
    ) -> None:
        if dtype is not None and dtype not in _ALLOWED_DTYPES:
            allowed = ", ".join(sorted(_ALLOWED_DTYPES))
            raise ValueError(f"dtype must be None or one of {allowed}, got {dtype!r}")
        self._model_id = model
        self._revision = revision
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = _ALLOWED_DTYPES[dtype] if dtype is not None else torch.float32
        self._trust_remote_code = trust_remote_code
        self._local_files_only = local_files_only
        self._chat_template_kwargs: dict[str, JSONValue] = (
            dict(chat_template_kwargs) if chat_template_kwargs is not None else {}
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        config = AutoConfig.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            raise ValueError(
                f"{model} has no text_config; Qwen35TextBackend expects a composite "
                f"vision-language checkpoint, got architectures="
                f"{getattr(config, 'architectures', None)!r}"
            )
        self._model = Qwen3_5ForCausalLM(text_config)
        self._load_text_tower(model, revision)
        self._model.eval()
        # Constructing from text_config gives float32 parameters, and
        # load_state_dict only widens the checkpoint's bfloat16 values into them.
        # Cast explicitly so the model actually runs at the requested dtype
        # instead of silently reporting one it never used.
        self._model.to(device=self._device, dtype=self._dtype)
        self._verbalizer_cache: dict[tuple[str | None, str, str, str, str], VerbalizerTokens] = {}
        self._template_kwargs_json = canonical_json(self._chat_template_kwargs)

    def _load_text_tower(self, model: str, revision: str | None) -> None:
        model_dir = snapshot_download(
            model,
            revision=revision,
            local_files_only=self._local_files_only,
            allow_patterns=SNAPSHOT_ALLOW_PATTERNS,
        )
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no safetensors shards found under {model_dir}")
        checkpoint: dict[str, torch.Tensor] = {}
        for shard in shards:
            checkpoint.update(load_file(shard))

        remapped: dict[str, torch.Tensor] = {}
        dropped = 0
        for key, value in checkpoint.items():
            if key.startswith(TEXT_TOWER_PREFIX):
                remapped["model." + key[len(TEXT_TOWER_PREFIX) :]] = value
            elif key.startswith(NON_TEXT_PREFIXES):
                dropped += 1
            else:
                remapped[key] = value
        del checkpoint

        report = self._model.load_state_dict(remapped, strict=False)
        del remapped
        self.load_report = {
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
            "dropped": [f"{dropped} non-text tower tensors"],
        }
