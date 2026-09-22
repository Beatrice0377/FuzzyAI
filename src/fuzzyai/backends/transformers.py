"""Local Hugging Face causal-LM backend (optional ``transformers`` extra).

Scores the positive/negative verbalizer tokens at the last input position; it
NEVER generates text, never samples, and never parses generated output. This
module is importable only when the optional dependency group is installed:

    uv sync --extra transformers
"""

from collections.abc import Mapping
from typing import Any

from fuzzyai.backends.verbalizers import (
    VerbalizerTokens,
    render_input_text,
    resolve_verbalizers,
)
from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.errors import UnsupportedCapabilityError
from fuzzyai.fingerprint import JSONValue, canonical_json
from fuzzyai.plans import EvidenceKind, InferencePlan, RawEvidence, ScoringStrategy

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "fuzzyai.backends.transformers requires the optional dependency group: "
        "install with 'uv sync --extra transformers' (or 'pip install fuzzyai[transformers]')"
    ) from exc

_ALLOWED_DTYPES: dict[str, Any] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TransformersBackend:
    """Local Hugging Face causal-LM backend. Scores tokens; it never generates text."""

    # Duck-typed third-party handles: transformers 5.x does not statically
    # declare the protocol/call shapes used below (and test fakes substitute
    # for these objects), so Any is the honest boundary annotation. Runtime
    # shape is validated in execute().
    _tokenizer: Any
    _model: Any

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
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=self._dtype,
        )
        self._model.eval()
        self._model.to(self._device)
        # Cache of resolved verbalizers keyed by the plan fields and backend
        # config that determine them. canonical_json sorts dict keys, so the
        # kwargs component is deterministic and order-insensitive.
        self._verbalizer_cache: dict[tuple[str | None, str, str, str, str], VerbalizerTokens] = {}
        self._template_kwargs_json = canonical_json(self._chat_template_kwargs)

    @property
    def capabilities(self) -> BackendCapabilities:
        """Honest capability declaration: binary token logits only."""
        return BackendCapabilities(binary_token_logits=True)

    def _plan_verbalizers(self, plan: InferencePlan) -> VerbalizerTokens:
        """Resolve (and cache) the plan's verbalizers against the rendered prefix."""
        if plan.system_prompt is None or plan.positive_verbalizer is None:
            raise UnsupportedCapabilityError(
                "plan lacks the system prompt / verbalizers required by the "
                f"{ScoringStrategy.BINARY_TOKEN_LOGITS.value} strategy"
            )
        negative = plan.negative_verbalizer
        if negative is None:  # pragma: no cover - InferencePlan validation forbids this
            raise UnsupportedCapabilityError(
                "plan lacks a negative_verbalizer for the "
                f"{ScoringStrategy.BINARY_TOKEN_LOGITS.value} strategy"
            )
        key = (
            plan.system_prompt,
            plan.prompt,
            plan.positive_verbalizer,
            negative,
            self._template_kwargs_json,
        )
        cached = self._verbalizer_cache.get(key)
        if cached is not None:
            return cached
        prefix_text = render_input_text(
            self._tokenizer,
            system_prompt=plan.system_prompt,
            user_prompt=plan.prompt,
            template_kwargs=self._chat_template_kwargs,
        )
        resolved = resolve_verbalizers(
            self._tokenizer,
            prefix_text=prefix_text,
            positive_verbalizer=plan.positive_verbalizer,
            negative_verbalizer=negative,
        )
        self._verbalizer_cache[key] = resolved
        return resolved

    def execute(self, plan: InferencePlan) -> RawEvidence:
        """Score the plan's verbalizer tokens at the last input position."""
        if plan.strategy is not ScoringStrategy.BINARY_TOKEN_LOGITS:
            raise UnsupportedCapabilityError(
                f"TransformersBackend supports only the "
                f"{ScoringStrategy.BINARY_TOKEN_LOGITS.value} strategy, got "
                f"{plan.strategy.value}"
            )
        verbalizers = self._plan_verbalizers(plan)
        prefix_text = render_input_text(
            self._tokenizer,
            system_prompt=plan.system_prompt,
            user_prompt=plan.prompt,
            template_kwargs=self._chat_template_kwargs,
        )
        used_chat_template = bool(getattr(self._tokenizer, "chat_template", None))
        encoded = self._tokenizer(prefix_text, add_special_tokens=not used_chat_template)
        input_ids = encoded.input_ids
        if not isinstance(input_ids, list) or not all(isinstance(t, int) for t in input_ids):
            raise TypeError(
                "tokenizer must return a list of int token ids for input_ids, "
                f"got {type(input_ids).__name__}"
            )
        inputs = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            logits = self._model(input_ids=inputs).logits
        # Read ONLY the final position: the distribution over the next token.
        row = logits[0, -1, :]
        logit_positive = float(row[verbalizers.positive_token_id].item())
        logit_negative = float(row[verbalizers.negative_token_id].item())
        model_revision = getattr(self._model.config, "_commit_hash", None) or self._revision
        return RawEvidence(
            kind=EvidenceKind.LOGITS,
            labels=("false", "true"),
            values=(logit_negative, logit_positive),
            plan_fingerprint=plan.fingerprint,
            metadata={
                "positive_token_id": verbalizers.positive_token_id,
                "negative_token_id": verbalizers.negative_token_id,
                "positive_verbalizer": verbalizers.positive_verbalizer,
                "negative_verbalizer": verbalizers.negative_verbalizer,
                "model": self._model_id,
                "model_revision": model_revision,
                "device": self._device,
                "rendered_input": prefix_text,
                "input_token_count": int(logits.shape[1]),
            },
        )
