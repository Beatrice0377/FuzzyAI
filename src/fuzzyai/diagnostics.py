"""Scoring diagnostics: where a model's next-token probability mass actually went.

A restricted candidate probability answers "which candidate would the model
prefer, if it had to choose between them". It does NOT answer "was the model
actually choosing between them". :class:`ScoringDiagnostics` carries the second
kind of information.

These values are derived from raw evidence. They are NOT decision
probabilities, NOT calibrated probabilities, and NOT a validity verdict: this
module applies no threshold anywhere, because no threshold has experimental
support across models, tokenizers, chat templates, verbalizers and prompts.
Measure and record now; let a future policy layer decide later.
"""

import math
from dataclasses import dataclass

from fuzzyai.assembler import BINARY_EVIDENCE_LABELS
from fuzzyai.errors import InvalidDecisionError, InvalidProbabilityError
from fuzzyai.plans import EvidenceKind, RawEvidence

#: ``log(sum(exp(logit) for logit in full vocabulary))`` for the scored position.
VOCAB_LOGSUMEXP_KEY = "vocab_logsumexp"
#: Token id of the highest-probability next token over the full vocabulary.
TOP_TOKEN_ID_KEY = "top_token_id"
#: Raw logit of that highest-probability next token.
TOP_TOKEN_LOGIT_KEY = "top_token_logit"
#: Best-effort decoded text of that token. Debug aid only; never required.
TOP_TOKEN_TEXT_KEY = "top_token_text"

_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    VOCAB_LOGSUMEXP_KEY,
    TOP_TOKEN_ID_KEY,
    TOP_TOKEN_LOGIT_KEY,
)

#: Ordering facts of a full-vocabulary distribution are exact in real
#: arithmetic, but they are reconstructed here through ``exp``/``logaddexp``,
#: which accumulate rounding error. They are therefore checked with a small
#: absolute tolerance rather than exactly.
_ORDERING_TOLERANCE = 1e-6


def _logaddexp(a: float, b: float) -> float:
    """``log(exp(a) + exp(b))``, computed without overflow.

    ``math.logaddexp`` does not exist in the standard library (it is a numpy /
    torch name), so the stable form is spelled out: factor out the larger
    operand and take ``log1p`` of an exponent that is never positive.
    """
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _probability_from_log_ratio(log_delta: float) -> float:
    """``exp(log_delta)`` clamped into ``[0, 1]`` without ever overflowing.

    ``log_delta`` is a raw log-probability (``logit - logsumexp``), so it is
    ``<= 0`` for a truthful backend. A backend that reports an inconsistent
    ``vocab_logsumexp`` would hand us a positive value; clamping rather than
    raising keeps a rounding-level inconsistency from failing an inference.
    """
    if log_delta >= 0.0:
        return 1.0
    return math.exp(log_delta)


def log_verbalizer_mass(*, logit_true: float, logit_false: float, vocab_logsumexp: float) -> float:
    """``log P(next token is either verbalizer)`` under full-vocabulary normalization."""
    return _logaddexp(logit_true, logit_false) - vocab_logsumexp


def verbalizer_mass(*, logit_true: float, logit_false: float, vocab_logsumexp: float) -> float:
    """``P(next token is either verbalizer)`` under full-vocabulary normalization."""
    return _probability_from_log_ratio(
        log_verbalizer_mass(
            logit_true=logit_true,
            logit_false=logit_false,
            vocab_logsumexp=vocab_logsumexp,
        )
    )


def full_vocab_probability(*, logit: float, vocab_logsumexp: float) -> float:
    """``P(next token is this token)`` under full-vocabulary normalization."""
    return _probability_from_log_ratio(logit - vocab_logsumexp)


def _validated_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidProbabilityError(
            f"{name} must be a real number, got {type(value).__name__} ({value!r})"
        )
    result = float(value)
    if result != result or result in (math.inf, -math.inf):
        raise InvalidProbabilityError(f"{name} must be finite, got {result!r}")
    if not 0.0 <= result <= 1.0:
        raise InvalidProbabilityError(f"{name} must be within [0, 1], got {result!r}")
    return result


@dataclass(frozen=True, slots=True)
class ScoringDiagnostics:
    """Evidence-quality facts about a scoring position.

    Every probability here is a FULL-VOCABULARY next-token probability. None of
    them is a decision probability and none of them is calibrated. The class
    carries no verdict: there is no ``valid`` flag, because deciding that
    requires a threshold policy that does not exist yet.
    """

    verbalizer_mass: float
    top_token_id: int
    top_token_probability: float
    positive_token_probability: float
    negative_token_probability: float
    top_token_text: str | None = None

    def __post_init__(self) -> None:
        mass = _validated_probability(self.verbalizer_mass, name="verbalizer_mass")
        top = _validated_probability(self.top_token_probability, name="top_token_probability")
        positive = _validated_probability(
            self.positive_token_probability, name="positive_token_probability"
        )
        negative = _validated_probability(
            self.negative_token_probability, name="negative_token_probability"
        )
        object.__setattr__(self, "verbalizer_mass", mass)
        object.__setattr__(self, "top_token_probability", top)
        object.__setattr__(self, "positive_token_probability", positive)
        object.__setattr__(self, "negative_token_probability", negative)
        if isinstance(self.top_token_id, bool) or not isinstance(self.top_token_id, int):
            raise InvalidDecisionError(
                "top_token_id must be an int, got "
                f"{type(self.top_token_id).__name__} ({self.top_token_id!r})"
            )
        if self.top_token_id < 0:
            raise InvalidDecisionError(f"top_token_id must be >= 0, got {self.top_token_id!r}")
        if self.top_token_text is not None and not isinstance(self.top_token_text, str):
            raise InvalidDecisionError(
                "top_token_text must be None or a string, got "
                f"{type(self.top_token_text).__name__} ({self.top_token_text!r})"
            )
        # Structural facts of any full-vocabulary distribution: the argmax is
        # at least as probable as any other single token, and the mass assigned
        # to the two candidates is at least each candidate's own probability.
        largest_candidate = max(positive, negative)
        if top < largest_candidate - _ORDERING_TOLERANCE:
            raise InvalidProbabilityError(
                "top_token_probability must be >= every other token probability: got "
                f"top_token_probability={top!r} but positive={positive!r}, negative={negative!r}"
            )
        if mass < largest_candidate - _ORDERING_TOLERANCE:
            raise InvalidProbabilityError(
                "verbalizer_mass must be >= each verbalizer token probability: got "
                f"verbalizer_mass={mass!r} but positive={positive!r}, negative={negative!r}"
            )


def _metadata_float(evidence: RawEvidence, key: str) -> float:
    value = evidence.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDecisionError(
            f"evidence metadata key {key!r} must be a real number, "
            f"got {type(value).__name__} ({value!r})"
        )
    result = float(value)
    if result != result or result in (math.inf, -math.inf):
        raise InvalidProbabilityError(
            f"evidence metadata key {key!r} must be finite, got {result!r}"
        )
    return result


def diagnose_bool_evidence(evidence: RawEvidence) -> ScoringDiagnostics:
    """Derive :class:`ScoringDiagnostics` from binary token-logit evidence.

    The evidence must carry the raw normalization statistics reported by the
    backend (``vocab_logsumexp``, ``top_token_id``, ``top_token_logit``); the
    backend measures them from the same forward pass it already performed, so
    this adds no model work. ``top_token_text`` is optional.

    Raises:
        InvalidDecisionError: if the evidence kind or labels are not binary
            token logits, or a required metadata key is missing or of the
            wrong type.
        InvalidProbabilityError: if a required metadata value is not finite.
    """
    if evidence.kind is not EvidenceKind.LOGITS:
        raise InvalidDecisionError(
            f"binary diagnostics require {EvidenceKind.LOGITS.value!r} evidence, "
            f"got {evidence.kind.value!r}"
        )
    if len(evidence.labels) != 2 or set(evidence.labels) != set(BINARY_EVIDENCE_LABELS):
        raise InvalidDecisionError(
            f"binary evidence labels must be exactly {BINARY_EVIDENCE_LABELS} in either "
            f"order, got {evidence.labels!r}"
        )
    for key in _REQUIRED_METADATA_KEYS:
        if key not in evidence.metadata:
            raise InvalidDecisionError(
                f"evidence metadata is missing required key {key!r}: scoring "
                "diagnostics need the full-vocabulary normalization statistics"
            )
    logit_true = evidence.values[evidence.labels.index("true")]
    logit_false = evidence.values[evidence.labels.index("false")]
    vocab_logsumexp = _metadata_float(evidence, VOCAB_LOGSUMEXP_KEY)
    top_token_id = evidence.metadata[TOP_TOKEN_ID_KEY]
    if isinstance(top_token_id, bool) or not isinstance(top_token_id, int):
        raise InvalidDecisionError(
            f"evidence metadata key {TOP_TOKEN_ID_KEY!r} must be an int, "
            f"got {type(top_token_id).__name__} ({top_token_id!r})"
        )
    top_token_logit = _metadata_float(evidence, TOP_TOKEN_LOGIT_KEY)
    top_token_text = evidence.metadata.get(TOP_TOKEN_TEXT_KEY)
    if top_token_text is not None and not isinstance(top_token_text, str):
        raise InvalidDecisionError(
            f"evidence metadata key {TOP_TOKEN_TEXT_KEY!r} must be None or a string, "
            f"got {type(top_token_text).__name__} ({top_token_text!r})"
        )
    return ScoringDiagnostics(
        verbalizer_mass=verbalizer_mass(
            logit_true=logit_true,
            logit_false=logit_false,
            vocab_logsumexp=vocab_logsumexp,
        ),
        top_token_id=top_token_id,
        top_token_probability=full_vocab_probability(
            logit=top_token_logit, vocab_logsumexp=vocab_logsumexp
        ),
        positive_token_probability=full_vocab_probability(
            logit=logit_true, vocab_logsumexp=vocab_logsumexp
        ),
        negative_token_probability=full_vocab_probability(
            logit=logit_false, vocab_logsumexp=vocab_logsumexp
        ),
        top_token_text=top_token_text,
    )
