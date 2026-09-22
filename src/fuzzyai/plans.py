"""Inference plans and raw evidence.

An :class:`InferencePlan` is a provider-independent description of what a
backend must compute for one decision. It deliberately contains no
provider-specific fields (no temperatures, no HTTP parameters, no vendor
options) — those belong to future backend configuration.

:class:`RawEvidence` is model output BEFORE conversion to a
:class:`~fuzzyai.results.DecisionResult`. It is explicitly NOT a probability
and NOT a calibrated probability.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.errors import InvalidDecisionError, InvalidProbabilityError
from fuzzyai.fingerprint import JSONValue, canonical_json, fingerprint


class ScoringStrategy(StrEnum):
    """How a backend should score a decision's outcome space."""

    BINARY_TOKEN_LOGITS = "binary_token_logits"
    CATEGORICAL_TOKEN_LOGITS = "categorical_token_logits"
    TOKEN_LOGPROBS = "token_logprobs"


@dataclass(frozen=True, slots=True)
class InferencePlan:
    """Provider-independent plan for one decision, executable by any backend."""

    decision_fingerprint: str
    strategy: ScoringStrategy
    prompt: str
    targets: tuple[str, ...] = ()
    required_capabilities: BackendCapabilities = field(default_factory=BackendCapabilities.none)
    system_prompt: str | None = None
    positive_verbalizer: str | None = None
    negative_verbalizer: str | None = None
    doctrine_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision_fingerprint, str) or not self.decision_fingerprint:
            raise InvalidDecisionError(
                "decision_fingerprint must be a non-empty string, "
                f"got {self.decision_fingerprint!r}"
            )
        if not isinstance(self.strategy, ScoringStrategy):
            raise InvalidDecisionError(
                f"strategy must be a ScoringStrategy, got {type(self.strategy).__name__} "
                f"({self.strategy!r})"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise InvalidDecisionError(f"prompt must be a non-empty string, got {self.prompt!r}")
        if not isinstance(self.targets, tuple):
            raise InvalidDecisionError(
                f"targets must be a tuple of str, got {type(self.targets).__name__}"
            )
        for target in self.targets:
            if not isinstance(target, str) or not target.strip():
                raise InvalidDecisionError(
                    f"targets must be non-empty stripped strings, got {target!r}"
                )
        if len(set(self.targets)) != len(self.targets):
            raise InvalidDecisionError(f"targets must be unique, got {self.targets!r}")
        if not isinstance(self.required_capabilities, BackendCapabilities):
            raise InvalidDecisionError(
                "required_capabilities must be a BackendCapabilities, got "
                f"{type(self.required_capabilities).__name__}"
            )
        if self.system_prompt is not None and (
            not isinstance(self.system_prompt, str) or not self.system_prompt.strip()
        ):
            raise InvalidDecisionError(
                f"system_prompt must be None or a non-empty string, got {self.system_prompt!r}"
            )
        if self.doctrine_id is not None and (
            not isinstance(self.doctrine_id, str) or not self.doctrine_id.strip()
        ):
            raise InvalidDecisionError(
                f"doctrine_id must be None or a non-empty string, got {self.doctrine_id!r}"
            )
        if self.strategy is ScoringStrategy.BINARY_TOKEN_LOGITS:
            if (
                not isinstance(self.positive_verbalizer, str)
                or not self.positive_verbalizer.strip()
            ):
                raise InvalidDecisionError(
                    "positive_verbalizer must be a non-empty string for the "
                    f"{ScoringStrategy.BINARY_TOKEN_LOGITS.value} strategy, "
                    f"got {self.positive_verbalizer!r}"
                )
            if (
                not isinstance(self.negative_verbalizer, str)
                or not self.negative_verbalizer.strip()
            ):
                raise InvalidDecisionError(
                    "negative_verbalizer must be a non-empty string for the "
                    f"{ScoringStrategy.BINARY_TOKEN_LOGITS.value} strategy, "
                    f"got {self.negative_verbalizer!r}"
                )
            if self.positive_verbalizer == self.negative_verbalizer:
                raise InvalidDecisionError(
                    "positive_verbalizer and negative_verbalizer must differ, got "
                    f"{self.positive_verbalizer!r} for both"
                )
        elif self.positive_verbalizer is not None or self.negative_verbalizer is not None:
            raise InvalidDecisionError(
                "verbalizers are only meaningful for the "
                f"{ScoringStrategy.BINARY_TOKEN_LOGITS.value} strategy, got "
                f"positive_verbalizer={self.positive_verbalizer!r}, "
                f"negative_verbalizer={self.negative_verbalizer!r}"
            )

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of this plan's semantic content."""
        return fingerprint(
            {
                "v": 2,
                "kind": "inference_plan",
                "decision_fingerprint": self.decision_fingerprint,
                "strategy": str(self.strategy),
                "prompt": self.prompt,
                "system_prompt": self.system_prompt,
                "targets": list(self.targets),
                "positive_verbalizer": self.positive_verbalizer,
                "negative_verbalizer": self.negative_verbalizer,
                "doctrine_id": self.doctrine_id,
                "required_capabilities": asdict(self.required_capabilities),
            }
        )


class EvidenceKind(StrEnum):
    """The kind of raw model output carried by :class:`RawEvidence`."""

    LOGITS = "logits"
    LOGPROBS = "logprobs"
    SAMPLE_COUNTS = "sample_counts"


@dataclass(frozen=True, slots=True)
class RawEvidence:
    """Raw model output for one plan, BEFORE conversion to a decision result.

    The ``values`` are explicitly NOT probabilities and NOT calibrated
    probabilities — their meaning depends on :attr:`kind` (raw logits,
    log-probabilities, or sample counts). ``metadata`` is the controlled
    extension mechanism: values must be JSON-compatible.
    """

    kind: EvidenceKind
    labels: tuple[str, ...]
    values: tuple[float, ...]
    plan_fingerprint: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            raise InvalidDecisionError(
                f"kind must be an EvidenceKind, got {type(self.kind).__name__} ({self.kind!r})"
            )
        if not isinstance(self.labels, tuple):
            raise InvalidDecisionError(
                f"labels must be a tuple of str, got {type(self.labels).__name__}"
            )
        if not isinstance(self.values, tuple):
            raise InvalidDecisionError(
                f"values must be a tuple of float, got {type(self.values).__name__}"
            )
        if len(self.labels) != len(self.values):
            raise InvalidDecisionError(
                f"labels and values must have the same length, got {len(self.labels)} labels "
                f"and {len(self.values)} values"
            )
        if not self.labels:
            raise InvalidDecisionError("labels must be non-empty")
        if len(set(self.labels)) != len(self.labels):
            raise InvalidDecisionError(f"labels must be unique, got {self.labels!r}")
        for label in self.labels:
            if not isinstance(label, str) or not label:
                raise InvalidDecisionError(f"labels must be non-empty strings, got {label!r}")
        coerced: list[float] = []
        for i, value in enumerate(self.values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidProbabilityError(
                    f"values[{i}] must be a real number, got {type(value).__name__} ({value!r})"
                )
            if not isinstance(value, float):
                value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise InvalidProbabilityError(f"values[{i}] must be finite, got {value!r}")
            coerced.append(value)
        object.__setattr__(self, "values", tuple(coerced))
        if self.plan_fingerprint is not None and (
            not isinstance(self.plan_fingerprint, str) or not self.plan_fingerprint
        ):
            raise InvalidDecisionError(
                "plan_fingerprint must be None or a non-empty string, "
                f"got {self.plan_fingerprint!r}"
            )
        if not isinstance(self.metadata, Mapping):
            raise InvalidDecisionError(
                f"metadata must be a mapping, got {type(self.metadata).__name__}"
            )
        try:
            canonical_json(dict(self.metadata))
        except Exception as exc:
            raise InvalidDecisionError(f"metadata must be JSON-compatible: {exc}") from exc
        # Deep-copy (not just top-level dict()) so later mutation of the
        # caller's mapping or its nested values cannot affect this evidence.
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))
