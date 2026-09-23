"""Inference plans and raw evidence.

An :class:`InferencePlan` is a provider-independent description of what a
backend must compute for one decision. It deliberately contains no
provider-specific fields (no temperatures, no HTTP parameters, no vendor
options) — those belong to future backend configuration.

:class:`RawEvidence` is model output BEFORE conversion to a
:class:`~probvenance.results.DecisionResult`. It is explicitly NOT a probability
and NOT a calibrated probability.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from probvenance.capabilities import BackendCapabilities
from probvenance.errors import InvalidDecisionError, InvalidProbabilityError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint


class ScoringStrategy(StrEnum):
    """How a backend should score a decision's outcome space."""

    BINARY_TOKEN_LOGITS = "binary_token_logits"
    CATEGORICAL_TOKEN_LOGITS = "categorical_token_logits"
    TOKEN_LOGPROBS = "token_logprobs"


# Decision families. The family is DECLARED on the plan (by the compiler or the
# caller), never inferred from the scoring strategy: a strategy describes how a
# backend scores, while the family describes what the outcome space is.
DECISION_FAMILY_BOOL = "bool"
DECISION_FAMILY_CHOICE = "choice"
DECISION_FAMILIES: tuple[str, ...] = (DECISION_FAMILY_BOOL, DECISION_FAMILY_CHOICE)

# Legal implemented-strategy/family combinations. This is an explicit table, not
# an inference rule: a future strategy (for example one_vs_rest_binary_logits
# declaring "choice") is a data change here, not a rewrite.
_IMPLEMENTED_STRATEGY_FAMILIES: dict[ScoringStrategy, str] = {
    ScoringStrategy.BINARY_TOKEN_LOGITS: DECISION_FAMILY_BOOL,
    ScoringStrategy.CATEGORICAL_TOKEN_LOGITS: DECISION_FAMILY_CHOICE,
}

# Strategies that have a real compiler and probability assembler today. Plans
# using these MUST declare which one produced them; an unimplemented strategy
# has no compiler yet, so it declares no provenance.
_STRATEGIES_WITH_PROVENANCE: frozenset[ScoringStrategy] = frozenset(_IMPLEMENTED_STRATEGY_FAMILIES)

# Bump when the canonical fingerprint payload changes shape; never hash across versions.
PLAN_FINGERPRINT_VERSION = 6


def _require_atomic_party(identifier: object, version: object, party: str) -> None:
    """Enforce the atomic identity contract for one implementation party.

    A versioned implementation identity (compiler, assembler, doctrine) is
    either fully concrete or fully unknown; half-known states are rejected.
    This deliberately does NOT apply to model/tokenizer ids and revisions:
    a known model with an unknown revision is a real provenance state.
    """
    id_field = f"{party}_id"
    version_field = f"{party}_version"
    id_known = identifier is not None
    version_known = version is not None
    if id_known != version_known:
        raise InvalidDecisionError(
            f"{id_field} and {version_field} must be set together (atomic identity), "
            f"got {id_field}={identifier!r}, {version_field}={version!r}"
        )
    if id_known:
        if not isinstance(identifier, str) or not identifier.strip():
            raise InvalidDecisionError(
                f"{id_field} must be None or a non-empty string, got "
                f"{type(identifier).__name__} ({identifier!r})"
            )
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise InvalidDecisionError(
                f"{version_field} must be None or an int >= 1, got "
                f"{type(version).__name__} ({version!r})"
            )


@dataclass(frozen=True, slots=True)
class CandidateLabelMapping:
    """One semantic candidate bound to one execution-only scoring label.

    The mapping is provider-independent: it deliberately carries no token id,
    because a token id does not exist until a concrete tokenizer renders a
    concrete input. Token ids are execution provenance, produced by the backend,
    and are never compiled into a plan.
    """

    candidate_index: int
    candidate_name: str
    candidate_description: str | None
    scoring_label: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise InvalidDecisionError(
                "candidate_index must be a non-negative int, "
                f"got {type(self.candidate_index).__name__} ({self.candidate_index!r})"
            )
        if not isinstance(self.candidate_name, str) or not self.candidate_name.strip():
            raise InvalidDecisionError(
                f"candidate_name must be a non-empty string, got {self.candidate_name!r}"
            )
        if self.candidate_description is not None and not isinstance(
            self.candidate_description, str
        ):
            raise InvalidDecisionError(
                "candidate_description must be None or str, got "
                f"{type(self.candidate_description).__name__} ({self.candidate_description!r})"
            )
        if not isinstance(self.scoring_label, str) or not self.scoring_label.strip():
            raise InvalidDecisionError(
                f"scoring_label must be a non-empty string, got {self.scoring_label!r}"
            )


@dataclass(frozen=True, slots=True)
class InferencePlan:
    """Provider-independent plan for one decision, executable by any backend."""

    decision_fingerprint: str
    strategy: ScoringStrategy
    prompt: str
    decision_family: str
    targets: tuple[str, ...] = ()
    required_capabilities: BackendCapabilities = field(default_factory=BackendCapabilities.none)
    system_prompt: str | None = None
    positive_verbalizer: str | None = None
    negative_verbalizer: str | None = None
    doctrine_id: str | None = None
    doctrine_version: int | None = None
    label_scheme_id: str | None = None
    candidate_mapping: tuple[CandidateLabelMapping, ...] = ()
    compiler_id: str | None = None
    compiler_version: int | None = None
    assembler_id: str | None = None
    assembler_version: int | None = None

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
        if not isinstance(self.decision_family, str) or not self.decision_family.strip():
            raise InvalidDecisionError(
                "decision_family must be a non-empty string, got "
                f"{type(self.decision_family).__name__} ({self.decision_family!r})"
            )
        if self.decision_family not in DECISION_FAMILIES:
            raise InvalidDecisionError(
                f"decision_family must be one of {DECISION_FAMILIES}, got {self.decision_family!r}"
            )
        expected_family = _IMPLEMENTED_STRATEGY_FAMILIES.get(self.strategy)
        if expected_family is not None and self.decision_family != expected_family:
            raise InvalidDecisionError(
                f"decision_family {self.decision_family!r} is not compatible with "
                f"strategy {self.strategy.value!r}, expected {expected_family!r}"
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
        if self.doctrine_version is not None and (
            isinstance(self.doctrine_version, bool)
            or not isinstance(self.doctrine_version, int)
            or self.doctrine_version < 1
        ):
            raise InvalidDecisionError(
                "doctrine_version must be None or an int >= 1, got "
                f"{type(self.doctrine_version).__name__} ({self.doctrine_version!r})"
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
        if self.label_scheme_id is not None and (
            not isinstance(self.label_scheme_id, str) or not self.label_scheme_id.strip()
        ):
            raise InvalidDecisionError(
                f"label_scheme_id must be None or a non-empty string, got {self.label_scheme_id!r}"
            )
        if not isinstance(self.candidate_mapping, tuple):
            raise InvalidDecisionError(
                "candidate_mapping must be a tuple of CandidateLabelMapping, got "
                f"{type(self.candidate_mapping).__name__}"
            )
        for entry in self.candidate_mapping:
            if not isinstance(entry, CandidateLabelMapping):
                raise InvalidDecisionError(
                    "candidate_mapping entries must be CandidateLabelMapping, got "
                    f"{type(entry).__name__} ({entry!r})"
                )
        if self.strategy is ScoringStrategy.CATEGORICAL_TOKEN_LOGITS:
            self._validate_categorical_mapping()
        elif self.candidate_mapping:
            raise InvalidDecisionError(
                "candidate_mapping is only meaningful for the "
                f"{ScoringStrategy.CATEGORICAL_TOKEN_LOGITS.value} strategy, got "
                f"{len(self.candidate_mapping)} entries"
            )
        # Implementation identity parties are atomic: both concrete or both
        # unknown, never half-known. Doctrine id and version are separate
        # declared fields; never parse a version out of the id string.
        _require_atomic_party(self.compiler_id, self.compiler_version, "compiler")
        _require_atomic_party(self.assembler_id, self.assembler_version, "assembler")
        _require_atomic_party(self.doctrine_id, self.doctrine_version, "doctrine")
        if self.strategy in _STRATEGIES_WITH_PROVENANCE:
            for party, id_field, version_field in (
                ("compiler", "compiler_id", "compiler_version"),
                ("assembler", "assembler_id", "assembler_version"),
                ("doctrine", "doctrine_id", "doctrine_version"),
            ):
                identifier = getattr(self, id_field)
                version = getattr(self, version_field)
                if identifier is None or version is None:
                    raise InvalidDecisionError(
                        f"{party} identity must be concrete for the "
                        f"{self.strategy.value!r} strategy, got "
                        f"{id_field}={identifier!r}, {version_field}={version!r}"
                    )

    def _validate_categorical_mapping(self) -> None:
        mapping = self.candidate_mapping
        if len(self.targets) < 2:
            raise InvalidDecisionError(
                "the categorical strategy requires at least 2 ordered targets, got "
                f"{self.targets!r}"
            )
        if not isinstance(self.label_scheme_id, str) or not self.label_scheme_id.strip():
            raise InvalidDecisionError(
                "label_scheme_id must be a non-empty string for the "
                f"{ScoringStrategy.CATEGORICAL_TOKEN_LOGITS.value} strategy"
            )
        if len(mapping) != len(self.targets):
            raise InvalidDecisionError(
                "candidate_mapping must have exactly one entry per target, got "
                f"{len(mapping)} entries for {len(self.targets)} targets"
            )
        indices = [entry.candidate_index for entry in mapping]
        if indices != list(range(len(mapping))):
            raise InvalidDecisionError(
                f"candidate_mapping indices must be 0..{len(mapping) - 1} in order, got {indices}"
            )
        names = [entry.candidate_name for entry in mapping]
        if len(set(names)) != len(names):
            raise InvalidDecisionError(f"candidate_mapping names must be unique, got {names!r}")
        labels = [entry.scoring_label for entry in mapping]
        if len(set(labels)) != len(labels):
            raise InvalidDecisionError(f"candidate_mapping labels must be unique, got {labels!r}")
        for index, entry in enumerate(mapping):
            if entry.scoring_label != self.targets[index]:
                raise InvalidDecisionError(
                    "candidate_mapping must follow target order: entry "
                    f"{index} has scoring_label {entry.scoring_label!r} but "
                    f"targets[{index}] is {self.targets[index]!r}"
                )

    @property
    def fingerprint_version(self) -> int:
        """Schema version of the canonical payload :attr:`fingerprint` hashes."""
        return PLAN_FINGERPRINT_VERSION

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of this plan's semantic content."""
        return fingerprint(
            {
                "v": PLAN_FINGERPRINT_VERSION,
                "kind": "inference_plan",
                "decision_fingerprint": self.decision_fingerprint,
                "strategy": str(self.strategy),
                "decision_family": self.decision_family,
                "prompt": self.prompt,
                "system_prompt": self.system_prompt,
                "targets": list(self.targets),
                "positive_verbalizer": self.positive_verbalizer,
                "negative_verbalizer": self.negative_verbalizer,
                "doctrine_id": self.doctrine_id,
                "doctrine_version": self.doctrine_version,
                "label_scheme_id": self.label_scheme_id,
                "candidate_mapping": [asdict(entry) for entry in self.candidate_mapping],
                "required_capabilities": asdict(self.required_capabilities),
                "compiler_id": self.compiler_id,
                "compiler_version": self.compiler_version,
                "assembler_id": self.assembler_id,
                "assembler_version": self.assembler_version,
            }
        )

    # The probability identity helpers are imported inside the properties rather
    # than at module scope: ``probability_identity`` reads plan fields and so
    # imports this module, and keeping the reference deferred avoids a cycle.
    # These identities are DERIVED and are deliberately not part of the plan
    # fingerprint payload, which would create a second source of truth.
    @property
    def probability_formulation_fingerprint(self) -> str:
        """Provider-independent exact formulation identity for this plan.

        Derived from the plan alone: it excludes question, context, model,
        tokenizer, rendering configuration, and resolved scoring token ids.
        """
        from probvenance.probability_identity import probability_formulation_fingerprint

        return probability_formulation_fingerprint(self)

    @property
    def probability_formulation_fingerprint_version(self) -> int:
        """Schema version of the canonical exact-formulation payload."""
        from probvenance.probability_identity import (
            PROBABILITY_FORMULATION_FINGERPRINT_VERSION,
        )

        return PROBABILITY_FORMULATION_FINGERPRINT_VERSION

    @property
    def formulation_family_fingerprint(self) -> str:
        """Coarser mechanism-class identity for this plan.

        Candidate names, descriptions, and the candidate-to-label assignment are
        collapsed. Equality does NOT imply interchangeability, poolability, or
        shared calibration (INV-24).
        """
        from probvenance.probability_identity import formulation_family_fingerprint

        return formulation_family_fingerprint(self)

    @property
    def formulation_family_fingerprint_version(self) -> int:
        """Schema version of the canonical formulation-family payload."""
        from probvenance.probability_identity import (
            FORMULATION_FAMILY_FINGERPRINT_VERSION,
        )

        return FORMULATION_FAMILY_FINGERPRINT_VERSION


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
