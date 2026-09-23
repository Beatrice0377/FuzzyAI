"""Compilers: decisions -> :class:`~fuzzyai.plans.InferencePlan`.

A compiler is a pure planner. It NEVER computes probabilities, never sees
evidence, and never touches a backend or a tokenizer. Its only job is to turn
a decision plus the backend's declared capabilities into a fully-specified
plan.

Two compilers exist:

- :class:`BoolCompiler` compiles :class:`~fuzzyai.decisions.BoolDecision`
  objects under the binary token-logit strategy.
- :class:`ChoiceCompiler` compiles :class:`~fuzzyai.decisions.ChoiceDecision`
  objects under the categorical token-logit strategy, assigning each semantic
  candidate an execution-only scoring label from a versioned label scheme.
"""

from fuzzyai.assembler import (
    BINARY_ASSEMBLER_ID,
    BINARY_ASSEMBLER_VERSION,
    CATEGORICAL_ASSEMBLER_ID,
    CATEGORICAL_ASSEMBLER_VERSION,
)
from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.decisions import BoolDecision, ChoiceDecision
from fuzzyai.doctrine import (
    BINARY_SEMANTIC_JUDGMENT_V1,
    CATEGORICAL_SEMANTIC_JUDGMENT_V1,
    CategoricalScoringDoctrine,
    ScoringDoctrine,
)
from fuzzyai.errors import (
    InvalidDecisionError,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
)
from fuzzyai.plans import CandidateLabelMapping, InferencePlan, ScoringStrategy

BINARY_COMPILER_ID = "bool-compiler"
BINARY_COMPILER_VERSION = 1
DEFAULT_POSITIVE_VERBALIZER = "yes"
DEFAULT_NEGATIVE_VERBALIZER = "no"

#: Versioned identifier of the ordered scoring-label scheme used by
#: :class:`ChoiceCompiler`. Bumping the scheme changes plan fingerprints.
CATEGORICAL_LABEL_SCHEME_ID = "categorical-labels-v1"

#: The ordered scoring labels. Candidate ``i`` receives
#: ``CATEGORICAL_LABELS[i]``; a decision with more candidates than labels
#: cannot be compiled under this scheme.
CATEGORICAL_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

CATEGORICAL_COMPILER_ID = "choice-compiler"
CATEGORICAL_COMPILER_VERSION = 1


class BoolCompiler:
    """Compiles :class:`~fuzzyai.decisions.BoolDecision` objects into plans."""

    def __init__(
        self,
        *,
        doctrine: ScoringDoctrine = BINARY_SEMANTIC_JUDGMENT_V1,
        positive_verbalizer: str = DEFAULT_POSITIVE_VERBALIZER,
        negative_verbalizer: str = DEFAULT_NEGATIVE_VERBALIZER,
    ) -> None:
        if not isinstance(doctrine, ScoringDoctrine):
            raise InvalidDecisionError(
                f"doctrine must be a ScoringDoctrine, got {type(doctrine).__name__}"
            )
        if not isinstance(positive_verbalizer, str) or not positive_verbalizer.strip():
            raise InvalidDecisionError(
                f"positive_verbalizer must be a non-empty string, got {positive_verbalizer!r}"
            )
        if not isinstance(negative_verbalizer, str) or not negative_verbalizer.strip():
            raise InvalidDecisionError(
                f"negative_verbalizer must be a non-empty string, got {negative_verbalizer!r}"
            )
        if positive_verbalizer == negative_verbalizer:
            raise InvalidDecisionError(
                "positive_verbalizer and negative_verbalizer must be distinct, "
                f"got {positive_verbalizer!r} for both"
            )
        self._doctrine = doctrine
        self._positive_verbalizer = positive_verbalizer
        self._negative_verbalizer = negative_verbalizer

    @property
    def doctrine(self) -> ScoringDoctrine:
        return self._doctrine

    @property
    def positive_verbalizer(self) -> str:
        return self._positive_verbalizer

    @property
    def negative_verbalizer(self) -> str:
        return self._negative_verbalizer

    def compile(
        self,
        decision: BoolDecision | ChoiceDecision,
        capabilities: BackendCapabilities,
    ) -> InferencePlan:
        """Compile ``decision`` into an :class:`InferencePlan`.

        Raises:
            UnsupportedDecisionError: if ``decision`` is not a
                :class:`~fuzzyai.decisions.BoolDecision`.
            UnsupportedCapabilityError: if the backend does not declare
                ``binary_token_logits``.
        """
        if not isinstance(decision, BoolDecision):
            raise UnsupportedDecisionError(
                f"BoolCompiler supports only BoolDecision, got {type(decision).__name__}"
            )
        if not capabilities.binary_token_logits:
            raise UnsupportedCapabilityError(
                "backend does not declare required capability 'binary_token_logits' "
                f"(declared: {capabilities.binary_token_logits!r})"
            )
        system_prompt, user_prompt = self._doctrine.render(
            question=decision.question,
            context=decision.context,
            positive_verbalizer=self._positive_verbalizer,
            negative_verbalizer=self._negative_verbalizer,
        )
        return InferencePlan(
            decision_fingerprint=decision.fingerprint,
            strategy=ScoringStrategy.BINARY_TOKEN_LOGITS,
            prompt=user_prompt,
            targets=(),
            system_prompt=system_prompt,
            positive_verbalizer=self._positive_verbalizer,
            negative_verbalizer=self._negative_verbalizer,
            doctrine_id=self._doctrine.doctrine_id,
            required_capabilities=BackendCapabilities(binary_token_logits=True),
            compiler_id=BINARY_COMPILER_ID,
            compiler_version=BINARY_COMPILER_VERSION,
            assembler_id=BINARY_ASSEMBLER_ID,
            assembler_version=BINARY_ASSEMBLER_VERSION,
        )


class ChoiceCompiler:
    """Compiles :class:`~fuzzyai.decisions.ChoiceDecision` objects into plans.

    Each semantic candidate receives the scoring label at its candidate index
    from :data:`CATEGORICAL_LABELS`. The compiler never consults a tokenizer,
    a model, or a token id: whether a label resolves to exactly one token is
    the backend's validation, not the compiler's.
    """

    def __init__(
        self,
        *,
        doctrine: CategoricalScoringDoctrine = CATEGORICAL_SEMANTIC_JUDGMENT_V1,
    ) -> None:
        if not isinstance(doctrine, CategoricalScoringDoctrine):
            raise InvalidDecisionError(
                f"doctrine must be a CategoricalScoringDoctrine, got {type(doctrine).__name__}"
            )
        self._doctrine = doctrine

    @property
    def doctrine(self) -> CategoricalScoringDoctrine:
        return self._doctrine

    def compile(
        self,
        decision: BoolDecision | ChoiceDecision,
        capabilities: BackendCapabilities,
    ) -> InferencePlan:
        """Compile ``decision`` into an :class:`InferencePlan`.

        Raises:
            UnsupportedDecisionError: if ``decision`` is not a
                :class:`~fuzzyai.decisions.ChoiceDecision`.
            UnsupportedCapabilityError: if the backend does not declare
                ``categorical_token_logits``.
            InvalidDecisionError: if the decision has more candidates than
                :data:`CATEGORICAL_LABELS` provides.
        """
        if not isinstance(decision, ChoiceDecision):
            raise UnsupportedDecisionError(
                f"ChoiceCompiler supports only ChoiceDecision, got {type(decision).__name__}"
            )
        if not capabilities.categorical_token_logits:
            raise UnsupportedCapabilityError(
                "backend does not declare required capability 'categorical_token_logits' "
                f"(declared: {capabilities.categorical_token_logits!r})"
            )
        choices = decision.choices
        if len(choices) > len(CATEGORICAL_LABELS):
            raise InvalidDecisionError(
                f"a choice decision with {len(choices)} candidates exceeds the "
                f"{CATEGORICAL_LABEL_SCHEME_ID} scheme, which provides at most "
                f"{len(CATEGORICAL_LABELS)} scoring labels"
            )
        targets = CATEGORICAL_LABELS[: len(choices)]
        candidate_mapping = tuple(
            CandidateLabelMapping(
                candidate_index=index,
                candidate_name=choice.name,
                candidate_description=choice.description,
                scoring_label=targets[index],
            )
            for index, choice in enumerate(choices)
        )
        mapping_lines = [
            f"{entry.scoring_label} = {entry.candidate_name}"
            if entry.candidate_description is None
            else f"{entry.scoring_label} = {entry.candidate_name} ({entry.candidate_description})"
            for entry in candidate_mapping
        ]
        system_prompt, user_prompt = self._doctrine.render(
            question=decision.question,
            context=decision.context,
            mapping_lines=mapping_lines,
        )
        return InferencePlan(
            decision_fingerprint=decision.fingerprint,
            strategy=ScoringStrategy.CATEGORICAL_TOKEN_LOGITS,
            prompt=user_prompt,
            targets=targets,
            system_prompt=system_prompt,
            doctrine_id=self._doctrine.doctrine_id,
            label_scheme_id=CATEGORICAL_LABEL_SCHEME_ID,
            candidate_mapping=candidate_mapping,
            required_capabilities=BackendCapabilities(categorical_token_logits=True),
            compiler_id=CATEGORICAL_COMPILER_ID,
            compiler_version=CATEGORICAL_COMPILER_VERSION,
            assembler_id=CATEGORICAL_ASSEMBLER_ID,
            assembler_version=CATEGORICAL_ASSEMBLER_VERSION,
        )


__all__ = [
    "BINARY_COMPILER_ID",
    "BINARY_COMPILER_VERSION",
    "CATEGORICAL_COMPILER_ID",
    "CATEGORICAL_COMPILER_VERSION",
    "CATEGORICAL_LABELS",
    "CATEGORICAL_LABEL_SCHEME_ID",
    "DEFAULT_NEGATIVE_VERBALIZER",
    "DEFAULT_POSITIVE_VERBALIZER",
    "BoolCompiler",
    "ChoiceCompiler",
]
