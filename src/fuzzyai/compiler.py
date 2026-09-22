"""BoolCompiler: compiles a :class:`~fuzzyai.decisions.BoolDecision` into an
:class:`~fuzzyai.plans.InferencePlan`.

The compiler is a pure planner. It NEVER computes probabilities, never sees
evidence, and never touches a backend. Its only job is to turn a decision
plus the backend's declared capabilities into a fully-specified plan.
"""

from fuzzyai.capabilities import BackendCapabilities
from fuzzyai.decisions import BoolDecision, ChoiceDecision
from fuzzyai.doctrine import BINARY_SEMANTIC_JUDGMENT_V1, ScoringDoctrine
from fuzzyai.errors import (
    InvalidDecisionError,
    UnsupportedCapabilityError,
    UnsupportedDecisionError,
)
from fuzzyai.plans import InferencePlan, ScoringStrategy

BINARY_COMPILER_VERSION = 1
DEFAULT_POSITIVE_VERBALIZER = "yes"
DEFAULT_NEGATIVE_VERBALIZER = "no"


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
        )


__all__ = [
    "BINARY_COMPILER_VERSION",
    "DEFAULT_NEGATIVE_VERBALIZER",
    "DEFAULT_POSITIVE_VERBALIZER",
    "BoolCompiler",
]
