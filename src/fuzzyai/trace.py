"""DecisionTrace: a complete provenance record of one evaluated decision.

A trace links the decision, the plan, the raw evidence, and the resulting
result together. It is built exclusively from values already present on those
objects (plus backend-reported metadata) — nothing is re-derived, and the
trace id is never derived from any fingerprint.
"""

from dataclasses import dataclass

from fuzzyai.errors import InvalidDecisionError
from fuzzyai.fingerprint import JSONValue, fingerprint
from fuzzyai.plans import InferencePlan, RawEvidence
from fuzzyai.results import BoolResult

POSITIVE_TOKEN_ID_KEY = "positive_token_id"
NEGATIVE_TOKEN_ID_KEY = "negative_token_id"
MODEL_KEY = "model"
MODEL_REVISION_KEY = "model_revision"
INPUT_TOKEN_COUNT_KEY = "input_token_count"
RENDERED_INPUT_KEY = "rendered_input"

_REQUIRED_TOKEN_METADATA_KEYS: tuple[str, ...] = (
    POSITIVE_TOKEN_ID_KEY,
    NEGATIVE_TOKEN_ID_KEY,
)


def _metadata_int(evidence: RawEvidence, key: str) -> int:
    value = evidence.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidDecisionError(
            f"evidence metadata key {key!r} must be an int, got {type(value).__name__} ({value!r})"
        )
    return value


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Immutable provenance record of one evaluated decision."""

    trace_id: str
    timestamp: str
    decision_fingerprint: str
    plan_fingerprint: str
    scoring_strategy: str
    doctrine_id: str
    positive_verbalizer: str
    negative_verbalizer: str
    positive_token_id: int
    negative_token_id: int
    evidence: RawEvidence
    probability_true: float
    input_fingerprint: str
    backend_type: str
    latency_ms: float
    model: str | None = None
    model_revision: str | None = None
    input_token_count: int | None = None
    rendered_input: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """A JSON-compatible plain dict of every field (evidence nested)."""
        payload: dict[str, JSONValue] = {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "decision_fingerprint": self.decision_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "scoring_strategy": self.scoring_strategy,
            "doctrine_id": self.doctrine_id,
            "positive_verbalizer": self.positive_verbalizer,
            "negative_verbalizer": self.negative_verbalizer,
            "positive_token_id": self.positive_token_id,
            "negative_token_id": self.negative_token_id,
            "evidence": {
                "kind": str(self.evidence.kind.value),
                "labels": list(self.evidence.labels),
                "values": list(self.evidence.values),
                "plan_fingerprint": self.evidence.plan_fingerprint,
                "metadata": dict(self.evidence.metadata),
            },
            "probability_true": self.probability_true,
            "input_fingerprint": self.input_fingerprint,
            "backend_type": self.backend_type,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "model_revision": self.model_revision,
            "input_token_count": self.input_token_count,
            "rendered_input": self.rendered_input,
        }
        return payload


def _metadata_str(evidence: RawEvidence, key: str) -> str | None:
    value: JSONValue | None = evidence.metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidDecisionError(
            f"evidence metadata key {key!r} must be a string, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _metadata_optional_int(evidence: RawEvidence, key: str) -> int | None:
    value = evidence.metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidDecisionError(
            f"evidence metadata key {key!r} must be an int, got {type(value).__name__} ({value!r})"
        )
    return value


def build_decision_trace(
    *,
    trace_id: str,
    timestamp: str,
    decision_fingerprint: str,
    plan: InferencePlan,
    evidence: RawEvidence,
    result: BoolResult,
    backend_type: str,
    latency_ms: float,
    capture_rendered_input: bool = False,
) -> DecisionTrace:
    """Assemble a :class:`DecisionTrace` from a plan, its evidence, and result.

    Verbalizers and the doctrine id are taken from the plan; scoring token
    ids are REQUIRED in ``evidence.metadata`` (``positive_token_id`` /
    ``negative_token_id`` as ints). ``model``, ``model_revision`` and
    ``input_token_count`` are optional metadata. ``rendered_input`` is
    captured only when ``capture_rendered_input`` is True.

    Raises:
        InvalidDecisionError: if required metadata keys are missing or of the
            wrong type.
    """
    for key in _REQUIRED_TOKEN_METADATA_KEYS:
        if key not in evidence.metadata:
            raise InvalidDecisionError(f"evidence metadata is missing required key {key!r}")
        _ = _metadata_int(evidence, key)
    rendered_input: str | None = None
    if capture_rendered_input:
        rendered_input = _metadata_str(evidence, RENDERED_INPUT_KEY)
    positive_token_id = _metadata_int(evidence, POSITIVE_TOKEN_ID_KEY)
    negative_token_id = _metadata_int(evidence, NEGATIVE_TOKEN_ID_KEY)
    # Fingerprint the text ACTUALLY fed to the model, not the plan: two
    # renderings of one plan are different inputs. Read independently of
    # ``capture_rendered_input`` — that flag only gates copying the text onto
    # the trace, never the fingerprint.
    rendered_for_fingerprint = _metadata_str(evidence, RENDERED_INPUT_KEY)
    if rendered_for_fingerprint is not None:
        input_fingerprint = fingerprint({"rendered_input": rendered_for_fingerprint})
    else:
        # Fallback for evidence without a rendered input: plan fields, as before.
        input_fingerprint = fingerprint(
            {"system_prompt": plan.system_prompt, "prompt": plan.prompt}
        )
    return DecisionTrace(
        trace_id=trace_id,
        timestamp=timestamp,
        decision_fingerprint=decision_fingerprint,
        plan_fingerprint=plan.fingerprint,
        scoring_strategy=str(plan.strategy.value),
        doctrine_id=plan.doctrine_id if plan.doctrine_id is not None else "none",
        positive_verbalizer=plan.positive_verbalizer
        if plan.positive_verbalizer is not None
        else "none",
        negative_verbalizer=plan.negative_verbalizer
        if plan.negative_verbalizer is not None
        else "none",
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        evidence=evidence,
        probability_true=result.probability_true,
        input_fingerprint=input_fingerprint,
        backend_type=backend_type,
        latency_ms=latency_ms,
        model=_metadata_str(evidence, MODEL_KEY),
        model_revision=_metadata_str(evidence, MODEL_REVISION_KEY),
        input_token_count=_metadata_optional_int(evidence, INPUT_TOKEN_COUNT_KEY),
        rendered_input=rendered_input,
    )
