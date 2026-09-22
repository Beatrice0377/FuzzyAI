"""DecisionTrace: replay-oriented provenance for one evaluated decision.

A trace links the decision, the plan, the raw evidence, the scoring
diagnostics, and the resulting result together. It is built exclusively from
values already present on those objects (plus backend-reported metadata) —
nothing is re-derived, and the trace id is never derived from any fingerprint.

This is replay-oriented provenance: it records what a future replay would
need, but it is NOT a claim of strict full replayability. Not every artifact
that affects inference is captured yet — there is no backend or tokenizer code
snapshot, and no retention mode.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from fuzzyai.diagnostics import ScoringDiagnostics
from fuzzyai.errors import InvalidDecisionError
from fuzzyai.fingerprint import JSONValue, canonical_json, fingerprint
from fuzzyai.plans import InferencePlan, RawEvidence
from fuzzyai.results import BoolResult

POSITIVE_TOKEN_ID_KEY = "positive_token_id"
NEGATIVE_TOKEN_ID_KEY = "negative_token_id"
MODEL_KEY = "model"
MODEL_REVISION_KEY = "model_revision"
INPUT_TOKEN_COUNT_KEY = "input_token_count"
RENDERED_INPUT_KEY = "rendered_input"
BACKEND_VERSION_KEY = "backend_version"
TOKENIZER_KEY = "tokenizer"
TOKENIZER_REVISION_KEY = "tokenizer_revision"
RUNTIME_VERSION_KEY = "runtime_version"
DTYPE_KEY = "dtype"
RENDERING_CONFIG_KEY = "rendering_config"

EXECUTION_FINGERPRINT_VERSION = 1

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
    scoring_diagnostics: ScoringDiagnostics
    execution_fingerprint: str
    model: str | None = None
    model_revision: str | None = None
    input_token_count: int | None = None
    rendered_input: str | None = None
    backend_version: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    runtime_version: str | None = None
    dtype: str | None = None
    rendering_config: Mapping[str, JSONValue] = field(default_factory=dict)

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
            "scoring_diagnostics": {
                "verbalizer_mass": self.scoring_diagnostics.verbalizer_mass,
                "top_token_id": self.scoring_diagnostics.top_token_id,
                "top_token_probability": self.scoring_diagnostics.top_token_probability,
                "positive_token_probability": (self.scoring_diagnostics.positive_token_probability),
                "negative_token_probability": (self.scoring_diagnostics.negative_token_probability),
                "top_token_text": self.scoring_diagnostics.top_token_text,
            },
            "execution_fingerprint": self.execution_fingerprint,
            "model": self.model,
            "model_revision": self.model_revision,
            "input_token_count": self.input_token_count,
            "rendered_input": self.rendered_input,
            "backend_version": self.backend_version,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
            "runtime_version": self.runtime_version,
            "dtype": self.dtype,
            "rendering_config": dict(self.rendering_config),
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


def _metadata_rendering_config(evidence: RawEvidence) -> dict[str, JSONValue]:
    """Rendering config from evidence metadata; ``{}`` when absent."""
    value = evidence.metadata.get(RENDERING_CONFIG_KEY)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidDecisionError(
            f"evidence metadata key {RENDERING_CONFIG_KEY!r} must be a mapping, "
            f"got {type(value).__name__} ({value!r})"
        )
    config = dict(value)
    try:
        canonical_json(config)
    except Exception as exc:
        raise InvalidDecisionError(
            f"evidence metadata key {RENDERING_CONFIG_KEY!r} must be JSON-compatible: {exc}"
        ) from exc
    return config


def build_decision_trace(
    *,
    trace_id: str,
    timestamp: str,
    decision_fingerprint: str,
    plan: InferencePlan,
    evidence: RawEvidence,
    result: BoolResult,
    diagnostics: ScoringDiagnostics,
    backend_type: str,
    latency_ms: float,
    capture_rendered_input: bool = False,
) -> DecisionTrace:
    """Assemble a :class:`DecisionTrace` from a plan, its evidence, and result.

    Verbalizers and the doctrine id are taken from the plan; scoring token
    ids are REQUIRED in ``evidence.metadata`` (``positive_token_id`` /
    ``negative_token_id`` as ints). ``model``, ``model_revision``,
    ``input_token_count``, ``backend_version``, ``tokenizer``,
    ``tokenizer_revision``, ``runtime_version`` and ``dtype`` are optional
    metadata. ``rendered_input`` is captured only when
    ``capture_rendered_input`` is True. ``rendering_config`` is optional
    metadata that enters the execution fingerprint (it changes the real model
    input) but never the plan or its fingerprint. ``diagnostics`` is the
    REQUIRED :class:`ScoringDiagnostics` derived from the same evidence.

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
    model = _metadata_str(evidence, MODEL_KEY)
    model_revision = _metadata_str(evidence, MODEL_REVISION_KEY)
    backend_version = _metadata_str(evidence, BACKEND_VERSION_KEY)
    tokenizer = _metadata_str(evidence, TOKENIZER_KEY)
    tokenizer_revision = _metadata_str(evidence, TOKENIZER_REVISION_KEY)
    runtime_version = _metadata_str(evidence, RUNTIME_VERSION_KEY)
    dtype = _metadata_str(evidence, DTYPE_KEY)
    rendering_config = _metadata_rendering_config(evidence)
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
    # The execution fingerprint identifies one execution configuration: the
    # plan plus everything about HOW it was executed (backend, versions,
    # rendering config). It is derived from the already-computed
    # ``input_fingerprint`` — the actually rendered text — never re-derived
    # from the plan.
    execution_fingerprint = fingerprint(
        {
            "v": EXECUTION_FINGERPRINT_VERSION,
            "kind": "execution",
            "plan_fingerprint": plan.fingerprint,
            "backend_type": backend_type,
            "backend_version": backend_version,
            "model": model,
            "model_revision": model_revision,
            "tokenizer": tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "runtime_version": runtime_version,
            "dtype": dtype,
            "rendering_config": rendering_config,
            "input_fingerprint": input_fingerprint,
            "positive_token_id": positive_token_id,
            "negative_token_id": negative_token_id,
        }
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
        scoring_diagnostics=diagnostics,
        execution_fingerprint=execution_fingerprint,
        model=model,
        model_revision=model_revision,
        input_token_count=_metadata_optional_int(evidence, INPUT_TOKEN_COUNT_KEY),
        rendered_input=rendered_input,
        backend_version=backend_version,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        runtime_version=runtime_version,
        dtype=dtype,
        rendering_config=rendering_config,
    )
