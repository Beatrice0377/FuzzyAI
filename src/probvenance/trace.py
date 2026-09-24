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

from probvenance.diagnostics import ChoiceScoringDiagnostics, ScoringDiagnostics
from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, canonical_json, fingerprint
from probvenance.plans import CandidateLabelMapping, InferencePlan, RawEvidence, ScoringStrategy
from probvenance.results import BoolResult, DecisionResult, _validate_calibration_provenance

POSITIVE_TOKEN_ID_KEY = "positive_token_id"
NEGATIVE_TOKEN_ID_KEY = "negative_token_id"
RESOLVED_TARGET_TOKEN_IDS_KEY = "resolved_target_token_ids"
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

EXECUTION_FINGERPRINT_VERSION = 2

_REQUIRED_BINARY_TOKEN_METADATA_KEYS: tuple[str, ...] = (
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
    """Immutable provenance record of one evaluated decision.

    ``scoring_diagnostics`` is a :class:`~probvenance.diagnostics.ScoringDiagnostics`
    for the binary strategy or a
    :class:`~probvenance.diagnostics.ChoiceScoringDiagnostics` for the categorical
    strategy; consumers must narrow on ``trace.scoring_strategy``.
    ``positive_token_id`` / ``negative_token_id`` are the binary scoring token
    ids (``-1`` when unused); ``resolved_target_token_ids`` carries the
    categorical ``(label, token_id)`` pairs in target order (empty when
    unused). ``candidate_mapping`` mirrors the plan's mapping.
    ``compiler_id`` / ``compiler_version`` and ``assembler_id`` /
    ``assembler_version`` mirror the plan's provenance, so lineage is readable
    without looking up historical code. ``plan_fingerprint_version`` records
    which canonical payload schema produced ``plan_fingerprint``, so a stored
    hash self-describes its version instead of needing historical code.
    ``probability_formulation_fingerprint`` and ``formulation_family_fingerprint``
    (each with its schema version) are the derived provider-independent
    formulation identities read from the plan; they exclude source and evidence
    and are deliberately not part of any fingerprint payload. Neither is a
    statement about comparability or calibration.
    ``probability_true`` is the binary ``P(True)`` and ``None`` for
    categorical decisions, which have no true/false outcome space.
    ``calibration_profile_fingerprint`` and
    ``calibration_profile_fingerprint_version`` identify the exact calibration
    artifact used for the returned result, and are ``None`` for an uncalibrated
    trace. They are primitive identity mirrors of the result's own provenance,
    so this module never imports the calibration layer. They do NOT mean the
    profile is valid for every population, that it improves calibration, or
    that the training data was independent.

    There is deliberately no whole-``DecisionTrace`` serialization schema
    version: ``to_dict`` is a stable key set, and
    ``execution_fingerprint`` (``EXECUTION_FINGERPRINT_VERSION``) identifies
    execution semantics only, so it is never bumped by calibration provenance.
    """

    trace_id: str
    timestamp: str
    decision_fingerprint: str
    plan_fingerprint: str
    plan_fingerprint_version: int
    probability_formulation_fingerprint: str
    probability_formulation_fingerprint_version: int
    formulation_family_fingerprint: str
    formulation_family_fingerprint_version: int
    scoring_strategy: str
    decision_family: str
    doctrine_id: str | None
    doctrine_version: int | None
    positive_verbalizer: str
    negative_verbalizer: str
    positive_token_id: int
    negative_token_id: int
    evidence: RawEvidence
    probability_true: float | None
    input_fingerprint: str
    backend_type: str
    latency_ms: float
    scoring_diagnostics: ScoringDiagnostics | ChoiceScoringDiagnostics
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
    candidate_mapping: tuple[CandidateLabelMapping, ...] = ()
    resolved_target_token_ids: tuple[tuple[str, int], ...] = ()
    compiler_id: str | None = None
    compiler_version: int | None = None
    assembler_id: str | None = None
    assembler_version: int | None = None
    calibration_profile_fingerprint: str | None = None
    calibration_profile_fingerprint_version: int | None = None

    def __post_init__(self) -> None:
        _validate_calibration_provenance(
            self.calibration_profile_fingerprint,
            self.calibration_profile_fingerprint_version,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """A JSON-compatible plain dict of every field (evidence nested)."""
        payload: dict[str, JSONValue] = {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "decision_fingerprint": self.decision_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "plan_fingerprint_version": self.plan_fingerprint_version,
            "probability_formulation_fingerprint": self.probability_formulation_fingerprint,
            "probability_formulation_fingerprint_version": (
                self.probability_formulation_fingerprint_version
            ),
            "formulation_family_fingerprint": self.formulation_family_fingerprint,
            "formulation_family_fingerprint_version": self.formulation_family_fingerprint_version,
            "scoring_strategy": self.scoring_strategy,
            "decision_family": self.decision_family,
            "doctrine_id": self.doctrine_id,
            "doctrine_version": self.doctrine_version,
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
            "scoring_diagnostics": _diagnostics_to_dict(self.scoring_diagnostics),
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
            "candidate_mapping": [
                {
                    "candidate_index": entry.candidate_index,
                    "candidate_name": entry.candidate_name,
                    "candidate_description": entry.candidate_description,
                    "scoring_label": entry.scoring_label,
                }
                for entry in self.candidate_mapping
            ],
            "resolved_target_token_ids": [
                [label, token_id] for label, token_id in self.resolved_target_token_ids
            ],
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "assembler_id": self.assembler_id,
            "assembler_version": self.assembler_version,
            "calibration_profile_fingerprint": self.calibration_profile_fingerprint,
            "calibration_profile_fingerprint_version": (
                self.calibration_profile_fingerprint_version
            ),
        }
        return payload


def _diagnostics_to_dict(
    diagnostics: ScoringDiagnostics | ChoiceScoringDiagnostics,
) -> dict[str, JSONValue]:
    """A JSON dict for either diagnostics variant, tagged by kind."""
    if isinstance(diagnostics, ChoiceScoringDiagnostics):
        return {
            "kind": "choice",
            "scoring_label_mass": diagnostics.scoring_label_mass,
            "scoring_label_token_probabilities": list(
                diagnostics.scoring_label_token_probabilities
            ),
            "top_token_id": diagnostics.top_token_id,
            "top_token_probability": diagnostics.top_token_probability,
            "top_token_text": diagnostics.top_token_text,
        }
    return {
        "kind": "binary",
        "verbalizer_mass": diagnostics.verbalizer_mass,
        "top_token_id": diagnostics.top_token_id,
        "top_token_probability": diagnostics.top_token_probability,
        "positive_token_probability": diagnostics.positive_token_probability,
        "negative_token_probability": diagnostics.negative_token_probability,
        "top_token_text": diagnostics.top_token_text,
    }


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


def _metadata_resolved_target_token_ids(
    evidence: RawEvidence,
) -> tuple[tuple[str, int], ...]:
    """Resolved ``(label, token_id)`` pairs from evidence metadata.

    The backend reports them as a JSON list of ``[label, token_id]`` pairs in
    ``plan.targets`` order (e.g. ``[["A", 123], ["B", 456]]``).
    """
    value = evidence.metadata.get(RESOLVED_TARGET_TOKEN_IDS_KEY)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidDecisionError(
            f"evidence metadata key {RESOLVED_TARGET_TOKEN_IDS_KEY!r} must be a list of "
            f"[label, token_id] pairs, got {type(value).__name__} ({value!r})"
        )
    resolved: list[tuple[str, int]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise InvalidDecisionError(
                f"evidence metadata key {RESOLVED_TARGET_TOKEN_IDS_KEY!r} entry {index} "
                f"must be a [label, token_id] pair, got {pair!r}"
            )
        label, token_id = pair
        if not isinstance(label, str) or not label:
            raise InvalidDecisionError(
                f"evidence metadata key {RESOLVED_TARGET_TOKEN_IDS_KEY!r} entry {index} "
                f"label must be a non-empty string, got {label!r}"
            )
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise InvalidDecisionError(
                f"evidence metadata key {RESOLVED_TARGET_TOKEN_IDS_KEY!r} entry {index} "
                f"token_id must be a non-negative int, got {token_id!r}"
            )
        resolved.append((label, token_id))
    return tuple(resolved)


def build_decision_trace(
    *,
    trace_id: str,
    timestamp: str,
    decision_fingerprint: str,
    plan: InferencePlan,
    evidence: RawEvidence,
    result: DecisionResult,
    diagnostics: ScoringDiagnostics | ChoiceScoringDiagnostics,
    backend_type: str,
    latency_ms: float,
    capture_rendered_input: bool = False,
) -> DecisionTrace:
    """Assemble a :class:`DecisionTrace` from a plan, its evidence, and result.

    For the binary strategy, verbalizers and the doctrine id are taken from
    the plan and the scoring token ids are REQUIRED in ``evidence.metadata``
    (``positive_token_id`` / ``negative_token_id`` as ints). For the
    categorical strategy, ``resolved_target_token_ids`` is REQUIRED in
    ``evidence.metadata`` (a JSON list of ``[label, token_id]`` pairs in
    ``plan.targets`` order) and the verbalizers are ``"none"``. ``model``,
    ``model_revision``, ``input_token_count``, ``backend_version``,
    ``tokenizer``, ``tokenizer_revision``, ``runtime_version`` and ``dtype``
    are optional metadata. ``rendered_input`` is captured only when
    ``capture_rendered_input`` is True. ``rendering_config`` is optional
    metadata that enters the execution fingerprint (it changes the real model
    input) but never the plan or its fingerprint. ``diagnostics`` is the
    REQUIRED scoring diagnostics derived from the same evidence, matching the
    plan's strategy.

    Raises:
        InvalidDecisionError: if required metadata keys are missing or of the
            wrong type, or the diagnostics variant does not match the plan
            strategy.
    """
    if plan.strategy is ScoringStrategy.BINARY_TOKEN_LOGITS:
        for key in _REQUIRED_BINARY_TOKEN_METADATA_KEYS:
            if key not in evidence.metadata:
                raise InvalidDecisionError(f"evidence metadata is missing required key {key!r}")
            _ = _metadata_int(evidence, key)
        positive_token_id = _metadata_int(evidence, POSITIVE_TOKEN_ID_KEY)
        negative_token_id = _metadata_int(evidence, NEGATIVE_TOKEN_ID_KEY)
        resolved_target_token_ids: tuple[tuple[str, int], ...] = ()
        if not isinstance(diagnostics, ScoringDiagnostics):
            raise InvalidDecisionError(
                f"binary plans require ScoringDiagnostics, got {type(diagnostics).__name__}"
            )
    elif plan.strategy is ScoringStrategy.CATEGORICAL_TOKEN_LOGITS:
        if RESOLVED_TARGET_TOKEN_IDS_KEY not in evidence.metadata:
            raise InvalidDecisionError(
                "evidence metadata is missing required key "
                f"{RESOLVED_TARGET_TOKEN_IDS_KEY!r}: categorical traces need the "
                "resolved scoring token ids"
            )
        resolved_target_token_ids = _metadata_resolved_target_token_ids(evidence)
        positive_token_id = -1
        negative_token_id = -1
        if not isinstance(diagnostics, ChoiceScoringDiagnostics):
            raise InvalidDecisionError(
                "categorical plans require ChoiceScoringDiagnostics, got "
                f"{type(diagnostics).__name__}"
            )
    else:
        raise InvalidDecisionError(
            f"unsupported scoring strategy for tracing: {plan.strategy.value!r}"
        )
    rendered_input: str | None = None
    if capture_rendered_input:
        rendered_input = _metadata_str(evidence, RENDERED_INPUT_KEY)
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
    # rendering config, resolved scoring token ids). It is derived from the
    # already-computed ``input_fingerprint`` — the actually rendered text —
    # never re-derived from the plan.
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
            "resolved_target_token_ids": [
                [label, token_id] for label, token_id in resolved_target_token_ids
            ],
        }
    )
    return DecisionTrace(
        trace_id=trace_id,
        timestamp=timestamp,
        decision_fingerprint=decision_fingerprint,
        plan_fingerprint=plan.fingerprint,
        plan_fingerprint_version=plan.fingerprint_version,
        probability_formulation_fingerprint=plan.probability_formulation_fingerprint,
        probability_formulation_fingerprint_version=(
            plan.probability_formulation_fingerprint_version
        ),
        formulation_family_fingerprint=plan.formulation_family_fingerprint,
        formulation_family_fingerprint_version=plan.formulation_family_fingerprint_version,
        scoring_strategy=str(plan.strategy.value),
        decision_family=plan.decision_family,
        doctrine_id=plan.doctrine_id,
        doctrine_version=plan.doctrine_version,
        positive_verbalizer=plan.positive_verbalizer
        if plan.positive_verbalizer is not None
        else "none",
        negative_verbalizer=plan.negative_verbalizer
        if plan.negative_verbalizer is not None
        else "none",
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        evidence=evidence,
        probability_true=result.probability_true if isinstance(result, BoolResult) else None,
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
        candidate_mapping=plan.candidate_mapping,
        resolved_target_token_ids=resolved_target_token_ids,
        compiler_id=plan.compiler_id,
        compiler_version=plan.compiler_version,
        assembler_id=plan.assembler_id,
        assembler_version=plan.assembler_version,
    )
