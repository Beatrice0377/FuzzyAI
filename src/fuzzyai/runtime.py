"""FuzzyAI runtime facade: the one-object entry point for evaluation.

The runtime orchestrates the existing pieces — compiler, backend, assembler,
trace builder — in a fixed order. It re-implements none of them and adds no
fallbacks: any unsupported decision or missing capability propagates as-is.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from fuzzyai.assembler import assemble_bool_probability
from fuzzyai.backends.base import Backend
from fuzzyai.compiler import BoolCompiler
from fuzzyai.decisions import BoolDecision, ChoiceDecision
from fuzzyai.diagnostics import diagnose_bool_evidence
from fuzzyai.errors import InvalidDecisionError
from fuzzyai.plans import InferencePlan, RawEvidence
from fuzzyai.results import BoolResult
from fuzzyai.trace import DecisionTrace, build_decision_trace


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One evaluated decision: its result plus the full provenance trace."""

    result: BoolResult
    trace: DecisionTrace


class FuzzyAI:
    """Evaluates decisions against a backend, end to end, with provenance."""

    def __init__(
        self,
        *,
        backend: Backend,
        compiler: BoolCompiler | None = None,
        capture_rendered_input: bool = False,
    ) -> None:
        self._backend = backend
        self._compiler = compiler if compiler is not None else BoolCompiler()
        self._capture_rendered_input = capture_rendered_input

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def compiler(self) -> BoolCompiler:
        return self._compiler

    @property
    def capture_rendered_input(self) -> bool:
        return self._capture_rendered_input

    def evaluate(self, decision: BoolDecision | ChoiceDecision) -> BoolResult:
        """Evaluate ``decision`` and return only its result."""
        return self.evaluate_with_trace(decision).result

    def evaluate_with_trace(self, decision: BoolDecision | ChoiceDecision) -> Evaluation:
        """Evaluate ``decision`` and return result + trace, in a fixed order.

        The exact order:

        1. Generate a fresh ``trace_id = str(uuid.uuid4())``.
        2. Compile the decision into a plan (compiler checks capabilities).
        3. Execute the plan on the backend, measuring latency.
        4. Check evidence/plan lineage: evidence that does not carry the
           executed plan's fingerprint — either a DIFFERENT fingerprint or
           none at all — is an ``InvalidDecisionError``.
        5. Assemble the decision result and the scoring diagnostics from the
           evidence, stamped with the trace id.
        6. Build the decision trace.
        7. Return the :class:`Evaluation`.

        Raises:
            UnsupportedDecisionError: if the compiler does not support the
                decision type.
            UnsupportedCapabilityError: if the backend lacks a required
                capability.
            InvalidDecisionError: if the evidence does not carry the plan's
                fingerprint (missing or mismatched).
        """
        # 1. Fresh trace id, never derived from any fingerprint.
        trace_id = str(uuid.uuid4())
        # 2. Compile (may raise UnsupportedDecisionError / UnsupportedCapabilityError).
        plan: InferencePlan = self._compiler.compile(decision, self._backend.capabilities)
        # 3. Execute and measure latency.
        t0 = perf_counter()
        evidence: RawEvidence = self._backend.execute(plan)
        latency_ms = (perf_counter() - t0) * 1000.0
        # 4. Lineage check: the runtime is the only lineage enforcement point,
        # so evidence without a fingerprint must not pass either.
        if evidence.plan_fingerprint is None:
            raise InvalidDecisionError(
                "evidence carries no plan_fingerprint: the backend must record "
                "the fingerprint of the plan it executed "
                f"(expected {plan.fingerprint!r}, received none)"
            )
        if evidence.plan_fingerprint != plan.fingerprint:
            raise InvalidDecisionError(
                "evidence plan_fingerprint does not match the executed plan: "
                f"expected {plan.fingerprint!r}, got {evidence.plan_fingerprint!r}"
            )
        # 5. Assemble the result and the scoring diagnostics, stamped with
        #    the trace id. The diagnostics are derived from evidence the
        #    backend already produced — no extra model work, no verdict.
        result = assemble_bool_probability(evidence, trace_id=trace_id)
        diagnostics = diagnose_bool_evidence(evidence)
        # 6. Build the trace.
        timestamp = datetime.now(UTC).isoformat()
        trace = build_decision_trace(
            trace_id=trace_id,
            timestamp=timestamp,
            decision_fingerprint=plan.decision_fingerprint,
            plan=plan,
            evidence=evidence,
            result=result,
            diagnostics=diagnostics,
            backend_type=type(self._backend).__name__,
            latency_ms=latency_ms,
            capture_rendered_input=self._capture_rendered_input,
        )
        # 7. Return.
        return Evaluation(result=result, trace=trace)
