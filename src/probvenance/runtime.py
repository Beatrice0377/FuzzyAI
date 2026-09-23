"""Probvenance runtime facade: the one-object entry point for evaluation.

The runtime orchestrates the existing pieces — compiler, backend, assembler,
trace builder — in a fixed order. It re-implements none of them and adds no
fallbacks: any unsupported decision or missing capability propagates as-is.

Dispatch is explicit on the decision type: :class:`BoolDecision` takes the
binary path, :class:`ChoiceDecision` takes the categorical path, and anything
else is rejected. Assembler selection is separate: the plan's declared
``(strategy, assembler_id, assembler_version)`` tuple selects the exact
implementation, so what runs is what the plan names.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from probvenance.assembler import assemble_probability
from probvenance.backends.base import Backend
from probvenance.compiler import BoolCompiler, ChoiceCompiler
from probvenance.decisions import BoolDecision, ChoiceDecision
from probvenance.errors import InvalidDecisionError, UnsupportedDecisionError
from probvenance.plans import InferencePlan, RawEvidence
from probvenance.results import BoolResult, ChoiceResult
from probvenance.trace import DecisionTrace, build_decision_trace


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One evaluated decision: its result plus the full provenance trace."""

    result: BoolResult | ChoiceResult
    trace: DecisionTrace


class Probvenance:
    """Evaluates decisions against a backend, end to end, with provenance."""

    def __init__(
        self,
        *,
        backend: Backend,
        compiler: BoolCompiler | None = None,
        choice_compiler: ChoiceCompiler | None = None,
        capture_rendered_input: bool = False,
    ) -> None:
        self._backend = backend
        self._compiler = compiler if compiler is not None else BoolCompiler()
        self._choice_compiler = choice_compiler if choice_compiler is not None else ChoiceCompiler()
        self._capture_rendered_input = capture_rendered_input

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def compiler(self) -> BoolCompiler:
        return self._compiler

    @property
    def choice_compiler(self) -> ChoiceCompiler:
        return self._choice_compiler

    @property
    def capture_rendered_input(self) -> bool:
        return self._capture_rendered_input

    def evaluate(self, decision: BoolDecision | ChoiceDecision) -> BoolResult | ChoiceResult:
        """Evaluate ``decision`` and return only its result."""
        return self.evaluate_with_trace(decision).result

    def evaluate_with_trace(self, decision: BoolDecision | ChoiceDecision) -> Evaluation:
        """Evaluate ``decision`` and return result + trace, in a fixed order.

        The exact order:

        1. Generate a fresh ``trace_id = str(uuid.uuid4())``.
        2. Dispatch on the decision type and compile it into a plan (the
           compiler checks capabilities).
        3. Execute the plan on the backend, measuring latency.
        4. Check evidence/plan lineage: evidence that does not carry the
           executed plan's fingerprint — either a DIFFERENT fingerprint or
           none at all — is an ``InvalidDecisionError``.
        5. Assemble the decision result and the scoring diagnostics from the
           evidence through the assembler the plan declares, stamped with the
           trace id.
        6. Build the decision trace.
        7. Return the :class:`Evaluation`.

        Raises:
            UnsupportedDecisionError: if the decision type is neither
                :class:`BoolDecision` nor :class:`ChoiceDecision`.
            UnsupportedCapabilityError: if the backend lacks a required
                capability.
            InvalidDecisionError: if the evidence does not carry the plan's
                fingerprint (missing or mismatched).
            UnsupportedAssemblerError: if the plan's assembler declaration
                matches no known implementation.
        """
        # 1. Fresh trace id, never derived from any fingerprint.
        trace_id = str(uuid.uuid4())
        # 2. Explicit type dispatch; no hasattr, no duck typing.
        if isinstance(decision, BoolDecision):
            plan: InferencePlan = self._compiler.compile(decision, self._backend.capabilities)
        elif isinstance(decision, ChoiceDecision):
            plan = self._choice_compiler.compile(decision, self._backend.capabilities)
        else:
            raise UnsupportedDecisionError(
                f"Probvenance supports only BoolDecision and ChoiceDecision, "
                f"got {type(decision).__name__}"
            )
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
        # 5. Assemble through the exact implementation the plan declares; an
        #    unknown or mismatched assembler tuple raises here, before any
        #    probability or trace exists.
        result, diagnostics = assemble_probability(plan, evidence, trace_id=trace_id)
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
