"""Contract tests for provider-independent probability formulation identity.

The tests pin the SHAPE of both canonical payloads (which dimensions are
included, which are excluded, and how they are ordered) rather than only the
final hash, so a schema mistake cannot hide behind a stable SHA-256. The
fingerprint relationships between decision, plan, formulation and family
identity are asserted explicitly, because that separation is the whole point.
"""

from collections.abc import Mapping
from dataclasses import replace

from probvenance.assembler import (
    BINARY_ASSEMBLER_ID,
    BINARY_ASSEMBLER_VERSION,
    CATEGORICAL_ASSEMBLER_ID,
    CATEGORICAL_ASSEMBLER_VERSION,
    assemble_bool_probability,
    assemble_choice_probability,
)
from probvenance.capabilities import BackendCapabilities
from probvenance.compiler import (
    BINARY_COMPILER_VERSION,
    CATEGORICAL_COMPILER_VERSION,
    BoolCompiler,
    ChoiceCompiler,
)
from probvenance.decisions import BoolDecision, ChoiceDecision
from probvenance.diagnostics import diagnose_bool_evidence
from probvenance.doctrine import (
    BINARY_DOCTRINE_VERSION,
    CATEGORICAL_DOCTRINE_VERSION,
)
from probvenance.fingerprint import JSONValue, canonical_json
from probvenance.plans import (
    PLAN_FINGERPRINT_VERSION,
    CandidateLabelMapping,
    EvidenceKind,
    InferencePlan,
    RawEvidence,
    ScoringStrategy,
)
from probvenance.probability_identity import (
    FORMULATION_FAMILY_FINGERPRINT_VERSION,
    PROBABILITY_FORMULATION_FINGERPRINT_VERSION,
    formulation_family_fingerprint,
    formulation_family_payload,
    probability_formulation_fingerprint,
    probability_formulation_payload,
)
from probvenance.trace import build_decision_trace

BINARY_CAPS = BackendCapabilities(binary_token_logits=True)
CATEGORICAL_CAPS = BackendCapabilities(categorical_token_logits=True)

EXACT_PAYLOAD_KEYS = {
    "v",
    "kind",
    "decision_family",
    "strategy",
    "assembler",
    "doctrine",
    "compiler",
    "outcome_space",
    "scoring_representation",
}

FAMILY_PAYLOAD_KEYS = {
    "v",
    "kind",
    "decision_family",
    "strategy",
    "assembler",
    "doctrine",
    "compiler",
    "label_scheme_id",
    "arity",
}

# Dimensions that must never appear in a provider-independent formulation
# identity: they are the source axis or execution provenance.
SOURCE_AND_EXECUTION_MARKERS = (
    "model",
    "tokenizer",
    "rendering",
    "enable_thinking",
    "dtype",
    "device",
    "resolved",
    "token_id",
    "input_fingerprint",
    "backend",
    "runtime_version",
    "question",
    "context",
    "prompt",
)


def default_choices() -> dict[str, str]:
    return {
        "billing": "Payment, charges, and invoices",
        "shipping": "Delivery and logistics",
        "technical": "Technical problems",
    }


def bool_plan(
    question: str = "Is this suspicious?",
    context: JSONValue = None,
    *,
    positive_verbalizer: str = "yes",
    negative_verbalizer: str = "no",
) -> InferencePlan:
    decision = BoolDecision(question, context)
    compiler = BoolCompiler(
        positive_verbalizer=positive_verbalizer,
        negative_verbalizer=negative_verbalizer,
    )
    return compiler.compile(decision, BINARY_CAPS)


def choice_plan(
    choices: dict[str, str] | None = None,
    question: str = "Which team owns this ticket?",
    context: JSONValue = "ticket",
) -> InferencePlan:
    decision = ChoiceDecision(question, context, choices=choices or default_choices())
    return ChoiceCompiler().compile(decision, CATEGORICAL_CAPS)


def relabel(plan: InferencePlan, labels: tuple[str, ...]) -> InferencePlan:
    mapping = tuple(
        CandidateLabelMapping(
            candidate_index=entry.candidate_index,
            candidate_name=entry.candidate_name,
            candidate_description=entry.candidate_description,
            scoring_label=label,
        )
        for entry, label in zip(plan.candidate_mapping, labels, strict=True)
    )
    return replace(plan, targets=labels, candidate_mapping=mapping)


def binary_evidence(
    plan: InferencePlan, metadata_overrides: Mapping[str, JSONValue] | None = None
) -> RawEvidence:
    metadata: dict[str, JSONValue] = {
        "positive_token_id": 11,
        "negative_token_id": 22,
        "vocab_logsumexp": 2.0,
        "top_token_id": 7,
        "top_token_logit": 1.0,
        "model": "fake-model",
        "tokenizer": "fake-tokenizer",
    }
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    return RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=("false", "true"),
        values=(0.5, 1.0),
        plan_fingerprint=plan.fingerprint,
        metadata=metadata,
    )


def categorical_evidence(
    plan: InferencePlan, metadata_overrides: Mapping[str, JSONValue] | None = None
) -> RawEvidence:
    metadata: dict[str, JSONValue] = {
        "resolved_target_token_ids": [
            [label, 100 + index] for index, label in enumerate(plan.targets)
        ],
        "vocab_logsumexp": 3.0,
        "top_token_id": 7,
        "top_token_logit": 2.0,
        "model": "fake-model",
        "tokenizer": "fake-tokenizer",
    }
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    return RawEvidence(
        kind=EvidenceKind.LOGITS,
        labels=plan.targets,
        values=(2.0, 1.5, 1.0)[: len(plan.targets)],
        plan_fingerprint=plan.fingerprint,
        metadata=metadata,
    )


def bool_trace(plan: InferencePlan, metadata_overrides: Mapping[str, JSONValue] | None = None):
    evidence = binary_evidence(plan, metadata_overrides)
    return build_decision_trace(
        trace_id="trace-1",
        timestamp="2026-01-01T00:00:00Z",
        decision_fingerprint=plan.decision_fingerprint,
        plan=plan,
        evidence=evidence,
        result=assemble_bool_probability(evidence),
        diagnostics=diagnose_bool_evidence(evidence),
        backend_type="FakeBackend",
        latency_ms=1.0,
    )


def choice_trace(plan: InferencePlan, metadata_overrides: Mapping[str, JSONValue] | None = None):
    evidence = categorical_evidence(plan, metadata_overrides)
    result, diagnostics = assemble_choice_probability(evidence, plan=plan)
    return build_decision_trace(
        trace_id="trace-1",
        timestamp="2026-01-01T00:00:00Z",
        decision_fingerprint=plan.decision_fingerprint,
        plan=plan,
        evidence=evidence,
        result=result,
        diagnostics=diagnostics,
        backend_type="FakeBackend",
        latency_ms=1.0,
    )


class TestPayloadShape:
    def test_exact_payload_has_exactly_the_frozen_dimensions(self) -> None:
        assert set(probability_formulation_payload(choice_plan())) == EXACT_PAYLOAD_KEYS
        assert set(probability_formulation_payload(bool_plan())) == EXACT_PAYLOAD_KEYS

    def test_family_payload_has_exactly_the_frozen_dimensions(self) -> None:
        assert set(formulation_family_payload(choice_plan())) == FAMILY_PAYLOAD_KEYS
        assert set(formulation_family_payload(bool_plan())) == FAMILY_PAYLOAD_KEYS

    def test_payloads_are_json_compatible(self) -> None:
        canonical_json(probability_formulation_payload(choice_plan()))
        canonical_json(formulation_family_payload(choice_plan()))
        canonical_json(probability_formulation_payload(bool_plan()))
        canonical_json(formulation_family_payload(bool_plan()))

    def test_exact_payload_declares_its_kind_and_version(self) -> None:
        payload = probability_formulation_payload(choice_plan())
        assert payload["kind"] == "probability_formulation"
        assert payload["v"] == PROBABILITY_FORMULATION_FINGERPRINT_VERSION

    def test_family_payload_declares_its_kind_and_version(self) -> None:
        payload = formulation_family_payload(choice_plan())
        assert payload["kind"] == "formulation_family"
        assert payload["v"] == FORMULATION_FAMILY_FINGERPRINT_VERSION

    def test_versions_are_independent_constants(self) -> None:
        assert PROBABILITY_FORMULATION_FINGERPRINT_VERSION == 2
        assert FORMULATION_FAMILY_FINGERPRINT_VERSION == 2

    def test_bool_outcome_space_is_explicit_and_ordered(self) -> None:
        payload = probability_formulation_payload(bool_plan())
        assert payload["decision_family"] == "bool"
        assert payload["outcome_space"] == {
            "outcomes": ["false", "true"],
            "closed_set": True,
            "single_label": True,
        }
        assert payload["scoring_representation"] == {
            "positive_verbalizer": "yes",
            "negative_verbalizer": "no",
        }

    def test_choice_outcome_space_keeps_order_names_and_descriptions(self) -> None:
        payload = probability_formulation_payload(choice_plan())
        assert payload["decision_family"] == "choice"
        assert payload["outcome_space"] == {
            "arity": 3,
            "candidates": [
                {"name": "billing", "description": "Payment, charges, and invoices"},
                {"name": "shipping", "description": "Delivery and logistics"},
                {"name": "technical", "description": "Technical problems"},
            ],
            "closed_set": True,
            "single_label": True,
        }

    def test_choice_scoring_representation_keeps_scheme_and_mapping(self) -> None:
        payload = probability_formulation_payload(choice_plan())
        assert payload["scoring_representation"] == {
            "label_scheme_id": "categorical-labels-v1",
            "mapping": [
                {
                    "candidate_index": 0,
                    "candidate_name": "billing",
                    "candidate_description": "Payment, charges, and invoices",
                    "scoring_label": "A",
                },
                {
                    "candidate_index": 1,
                    "candidate_name": "shipping",
                    "candidate_description": "Delivery and logistics",
                    "scoring_label": "B",
                },
                {
                    "candidate_index": 2,
                    "candidate_name": "technical",
                    "candidate_description": "Technical problems",
                    "scoring_label": "C",
                },
            ],
        }

    def test_family_payload_drops_the_representation_instance(self) -> None:
        payload = formulation_family_payload(choice_plan())
        serialized = canonical_json(payload)
        for marker in ("billing", "shipping", "technical", "Payment", "mapping", "candidates"):
            assert marker not in serialized

    def test_exact_payload_contains_no_source_or_evidence_marker(self) -> None:
        for plan in (bool_plan(), choice_plan()):
            serialized = canonical_json(probability_formulation_payload(plan))
            for marker in SOURCE_AND_EXECUTION_MARKERS:
                assert marker not in serialized


class TestExactFormulationBasics:
    def test_same_inputs_produce_the_same_fingerprint(self) -> None:
        assert probability_formulation_fingerprint(choice_plan()) == (
            probability_formulation_fingerprint(choice_plan())
        )
        assert probability_formulation_fingerprint(bool_plan()) == (
            probability_formulation_fingerprint(bool_plan())
        )

    def test_different_assembler_version_changes_the_fingerprint(self) -> None:
        plan = bool_plan()
        other = replace(plan, assembler_version=BINARY_ASSEMBLER_VERSION + 1)
        assert probability_formulation_fingerprint(plan) != probability_formulation_fingerprint(
            other
        )
        assert formulation_family_fingerprint(plan) != formulation_family_fingerprint(other)

    def test_different_assembler_id_changes_the_fingerprint(self) -> None:
        plan = choice_plan()
        other = replace(plan, assembler_id="some-other-assembler")
        assert probability_formulation_fingerprint(plan) != probability_formulation_fingerprint(
            other
        )

    def test_different_compiler_version_changes_the_fingerprint(self) -> None:
        plan = choice_plan()
        other = replace(plan, compiler_version=CATEGORICAL_COMPILER_VERSION + 1)
        assert probability_formulation_fingerprint(plan) != probability_formulation_fingerprint(
            other
        )

    def test_different_doctrine_changes_the_fingerprint(self) -> None:
        plan = choice_plan()
        other = replace(plan, doctrine_id="categorical-semantic-judgment-v2")
        assert probability_formulation_fingerprint(plan) != probability_formulation_fingerprint(
            other
        )

    def test_different_strategy_changes_the_fingerprint(self) -> None:
        plan = bool_plan()
        other = replace(
            plan,
            strategy=ScoringStrategy.TOKEN_LOGPROBS,
            positive_verbalizer=None,
            negative_verbalizer=None,
        )
        assert probability_formulation_payload(other)["decision_family"] == "unknown"
        assert probability_formulation_payload(other)["outcome_space"] is None
        assert probability_formulation_fingerprint(plan) != probability_formulation_fingerprint(
            other
        )

    def test_assembler_provenance_uses_the_real_ids(self) -> None:
        assert probability_formulation_payload(bool_plan())["assembler"] == {
            "id": BINARY_ASSEMBLER_ID,
            "version": BINARY_ASSEMBLER_VERSION,
        }
        assert probability_formulation_payload(choice_plan())["assembler"] == {
            "id": CATEGORICAL_ASSEMBLER_ID,
            "version": CATEGORICAL_ASSEMBLER_VERSION,
        }

    def test_compiler_provenance_carries_versions(self) -> None:
        assert probability_formulation_payload(bool_plan())["compiler"] == {
            "id": "bool-compiler",
            "version": BINARY_COMPILER_VERSION,
        }

    def test_plan_fingerprint_schema_is_not_bumped_by_derived_identities(self) -> None:
        assert PLAN_FINGERPRINT_VERSION == 5
        assert bool_plan().fingerprint_version == 5

    def test_doctrine_block_records_the_real_identity_and_version(self) -> None:
        assert probability_formulation_payload(bool_plan())["doctrine"] == {
            "doctrine_id": "binary-semantic-judgment-v1",
            "version": BINARY_DOCTRINE_VERSION,
        }
        assert probability_formulation_payload(choice_plan())["doctrine"] == {
            "doctrine_id": "categorical-semantic-judgment-v1",
            "version": CATEGORICAL_DOCTRINE_VERSION,
        }

    def test_doctrine_version_change_moves_every_formulation_identity(self) -> None:
        plan = bool_plan()
        bumped = replace(plan, doctrine_version=BINARY_DOCTRINE_VERSION + 1)

        assert bumped.decision_fingerprint == plan.decision_fingerprint
        assert bumped.fingerprint != plan.fingerprint
        assert probability_formulation_fingerprint(bumped) != probability_formulation_fingerprint(
            plan
        )
        assert formulation_family_fingerprint(bumped) != formulation_family_fingerprint(plan)

    def test_doctrine_id_change_moves_every_formulation_identity(self) -> None:
        plan = bool_plan()
        renamed = replace(plan, doctrine_id="other-doctrine")

        assert renamed.decision_fingerprint == plan.decision_fingerprint
        assert renamed.fingerprint != plan.fingerprint
        assert probability_formulation_fingerprint(renamed) != probability_formulation_fingerprint(
            plan
        )
        assert formulation_family_fingerprint(renamed) != formulation_family_fingerprint(plan)


class TestEvidenceExclusion:
    def test_bool_same_formulation_different_evidence(self) -> None:
        first = bool_plan(question="Is A suspicious?", context={"amount": 1})
        second = bool_plan(question="Is B suspicious?", context={"amount": 999})

        assert first.fingerprint != second.fingerprint
        assert first.decision_fingerprint != second.decision_fingerprint
        assert probability_formulation_fingerprint(first) == (
            probability_formulation_fingerprint(second)
        )
        assert formulation_family_fingerprint(first) == formulation_family_fingerprint(second)

    def test_choice_same_formulation_different_evidence(self) -> None:
        first = choice_plan(context="ticket one")
        second = choice_plan(context={"nested": ["ticket", "two"]})

        assert first.fingerprint != second.fingerprint
        assert first.decision_fingerprint != second.decision_fingerprint
        assert probability_formulation_fingerprint(first) == (
            probability_formulation_fingerprint(second)
        )
        assert formulation_family_fingerprint(first) == formulation_family_fingerprint(second)


class TestChoiceRepresentation:
    def test_label_permutation_keeps_decision_and_family_only(self) -> None:
        natural = choice_plan()
        permuted = relabel(natural, ("B", "C", "A"))

        assert permuted.decision_fingerprint == natural.decision_fingerprint
        assert permuted.fingerprint != natural.fingerprint
        assert probability_formulation_fingerprint(permuted) != (
            probability_formulation_fingerprint(natural)
        )
        assert formulation_family_fingerprint(permuted) == (formulation_family_fingerprint(natural))

    def test_all_six_permutations_share_one_family_and_six_exact_identities(self) -> None:
        natural = choice_plan()
        permutations = [
            ("A", "B", "C"),
            ("A", "C", "B"),
            ("B", "A", "C"),
            ("B", "C", "A"),
            ("C", "A", "B"),
            ("C", "B", "A"),
        ]
        exact = {probability_formulation_fingerprint(relabel(natural, p)) for p in permutations}
        family = {formulation_family_fingerprint(relabel(natural, p)) for p in permutations}

        assert len(exact) == 6
        assert len(family) == 1

    def test_description_paraphrase_changes_exact_but_not_family(self) -> None:
        natural = choice_plan()
        paraphrased_choices = default_choices()
        paraphrased_choices["billing"] = "Problems involving invoices, charges, or payments"
        paraphrased = choice_plan(choices=paraphrased_choices)

        assert paraphrased.decision_fingerprint != natural.decision_fingerprint
        assert probability_formulation_fingerprint(paraphrased) != (
            probability_formulation_fingerprint(natural)
        )
        assert formulation_family_fingerprint(paraphrased) == (
            formulation_family_fingerprint(natural)
        )

    def test_candidate_order_changes_exact_but_not_family(self) -> None:
        natural = choice_plan()
        reordered = choice_plan(
            choices={
                "technical": "Technical problems",
                "billing": "Payment, charges, and invoices",
                "shipping": "Delivery and logistics",
            }
        )

        assert probability_formulation_fingerprint(reordered) != (
            probability_formulation_fingerprint(natural)
        )
        assert formulation_family_fingerprint(reordered) == (
            formulation_family_fingerprint(natural)
        )

    def test_different_taxonomy_same_mechanism(self) -> None:
        animals = choice_plan(choices={"cat": "Felines", "dog": "Canines", "bird": "Avians"})

        assert probability_formulation_fingerprint(animals) != (
            probability_formulation_fingerprint(choice_plan())
        )
        assert formulation_family_fingerprint(animals) == (
            formulation_family_fingerprint(choice_plan())
        )

    def test_arity_changes_both_exact_and_family(self) -> None:
        three = choice_plan()
        five = choice_plan(
            choices={
                "billing": "Payment, charges, and invoices",
                "shipping": "Delivery and logistics",
                "technical": "Technical problems",
                "returns": "Returns and refunds",
                "account": "Account management",
            }
        )

        assert three.probability_formulation_fingerprint != (
            five.probability_formulation_fingerprint
        )
        assert three.formulation_family_fingerprint != five.formulation_family_fingerprint


class TestBoolRepresentation:
    def test_verbalizer_pair_changes_exact_but_not_family(self) -> None:
        yes_no = bool_plan()
        true_false = bool_plan(positive_verbalizer="true", negative_verbalizer="false")

        assert probability_formulation_fingerprint(true_false) != (
            probability_formulation_fingerprint(yes_no)
        )
        assert formulation_family_fingerprint(true_false) == (
            formulation_family_fingerprint(yes_no)
        )

    def test_verbalizer_roles_are_not_interchangeable(self) -> None:
        positive_yes = bool_plan(positive_verbalizer="yes", negative_verbalizer="no")
        positive_no = bool_plan(positive_verbalizer="no", negative_verbalizer="yes")

        assert probability_formulation_fingerprint(positive_no) != (
            probability_formulation_fingerprint(positive_yes)
        )

    def test_bool_family_payload_does_not_leak_verbalizer_strings(self) -> None:
        serialized = canonical_json(
            formulation_family_payload(
                bool_plan(positive_verbalizer="absolutely", negative_verbalizer="absurd")
            )
        )

        assert "absolutely" not in serialized
        assert "absurd" not in serialized
        assert formulation_family_payload(bool_plan())["arity"] == 2


class TestSourceAndExecutionExclusion:
    def test_execution_metadata_does_not_reach_the_formulation_identity(self) -> None:
        plan = bool_plan()
        first = bool_trace(plan, {"model": "model-a", "tokenizer": "tok-a"})
        second = bool_trace(
            plan,
            {
                "model": "model-b",
                "tokenizer": "tok-b",
                "rendering_config": {"enable_thinking": False},
            },
        )

        assert first.execution_fingerprint != second.execution_fingerprint
        assert first.probability_formulation_fingerprint == (
            second.probability_formulation_fingerprint
        )
        assert first.formulation_family_fingerprint == second.formulation_family_fingerprint

    def test_resolved_token_ids_do_not_reach_the_formulation_identity(self) -> None:
        plan = choice_plan()
        first = choice_trace(plan, {"resolved_target_token_ids": [["A", 32], ["B", 33], ["C", 34]]})
        second = choice_trace(
            plan, {"resolved_target_token_ids": [["A", 9001], ["B", 9002], ["C", 9003]]}
        )

        assert first.resolved_target_token_ids != second.resolved_target_token_ids
        assert first.execution_fingerprint != second.execution_fingerprint
        assert first.probability_formulation_fingerprint == (
            second.probability_formulation_fingerprint
        )
        assert first.formulation_family_fingerprint == second.formulation_family_fingerprint


class TestTraceExposure:
    def test_trace_mirrors_the_plan_identities(self) -> None:
        plan = choice_plan()
        trace = choice_trace(plan)

        assert trace.probability_formulation_fingerprint == (
            plan.probability_formulation_fingerprint
        )
        assert trace.formulation_family_fingerprint == plan.formulation_family_fingerprint
        assert trace.probability_formulation_fingerprint_version == (
            PROBABILITY_FORMULATION_FINGERPRINT_VERSION
        )
        assert trace.formulation_family_fingerprint_version == (
            FORMULATION_FAMILY_FINGERPRINT_VERSION
        )

    def test_to_dict_exposes_all_four_identity_fields(self) -> None:
        trace = bool_trace(bool_plan())
        payload = trace.to_dict()

        assert payload["probability_formulation_fingerprint"] == (
            trace.probability_formulation_fingerprint
        )
        assert payload["probability_formulation_fingerprint_version"] == (
            PROBABILITY_FORMULATION_FINGERPRINT_VERSION
        )
        assert payload["formulation_family_fingerprint"] == trace.formulation_family_fingerprint
        assert payload["formulation_family_fingerprint_version"] == (
            FORMULATION_FAMILY_FINGERPRINT_VERSION
        )

    def test_trace_does_not_expose_any_comparability_field(self) -> None:
        payload = bool_trace(bool_plan()).to_dict()

        for forbidden in (
            "formulation_relation",
            "source_relation",
            "comparable",
            "calibratable",
        ):
            assert forbidden not in payload
