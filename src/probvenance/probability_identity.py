"""Probability formulation identity: provider-independent formulation fingerprints.

This module materializes the identity contract frozen in
``docs/probability-semantics-identity.md`` (INV-23 through INV-27). Both
identities are DERIVED from an :class:`~probvenance.plans.InferencePlan` alone
and are never stored back on the plan or its fingerprint.

Two identities exist:

- :func:`probability_formulation_fingerprint` is the exact formulation
  identity. It commits the provider-independent probability formulation:
  decision family, scoring strategy, compiler, probability assembler,
  doctrine, semantic outcome space, and scoring representation.
- :func:`formulation_family_fingerprint` is the coarser mechanism-class
  identity. It commits the same mechanism dimensions but collapses the concrete
  representation instance: candidate names, candidate descriptions, and the
  candidate-to-label assignment are excluded.

Neither identity commits the model, the tokenizer, the rendering configuration,
the resolved scoring token ids, or the decision's question and context. The
first four belong to the source axis and to execution provenance; the last two
are instance evidence. They are recorded by the execution fingerprint and the
decision fingerprint instead.

Two boundaries this module exists to protect:

- A formulation identity is NOT a complete statement about a probability. The
  provider-independent formulation must be composed with the source axis and
  with instance evidence before anything can be said about a realized value.
- Equality of a formulation fingerprint is NOT probability equality, and
  membership of one formulation family does NOT imply interchangeability,
  poolability, or calibration compatibility (INV-23, INV-24).
"""

from probvenance.errors import InvalidDecisionError
from probvenance.fingerprint import JSONValue, fingerprint
from probvenance.plans import (
    DECISION_FAMILY_BOOL,
    DECISION_FAMILY_CHOICE,
    InferencePlan,
    ScoringStrategy,
)

# Bump when the canonical payload shape changes. The two identities evolve
# independently, so they carry separate versions.
# Version 2 commits a real doctrine version; v1 could only record an unknown.
PROBABILITY_FORMULATION_FINGERPRINT_VERSION = 2
FORMULATION_FAMILY_FINGERPRINT_VERSION = 2

_BOOL_DECISION_FAMILY = DECISION_FAMILY_BOOL
_CHOICE_DECISION_FAMILY = DECISION_FAMILY_CHOICE

# The Bool outcome space is fixed: Omega = {False, True}. It is rendered as
# explicit semantic outcome names so the payload never depends on Python class
# paths, enum reprs, or dict iteration accidents.
_BOOL_OUTCOMES: tuple[str, ...] = ("false", "true")

# The Bool outcome space has exactly two outcomes, always. Arity is part of the
# formulation family because it changes the restricted softmax support.
_BOOL_ARITY = 2


def _identity_party(identifier: str | None, version: int | None) -> dict[str, JSONValue]:
    """An ``{id, version}`` block, explicit in both the concrete and unknown form.

    An undeclared identity is an explicit unknown (INV-26): ``{"id": None,
    "version": None}``. It is never omitted, never rendered as an empty
    identifier or a zero version, and never a sentinel string, which would
    claim a concrete identity the plan does not have. Half-known states are
    rejected here as defence in depth; the plan already rejects them.
    """
    if (identifier is None) != (version is None):
        raise InvalidDecisionError(
            "identity party must be atomic (both id and version, or neither), got "
            f"id={identifier!r}, version={version!r}"
        )
    if identifier is None:
        return {"id": None, "version": None}
    if not isinstance(identifier, str) or not identifier.strip():
        raise InvalidDecisionError(
            f"identity party id must be a non-empty string, got {identifier!r}"
        )
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise InvalidDecisionError(f"identity party version must be an int >= 1, got {version!r}")
    return {"id": identifier, "version": version}


def _doctrine_block(plan: InferencePlan) -> dict[str, JSONValue]:
    """The doctrine identity block: id and version, both explicit.

    Doctrine is a probability-formulation-relevant contract, so its identity and
    its revision are separate plan fields and both enter the payload. The version
    is never derived from the id string, even though the shipped ids happen to be
    version-suffixed. A plan with no doctrine contributes the explicit unknown
    block (INV-26); the key is never omitted.
    """
    if (plan.doctrine_id is None) != (plan.doctrine_version is None):
        raise InvalidDecisionError(
            "doctrine identity must be atomic (both id and version, or neither), got "
            f"doctrine_id={plan.doctrine_id!r}, "
            f"doctrine_version={plan.doctrine_version!r}"
        )
    if plan.doctrine_id is None:
        return {"doctrine_id": None, "version": None}
    if not isinstance(plan.doctrine_id, str) or not plan.doctrine_id.strip():
        raise InvalidDecisionError(
            f"doctrine_id must be a non-empty string, got {plan.doctrine_id!r}"
        )
    version = plan.doctrine_version
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise InvalidDecisionError(f"doctrine_version must be an int >= 1, got {version!r}")
    return {"doctrine_id": plan.doctrine_id, "version": version}


def _bool_outcome_space() -> dict[str, JSONValue]:
    return {
        "outcomes": list(_BOOL_OUTCOMES),
        "closed_set": True,
        "single_label": True,
    }


def _bool_scoring_representation(plan: InferencePlan) -> dict[str, JSONValue] | None:
    """The Bool scoring representation: the verbalizer pair WITH its roles.

    The pair is ordered by semantic role (positive means True, negative means
    False), never as an unordered set of surface strings, so switching the
    roles produces a different identity.
    """
    if plan.positive_verbalizer is None or plan.negative_verbalizer is None:
        return None
    return {
        "positive_verbalizer": plan.positive_verbalizer,
        "negative_verbalizer": plan.negative_verbalizer,
    }


def _choice_outcome_space(plan: InferencePlan) -> dict[str, JSONValue]:
    """The ordered Choice outcome space: names, descriptions, and assumptions."""
    return {
        "arity": len(plan.candidate_mapping),
        "candidates": [
            {"name": entry.candidate_name, "description": entry.candidate_description}
            for entry in plan.candidate_mapping
        ],
        "closed_set": True,
        "single_label": True,
    }


def _choice_scoring_representation(plan: InferencePlan) -> dict[str, JSONValue]:
    """The Choice scoring representation: label scheme plus candidate-to-label map.

    Ordered exactly like the plan's mapping. Resolved token ids are deliberately
    absent: they are execution provenance, not formulation.
    """
    return {
        "label_scheme_id": plan.label_scheme_id,
        "mapping": [
            {
                "candidate_index": entry.candidate_index,
                "candidate_name": entry.candidate_name,
                "candidate_description": entry.candidate_description,
                "scoring_label": entry.scoring_label,
            }
            for entry in plan.candidate_mapping
        ],
    }


def _family_arity(plan: InferencePlan) -> int | None:
    """Candidate cardinality for the family identity, or ``None`` if unknown.

    Branches on the DECLARED decision family, not on the strategy: the family
    is a plan property, and arity is a property of the outcome space.
    """
    if plan.decision_family == _BOOL_DECISION_FAMILY:
        return _BOOL_ARITY
    if plan.decision_family == _CHOICE_DECISION_FAMILY:
        return len(plan.candidate_mapping)
    return None


def probability_formulation_payload(plan: InferencePlan) -> dict[str, JSONValue]:
    """Canonical payload of the exact probability formulation identity.

    Provider-independent and evidence-independent: it commits the semantic
    outcome space and the scoring representation together with the mechanism
    that produced them, and it excludes question, context, model, tokenizer,
    rendering configuration, and resolved token ids.

    A strategy with no compiled formulation yields explicit unknown blocks
    rather than fabricated defaults.
    """
    strategy = plan.strategy
    outcome_space: JSONValue
    scoring_representation: JSONValue
    if strategy is ScoringStrategy.BINARY_TOKEN_LOGITS:
        outcome_space = _bool_outcome_space()
        scoring_representation = _bool_scoring_representation(plan)
    elif strategy is ScoringStrategy.CATEGORICAL_TOKEN_LOGITS:
        outcome_space = _choice_outcome_space(plan)
        scoring_representation = _choice_scoring_representation(plan)
    else:
        outcome_space = None
        scoring_representation = None
    return {
        "v": PROBABILITY_FORMULATION_FINGERPRINT_VERSION,
        "kind": "probability_formulation",
        "decision_family": plan.decision_family,
        "strategy": str(strategy),
        "assembler": _identity_party(plan.assembler_id, plan.assembler_version),
        "doctrine": _doctrine_block(plan),
        "compiler": _identity_party(plan.compiler_id, plan.compiler_version),
        "outcome_space": outcome_space,
        "scoring_representation": scoring_representation,
    }


def probability_formulation_fingerprint(plan: InferencePlan) -> str:
    """SHA-256 of :func:`probability_formulation_payload` (the exact identity)."""
    return fingerprint(probability_formulation_payload(plan))


def formulation_family_payload(plan: InferencePlan) -> dict[str, JSONValue]:
    """Canonical payload of the formulation family identity.

    Same mechanism dimensions as the exact identity, with the representation
    instance collapsed: no candidate names, no descriptions, no
    candidate-to-label assignment. ``label_scheme_id`` and ``arity`` survive
    because they describe the mechanism, not the particular instance.
    """
    return {
        "v": FORMULATION_FAMILY_FINGERPRINT_VERSION,
        "kind": "formulation_family",
        "decision_family": plan.decision_family,
        "strategy": str(plan.strategy),
        "assembler": _identity_party(plan.assembler_id, plan.assembler_version),
        "doctrine": _doctrine_block(plan),
        "compiler": _identity_party(plan.compiler_id, plan.compiler_version),
        "label_scheme_id": plan.label_scheme_id,
        "arity": _family_arity(plan),
    }


def formulation_family_fingerprint(plan: InferencePlan) -> str:
    """SHA-256 of :func:`formulation_family_payload` (the family identity)."""
    return fingerprint(formulation_family_payload(plan))
