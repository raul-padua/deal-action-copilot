"""Deterministic policy gates. Plain Python, no LLM.

Gate 1 (eligibility) runs before generation: which action types, products, and
constraints apply to this opportunity.
Gate 2 (validation) runs after generation: block or downgrade noncompliant output.
"""

import re

from .schemas import CUSTOMER_FACING_ACTIONS, ActionType, Confidence, Recommendation

# Evidence richness below this → customer-facing actions are ineligible.
MIN_RICHNESS_FOR_OUTREACH = 3


def eligibility_gate(opp: dict, signals: dict) -> dict:
    """Returns eligible action types, product constraints, and prompt constraints."""
    eligible = {a for a in ActionType}
    constraints: list[str] = []
    escalations: list[str] = []

    # Rule: thin evidence → no customer-facing recommendation.
    if signals["evidence_richness"] < MIN_RICHNESS_FOR_OUTREACH:
        eligible -= CUSTOMER_FACING_ACTIONS
        constraints.append(
            "Evidence is too thin for customer outreach: recommend gathering "
            "information or an internal step instead."
        )

    # Rule: already-deployed products can only be positioned in expansion motions.
    deployed = opp.get("products_deployed", [])
    if deployed and opp["motion"] != "expansion":
        constraints.append(
            f"Do not position already-deployed products: {', '.join(deployed)}."
        )

    # Rule: pricing/legal/contractual blockers are escalations, not copilot drafts.
    blocker_terms = ("pricing", "quote", "legal", "contract", "redline", "data-processing")
    open_blockers = [
        note for note in opp.get("call_notes", []) + opp.get("objections", [])
        if any(t in note.lower() for t in blocker_terms)
    ]
    if open_blockers:
        escalations.append(
            "Open pricing/legal/contractual items detected — these must be routed to "
            "the deal desk / legal, and the copilot must not draft commercial terms."
        )
        constraints.append(
            "Pricing, discounts, and contractual language are out of bounds; the "
            "recommendation may coordinate the internal escalation but not draft terms."
        )

    # Rule: no executive escalation while technical validation is unresolved.
    if opp["stage"] in ("discovery", "technical_evaluation", "solution_evaluation"):
        constraints.append(
            "Do not recommend executive-level escalation while evaluation is in progress."
        )

    # Rule: respect contact fatigue — if last touch was very recent, avoid another generic ping.
    if opp["days_since_last_activity"] <= 2:
        constraints.append(
            "Contact was very recent: only recommend outreach that adds new, specific value."
        )

    return {
        "eligible_action_types": sorted(a.value for a in eligible),
        "constraints": constraints,
        "escalations": escalations,
    }


_NUMBER_CLAIM = re.compile(r"\d+(\.\d+)?\s*(%|percent|x\b)|\$\s*\d", re.IGNORECASE)


def validation_gate(rec: Recommendation, policy: dict, known_source_ids: set[str]) -> dict:
    """Final guardrail check on the generated recommendation."""
    failures: list[str] = []
    warnings: list[str] = []

    if rec.action_type.value not in policy["eligible_action_types"]:
        failures.append(
            f"Action type '{rec.action_type.value}' is not eligible for this "
            f"opportunity (eligible: {', '.join(policy['eligible_action_types'])})."
        )

    if not rec.evidence:
        failures.append("Recommendation has no cited evidence.")
    else:
        unknown = [e.source_id for e in rec.evidence if e.source_id not in known_source_ids]
        if unknown:
            failures.append(
                f"Evidence cites unknown source ids: {', '.join(sorted(set(unknown)))}."
            )

    if rec.action_type in CUSTOMER_FACING_ACTIONS and rec.confidence == Confidence.low:
        failures.append(
            "Low-confidence output cannot recommend a customer-facing action; "
            "downgrade to gather_information."
        )

    # Quantified claims in the draft must be traceable to retrieved approved knowledge.
    if rec.draft_message and _NUMBER_CLAIM.search(rec.draft_message):
        kb_backed = any(e.source_id.startswith("KB:") for e in rec.evidence)
        if not kb_backed:
            failures.append(
                "Draft contains quantified claims with no approved knowledge-base citation."
            )
        else:
            warnings.append(
                "Draft contains quantified claims — reviewer should verify they match "
                "the cited approved source verbatim."
            )

    if rec.action_type not in CUSTOMER_FACING_ACTIONS and rec.draft_message:
        warnings.append("Draft message present on a non-customer-facing action; reviewer may discard it.")

    return {"passed": not failures, "failures": failures, "warnings": warnings}
