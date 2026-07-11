"""Unit tests for the deterministic policy gates (no LLM, no network).

These cover the two governance layers described in the outline:
- Gate 1 (eligibility) before generation
- Gate 2 (validation) after generation
"""

import pytest

from api._copilot.policies import (
    MIN_RICHNESS_FOR_OUTREACH,
    eligibility_gate,
    validation_gate,
)
from api._copilot.schemas import (
    CUSTOMER_FACING_ACTIONS,
    ActionType,
    Confidence,
    Evidence,
    Recommendation,
)


def make_opp(**overrides) -> dict:
    """A healthy baseline opportunity that trips no rules."""
    opp = {
        "motion": "new_logo",
        "stage": "business_validation",
        "days_since_last_activity": 7,
        "products_deployed": [],
        "call_notes": ["Great discovery call about fraud stack."],
        "objections": [],
    }
    opp.update(overrides)
    return opp


def make_signals(richness: int = 5) -> dict:
    return {"signals": [], "gaps": [], "evidence_richness": richness}


def make_rec(**overrides) -> Recommendation:
    fields = {
        "action_type": ActionType.customer_outreach,
        "action": "Invite the Head of Fraud to a working session.",
        "owner": "AE",
        "target_stakeholder": "Head of Fraud",
        "timing": "within 3 business days",
        "rationale": "Unresolved manual-review concern with active engagement.",
        "evidence": [Evidence(source_id="CRM:opportunity", claim="17 days without stage movement.")],
        "confidence": Confidence.medium,
        "missing_information": [],
        "draft_message": None,
    }
    fields.update(overrides)
    return Recommendation(**fields)


ALLOWED_SOURCES = {"CRM:opportunity", "KB:sigma-synthetic-fraud"}


# ---------------------------------------------------------------------------
# Gate 1: eligibility
# ---------------------------------------------------------------------------


class TestEligibilityGate:
    def test_healthy_opportunity_allows_all_action_types(self):
        policy = eligibility_gate(make_opp(), make_signals())
        assert policy["eligible_action_types"] == sorted(a.value for a in ActionType)
        assert policy["escalations"] == []

    def test_thin_evidence_strips_customer_facing_actions(self):
        policy = eligibility_gate(
            make_opp(), make_signals(richness=MIN_RICHNESS_FOR_OUTREACH - 1)
        )
        eligible = set(policy["eligible_action_types"])
        assert not eligible & {a.value for a in CUSTOMER_FACING_ACTIONS}
        assert {
            ActionType.gather_information.value,
            ActionType.internal_preparation.value,
            ActionType.no_action_yet.value,
        } <= eligible

    def test_richness_at_threshold_keeps_outreach_eligible(self):
        policy = eligibility_gate(
            make_opp(), make_signals(richness=MIN_RICHNESS_FOR_OUTREACH)
        )
        assert ActionType.customer_outreach.value in policy["eligible_action_types"]

    def test_deployed_products_constrained_outside_expansion(self):
        opp = make_opp(products_deployed=["Socure Verify"], motion="new_logo")
        policy = eligibility_gate(opp, make_signals())
        assert any("Socure Verify" in c for c in policy["constraints"])

    def test_deployed_products_unconstrained_in_expansion(self):
        opp = make_opp(products_deployed=["Socure Verify"], motion="expansion")
        policy = eligibility_gate(opp, make_signals())
        assert not any("already-deployed" in c for c in policy["constraints"])

    @pytest.mark.parametrize("note", ["Pricing pushback on the quote", "Legal wants redlines"])
    def test_commercial_blockers_route_to_escalation(self, note):
        opp = make_opp(call_notes=[note])
        policy = eligibility_gate(opp, make_signals())
        assert policy["escalations"], f"expected escalation for note: {note!r}"
        assert any("out of bounds" in c for c in policy["constraints"])

    @pytest.mark.parametrize(
        "stage", ["discovery", "technical_evaluation", "solution_evaluation"]
    )
    def test_no_exec_escalation_during_evaluation(self, stage):
        policy = eligibility_gate(make_opp(stage=stage), make_signals())
        assert any("executive-level" in c for c in policy["constraints"])

    def test_recent_contact_adds_fatigue_constraint(self):
        policy = eligibility_gate(make_opp(days_since_last_activity=1), make_signals())
        assert any("very recent" in c for c in policy["constraints"])


# ---------------------------------------------------------------------------
# Gate 2: validation
# ---------------------------------------------------------------------------


def full_policy() -> dict:
    return {
        "eligible_action_types": sorted(a.value for a in ActionType),
        "constraints": [],
        "escalations": [],
    }


class TestValidationGate:
    def test_compliant_recommendation_passes(self):
        result = validation_gate(make_rec(), full_policy(), ALLOWED_SOURCES)
        assert result["passed"], result["failures"]

    def test_ineligible_action_type_fails(self):
        policy = full_policy()
        policy["eligible_action_types"] = [ActionType.gather_information.value]
        result = validation_gate(make_rec(), policy, ALLOWED_SOURCES)
        assert not result["passed"]
        assert any("not eligible" in f for f in result["failures"])

    def test_missing_evidence_fails(self):
        rec = make_rec(evidence=[])
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert not result["passed"]
        assert any("no cited evidence" in f for f in result["failures"])

    def test_unknown_source_id_fails(self):
        rec = make_rec(
            evidence=[Evidence(source_id="KB:made-up-doc", claim="Invented fact.")]
        )
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert not result["passed"]
        assert any("unknown source ids" in f for f in result["failures"])

    def test_low_confidence_customer_facing_fails(self):
        rec = make_rec(confidence=Confidence.low)
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert not result["passed"]
        assert any("Low-confidence" in f for f in result["failures"])

    def test_low_confidence_internal_action_passes(self):
        rec = make_rec(
            action_type=ActionType.gather_information,
            target_stakeholder=None,
            confidence=Confidence.low,
        )
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert result["passed"], result["failures"]

    def test_quantified_claim_without_kb_citation_fails(self):
        rec = make_rec(draft_message="Hi Dana,\n\nWe cut manual reviews by 40%.\n\nBest,\n[Your name]")
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert not result["passed"]
        assert any("quantified claims" in f for f in result["failures"])

    def test_quantified_claim_with_kb_citation_passes_with_warning(self):
        rec = make_rec(
            evidence=[
                Evidence(source_id="KB:sigma-synthetic-fraud", claim="40% manual-review reduction in case study.")
            ],
            draft_message="Hi Dana,\n\nA peer bank cut manual reviews by 40%.\n\nBest,\n[Your name]",
        )
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert result["passed"], result["failures"]
        assert any("verify" in w for w in result["warnings"])

    def test_draft_on_internal_action_warns_but_passes(self):
        rec = make_rec(
            action_type=ActionType.internal_preparation,
            draft_message="Hi Dana,\n\nQuick sync?\n\nBest,\n[Your name]",
        )
        result = validation_gate(rec, full_policy(), ALLOWED_SOURCES)
        assert result["passed"], result["failures"]
        assert any("non-customer-facing" in w for w in result["warnings"])
