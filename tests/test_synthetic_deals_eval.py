"""Offline eval of the deterministic pipeline over the 9 synthetic deals.

Runs signal detection + the eligibility gate (no LLM, no network) and asserts
the governance behavior each deal was designed to demonstrate — a miniature
version of the outline's "inspect historical opportunities" validation step.
"""

import pytest

from api._copilot.data_access import detect_signals, get_opportunity, load_opportunities
from api._copilot.policies import MIN_RICHNESS_FOR_OUTREACH, eligibility_gate
from api._copilot.schemas import CUSTOMER_FACING_ACTIONS


def run_gate(opportunity_id: str) -> tuple[dict, dict, dict]:
    opp = get_opportunity(opportunity_id)
    signals = detect_signals(opp)
    policy = eligibility_gate(opp, signals)
    return opp, signals, policy


def outreach_eligible(policy: dict) -> bool:
    return bool(
        set(policy["eligible_action_types"]) & {a.value for a in CUSTOMER_FACING_ACTIONS}
    )


def test_corpus_has_nine_deals():
    assert len(load_opportunities()) == 9


@pytest.mark.parametrize("opp_id", [o["id"] for o in load_opportunities()])
def test_every_deal_produces_a_valid_policy(opp_id):
    """The gate must always return a well-formed policy with at least one way forward."""
    _, signals, policy = run_gate(opp_id)
    assert signals["evidence_richness"] >= 0
    assert policy["eligible_action_types"], "no eligible action types at all"
    # Abstention paths must never be gated off.
    assert "gather_information" in policy["eligible_action_types"]
    assert "no_action_yet" in policy["eligible_action_types"]


def test_meridian_rich_evidence_allows_outreach():
    """Meridian Digital Bank: rich evidence -> customer-facing action allowed."""
    _, signals, policy = run_gate("opp-meridian")
    assert signals["evidence_richness"] >= MIN_RICHNESS_FOR_OUTREACH
    assert outreach_eligible(policy)


def test_loopride_empty_record_blocks_outreach():
    """LoopRide: near-empty CRM record -> outreach ineligible, abstention forced."""
    _, signals, policy = run_gate("opp-loopride")
    assert signals["evidence_richness"] < MIN_RICHNESS_FOR_OUTREACH
    assert not outreach_eligible(policy)
    assert signals["gaps"], "an empty record should surface explicit gaps"


def test_helios_procurement_blockers_escalate():
    """Helios Telecom: pricing/legal items -> escalation, no drafted commercial terms."""
    _, _, policy = run_gate("opp-helios")
    assert policy["escalations"]
    assert any("out of bounds" in c for c in policy["constraints"])


def test_brightpath_stalled_deal_is_flagged():
    """BrightPath Health: ~20 days quiet -> inactivity signal detected."""
    _, signals, _ = run_gate("opp-brightpath")
    assert any("Inactive" in s for s in signals["signals"])


def test_evaluation_stage_deals_block_exec_escalation():
    """Deals still in evaluation stages must carry the no-exec-escalation constraint."""
    for opp in load_opportunities():
        if opp["stage"] in ("discovery", "technical_evaluation", "solution_evaluation"):
            _, _, policy = run_gate(opp["id"])
            assert any("executive-level" in c for c in policy["constraints"]), opp["id"]
