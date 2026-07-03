"""Synthetic CRM access + deterministic signal detection (pipeline stage 2)."""

import json
import time
from functools import lru_cache

from .config import DATA_DIR

# Approved follow-ups recorded back onto the opportunity (demo: in-memory overlay
# over the read-only JSON "CRM"). The agent's next run sees the updated next step.
_followups: dict[str, dict] = {}


def record_followup(opportunity_id: str, action: str, decision: str) -> dict:
    _followups[opportunity_id] = {
        "action": action,
        "decision": decision,
        "ts": time.time(),
    }
    return _followups[opportunity_id]


def get_followup(opportunity_id: str) -> dict | None:
    return _followups.get(opportunity_id)


@lru_cache(maxsize=1)
def load_opportunities() -> list[dict]:
    with open(DATA_DIR / "opportunities.json") as f:
        return json.load(f)


def get_opportunity(opportunity_id: str) -> dict:
    for opp in load_opportunities():
        if opp["id"] == opportunity_id:
            followup = _followups.get(opportunity_id)
            if followup:
                return {
                    **opp,
                    "next_step_on_record": f"Approved follow-up: {followup['action']}",
                }
            return opp
    raise KeyError(f"Unknown opportunity: {opportunity_id}")


def detect_signals(opp: dict) -> dict:
    """Deterministic signal & gap detection. No LLM involved."""
    signals: list[str] = []
    gaps: list[str] = []

    if opp["days_since_last_activity"] >= 14:
        signals.append(f"Inactive for {opp['days_since_last_activity']} days")
    if opp["days_in_stage"] >= 15:
        signals.append(f"Stuck in {opp['stage']} for {opp['days_in_stage']} days")
    if opp["days_to_close"] <= 21:
        signals.append(f"Close date in {opp['days_to_close']} days")
    for item in opp.get("recent_engagement", []):
        signals.append(f"Engagement: {item}")
    for obj in opp.get("objections", []):
        signals.append(f"Open objection: {obj}")

    if not opp.get("stakeholders"):
        gaps.append("No stakeholders mapped")
    elif len(opp["stakeholders"]) == 1:
        gaps.append("Single-threaded: only one stakeholder engaged")
    if not opp.get("call_notes"):
        gaps.append("No call notes / discovery record")
    if opp.get("next_step_on_record") in (None, "", "None scheduled"):
        gaps.append("No next step on record")
    if not opp.get("recent_engagement"):
        gaps.append("No recent marketing engagement")

    # Crude evidence-richness score used by the eligibility gate.
    richness = 0
    richness += min(len(opp.get("call_notes", [])), 3)
    richness += min(len(opp.get("recent_engagement", [])), 3)
    richness += min(len(opp.get("stakeholders", [])), 2)

    return {"signals": signals, "gaps": gaps, "evidence_richness": richness}


def opportunity_summary_text(opp: dict) -> str:
    """Compact plain-text snapshot handed to the LLM as CRM context."""
    stakeholders = "; ".join(
        f"{s['name']} ({s['role']}, engagement {s['engagement']}, last touch {s['last_touch_days']}d ago)"
        for s in opp.get("stakeholders", [])
    ) or "none mapped"
    lines = [
        f"Account: {opp['account']} | Vertical: {opp['vertical']} | Motion: {opp['motion']}",
        f"Stage: {opp['stage']} ({opp['days_in_stage']}d in stage) | Value: ${opp['value_usd']:,} | Close in {opp['days_to_close']}d",
        f"Last activity: {opp['days_since_last_activity']}d ago | Next step on record: {opp.get('next_step_on_record') or 'none'}",
        f"Products evaluated: {', '.join(opp.get('products_evaluated', [])) or 'none'}",
        f"Products already deployed: {', '.join(opp.get('products_deployed', [])) or 'none'}",
        f"Stakeholders: {stakeholders}",
        f"Recent engagement: {'; '.join(opp.get('recent_engagement', [])) or 'none'}",
        f"Call notes: {' | '.join(opp.get('call_notes', [])) or 'none'}",
        f"Open objections: {'; '.join(opp.get('objections', [])) or 'none'}",
        f"Competitors: {', '.join(opp.get('competitors', [])) or 'none known'}",
        f"Notes: {opp.get('notes', '')}",
    ]
    return "\n".join(lines)
