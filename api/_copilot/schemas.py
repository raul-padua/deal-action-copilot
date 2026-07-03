"""Fixed output schema for the recommendation — the 'structured reasoning' contract.

Abstention is not a failure mode: `gather_information` and `no_action_yet`
are first-class action types, per the Deal Action Copilot outline.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    customer_outreach = "customer_outreach"
    coordinated_sales_marketing = "coordinated_sales_marketing"
    internal_preparation = "internal_preparation"
    gather_information = "gather_information"
    no_action_yet = "no_action_yet"


CUSTOMER_FACING_ACTIONS = {
    ActionType.customer_outreach,
    ActionType.coordinated_sales_marketing,
}


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Evidence(BaseModel):
    source_id: str = Field(
        description=(
            "Identifier of the source this claim rests on. Must be one of the "
            "source ids provided in context, e.g. 'CRM:opportunity', "
            "'KB:sigma-synthetic-fraud', 'WEB:tavily-1'."
        )
    )
    claim: str = Field(description="The specific fact or signal taken from this source.")


class Recommendation(BaseModel):
    action_type: ActionType
    action: str = Field(description="One concrete recommended next action, 1-3 sentences.")
    owner: str = Field(description="Who on the GTM side executes it, e.g. 'AE', 'AE + Product Marketing'.")
    target_stakeholder: Optional[str] = Field(
        default=None, description="Customer-side person/role the action targets, if customer-facing."
    )
    timing: str = Field(description="When to act, e.g. 'within 3 business days'.")
    rationale: str = Field(description="Why this action, grounded in the evidence. 2-4 sentences.")
    socure_angle: Optional[str] = Field(
        default=None, description="Relevant Socure capability/positioning to lead with, if any."
    )
    supporting_asset: Optional[str] = Field(
        default=None, description="Approved asset or proof point to use, if any (must come from retrieved knowledge)."
    )
    evidence: list[Evidence] = Field(description="Every material claim, each tied to a provided source id.")
    confidence: Confidence
    missing_information: list[str] = Field(
        default_factory=list, description="What is unknown and would change or strengthen the recommendation."
    )
    draft_message: Optional[str] = Field(
        default=None,
        description=(
            "Optional draft email for customer-facing actions. Must read like a natural, "
            "personal email from one colleague to another: greet one person by first name, "
            "60-120 words, short sentences, one specific ask, sign off with 'Best,'. "
            "No quantified claims unless backed by cited evidence."
        ),
    )
