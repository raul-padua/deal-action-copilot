"""Agent core — research + structured generation (no LangGraph on the import path)."""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .config import MAX_RESEARCH_TURNS, OPENAI_MODEL
from .data_access import detect_signals, get_opportunity, opportunity_summary_text
from .policies import eligibility_gate, validation_gate
from .schemas import ActionType, Confidence, Evidence, Recommendation
from .tools import TOOL_BELT

SOURCE_ID_PATTERN = re.compile(r"\[((?:KB|WEB):[A-Za-z0-9_-]+)\]")


def _llm() -> ChatOpenAI:
    if OPENAI_MODEL.startswith(("gpt-5", "o")):
        return ChatOpenAI(model=OPENAI_MODEL)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def assemble_context(state: dict) -> dict:
    opp = get_opportunity(state["opportunity_id"])
    signals = detect_signals(opp)
    return {"opportunity": opp, "signals": signals, "known_source_ids": ["CRM:opportunity"]}


RESEARCH_SYSTEM = """You are the research step of the Deal Action Copilot, a policy-governed \
GTM assistant. Your only job is to gather evidence before a recommendation is generated.

Use the tools to:
1. retrieve_socure_knowledge — find the approved Socure products, case studies, playbooks, \
and messaging policy relevant to this deal's problems and vertical.
2. research_account — one web search for recent public news about the account, if the \
account looks substantial enough to have public news.

Make at most {max_turns} rounds of tool calls, then stop and summarize what you found in \
2-3 sentences. Do NOT produce a recommendation."""


def research_agent(state: dict) -> dict:
    llm = _llm().bind_tools(TOOL_BELT)
    if not state.get("messages"):
        constraints = "\n".join(f"- {c}" for c in state["policy"]["constraints"]) or "- none"
        analyst = state["signals"].get("analyst_notes") or "none"
        seed = [
            SystemMessage(content=RESEARCH_SYSTEM.format(max_turns=MAX_RESEARCH_TURNS)),
            HumanMessage(
                content=(
                    f"Opportunity snapshot [CRM:opportunity]:\n{opportunity_summary_text(state['opportunity'])}\n\n"
                    f"Detected signals: {'; '.join(state['signals']['signals']) or 'none'}\n"
                    f"Detected gaps: {'; '.join(state['signals']['gaps']) or 'none'}\n"
                    f"Analyst notes (human-approved): {analyst}\n"
                    f"Policy constraints already applied:\n{constraints}"
                )
            ),
        ]
        response = llm.invoke(seed)
        return {"messages": seed + [response]}
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def route_research(state: dict) -> str:
    last = state["messages"][-1]
    ai_turns = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if isinstance(last, AIMessage) and last.tool_calls and ai_turns <= MAX_RESEARCH_TURNS:
        return "tools"
    return "generate"


GENERATE_SYSTEM = """You are the structured-reasoning step of the Deal Action Copilot. \
Produce ONE recommendation for the next-best coordinated GTM action, following the fixed schema.

Rules:
- Separate fact from inference; every material claim in `evidence` must cite one of the \
allowed source ids EXACTLY as given. Never invent a source id.
- Only use quantified numbers that appear in retrieved KB content, and cite that KB source.
- Respect every policy constraint listed. If the eligible action types exclude \
customer-facing actions, recommend gather_information, internal_preparation, or no_action_yet.
- Abstaining (gather_information / no_action_yet) is the CORRECT output when evidence is \
insufficient — do not force outreach.
- Draft messages must read like an email a thoughtful colleague would actually send, not \
marketing copy:
  * Address ONE person by first name — the target stakeholder. Never combine names with \
slashes.
  * 60-120 words. Shape: a one-line opener referencing something specific and recent from \
the deal; one short paragraph connecting THEIR stated problem to why a conversation is \
worth their time; then a single, low-friction ask with a concrete time suggestion.
  * Plain, warm, direct language. Short sentences. No feature lists, no more than one \
number, no buzzwords ("seamless", "leverage", "streamline"), at most one dash in the \
whole message.
  * Do not cram the product pitch into the ask — the meeting is the pitch.
  * Never name competitors, pricing, or contractual terms.
  * End with "Best,\\n[Your name]"."""


def generate_recommendation(state: dict) -> dict:
    llm = _llm().with_structured_output(Recommendation)
    evidence_blobs = [str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]
    summary_msgs = [
        str(m.content)
        for m in state["messages"]
        if isinstance(m, AIMessage) and not m.tool_calls and m.content
    ]
    policy = state["policy"]
    prompt = (
        f"Opportunity snapshot [CRM:opportunity]:\n{opportunity_summary_text(state['opportunity'])}\n\n"
        f"Detected signals: {'; '.join(state['signals']['signals']) or 'none'}\n"
        f"Detected gaps: {'; '.join(state['signals']['gaps']) or 'none'}\n"
        f"Analyst notes (human-approved): {state['signals'].get('analyst_notes') or 'none'}\n\n"
        f"Eligible action types: {', '.join(policy['eligible_action_types'])}\n"
        f"Policy constraints:\n" + ("\n".join(f"- {c}" for c in policy["constraints"]) or "- none") + "\n"
        f"Escalations:\n" + ("\n".join(f"- {e}" for e in policy["escalations"]) or "- none") + "\n\n"
        f"Retrieved evidence:\n" + ("\n\n".join(evidence_blobs) or "none") + "\n\n"
        f"Research summary: {' '.join(summary_msgs) or 'none'}\n\n"
        f"Allowed source ids: {', '.join(state['known_source_ids'])}"
    )
    if state.get("validation_feedback"):
        prompt += (
            "\n\nYour previous attempt FAILED policy validation. Fix these issues:\n"
            + state["validation_feedback"]
        )
    rec = llm.invoke([SystemMessage(content=GENERATE_SYSTEM), HumanMessage(content=prompt)])
    return {"recommendation": rec.model_dump(mode="json")}


def apply_validation_gate(state: dict) -> dict:
    rec = Recommendation.model_validate(state["recommendation"])
    result = validation_gate(rec, state["policy"], set(state["known_source_ids"]))
    update: dict[str, Any] = {"validation": result}
    if not result["passed"]:
        update["retry_count"] = state.get("retry_count", 0) + 1
        update["validation_feedback"] = "\n".join(f"- {f}" for f in result["failures"])
    return update


def fallback_abstain(state: dict) -> dict:
    gaps = state["signals"]["gaps"] or ["Evidence base insufficient for a compliant recommendation"]
    rec = Recommendation(
        action_type=ActionType.gather_information,
        action=(
            "Do not act on this opportunity yet. The copilot could not produce a "
            "policy-compliant recommendation; gather the missing context first."
        ),
        owner="AE",
        target_stakeholder=None,
        timing="before any customer-facing follow-up",
        rationale=(
            "Generated recommendations failed policy validation twice. Per governance "
            "rules the system abstains rather than shipping an unsupported action."
        ),
        socure_angle=None,
        supporting_asset=None,
        evidence=[Evidence(source_id="CRM:opportunity", claim="Opportunity record lacks sufficient verified context.")],
        confidence=Confidence.low,
        missing_information=gaps,
        draft_message=None,
    )
    return {
        "recommendation": rec.model_dump(mode="json"),
        "validation": {
            "passed": True,
            "failures": [],
            "warnings": ["Deterministic fallback: model output failed validation twice; abstained."],
        },
    }
