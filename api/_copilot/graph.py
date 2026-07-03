"""Bounded hybrid decision workflow as a LangGraph.

assemble -> eligibility gate -> research loop (agent + tools, bounded)
        -> generate (structured output) -> validation gate (one retry, then abstain)
"""

import re
from typing import Annotated, Any, Iterator, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import MAX_RESEARCH_TURNS, OPENAI_MODEL
from .data_access import detect_signals, get_opportunity, opportunity_summary_text
from .policies import eligibility_gate, validation_gate
from .schemas import ActionType, Confidence, Evidence, Recommendation
from .tools import TOOL_BELT

SOURCE_ID_PATTERN = re.compile(r"\[((?:KB|WEB):[A-Za-z0-9_-]+)\]")


class CopilotState(TypedDict):
    opportunity_id: str
    opportunity: dict
    signals: dict
    policy: dict
    messages: Annotated[list, add_messages]
    known_source_ids: list[str]
    recommendation: Optional[dict]
    validation: Optional[dict]
    validation_feedback: Optional[str]
    retry_count: int


def _llm() -> ChatOpenAI:
    # GPT-5 family rejects non-default temperature; older models get 0 for determinism.
    if OPENAI_MODEL.startswith(("gpt-5", "o")):
        return ChatOpenAI(model=OPENAI_MODEL)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# ── Nodes ────────────────────────────────────────────────────────────────────

def assemble_context(state: CopilotState) -> dict:
    opp = get_opportunity(state["opportunity_id"])
    signals = detect_signals(opp)
    return {"opportunity": opp, "signals": signals, "known_source_ids": ["CRM:opportunity"]}


def apply_eligibility_gate(state: CopilotState) -> dict:
    policy = eligibility_gate(state["opportunity"], state["signals"])
    return {"policy": policy}


RESEARCH_SYSTEM = """You are the research step of the Deal Action Copilot, a policy-governed \
GTM assistant. Your only job is to gather evidence before a recommendation is generated.

Use the tools to:
1. retrieve_socure_knowledge — find the approved Socure products, case studies, playbooks, \
and messaging policy relevant to this deal's problems and vertical.
2. research_account — one web search for recent public news about the account, if the \
account looks substantial enough to have public news.

Make at most {max_turns} rounds of tool calls, then stop and summarize what you found in \
2-3 sentences. Do NOT produce a recommendation."""


def research_agent(state: CopilotState) -> dict:
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


def route_research(state: CopilotState) -> str:
    last = state["messages"][-1]
    ai_turns = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if isinstance(last, AIMessage) and last.tool_calls and ai_turns <= MAX_RESEARCH_TURNS:
        return "tools"
    return "generate"


def collect_sources(state: CopilotState) -> dict:
    """Harvest citable source ids from tool outputs."""
    found = set(state.get("known_source_ids", ["CRM:opportunity"]))
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            found.update(SOURCE_ID_PATTERN.findall(str(msg.content)))
    return {"known_source_ids": sorted(found)}


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


def generate_recommendation(state: CopilotState) -> dict:
    llm = _llm().with_structured_output(Recommendation)
    evidence_blobs = [
        str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)
    ]
    summary_msgs = [
        str(m.content) for m in state["messages"]
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


def apply_validation_gate(state: CopilotState) -> dict:
    rec = Recommendation.model_validate(state["recommendation"])
    result = validation_gate(rec, state["policy"], set(state["known_source_ids"]))
    update: dict[str, Any] = {"validation": result}
    if not result["passed"]:
        update["retry_count"] = state.get("retry_count", 0) + 1
        update["validation_feedback"] = "\n".join(f"- {f}" for f in result["failures"])
    return update


def route_validation(state: CopilotState) -> str:
    if state["validation"]["passed"]:
        return END
    if state.get("retry_count", 0) <= 1:
        return "generate"
    return "fallback"


def fallback_abstain(state: CopilotState) -> dict:
    """Deterministic safe output when generation cannot pass validation."""
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


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(CopilotState)
    g.add_node("assemble", assemble_context)
    g.add_node("eligibility", apply_eligibility_gate)
    g.add_node("agent", research_agent)
    g.add_node("tools", ToolNode(TOOL_BELT))
    g.add_node("collect", collect_sources)
    g.add_node("generate", generate_recommendation)
    g.add_node("validate", apply_validation_gate)
    g.add_node("fallback", fallback_abstain)

    g.add_edge(START, "assemble")
    g.add_edge("assemble", "eligibility")
    g.add_edge("eligibility", "agent")
    g.add_conditional_edges("agent", route_research, {"tools": "tools", "generate": "collect"})
    g.add_edge("tools", "agent")
    g.add_edge("collect", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_validation, {END: END, "generate": "generate", "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


GRAPH = build_graph()


# ── Streaming runner (SSE-friendly events) ───────────────────────────────────

STAGE_LABELS = {
    "assemble": "Reviewing the deal",
    "eligibility": "Applying playbook rules",
    "agent": "Researching the account",
    "tools": "Looking things up",
    "collect": "Collecting sources",
    "generate": "Drafting the recommendation",
    "validate": "Final compliance check",
    "fallback": "Holding off — more info needed",
}


def _event(type_: str, **payload) -> dict:
    return {"type": type_, **payload}


def stream_run(opportunity_id: str) -> Iterator[dict]:
    """Run the graph and yield UI-friendly events per node update."""
    final_state: dict = {}
    yield _event("stage", stage="assemble", label=STAGE_LABELS["assemble"], status="running")
    for update in GRAPH.stream(
        {"opportunity_id": opportunity_id, "retry_count": 0},
        stream_mode="updates",
    ):
        for node, out in update.items():
            out = out or {}
            final_state.update(out)
            if node == "assemble":
                sig = out["signals"]
                yield _event(
                    "stage", stage=node, label=STAGE_LABELS[node], status="done",
                    detail=f"Found {len(sig['signals'])} things worth acting on and "
                           f"{len(sig['gaps'])} gaps in what we know",
                    signals=sig["signals"], gaps=sig["gaps"],
                )
                yield _event("stage", stage="eligibility", label=STAGE_LABELS["eligibility"], status="running")
            elif node == "eligibility":
                pol = out["policy"]
                yield _event(
                    "stage", stage=node, label=STAGE_LABELS[node], status="done",
                    detail=f"{len(pol['constraints'])} playbook rule(s) apply to this deal; "
                           f"{len(pol['eligible_action_types'])} of 5 action types allowed",
                    constraints=pol["constraints"], escalations=pol["escalations"],
                    eligible_action_types=pol["eligible_action_types"],
                )
                yield _event("stage", stage="agent", label=STAGE_LABELS["agent"], status="running")
            elif node == "agent":
                for msg in out.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield _event(
                                "tool_call", tool=tc["name"],
                                query=tc["args"].get("query", ""),
                            )
            elif node == "tools":
                for msg in out.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        sources = sorted(set(SOURCE_ID_PATTERN.findall(str(msg.content))))
                        yield _event(
                            "tool_result", tool=msg.name,
                            sources=sources,
                            preview=str(msg.content)[:280],
                        )
            elif node == "collect":
                yield _event(
                    "stage", stage="agent", label=STAGE_LABELS["agent"], status="done",
                    detail=f"{len(out['known_source_ids'])} sources gathered to back up the recommendation",
                    sources=out["known_source_ids"],
                )
                yield _event("stage", stage="generate", label=STAGE_LABELS["generate"], status="running")
            elif node == "generate":
                yield _event(
                    "stage", stage=node, label=STAGE_LABELS[node], status="done",
                    detail="Draft recommendation ready",
                )
                yield _event("stage", stage="validate", label=STAGE_LABELS["validate"], status="running")
            elif node == "validate":
                val = out["validation"]
                yield _event(
                    "stage", stage=node, label=STAGE_LABELS[node],
                    status="done" if val["passed"] else "failed",
                    detail="All checks passed — every claim is backed by a source"
                    if val["passed"]
                    else f"{len(val['failures'])} issue(s) caught — redrafting",
                    failures=val["failures"], warnings=val["warnings"],
                )
                if not val["passed"]:
                    yield _event("stage", stage="generate", label=STAGE_LABELS["generate"], status="running")
            elif node == "fallback":
                yield _event(
                    "stage", stage=node, label=STAGE_LABELS[node], status="done",
                    detail="Not enough solid evidence — recommending to gather more information instead",
                )

    yield _event(
        "result",
        recommendation=final_state.get("recommendation"),
        validation=final_state.get("validation"),
        sources=final_state.get("known_source_ids", []),
    )
