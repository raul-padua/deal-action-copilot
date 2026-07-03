"""LangGraph Studio entry — not loaded on Vercel (keeps serverless memory down)."""

from typing import Annotated, Any, Iterator, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import MAX_RESEARCH_TURNS
from .data_access import detect_signals, get_opportunity, opportunity_summary_text
from .graph import (
    GENERATE_SYSTEM,
    RESEARCH_SYSTEM,
    SOURCE_ID_PATTERN,
    apply_validation_gate,
    assemble_context,
    fallback_abstain,
    generate_recommendation,
    research_agent,
    route_research,
)
from .policies import eligibility_gate
from .tools import TOOL_BELT


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


def apply_eligibility_gate(state: CopilotState) -> dict:
    policy = eligibility_gate(state["opportunity"], state["signals"])
    return {"policy": policy}


def collect_sources(state: CopilotState) -> dict:
    found = set(state.get("known_source_ids", ["CRM:opportunity"]))
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            found.update(SOURCE_ID_PATTERN.findall(str(msg.content)))
    return {"known_source_ids": sorted(found)}


def route_validation(state: CopilotState) -> str:
    if state["validation"]["passed"]:
        return END
    if state.get("retry_count", 0) <= 1:
        return "generate"
    return "fallback"


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
    g.add_conditional_edges(
        "validate", route_validation, {END: END, "generate": "generate", "fallback": "fallback"}
    )
    g.add_edge("fallback", END)
    return g.compile()


GRAPH = build_graph()

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
                    "stage",
                    stage=node,
                    label=STAGE_LABELS[node],
                    status="done",
                    detail=f"Found {len(sig['signals'])} things worth acting on and "
                    f"{len(sig['gaps'])} gaps in what we know",
                    signals=sig["signals"],
                    gaps=sig["gaps"],
                )
                yield _event(
                    "stage", stage="eligibility", label=STAGE_LABELS["eligibility"], status="running"
                )
            elif node == "eligibility":
                pol = out["policy"]
                yield _event(
                    "stage",
                    stage=node,
                    label=STAGE_LABELS[node],
                    status="done",
                    detail=f"{len(pol['constraints'])} playbook rule(s) apply to this deal; "
                    f"{len(pol['eligible_action_types'])} of 5 action types allowed",
                    constraints=pol["constraints"],
                    escalations=pol["escalations"],
                    eligible_action_types=pol["eligible_action_types"],
                )
                yield _event("stage", stage="agent", label=STAGE_LABELS["agent"], status="running")
            elif node == "agent":
                for msg in out.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield _event(
                                "tool_call",
                                tool=tc["name"],
                                query=tc["args"].get("query", ""),
                            )
            elif node == "tools":
                for msg in out.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        sources = sorted(set(SOURCE_ID_PATTERN.findall(str(msg.content))))
                        yield _event(
                            "tool_result",
                            tool=msg.name,
                            sources=sources,
                            preview=str(msg.content)[:280],
                        )
            elif node == "collect":
                yield _event(
                    "stage",
                    stage="agent",
                    label=STAGE_LABELS["agent"],
                    status="done",
                    detail=f"{len(out['known_source_ids'])} sources gathered to back up the recommendation",
                    sources=out["known_source_ids"],
                )
                yield _event(
                    "stage", stage="generate", label=STAGE_LABELS["generate"], status="running"
                )
            elif node == "generate":
                yield _event(
                    "stage",
                    stage=node,
                    label=STAGE_LABELS[node],
                    status="done",
                    detail="Draft recommendation ready",
                )
                yield _event(
                    "stage", stage="validate", label=STAGE_LABELS["validate"], status="running"
                )
            elif node == "validate":
                val = out["validation"]
                yield _event(
                    "stage",
                    stage=node,
                    label=STAGE_LABELS[node],
                    status="done" if val["passed"] else "failed",
                    detail="All checks passed — every claim is backed by a source"
                    if val["passed"]
                    else f"{len(val['failures'])} issue(s) caught — redrafting",
                    failures=val["failures"],
                    warnings=val["warnings"],
                )
                if not val["passed"]:
                    yield _event(
                        "stage", stage="generate", label=STAGE_LABELS["generate"], status="running"
                    )
            elif node == "fallback":
                yield _event(
                    "stage",
                    stage=node,
                    label=STAGE_LABELS[node],
                    status="done",
                    detail="Not enough solid evidence — recommending to gather more information instead",
                )

    yield _event(
        "result",
        recommendation=final_state.get("recommendation"),
        validation=final_state.get("validation"),
        sources=final_state.get("known_source_ids", []),
    )
