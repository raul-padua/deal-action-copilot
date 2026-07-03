"""Human-in-the-loop session workflow — one approval gate per stage."""

from __future__ import annotations

import re
import uuid
from typing import Any, Iterator

from langchain_core.messages import AIMessage, ToolMessage

from .data_access import detect_signals, get_opportunity, opportunity_summary_text
from .graph import (
    GENERATE_SYSTEM,
    RESEARCH_SYSTEM,
    SOURCE_ID_PATTERN,
    _llm,
    apply_validation_gate,
    fallback_abstain,
    generate_recommendation,
    research_agent,
)
from .policies import eligibility_gate
from .tools import TOOL_BELT

SESSIONS: dict[str, dict[str, Any]] = {}

STAGE_NAMES = {
    1: "Review the deal",
    2: "Apply playbook rules",
    3: "Research the account",
    4: "Draft the next move",
}

ACTION_TYPES = [
    "customer_outreach",
    "coordinated_sales_marketing",
    "internal_preparation",
    "gather_information",
    "no_action_yet",
]


def _calc_richness(opp: dict, signals: list[str], analyst_notes: str) -> int:
    base = detect_signals(opp)["evidence_richness"]
    bonus = min(len(signals), 3) // 3
    if analyst_notes.strip():
        bonus += 1
    return min(base + bonus, 8)


def _merge_signals(opp: dict, signals: list[str], gaps: list[str], analyst_notes: str) -> dict:
    return {
        "signals": signals,
        "gaps": gaps,
        "analyst_notes": analyst_notes.strip(),
        "evidence_richness": _calc_richness(opp, signals, analyst_notes),
    }


def _session_public(s: dict) -> dict:
    return {
        "session_id": s["id"],
        "opportunity_id": s["opportunity_id"],
        "account": s["opportunity"]["account"],
        "current_stage": s["current_stage"],
        "stage_name": STAGE_NAMES[s["current_stage"]],
        "stage1": {
            "summary": opportunity_summary_text(s["opportunity"]),
            "signals": s["signals"]["signals"],
            "gaps": s["signals"]["gaps"],
            "analyst_notes": s["signals"].get("analyst_notes", ""),
        }
        if s["current_stage"] >= 1
        else None,
        "stage2": s.get("policy")
        if s["current_stage"] >= 2 and s.get("policy")
        else None,
        "stage3": {
            "feed": s.get("feed", []),
            "sources": s.get("known_source_ids", []),
            "research_summary": s.get("research_summary", ""),
            "status": s.get("research_status", "pending"),
        }
        if s["current_stage"] >= 3
        else None,
        "stage4": {
            "recommendation": s.get("recommendation"),
            "validation": s.get("validation"),
        }
        if s["current_stage"] >= 4 and s.get("recommendation")
        else None,
        "completed_stages": s.get("completed_stages", []),
    }


def start_session(opportunity_id: str) -> dict:
    opp = get_opportunity(opportunity_id)
    auto = detect_signals(opp)
    sid = uuid.uuid4().hex[:10]
    SESSIONS[sid] = {
        "id": sid,
        "opportunity_id": opportunity_id,
        "opportunity": opp,
        "current_stage": 1,
        "completed_stages": [],
        "signals": {
            "signals": auto["signals"],
            "gaps": auto["gaps"],
            "analyst_notes": "",
            "evidence_richness": auto["evidence_richness"],
        },
        "policy": None,
        "messages": [],
        "known_source_ids": ["CRM:opportunity"],
        "feed": [],
        "research_summary": "",
        "research_status": "pending",
        "recommendation": None,
        "validation": None,
        "retry_count": 0,
    }
    return _session_public(SESSIONS[sid])


def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise KeyError(session_id)
    return _session_public(SESSIONS[session_id])


def approve_stage1(
    session_id: str,
    signals: list[str],
    gaps: list[str],
    analyst_notes: str = "",
) -> dict:
    s = SESSIONS[session_id]
    if s["current_stage"] != 1:
        raise ValueError("Not on stage 1")
    s["signals"] = _merge_signals(s["opportunity"], signals, gaps, analyst_notes)
    s["policy"] = eligibility_gate(s["opportunity"], s["signals"])
    s["completed_stages"] = [1]
    s["current_stage"] = 2
    return _session_public(s)


def approve_stage2(
    session_id: str,
    constraints: list[str],
    escalations: list[str],
    eligible_action_types: list[str],
) -> dict:
    s = SESSIONS[session_id]
    if s["current_stage"] != 2:
        raise ValueError("Not on stage 2")
    invalid = set(eligible_action_types) - set(ACTION_TYPES)
    if invalid:
        raise ValueError(f"Invalid action types: {invalid}")
    s["policy"] = {
        "constraints": constraints,
        "escalations": escalations,
        "eligible_action_types": eligible_action_types,
    }
    s["completed_stages"] = [1, 2]
    s["current_stage"] = 3
    s["research_status"] = "ready"
    return _session_public(s)


def _copilot_state(s: dict) -> dict:
    return {
        "opportunity_id": s["opportunity_id"],
        "opportunity": s["opportunity"],
        "signals": s["signals"],
        "policy": s["policy"],
        "messages": s.get("messages", []),
        "known_source_ids": s.get("known_source_ids", ["CRM:opportunity"]),
        "recommendation": s.get("recommendation"),
        "validation": s.get("validation"),
        "validation_feedback": s.get("validation_feedback"),
        "retry_count": s.get("retry_count", 0),
    }


def _run_tools(messages: list) -> list:
    """Execute tool calls from the last AI message."""
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return []
    tool_map = {t.name: t for t in TOOL_BELT}
    out = []
    for tc in last.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        out.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
        )
    return out


def stream_research(session_id: str) -> Iterator[dict]:
    s = SESSIONS[session_id]
    if s["current_stage"] != 3:
        raise ValueError("Not on stage 3")
    s["research_status"] = "running"
    s["messages"] = []
    s["feed"] = []
    yield {"type": "research_start"}

    state = _copilot_state(s)

    for _ in range(6):  # safety cap
        out = research_agent(state)
        if not state.get("messages"):
            state["messages"] = out.get("messages", [])
        else:
            state["messages"] = state["messages"] + out.get("messages", [])

        last = state["messages"][-1]

        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                evt = {"type": "tool_call", "tool": tc["name"], "query": tc["args"].get("query", "")}
                s["feed"].append(evt)
                yield evt
            for msg in _run_tools(state["messages"]):
                state["messages"].append(msg)
                if isinstance(msg, ToolMessage):
                    sources = sorted(set(SOURCE_ID_PATTERN.findall(str(msg.content))))
                    evt = {
                        "type": "tool_result",
                        "tool": msg.name,
                        "sources": sources,
                        "preview": str(msg.content)[:280],
                    }
                    s["feed"].append(evt)
                    yield evt
            continue

        if isinstance(last, AIMessage) and last.content:
            s["research_summary"] = str(last.content)
            break

    found = set(["CRM:opportunity"])
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            found.update(SOURCE_ID_PATTERN.findall(str(msg.content)))
    s["known_source_ids"] = sorted(found)
    s["messages"] = state["messages"]
    s["research_status"] = "done"
    yield {
        "type": "research_done",
        "sources": s["known_source_ids"],
        "research_summary": s["research_summary"],
        "feed": s["feed"],
    }


def approve_stage3(session_id: str) -> dict:
    s = SESSIONS[session_id]
    if s["current_stage"] != 3:
        raise ValueError("Not on stage 3")
    if s["research_status"] != "done":
        raise ValueError("Research not complete")

    state = _copilot_state(s)
    out = generate_recommendation(state)
    state.update(out)
    state.update(apply_validation_gate(state))

    if not state["validation"]["passed"] and state.get("retry_count", 0) <= 1:
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["validation_feedback"] = "\n".join(f"- {f}" for f in state["validation"]["failures"])
        out = generate_recommendation(state)
        state.update(out)
        state.update(apply_validation_gate(state))

    if not state["validation"]["passed"]:
        fb = fallback_abstain(state)
        state.update(fb)

    s["recommendation"] = state["recommendation"]
    s["validation"] = state["validation"]
    s["known_source_ids"] = state["known_source_ids"]
    s["completed_stages"] = [1, 2, 3]
    s["current_stage"] = 4
    return _session_public(s)


def approve_stage4(session_id: str) -> dict:
    s = SESSIONS[session_id]
    if s["current_stage"] != 4:
        raise ValueError("Not on stage 4")
    s["completed_stages"] = [1, 2, 3, 4]
    return _session_public(s)
