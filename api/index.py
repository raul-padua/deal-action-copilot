"""FastAPI entry — served by @vercel/python at /api/*, or locally via uvicorn."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from _copilot.data_access import get_followup, load_opportunities, record_followup
from _copilot.sessions import (
    approve_stage1,
    approve_stage2,
    approve_stage3,
    approve_stage4,
    get_session,
    start_session,
    stream_research,
)

app = FastAPI(title="Deal Action Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REVIEWS: list[dict] = []


class StartSessionIn(BaseModel):
    opportunity_id: str


class Stage1In(BaseModel):
    signals: list[str]
    gaps: list[str]
    analyst_notes: str = ""


class Stage2In(BaseModel):
    constraints: list[str]
    escalations: list[str]
    eligible_action_types: list[str]


class ReviewIn(BaseModel):
    opportunity_id: str
    session_id: str | None = None
    decision: str
    action: str | None = None
    reason_code: str | None = None
    notes: str | None = None
    edited_action: str | None = None
    draft_message: str | None = None


@app.get("/api/py/health")
def health():
    langsmith_on = bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv(
        "LANGSMITH_TRACING", ""
    ).lower() in ("true", "1")
    return {
        "ok": True,
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "qdrant_cloud": bool(os.getenv("QDRANT_URL")),
        "langsmith": langsmith_on,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "deal-action-copilot"),
    }


@app.get("/api/py/opportunities")
def opportunities():
    return [
        {
            "id": o["id"],
            "account": o["account"],
            "vertical": o["vertical"],
            "motion": o["motion"],
            "stage": o["stage"],
            "value_usd": o["value_usd"],
            "days_in_stage": o["days_in_stage"],
            "days_since_last_activity": o["days_since_last_activity"],
            "notes": o.get("notes", ""),
            "followup": get_followup(o["id"]),
        }
        for o in load_opportunities()
    ]


@app.post("/api/py/sessions")
def create_session(body: StartSessionIn):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    valid = {o["id"] for o in load_opportunities()}
    if body.opportunity_id not in valid:
        raise HTTPException(status_code=404, detail="Unknown opportunity.")
    return start_session(body.opportunity_id)


@app.get("/api/py/sessions/{session_id}")
def read_session(session_id: str):
    try:
        return get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")


@app.post("/api/py/sessions/{session_id}/stage1/approve")
def stage1_approve(session_id: str, body: Stage1In):
    try:
        return approve_stage1(session_id, body.signals, body.gaps, body.analyst_notes)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/py/sessions/{session_id}/stage2/approve")
def stage2_approve(session_id: str, body: Stage2In):
    try:
        return approve_stage2(
            session_id, body.constraints, body.escalations, body.eligible_action_types
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/py/sessions/{session_id}/research")
def research_stream(session_id: str):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    def sse():
        try:
            for event in stream_research(session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/py/sessions/{session_id}/stage3/approve")
def stage3_approve(session_id: str):
    try:
        return approve_stage3(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/py/sessions/{session_id}/stage4/approve")
def stage4_approve(session_id: str):
    try:
        return approve_stage4(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/py/reviews")
def add_review(review: ReviewIn):
    entry = {**review.model_dump(), "ts": time.time()}
    REVIEWS.append(entry)
    followup = None
    if review.decision in ("approved", "edited"):
        action = review.edited_action or review.action or "Follow-up approved"
        followup = record_followup(review.opportunity_id, action, review.decision)
    return {"ok": True, "count": len(REVIEWS), "followup": followup}


@app.get("/api/py/reviews")
def list_reviews():
    return REVIEWS
