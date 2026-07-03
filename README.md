# Deal Action Copilot

Policy-governed agentic RAG demo: given an open B2B opportunity, the copilot assembles
fragmented deal context, applies **deterministic policy gates**, retrieves **approved
knowledge** (Qdrant RAG), enriches with **live web research** (Tavily), and produces a
**fixed-schema recommendation** for the next-best coordinated GTM action — which a human
reviews with approve / edit / reject + reason codes. Abstention ("gather information",
"no action yet") is a first-class output.

Companion MVP to the *Deal Action Copilot — MVP Outline* one-pager.

## Architecture

```
Next.js (Vercel)  ──SSE──  FastAPI (@vercel/python, api/index.py)
                              └─ LangGraph bounded workflow (api/_copilot/graph.py)
                                   assemble → eligibility gate → research loop
                                   (RAG tool + Tavily tool, max 3 turns)
                                   → structured output (Pydantic) → validation gate
                                   → one retry → deterministic abstain fallback
                              └─ Qdrant (Cloud if configured, else in-memory)
```

- **Synthetic data:** `api/_copilot/data/opportunities.json` (mock CRM, 5 deals including
  an evidence-poor one that triggers abstention) and `api/_copilot/data/knowledge/*.md`
  (mock approved product docs, case studies, playbooks, messaging policy).
- **Policy gates** (`api/_copilot/policies.py`) are plain Python, no LLM: eligibility rules
  before generation, guardrail validation after (citation integrity, action eligibility,
  quantified-claims check, confidence rules).

## Setup

1. Fill in the keys in `.env` (already created at repo root):
   - `OPENAI_API_KEY` (required)
   - `TAVILY_API_KEY` (recommended — free at [tavily.com](https://app.tavily.com); without
     it the copilot degrades gracefully and skips web enrichment)
   - `QDRANT_URL` / `QDRANT_API_KEY` (optional — [Qdrant Cloud free tier](https://cloud.qdrant.io);
     without them an in-memory index is rebuilt per cold start)
   - `LANGSMITH_API_KEY` (optional — [smith.langchain.com](https://smith.langchain.com); with
     `LANGSMITH_TRACING=true` every run is traced end-to-end in the
     `deal-action-copilot` project, and the UI header links to it)

2. Install:

```bash
uv sync          # python deps
npm install      # frontend deps
```

## Run locally (two terminals)

```bash
npm run fastapi-dev   # FastAPI on :8000
npm run dev           # Next.js on :3000 (proxies /api/py/* to :8000)
```

Open http://localhost:3000, pick an opportunity, and click **Start workflow**. The copilot
pauses at each of four steps for your review and approval — edits in early steps carry
forward into research and the final draft.

| Step | What happens | Can you edit? |
|------|----------------|---------------|
| 1 — Review the deal | CRM snapshot, signals, gaps | Yes — signals, gaps, analyst notes |
| 2 — Playbook rules | Constraints and allowed action types | Yes — constraints, escalations, eligible types |
| 3 — Research | Approved-knowledge RAG + Tavily web search | Review log, then approve |
| 4 — Draft | Structured recommendation + optional email | Yes — action, draft message; final approve/reject |

> Dev note: `.env.development` sets `NEXT_PUBLIC_API_BASE=http://localhost:8000` so the
> browser streams SSE directly from FastAPI — the Next dev proxy buffers `text/event-stream`
> and would make runs appear to hang. In production this variable is unset and the app uses
> same-origin `/api/py/*` routes.

### LangGraph Studio (optional)

The graph is also exposed via `langgraph.json`, so you can inspect and step through it in
Studio:

```bash
uv run langgraph dev   # opens Studio against the local graph on :2024
```

Suggested demo path (9 synthetic deals across 9 verticals, 3 of them expansions):
- **Meridian Digital Bank** (new logo, technical evaluation) — rich evidence → coordinated
  action with cited case study.
- **Apex Digital Exchange** (expansion, business validation) — deployed KYC customer with an
  ATO spike → device-intelligence expansion; exec alignment allowed at this stage.
- **GigBridge** (expansion, discovery) — no budget owner yet → problem-sizing motion, not a pitch.
- **BrightPath Health** (stalled, 20 days quiet) — legal blocker → diagnose inactivity, escalate
  BAA internally, no drafted terms.
- **Helios Telecom** (procurement) — pricing/legal blockers → internal escalation, no drafted terms.
- **LoopRide** — empty record → the eligibility gate blocks outreach and the copilot abstains.

## Deploy to Vercel

```bash
vercel
```

The repo follows the Next.js + FastAPI pattern: Vercel builds the Next.js app and deploys
`api/index.py` as a Python serverless function (deps from `requirements.txt`). Set the
same env vars in the Vercel project settings. Use Qdrant Cloud in production so the index
survives cold starts.

## Notes / known limits

- Review log is in-memory (per warm serverless instance) — a demo simplification; swap for
  Vercel KV/Postgres to persist.
- The agent research loop is bounded to 3 LLM turns; validation allows one regeneration
  before the deterministic abstain fallback.
