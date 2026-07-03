"use client";

import { useEffect, useRef, useState } from "react";

type Followup = { action: string; decision: string; ts: number } | null;

type Opportunity = {
  id: string;
  account: string;
  vertical: string;
  motion: string;
  stage: string;
  value_usd: number;
  days_in_stage: number;
  days_since_last_activity: number;
  notes: string;
  followup: Followup;
};

type Session = {
  session_id: string;
  opportunity_id: string;
  account: string;
  current_stage: number;
  stage_name: string;
  completed_stages: number[];
  stage1?: {
    summary: string;
    signals: string[];
    gaps: string[];
    analyst_notes: string;
  };
  stage2?: {
    constraints: string[];
    escalations: string[];
    eligible_action_types: string[];
  };
  stage3?: {
    feed: FeedItem[];
    sources: string[];
    research_summary: string;
    research_notes: string;
    status: string;
  };
  stage4?: {
    recommendation: Recommendation | null;
    validation: Validation | null;
  };
};

type FeedItem =
  | { type: "tool_call"; tool: string; query: string }
  | { type: "tool_result"; tool: string; sources: string[]; preview: string };

type Evidence = { source_id: string; claim: string };

type Recommendation = {
  action_type: string;
  action: string;
  owner: string;
  target_stakeholder?: string | null;
  timing: string;
  rationale: string;
  socure_angle?: string | null;
  supporting_asset?: string | null;
  evidence: Evidence[];
  confidence: string;
  missing_information: string[];
  draft_message?: string | null;
};

type Validation = { passed: boolean; failures: string[]; warnings: string[] };

type Health = {
  ok: boolean;
  openai: boolean;
  tavily: boolean;
  qdrant_cloud: boolean;
  langsmith: boolean;
  langsmith_project: string;
};

const API = process.env.NEXT_PUBLIC_API_BASE ?? "";

const STEPS = [
  "Review the deal",
  "Apply playbook rules",
  "Research the account",
  "Draft the next move",
];

const ACTION_TYPES = [
  "customer_outreach",
  "coordinated_sales_marketing",
  "internal_preparation",
  "gather_information",
  "no_action_yet",
];

const REASON_CODES = [
  "wrong_context",
  "wrong_persona",
  "wrong_product",
  "weak_evidence",
  "poor_timing",
  "policy_issue",
];

const fmtUsd = (n: number) => `$${(n / 1000).toFixed(0)}K`;
const titleize = (s: string) => s.replaceAll("_", " ");
const listToText = (items: string[]) => items.join("\n");
const textToList = (text: string) =>
  text.split("\n").map((l) => l.trim()).filter(Boolean);

export default function Page() {
  const [opps, setOpps] = useState<Opportunity[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Stage 1 edits
  const [signalsText, setSignalsText] = useState("");
  const [gapsText, setGapsText] = useState("");
  const [analystNotes, setAnalystNotes] = useState("");

  // Stage 2 edits
  const [constraintsText, setConstraintsText] = useState("");
  const [escalationsText, setEscalationsText] = useState("");
  const [eligibleTypes, setEligibleTypes] = useState<string[]>([]);

  // Stage 3
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [researchSummary, setResearchSummary] = useState("");
  const [researchNotes, setResearchNotes] = useState("");
  const [researchDone, setResearchDone] = useState(false);

  // Stage 4
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [editedAction, setEditedAction] = useState("");
  const [draftText, setDraftText] = useState("");
  const [copied, setCopied] = useState(false);
  const [reviewMode, setReviewMode] = useState<string | null>(null);
  const [reasonCode, setReasonCode] = useState(REASON_CODES[0]);
  const [reviewDone, setReviewDone] = useState<string | null>(null);

  const esRef = useRef<EventSource | null>(null);

  const loadOpportunities = () =>
    fetch(`${API}/api/py/opportunities`).then((r) => r.json()).then(setOpps).catch(() => {});

  useEffect(() => {
    loadOpportunities();
    fetch(`${API}/api/py/health`).then((r) => r.json()).then(setHealth).catch(() => {});
    return () => esRef.current?.close();
  }, []);

  const applySession = (s: Session) => {
    setSession(s);
    if (s.stage1) {
      setSignalsText(listToText(s.stage1.signals));
      setGapsText(listToText(s.stage1.gaps));
      setAnalystNotes(s.stage1.analyst_notes);
    }
    if (s.stage2) {
      setConstraintsText(listToText(s.stage2.constraints));
      setEscalationsText(listToText(s.stage2.escalations));
      setEligibleTypes(s.stage2.eligible_action_types);
    }
    if (s.stage3) {
      if (s.stage3.research_summary) setResearchSummary(s.stage3.research_summary);
      if (s.stage3.research_notes) setResearchNotes(s.stage3.research_notes);
      if (s.stage3.feed.length) setFeed(s.stage3.feed);
      if (s.stage3.status === "done") setResearchDone(true);
    }
    if (s.stage4?.recommendation) {
      setRec(s.stage4.recommendation);
      setValidation(s.stage4.validation);
      setEditedAction(s.stage4.recommendation.action);
      setDraftText(s.stage4.recommendation.draft_message ?? "");
    }
  };

  const resetWorkflow = () => {
    esRef.current?.close();
    setSession(null);
    setFeed([]);
    setResearchSummary("");
    setResearchNotes("");
    setResearchDone(false);
    setRec(null);
    setValidation(null);
    setError(null);
    setReviewDone(null);
    setReviewMode(null);
    setCopied(false);
  };

  const startWorkflow = async () => {
    if (!selected || busy) return;
    resetWorkflow();
    setBusy(true);
    try {
      const res = await fetch(`${API}/api/py/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ opportunity_id: selected }),
      });
      if (!res.ok) throw new Error(await res.text());
      applySession(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const approveStage1 = async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/py/sessions/${session.session_id}/stage1/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signals: textToList(signalsText),
          gaps: textToList(gapsText),
          analyst_notes: analystNotes,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      applySession(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const approveStage2 = async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/py/sessions/${session.session_id}/stage2/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          constraints: textToList(constraintsText),
          escalations: textToList(escalationsText),
          eligible_action_types: eligibleTypes,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const s = await res.json();
      applySession(s);
      startResearch(s.session_id);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  const startResearch = (sessionId: string) => {
    setFeed([]);
    setResearchDone(false);
    setResearchSummary("");
    const es = new EventSource(`${API}/api/py/sessions/${sessionId}/research`);
    esRef.current = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      if (ev.type === "tool_call" || ev.type === "tool_result") {
        setFeed((f) => [...f, ev]);
      } else if (ev.type === "research_done") {
        setResearchSummary(ev.research_summary ?? "");
        setResearchDone(true);
        setBusy(false);
        es.close();
      } else if (ev.type === "error") {
        setError(ev.message);
        setBusy(false);
        es.close();
      } else if (ev.type === "done") {
        setBusy(false);
        es.close();
      }
    };
    es.onerror = () => {
      setError("Research stream interrupted.");
      setBusy(false);
      es.close();
    };
  };

  const approveStage3 = async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/py/sessions/${session.session_id}/stage3/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          research_summary: researchSummary,
          research_notes: researchNotes,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      applySession(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const submitFinalReview = async (decision: string) => {
    if (!session || !selected) return;
    await fetch(`${API}/api/py/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        opportunity_id: selected,
        session_id: session.session_id,
        decision,
        action: rec?.action ?? null,
        reason_code: decision === "rejected" ? reasonCode : null,
        edited_action: decision === "edited" ? editedAction : null,
        draft_message: draftText || null,
      }),
    });
    await fetch(`${API}/api/py/sessions/${session.session_id}/stage4/approve`, {
      method: "POST",
    });
    setReviewDone(decision);
    setReviewMode(null);
    if (decision === "approved" || decision === "edited") {
      setTimeout(async () => {
        await loadOpportunities();
        resetWorkflow();
        setSelected(null);
      }, 1800);
    }
  };

  const currentStage = session?.current_stage ?? 0;
  const selectedOpp = opps.find((o) => o.id === selected);
  const abstained =
    rec && (rec.action_type === "gather_information" || rec.action_type === "no_action_yet");

  const toggleEligible = (t: string) => {
    setEligibleTypes((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  };

  return (
    <div className="page">
      <nav className="topnav">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">DA</div>
          Deal Action <span>Copilot</span>
        </div>
        {health && (
          <div className="nav-status">
            <span className={`pill ${health.openai ? "ok" : "off"}`}>OpenAI</span>
            <span className={`pill ${health.tavily ? "ok" : "off"}`}>Tavily</span>
            <span className={`pill ${health.qdrant_cloud ? "ok" : "off"}`}>Qdrant</span>
            {health.langsmith ? (
              <a className="pill ok" href="https://smith.langchain.com" target="_blank" rel="noreferrer">
                LangSmith ↗
              </a>
            ) : (
              <span className="pill off">LangSmith</span>
            )}
          </div>
        )}
      </nav>

      <section className="hero hero-compact">
        <div className="hero-inner">
          <h1>AI for the next-best <em>deal action</em></h1>
          <p className="sub">
            Four guided steps — each reviewed and approved by your team before the copilot moves on.
          </p>
        </div>
      </section>

      <main className="shell">
        {/* Opportunity picker — full-width grid */}
        <section className="workspace-section">
          <div className="section-title">Select an open opportunity</div>
          <div className="opp-grid">
            {opps.map((o) => (
              <button
                key={o.id}
                className={`opp-card ${selected === o.id ? "active" : ""}`}
                onClick={() => { setSelected(o.id); resetWorkflow(); }}
                disabled={!!session && !reviewDone}
              >
                <div className="opp-card-name">{o.account}</div>
                <div className="opp-card-tags">
                  <span className={`tag motion-${o.motion}`}>{titleize(o.motion)}</span>
                  <span className="tag">{titleize(o.vertical)}</span>
                  <span className="tag">{fmtUsd(o.value_usd)}</span>
                </div>
                <div className="opp-card-meta">
                  {titleize(o.stage)} · {o.days_in_stage}d in stage
                </div>
                {o.followup && (
                  <div className="followup">✓ Follow-up approved</div>
                )}
              </button>
            ))}
          </div>
          {selected && !session && (
            <div className="start-row">
              <button className="run-btn run-btn-inline" onClick={startWorkflow} disabled={busy}>
                {busy ? "Starting…" : `Start workflow for ${selectedOpp?.account}`}
              </button>
            </div>
          )}
        </section>

        {/* Stepper */}
        {session && (
          <div className="stepper">
            {STEPS.map((label, i) => {
              const n = i + 1;
              const done = session.completed_stages.includes(n);
              const active = currentStage === n;
              return (
                <div key={label} className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}>
                  <div className="step-num">{done ? "✓" : n}</div>
                  <div className="step-label">{label}</div>
                </div>
              );
            })}
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        {/* Stage panels */}
        {session?.current_stage === 1 && session.stage1 && (
          <div className="panel stage-panel">
            <div className="panel-h">Step 1 — Review the deal</div>
            <div className="panel-b">
              <p className="stage-intro">
                The copilot assembled context from CRM, engagement, and call notes. Edit anything
                that looks off — your changes carry into every later step.
              </p>
              <div className="section-label">Deal snapshot</div>
              <pre className="snapshot">{session.stage1.summary}</pre>
              <div className="section-label">Signals worth acting on (one per line)</div>
              <textarea className="list-edit" value={signalsText} onChange={(e) => setSignalsText(e.target.value)} rows={5} />
              <div className="section-label">Gaps in what we know (one per line)</div>
              <textarea className="list-edit" value={gapsText} onChange={(e) => setGapsText(e.target.value)} rows={3} />
              <div className="section-label">Your analyst notes</div>
              <textarea className="list-edit" value={analystNotes} onChange={(e) => setAnalystNotes(e.target.value)} rows={3} placeholder="Add context the copilot should weigh in later steps…" />
              <div className="stage-actions">
                <button className="run-btn run-btn-inline" onClick={approveStage1} disabled={busy}>
                  Approve &amp; continue to playbook rules
                </button>
              </div>
            </div>
          </div>
        )}

        {session?.current_stage === 2 && session.stage2 && (
          <div className="panel stage-panel">
            <div className="panel-h">Step 2 — Apply playbook rules</div>
            <div className="panel-b">
              <p className="stage-intro">
                Based on your approved deal review, these playbook rules apply. Edit constraints or
                which action types are allowed before research begins.
              </p>
              <div className="section-label">Playbook constraints (one per line)</div>
              <textarea className="list-edit" value={constraintsText} onChange={(e) => setConstraintsText(e.target.value)} rows={4} />
              <div className="section-label">Escalations (one per line)</div>
              <textarea className="list-edit" value={escalationsText} onChange={(e) => setEscalationsText(e.target.value)} rows={2} />
              <div className="section-label">Allowed action types</div>
              <div className="checkbox-grid">
                {ACTION_TYPES.map((t) => (
                  <label key={t} className="check-label">
                    <input type="checkbox" checked={eligibleTypes.includes(t)} onChange={() => toggleEligible(t)} />
                    {titleize(t)}
                  </label>
                ))}
              </div>
              <div className="stage-actions">
                <button className="run-btn run-btn-inline" onClick={approveStage2} disabled={busy || eligibleTypes.length === 0}>
                  Approve &amp; start account research
                </button>
              </div>
            </div>
          </div>
        )}

        {session?.current_stage === 3 && (
          <div className="panel stage-panel">
            <div className="panel-h">Step 3 — Research the account</div>
            <div className="panel-b">
              <p className="stage-intro">
                The copilot looked up approved sales content and public news using your
                approved deal review and playbook rules. Edit the research summary or add
                your own notes — your changes carry into the draft recommendation.
              </p>
              {feed.length === 0 && !researchDone && (
                <div className="loading-pulse">Researching…</div>
              )}
              {feed.map((f, i) =>
                f.type === "tool_call" ? (
                  <div className="feed-item" key={i}>
                    <span className="t">→ {f.tool}</span> “{f.query}”
                  </div>
                ) : (
                  <div className="feed-item" key={i}>
                    <span className="t">← {f.tool}</span>{" "}
                    {f.sources?.map((s) => <span className="src" key={s}>[{s}] </span>)}
                    {f.preview}…
                  </div>
                )
              )}
              {researchDone && (
                <>
                  <div className="section-label">Research summary — edit freely</div>
                  <textarea
                    className="list-edit"
                    value={researchSummary}
                    onChange={(e) => setResearchSummary(e.target.value)}
                    rows={5}
                    placeholder="Summarize what the research found…"
                  />
                  <div className="section-label">Your research notes</div>
                  <textarea
                    className="list-edit"
                    value={researchNotes}
                    onChange={(e) => setResearchNotes(e.target.value)}
                    rows={3}
                    placeholder="Add context, corrections, or extra findings the copilot should weigh when drafting…"
                  />
                  <div className="stage-actions">
                    <button
                      className="run-btn run-btn-inline"
                      onClick={approveStage3}
                      disabled={busy || !researchSummary.trim()}
                    >
                      Approve research &amp; draft recommendation
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {session?.current_stage === 4 && rec && (
          <div className="panel rec stage-panel">
            <div className="panel-h">{abstained ? "Step 4 — Copilot abstained" : "Step 4 — Draft the next move"}</div>
            <div className="panel-b">
              <div className="rec-head">
                <span className={`badge action-${rec.action_type}`}>{titleize(rec.action_type)}</span>
                <span className={`badge conf-${rec.confidence}`}>confidence: {rec.confidence}</span>
                {validation?.warnings.map((w) => <span className="warn" key={w}>⚠ {w}</span>)}
              </div>
              <h3>{rec.action}</h3>
              <dl className="kv">
                <dt>Owner</dt><dd>{rec.owner}</dd>
                {rec.target_stakeholder && (<><dt>Target stakeholder</dt><dd>{rec.target_stakeholder}</dd></>)}
                <dt>Timing</dt><dd>{rec.timing}</dd>
                {rec.socure_angle && (<><dt>Product angle</dt><dd>{rec.socure_angle}</dd></>)}
                {rec.supporting_asset && (<><dt>Supporting asset</dt><dd>{rec.supporting_asset}</dd></>)}
              </dl>
              <div className="section-label">Rationale</div>
              <div>{rec.rationale}</div>
              <div className="section-label">Evidence ({rec.evidence.length} cited)</div>
              <ul className="evidence">
                {rec.evidence.map((e, i) => (
                  <li key={i}><span className="src">[{e.source_id}]</span>{e.claim}</li>
                ))}
              </ul>
              {rec.missing_information.length > 0 && (
                <>
                  <div className="section-label">Missing information</div>
                  <ul className="sub-list">
                    {rec.missing_information.map((m) => <li key={m}>{m}</li>)}
                  </ul>
                </>
              )}
              {rec.draft_message && (
                <>
                  <div className="section-label">Draft message — edit freely, then copy</div>
                  <textarea className="draft-edit" value={draftText} onChange={(e) => setDraftText(e.target.value)} />
                  <div className="review-actions">
                    <button
                      className={`btn ${copied ? "approve" : ""}`}
                      onClick={async () => {
                        await navigator.clipboard.writeText(draftText);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 1600);
                      }}
                    >
                      {copied ? "Copied ✓" : "Copy to clipboard"}
                    </button>
                  </div>
                </>
              )}
              <div className="section-label">Final approval — Product Marketing</div>
              {reviewDone ? (
                <div className="review-done">
                  Review recorded: <strong>{reviewDone}</strong>.
                  {reviewDone === "approved" || reviewDone === "edited"
                    ? " Follow-up stored — returning to opportunities…"
                    : " Feedback logged for the next iteration."}
                </div>
              ) : (
                <>
                  <div className="review-actions">
                    <button className="btn approve" onClick={() => submitFinalReview("approved")}>Approve</button>
                    <button className={`btn ${reviewMode === "edit" ? "active" : ""}`} onClick={() => setReviewMode(reviewMode === "edit" ? null : "edit")}>Edit &amp; approve</button>
                    <button className={`btn reject ${reviewMode === "reject" ? "active" : ""}`} onClick={() => setReviewMode(reviewMode === "reject" ? null : "reject")}>Reject</button>
                    <button className="btn" onClick={() => submitFinalReview("deferred")}>Defer</button>
                  </div>
                  {reviewMode === "edit" && (
                    <>
                      <textarea value={editedAction} onChange={(e) => setEditedAction(e.target.value)} />
                      <div className="review-actions">
                        <button className="btn approve" onClick={() => submitFinalReview("edited")}>Submit edit &amp; approve</button>
                      </div>
                    </>
                  )}
                  {reviewMode === "reject" && (
                    <>
                      <select value={reasonCode} onChange={(e) => setReasonCode(e.target.value)}>
                        {REASON_CODES.map((r) => <option key={r} value={r}>{titleize(r)}</option>)}
                      </select>
                      <div className="review-actions">
                        <button className="btn reject" onClick={() => submitFinalReview("rejected")}>Submit rejection</button>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {!session && selected && (
          <div className="hint-panel">
            Click <strong>Start workflow</strong> above to begin the four-step review process for{" "}
            <strong>{selectedOpp?.account}</strong>.
          </div>
        )}

        {!selected && (
          <div className="hint-panel">
            Choose an opportunity above. Each step — deal review, playbook rules, account research,
            and the draft recommendation — requires your approval before the copilot continues.
          </div>
        )}

        <footer className="site-footer">
          <span>Demo with synthetic CRM data. Nothing reaches a customer without human approval.</span>
          <a href="https://www.socure.com/" target="_blank" rel="noreferrer">Inspired by Socure ↗</a>
        </footer>
      </main>
    </div>
  );
}
