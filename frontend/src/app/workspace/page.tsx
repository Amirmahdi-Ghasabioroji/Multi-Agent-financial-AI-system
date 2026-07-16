"use client";

import { JobMonitor } from "@/components/job-monitor";
import {
  Badge,
  ErrorState,
  Field,
  PageHeader,
  Panel,
  SectionHeading,
} from "@/components/ui";
import { api, errorMessage } from "@/lib/api";
import type { Conversation, Job, RunRequest } from "@/lib/types";
import {
  BrainCircuit,
  Database,
  GitBranch,
  Play,
  RotateCcw,
  ShieldCheck,
  Target,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";

const TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY", "QQQ"];

export default function PipelineWorkspace() {
  const [query, setQuery] = useState("");
  const [tickers, setTickers] = useState<string[]>(["AAPL"]);
  const [customTicker, setCustomTicker] = useState("");
  const [lookback, setLookback] = useState(90);
  const [useLlm, setUseLlm] = useState(true);
  const [demo, setDemo] = useState(true);
  const [contextMode, setContextMode] = useState<"one-shot" | "existing" | "new">(
    "one-shot",
  );
  const [conversationId, setConversationId] = useState("");
  const [conversationTitle, setConversationTitle] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [job, setJob] = useState<Job>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(() => {
      const requestedConversation = new URLSearchParams(
        window.location.search,
      ).get("conversation");
      if (requestedConversation) {
        setContextMode("existing");
        setConversationId(requestedConversation);
      }
      api
        .conversations()
        .then((response) => {
          if (active) setConversations(response.items);
        })
        .catch(() => {
          if (active) setConversations([]);
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, []);

  const toggleTicker = (ticker: string) => {
    setTickers((current) =>
      current.includes(ticker)
        ? current.filter((item) => item !== ticker)
        : [...current, ticker],
    );
  };

  const addCustomTicker = () => {
    const ticker = customTicker.trim().toUpperCase();
    if (ticker && !tickers.includes(ticker)) setTickers((items) => [...items, ticker]);
    setCustomTicker("");
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Describe the financial research question before starting.");
      return;
    }
    if (!tickers.length) {
      setError("Select at least one ticker or instrument.");
      return;
    }
    if (contextMode === "existing" && !conversationId) {
      setError("Choose an existing conversation.");
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      let activeConversationId =
        contextMode === "existing" ? conversationId : undefined;
      if (contextMode === "new") {
        const conversation = await api.createConversation(
          conversationTitle.trim() || query.trim().slice(0, 80),
        );
        activeConversationId = conversation.id;
        setConversationId(conversation.id);
        setContextMode("existing");
        setConversations((current) => [conversation, ...current]);
      }
      const payload: RunRequest = {
        query: query.trim(),
        tickers,
        lookback_days: lookback,
        use_llm: useLlm,
        demo_mode: demo,
        ...(activeConversationId
          ? { conversation_id: activeConversationId }
          : {}),
      };
      const created = await api.run("pipeline", payload);
      setJob(created);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Orchestrated workflow"
        title="Pipeline workspace"
        description="Frame a research question, define its context, and follow the structured hand-off through all four specialist agents."
      >
        <Badge tone={demo ? "amber" : "teal"}>
          {demo ? "Demo / frozen inputs" : "Live data requested"}
        </Badge>
      </PageHeader>

      <div className="grid grid-main">
        <form className="stack" onSubmit={submit}>
          <Panel>
            <SectionHeading
              eyebrow="Research brief"
              title="Question & instruments"
              detail="Specific questions produce more auditable outputs."
            />
            <div className="stack">
              <Field
                label="Research query"
                hint="Ask for evidence, conditions and risks—not a buy/sell instruction."
              >
                <textarea
                  className="textarea"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Assess the evidence for margin expansion at NVIDIA over the next two reported quarters, including material downside risks."
                  required
                />
              </Field>
              <Field label="Ticker universe">
                <div className="chips" aria-label="Ticker selection">
                  {TICKERS.map((ticker) => (
                    <button
                      className={`chip ${tickers.includes(ticker) ? "chip-active" : ""}`}
                      type="button"
                      onClick={() => toggleTicker(ticker)}
                      aria-pressed={tickers.includes(ticker)}
                      key={ticker}
                    >
                      {ticker}
                    </button>
                  ))}
                </div>
              </Field>
              <div className="inline-actions">
                <input
                  className="input"
                  style={{ maxWidth: 210 }}
                  value={customTicker}
                  onChange={(event) => setCustomTicker(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      addCustomTicker();
                    }
                  }}
                  placeholder="Add ticker"
                  aria-label="Custom ticker"
                />
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={addCustomTicker}
                >
                  Add
                </button>
              </div>
            </div>
          </Panel>

          <Panel>
            <SectionHeading
              eyebrow="Context"
              title="Conversation memory"
              detail="Keep the run isolated or attach it to reusable research context."
            />
            <div className="chips" role="group" aria-label="Conversation mode">
              {(["one-shot", "existing", "new"] as const).map((mode) => (
                <button
                  className={`chip ${contextMode === mode ? "chip-active" : ""}`}
                  type="button"
                  onClick={() => setContextMode(mode)}
                  aria-pressed={contextMode === mode}
                  key={mode}
                >
                  {mode === "one-shot"
                    ? "One-shot"
                    : mode === "existing"
                      ? "Continue conversation"
                      : "New conversation"}
                </button>
              ))}
            </div>
            {contextMode === "existing" && (
              <Field label="Conversation">
                <select
                  className="select"
                  value={conversationId}
                  onChange={(event) => setConversationId(event.target.value)}
                >
                  <option value="">Select research context…</option>
                  {conversations.map((conversation) => (
                    <option value={conversation.id} key={conversation.id}>
                      {conversation.title ?? conversation.id}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            {contextMode === "new" && (
              <Field label="Conversation title" hint="Optional—MAFAS may infer one from the query.">
                <input
                  className="input"
                  value={conversationTitle}
                  onChange={(event) => setConversationTitle(event.target.value)}
                  placeholder="Semiconductor margin watch"
                />
              </Field>
            )}
          </Panel>

          <Panel>
            <SectionHeading eyebrow="Runtime" title="Run controls" />
            <div className="grid grid-3">
              <Field label="Lookback window">
                <select
                  className="select"
                  value={lookback}
                  onChange={(event) => setLookback(Number(event.target.value))}
                >
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                  <option value={180}>180 days</option>
                  <option value={365}>1 year</option>
                  <option value={730}>2 years</option>
                </select>
              </Field>
              <label className="toggle-line">
                <span>LLM synthesis</span>
                <input
                  type="checkbox"
                  className="toggle"
                  checked={useLlm}
                  onChange={(event) => setUseLlm(event.target.checked)}
                />
              </label>
              <label className="toggle-line">
                <span>Demo / frozen data</span>
                <input
                  type="checkbox"
                  className="toggle"
                  checked={demo}
                  onChange={(event) => setDemo(event.target.checked)}
                />
              </label>
            </div>
            {error && <ErrorState message={error} />}
            <div className="button-row" style={{ marginTop: 18 }}>
              <button className="button button-primary" disabled={submitting}>
                <Play size={16} />
                {submitting ? "Starting pipeline…" : "Start full pipeline"}
              </button>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => {
                  setQuery("");
                  setTickers(["AAPL"]);
                  setError("");
                  setJob(undefined);
                }}
              >
                <RotateCcw size={15} />
                Reset
              </button>
            </div>
          </Panel>
        </form>

        <div className="stack">
          {job ? (
            <JobMonitor initialJob={job} kind="pipeline" />
          ) : (
            <>
              <Panel>
                <SectionHeading eyebrow="Route preview" title="Agent hand-offs" />
                <div className="timeline">
                  {[
                    [BrainCircuit, "Analyst", "Evidence and confidence"],
                    [ShieldCheck, "Risk", "Regime and constraints"],
                    [Target, "Strategy", "Scored playbooks"],
                    [Zap, "Execution", "Geometry and verdict"],
                  ].map(([Icon, title, detail]) => {
                    const StageIcon = Icon as typeof BrainCircuit;
                    return (
                      <div className="timeline-item" key={String(title)}>
                        <div className="timeline-dot">
                          <StageIcon size={15} />
                        </div>
                        <strong>{String(title)}</strong>
                        <span>{String(detail)}</span>
                      </div>
                    );
                  })}
                </div>
              </Panel>
              <div className="callout callout-amber">
                <Database size={20} />
                <div>
                  <strong>{demo ? "Frozen dataset selected" : "Live mode requested"}</strong>
                  <p>
                    {demo
                      ? "Useful for repeatable evaluation. Dates and market values may be historical."
                      : "Live labels indicate requested data mode, not guaranteed source freshness. Check citations and timestamps."}
                  </p>
                </div>
              </div>
              <div className="callout">
                <GitBranch size={20} />
                <div>
                  <strong>Conversation-aware research</strong>
                  <p>
                    Reusing context can improve continuity, but prior assumptions
                    should still be revalidated on every run.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
