"use client";

import {
  Badge,
  EmptyState,
  ErrorState,
  LoadingState,
  Metric,
  ModeBadge,
  PageHeader,
  Panel,
  SectionHeading,
  StatusBadge,
} from "@/components/ui";
import { api, errorMessage } from "@/lib/api";
import { compactNumber, jobLabel, shortDate } from "@/lib/format";
import type { AppConfig, Conversation, Health, Job } from "@/lib/types";
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  FileClock,
  GitBranch,
  Layers3,
  ShieldCheck,
  Sparkles,
  Target,
  Zap,
  type LucideIcon,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

const agents: Array<{
  name: string;
  detail: string;
  href: string;
  icon: LucideIcon;
  input: string;
}> = [
  {
    name: "Analyst",
    detail: "Retrieval-grounded evidence synthesis and confidence attribution.",
    href: "/agents/analyst",
    icon: BrainCircuit,
    input: "Query + market context",
  },
  {
    name: "Risk",
    detail: "Regime, volatility, correlation, concentration and sizing analysis.",
    href: "/agents/risk",
    icon: ShieldCheck,
    input: "Analyst output",
  },
  {
    name: "Strategy",
    detail: "Macro bias, scored playbooks and setup suppression guardrails.",
    href: "/agents/strategy",
    icon: Target,
    input: "Analysis + risk",
  },
  {
    name: "Execution",
    detail: "Entry geometry, expected R and probabilistic execution verdict.",
    href: "/agents/execution",
    icon: Zap,
    input: "Setup + risk",
  },
];

export default function Home() {
  const [health, setHealth] = useState<Health>();
  const [config, setConfig] = useState<AppConfig>();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    const results = await Promise.allSettled([
      api.health(),
      api.config(),
      api.jobs(),
      api.conversations(),
    ]);
    if (results[0].status === "fulfilled") setHealth(results[0].value);
    if (results[1].status === "fulfilled") setConfig(results[1].value);
    if (results[2].status === "fulfilled") setJobs(results[2].value.items);
    if (results[3].status === "fulfilled")
      setConversations(results[3].value.items);
    const failures = results.filter((result) => result.status === "rejected");
    setError(
      failures.length === results.length
        ? errorMessage((failures[0] as PromiseRejectedResult).reason)
        : "",
    );
    setLoading(false);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const qdrant = health?.services?.qdrant ?? {};
  const ollama = health?.services?.ollama ?? {};
  const liveReady = qdrant.status === "ok" && ollama.status === "ok";
  const demo = config?.demo_mode ?? !liveReady;
  const documentCount = compactNumber(
    qdrant.count ?? health?.collection_count ?? health?.document_count,
  );

  return (
    <>
      <PageHeader
        eyebrow="Command centre"
        title="Financial research, orchestrated."
        description="Run the full multi-agent pipeline or inspect each specialist. Every output stays traceable, guardrailed, and clearly separated from trading advice."
      >
        <ModeBadge demo={demo} />
        <Link href="/workspace" className="button button-primary">
          <Sparkles size={16} />
          New pipeline run
        </Link>
      </PageHeader>

      {error && <ErrorState message={error} retry={load} />}
      {loading && !health ? (
        <LoadingState label="Connecting to the MAFAS API…" />
      ) : (
        <>
          <div className="grid grid-4">
            <Metric
              label="API status"
              value={health?.status ?? (error ? "Offline" : "Unknown")}
              detail={health?.version ? `Version ${health.version}` : "Backend service"}
              icon={Activity}
              tone={health?.status === "ok" ? "teal" : "amber"}
            />
            <Metric
              label="Corpus records"
              value={documentCount}
              detail={
                health?.corpus_updated_at
                  ? `Fresh ${shortDate(health.corpus_updated_at)}`
                  : "Freshness not reported"
              }
              icon={Database}
            />
            <Metric
              label="Recent jobs"
              value={jobs.length}
              detail={`${jobs.filter((job) => job.status === "running").length} active now`}
              icon={Layers3}
            />
            <Metric
              label="Conversations"
              value={conversations.length}
              detail="Reusable research context"
              icon={GitBranch}
              tone="teal"
            />
          </div>

          <div className="content-section grid grid-main">
            <Panel>
              <SectionHeading
                eyebrow="Run ledger"
                title="Recent jobs"
                detail="Latest orchestrated and specialist analyses"
                action={
                  <Link href="/history" className="link">
                    View history
                  </Link>
                }
              />
              {jobs.length ? (
                <div className="data-table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Research</th>
                        <th>Status</th>
                        <th>Created</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {jobs.slice(0, 6).map((job) => (
                        <tr key={job.id}>
                          <td>
                            <strong>{jobLabel(job)}</strong>
                            <div className="faint mono">{job.id.slice(0, 12)}</div>
                          </td>
                          <td>
                            <StatusBadge status={job.status} />
                          </td>
                          <td>{shortDate(job.created_at)}</td>
                          <td>
                            <Link
                              className="icon-button"
                              href={`/reports/${job.id}`}
                              aria-label={`Open ${jobLabel(job)}`}
                            >
                              <ChevronRight size={16} />
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No research runs yet"
                  detail="Start the pipeline to create the first traceable result."
                  action={
                    <Link href="/workspace" className="button button-primary button-small">
                      Launch workspace
                    </Link>
                  }
                />
              )}
            </Panel>

            <div className="stack">
              <Panel>
                <SectionHeading eyebrow="Quick launch" title="Choose a workflow" />
                <div className="stack">
                  <Link href="/workspace" className="launch-card">
                    <Badge tone="teal">Recommended</Badge>
                    <h3>Full pipeline</h3>
                    <p>
                      One query routed through analysis, risk, strategy and
                      execution with aggregate guardrails.
                    </p>
                    <ArrowRight size={17} style={{ marginTop: "auto" }} />
                  </Link>
                  <Link href="/agents/analyst" className="launch-card">
                    <Badge tone="blue">Specialist</Badge>
                    <h3>Evidence research</h3>
                    <p>Start with an analyst packet, then reuse it downstream.</p>
                    <ArrowRight size={17} style={{ marginTop: "auto" }} />
                  </Link>
                </div>
              </Panel>
            </div>
          </div>

          <section className="content-section">
            <SectionHeading
              eyebrow="System architecture"
              title="Four specialists, one governed route"
              detail="Structured outputs become explicit inputs to the next stage."
            />
            <div className="architecture" aria-label="MAFAS agent pipeline">
              {agents.map((agent, index) => (
                <div style={{ display: "contents" }} key={agent.name}>
                  <div className="architecture-node">
                    <strong>{agent.name}</strong>
                    <small>{agent.input}</small>
                  </div>
                  {index < agents.length - 1 && (
                    <div className="architecture-arrow">
                      <ArrowRight size={15} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section className="content-section">
            <SectionHeading
              eyebrow="Specialist workbenches"
              title="Inspect or run agents independently"
            />
            <div className="grid grid-4">
              {agents.map((agent) => {
                const Icon = agent.icon;
                return (
                  <Link href={agent.href} className="agent-card" key={agent.name}>
                    <div className="agent-card-icon">
                      <Icon size={18} />
                    </div>
                    <h3>{agent.name}</h3>
                    <p>{agent.detail}</p>
                    <small>Open workbench →</small>
                  </Link>
                );
              })}
            </div>
          </section>

          <section className="content-section grid grid-3">
            <Panel>
              <Badge tone="teal">
                <CheckCircle2 size={12} /> Traceable
              </Badge>
              <h3 style={{ marginTop: 14 }}>Source-aware analysis</h3>
              <p className="muted mt-1">
                Citation relevance and confidence components stay visible.
              </p>
            </Panel>
            <Panel>
              <Badge tone="amber">
                <Clock3 size={12} /> Asynchronous
              </Badge>
              <h3 style={{ marginTop: 14 }}>Live job telemetry</h3>
              <p className="muted mt-1">
                Stream stage events with automatic polling fallback.
              </p>
            </Panel>
            <Panel>
              <Badge tone="blue">
                <FileClock size={12} /> Reproducible
              </Badge>
              <h3 style={{ marginTop: 14 }}>Persistent research history</h3>
              <p className="muted mt-1">
                Reopen reports, export outputs, and continue context.
              </p>
            </Panel>
          </section>
        </>
      )}
    </>
  );
}
