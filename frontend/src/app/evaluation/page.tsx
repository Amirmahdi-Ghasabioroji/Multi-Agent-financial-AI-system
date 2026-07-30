"use client";

import { EvaluationReport } from "@/components/evaluation-report";
import { JobMonitor } from "@/components/job-monitor";
import {
  Badge,
  ErrorState,
  Field,
  LoadingState,
  PageHeader,
  Panel,
  SectionHeading,
} from "@/components/ui";
import { api, errorMessage } from "@/lib/api";
import { shortDate } from "@/lib/format";
import type { Job } from "@/lib/types";
import { ClipboardCheck, ExternalLink, Play, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

const SUITE_OPTIONS = [
  { id: "all", label: "All suites" },
  { id: "rag", label: "RAG retrieval (live corpus)" },
  { id: "simulation", label: "Monte Carlo calibration" },
  { id: "risk", label: "Risk agent metrics" },
] as const;

export default function EvaluationPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [suite, setSuite] = useState<(typeof SUITE_OPTIONS)[number]["id"]>("all");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<Job | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api.jobs();
      const evalJobs = response.items.filter(
        (job) => (job.kind ?? job.agent) === "evaluation",
      );
      setJobs(evalJobs);
      if (!activeJobId && evalJobs.length > 0) {
        setActiveJobId(evalJobs[0].id);
      }
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }, [activeJobId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  useEffect(() => {
    if (!activeJobId) {
      setActiveJob(null);
      return;
    }
    let cancelled = false;
    const refresh = async () => {
      try {
        const job = await api.job(activeJobId);
        if (!cancelled) setActiveJob(job);
      } catch {
        if (!cancelled) setActiveJob(null);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJobId]);

  const runEvaluation = async () => {
    setSubmitting(true);
    setError("");
    try {
      const created = await api.runEvaluation({
        suites: [suite],
        top_k: 8,
        lookback_days: 252,
      });
      setActiveJobId(created.id);
      await load();
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  const report =
    activeJob?.result && typeof activeJob.result === "object"
      ? (activeJob.result as Record<string, unknown>)
      : null;

  const running =
    activeJob &&
    !["succeeded", "completed", "failed", "cancelled"].includes(activeJob.status);

  return (
    <>
      <PageHeader
        eyebrow="Quality assurance"
        title="Evaluation reports"
        description="On-demand checks for live RAG retrieval, Monte Carlo calibration, and risk metric completeness. Results are scores only — no automated pass/fail gates."
      >
        <Badge tone="teal">On demand</Badge>
      </PageHeader>

      <div className="grid grid-evaluation">
        <div className="evaluation-main stack">
          {running ? (
            <JobMonitor initialJob={activeJob} kind="pipeline" />
          ) : report ? (
            <EvaluationReport report={report} />
          ) : activeJob?.status === "failed" ? (
            <ErrorState
              message={
                typeof activeJob.error === "string"
                  ? activeJob.error
                  : "Evaluation job failed."
              }
            />
          ) : (
            <Panel>
              <SectionHeading
                eyebrow="Report"
                title="Select or run an evaluation"
                detail="Completed reports appear here with suite-level metrics and scenario tables."
              />
            </Panel>
          )}
        </div>

        <aside className="evaluation-sidebar stack">
          <Panel>
            <SectionHeading
              eyebrow="Run"
              title="Launch evaluation"
              detail="RAG needs corpus. Risk uses live data."
              action={<ClipboardCheck size={18} />}
            />
            <div className="stack">
              <Field label="Suite">
                <select
                  className="select"
                  value={suite}
                  onChange={(event) =>
                    setSuite(event.target.value as (typeof SUITE_OPTIONS)[number]["id"])
                  }
                >
                  {SUITE_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Field>
              {error && <ErrorState message={error} />}
              <div className="button-row">
                <button
                  className="button button-primary"
                  type="button"
                  disabled={submitting}
                  onClick={() => void runEvaluation()}
                >
                  <Play size={16} />
                  {submitting ? "Starting…" : "Run evaluation"}
                </button>
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={() => void load()}
                >
                  <RefreshCw size={15} />
                  Refresh
                </button>
              </div>
            </div>
          </Panel>

          <Panel>
            <SectionHeading
              eyebrow="History"
              title="Past runs"
              detail={`${jobs.length} total`}
            />
            {loading ? (
              <LoadingState label="Loading…" />
            ) : jobs.length === 0 ? (
              <p className="muted">No evaluation runs yet.</p>
            ) : (
              <div className="stack" style={{ gap: 10 }}>
                {jobs.map((job) => (
                  <div
                    key={job.id}
                    className={`evaluation-history-item ${
                      activeJobId === job.id ? "evaluation-history-item-active" : ""
                    }`}
                    onClick={() => setActiveJobId(job.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setActiveJobId(job.id);
                      }
                    }}
                    role="button"
                    tabIndex={0}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 8,
                        alignItems: "center",
                      }}
                    >
                      <strong>{job.status}</strong>
                      <Link
                        href={`/reports/${job.id}`}
                        className="button button-secondary"
                        style={{ padding: "6px 10px", fontSize: "0.78rem" }}
                        onClick={(event) => event.stopPropagation()}
                      >
                        <ExternalLink size={13} />
                        Report
                      </Link>
                    </div>
                    <span className="muted">{shortDate(job.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </aside>
      </div>
    </>
  );
}
