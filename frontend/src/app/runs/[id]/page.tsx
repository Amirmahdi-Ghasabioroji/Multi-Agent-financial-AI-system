"use client";

import { JobMonitor } from "@/components/job-monitor";
import {
  Badge,
  ErrorState,
  LoadingState,
  PageHeader,
  Panel,
  SectionHeading,
  StatusBadge,
} from "@/components/ui";
import { api, errorMessage } from "@/lib/api";
import { jobLabel, jobQuery, jobTickers, shortDate, titleCase } from "@/lib/format";
import type { Job, RunKind } from "@/lib/types";
import { ArrowLeft, FileText, GitBranch } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

const TERMINAL = new Set(["succeeded", "completed", "failed", "cancelled"]);

export default function RunPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [job, setJob] = useState<Job>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setJob(await api.job(id));
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const kind = (job?.kind ?? job?.agent ?? "pipeline") as RunKind;
  const sourceQuery = job ? jobQuery(job) : "";
  const sourceTickers = job ? jobTickers(job) : [];
  const terminal = job ? TERMINAL.has(job.status) : false;

  return (
    <>
      <PageHeader
        eyebrow="Live run"
        title={job ? jobLabel(job) : "Pipeline run"}
        description="Track agent stages, partial outputs and the final structured result."
      >
        <Link href="/workspace" className="button button-secondary">
          <ArrowLeft size={15} />
          Back to workspace
        </Link>
        {job && terminal && (job.result || job.partial_result) && (
          <Link href={`/reports/${job.id}`} className="button button-primary">
            <FileText size={15} />
            Executive report
          </Link>
        )}
      </PageHeader>

      {loading ? (
        <LoadingState label="Loading run status…" />
      ) : error ? (
        <ErrorState message={error} retry={load} />
      ) : job ? (
        <div className="stack">
          <Panel>
            <SectionHeading
              eyebrow="Run context"
              title={sourceQuery || "No source query was stored"}
              detail={
                sourceTickers.length === 1
                  ? `Instrument: ${sourceTickers[0]}`
                  : sourceTickers.length
                    ? `Instruments: ${sourceTickers.join(", ")}`
                    : "Instrument universe not reported"
              }
              action={<StatusBadge status={job.status} />}
            />
            <div className="toolbar">
              <Badge tone={job.demo === false || job.mode === "live" ? "teal" : "amber"}>
                {job.demo === false || job.mode === "live"
                  ? "Live data requested"
                  : "Demo / frozen data"}
              </Badge>
              <span className="muted mono">Job {job.id}</span>
              <span className="muted">Workflow {titleCase(kind)}</span>
              <span className="muted">Started {shortDate(job.created_at)}</span>
              {job.conversation_id && (
                <Link
                  href={`/workspace?conversation=${encodeURIComponent(job.conversation_id)}`}
                  className="link"
                >
                  <GitBranch size={14} />
                  Continue conversation
                </Link>
              )}
            </div>
          </Panel>
          <JobMonitor initialJob={job} kind={kind} />
        </div>
      ) : null}
    </>
  );
}
