"use client";

import {
  EmptyState,
  ErrorState,
  Field,
  LoadingState,
  PageHeader,
  Panel,
  SectionHeading,
  StatusBadge,
} from "@/components/ui";
import { api, errorMessage } from "@/lib/api";
import { jobLabel, shortDate, titleCase } from "@/lib/format";
import type { Job } from "@/lib/types";
import {
  ExternalLink,
  FileJson,
  GitBranch,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function HistoryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [kind, setKind] = useState("all");
  const [deleting, setDeleting] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api.jobs();
      setJobs(response.items);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const filtered = useMemo(
    () =>
      jobs.filter((job) => {
        const matchesQuery =
          !query ||
          jobLabel(job).toLowerCase().includes(query.toLowerCase()) ||
          job.id.toLowerCase().includes(query.toLowerCase());
        const jobKind = job.kind ?? job.agent ?? "pipeline";
        return (
          matchesQuery &&
          (status === "all" || job.status === status) &&
          (kind === "all" || jobKind === kind)
        );
      }),
    [jobs, query, status, kind],
  );

  const deleteJob = async (job: Job) => {
    if (!window.confirm(`Delete "${jobLabel(job)}"? This cannot be undone.`)) return;
    setDeleting(job.id);
    setError("");
    try {
      await api.deleteJob(job.id);
      setJobs((items) => items.filter((item) => item.id !== job.id));
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setDeleting("");
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Run ledger"
        title="Research history"
        description="Filter, reopen and export prior MAFAS jobs. Continue conversation-linked work without losing the original audit trail."
      >
        <button className="button button-secondary" onClick={load}>
          <RefreshCw size={15} />
          Refresh
        </button>
        <Link href="/workspace" className="button button-primary">
          New run
        </Link>
      </PageHeader>

      <Panel>
        <SectionHeading
          eyebrow="Filters"
          title="Find a run"
          detail={`${filtered.length} of ${jobs.length} jobs shown`}
        />
        <div className="grid grid-3">
          <Field label="Search">
            <div style={{ position: "relative" }}>
              <Search
                size={15}
                style={{ position: "absolute", left: 12, top: 14 }}
                className="faint"
              />
              <input
                className="input"
                style={{ paddingLeft: 36 }}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Question, ticker or job ID"
              />
            </div>
          </Field>
          <Field label="Status">
            <select
              className="select"
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="all">All statuses</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </Field>
          <Field label="Workflow">
            <select
              className="select"
              value={kind}
              onChange={(event) => setKind(event.target.value)}
            >
              <option value="all">All workflows</option>
              <option value="pipeline">Full pipeline</option>
              <option value="analyst">Analyst</option>
              <option value="risk">Risk</option>
              <option value="strategy">Strategy</option>
              <option value="execution">Execution</option>
            </select>
          </Field>
        </div>
      </Panel>

      <Panel>
        <SectionHeading
          eyebrow="Jobs"
          title="Persisted research runs"
          detail="Deleting removes the backend job record where supported."
        />
        {error && <ErrorState message={error} retry={load} />}
        {loading ? (
          <LoadingState label="Loading run history…" />
        ) : filtered.length ? (
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Research</th>
                  <th>Workflow</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Updated</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((job) => {
                  const jobKind = job.kind ?? job.agent ?? "pipeline";
                  return (
                    <tr key={job.id}>
                      <td>
                        <strong>{jobLabel(job)}</strong>
                        <div className="faint mono">{job.id}</div>
                      </td>
                      <td>{titleCase(jobKind)}</td>
                      <td>{job.demo === false || job.mode === "live" ? "Live" : "Demo"}</td>
                      <td>
                        <StatusBadge status={job.status} />
                      </td>
                      <td>{shortDate(job.updated_at ?? job.created_at)}</td>
                      <td>
                        <div className="inline-actions">
                          <Link
                            href={`/runs/${job.id}`}
                            className="icon-button"
                            title="Open run monitor"
                            aria-label={`Open run monitor for ${jobLabel(job)}`}
                          >
                            <ExternalLink size={15} />
                          </Link>
                          <a
                            href={api.exportUrl(job.id, "json")}
                            className="icon-button"
                            title="Download JSON"
                            aria-label={`Download JSON for ${jobLabel(job)}`}
                          >
                            <FileJson size={15} />
                          </a>
                          {job.conversation_id && (
                            <Link
                              href={`/workspace?conversation=${encodeURIComponent(job.conversation_id)}`}
                              className="icon-button"
                              title="Continue conversation"
                              aria-label={`Continue conversation from ${jobLabel(job)}`}
                            >
                              <GitBranch size={15} />
                            </Link>
                          )}
                          <button
                            className="icon-button"
                            onClick={() => deleteJob(job)}
                            disabled={deleting === job.id}
                            title="Delete job"
                            aria-label={`Delete ${jobLabel(job)}`}
                          >
                            <Trash2 size={15} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={jobs.length ? "No matching jobs" : "No job history"}
            detail={
              jobs.length
                ? "Adjust the search or filters."
                : "Completed and in-progress runs will appear here."
            }
          />
        )}
      </Panel>
    </>
  );
}
