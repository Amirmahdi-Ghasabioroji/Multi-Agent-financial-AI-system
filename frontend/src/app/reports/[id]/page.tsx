"use client";

import { ResultView } from "@/components/result-view";
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
import { jobLabel, shortDate, titleCase } from "@/lib/format";
import type { Job, RunKind } from "@/lib/types";
import {
  Download,
  FileJson,
  FileText,
  Printer,
  Scale,
} from "lucide-react";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

export default function ReportPage() {
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

  return (
    <>
      <div className="no-print">
        <PageHeader
          eyebrow="Executive report"
          title={job ? jobLabel(job) : "Research report"}
          description="Print this structured result or use your browser’s Save as PDF option. Export links preserve the backend representation."
        >
          <button
            className="button button-primary"
            onClick={() => window.print()}
            disabled={!job}
          >
            <Printer size={15} />
            Print / Save PDF
          </button>
        </PageHeader>
      </div>

      {loading ? (
        <LoadingState label="Compiling executive report…" />
      ) : error ? (
        <ErrorState message={error} retry={load} />
      ) : job ? (
        <article className="report">
          <header className="report-masthead">
            <div>
              <p className="eyebrow">MAFAS executive research report</p>
              <h1 style={{ marginTop: 7 }}>{jobLabel(job)}</h1>
              <div className="toolbar" style={{ marginTop: 12 }}>
                <StatusBadge status={job.status} />
                <Badge tone={job.demo === false || job.mode === "live" ? "teal" : "amber"}>
                  {job.demo === false || job.mode === "live"
                    ? "Live data requested"
                    : "Demo / frozen data"}
                </Badge>
              </div>
            </div>
            <div className="report-meta">
              <div>Job {job.id}</div>
              <div>Workflow {titleCase(kind)}</div>
              <div>Created {shortDate(job.created_at)}</div>
              <div>Completed {shortDate(job.completed_at)}</div>
            </div>
          </header>

          <Panel>
            <SectionHeading
              eyebrow="Research brief"
              title={job.query ?? "No source query was stored"}
              detail={
                job.tickers?.length
                  ? `Instruments: ${job.tickers.join(", ")}`
                  : job.ticker
                    ? `Instrument: ${job.ticker}`
                    : "Instrument universe not reported"
              }
            />
            <div className="callout callout-amber">
              <Scale size={19} />
              <div>
                <strong>Research and simulation only</strong>
                <p>
                  This report is not financial advice, a solicitation or a trade
                  instruction. Data and model outputs can be stale, incomplete or
                  wrong. Independently verify all material facts.
                </p>
              </div>
            </div>
          </Panel>

          <div className="no-print toolbar" style={{ margin: "18px 0" }}>
            <a
              href={api.exportUrl(job.id, "json")}
              className="button button-secondary"
            >
              <FileJson size={15} />
              Download JSON
            </a>
            <a
              href={api.exportUrl(job.id, "markdown")}
              className="button button-secondary"
            >
              <FileText size={15} />
              Download Markdown
            </a>
            <button className="button button-secondary" onClick={() => window.print()}>
              <Download size={15} />
              Save as PDF
            </button>
          </div>

          <div style={{ marginTop: 16 }}>
            {job.result ? (
              <ResultView result={job.result} kind={kind} />
            ) : job.partial_result ? (
              <ResultView result={job.partial_result} kind={kind} partial />
            ) : (
              <Panel>
                <SectionHeading
                  eyebrow="Result unavailable"
                  title={`Job is ${job.status}`}
                  detail={
                    typeof job.error === "string"
                      ? job.error
                      : "No structured result has been stored yet."
                  }
                />
              </Panel>
            )}
          </div>

          <p className="print-only" style={{ marginTop: 24, fontSize: 11 }}>
            Generated from MAFAS job {job.id}. Research and simulation only—not
            financial advice.
          </p>
        </article>
      ) : null}
    </>
  );
}
