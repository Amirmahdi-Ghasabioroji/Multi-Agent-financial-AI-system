"use client";

import { ResultView } from "@/components/result-view";
import { Badge, ErrorState, Panel, SectionHeading, StatusBadge } from "@/components/ui";
import { api, errorMessage, subscribeToJob } from "@/lib/api";
import { shortDate, titleCase } from "@/lib/format";
import type { Job, JobEvent, JobStatus, MAFASResult, RunKind, StageState } from "@/lib/types";
import {
  Check,
  Clock3,
  LoaderCircle,
  RadioTower,
  RotateCw,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const DEFAULT_STAGES = ["analyst", "risk", "strategy", "execution", "aggregate"];
const TERMINAL = new Set<JobStatus>([
  "succeeded",
  "completed",
  "failed",
  "cancelled",
]);

function mergeEvent(job: Job, event: JobEvent): Job {
  const payload =
    event.data && typeof event.data === "object"
      ? ({ ...event, ...(event.data as JobEvent) } as JobEvent)
      : event;
  const eventType = payload.event_type ?? payload.type;
  const eventStatus = (
    ["queued", "running", "succeeded", "failed", "cancelled"].includes(
      eventType ?? "",
    )
      ? eventType
      : undefined
  ) as JobStatus | undefined;
  return {
    ...job,
    status: payload.status ?? eventStatus ?? job.status,
    stage: payload.stage ?? job.stage,
    progress: payload.progress ?? job.progress,
    result:
      payload.result ??
      (payload.type === "result" && payload.data
        ? (payload.data as MAFASResult)
        : job.result),
    partial_result: payload.partial_result ?? job.partial_result,
    error:
      payload.type === "error"
        ? payload.message ?? job.error
        : job.error,
  };
}

function stageStates(job: Job, kind: RunKind): StageState[] {
  if (job.stages?.length) return job.stages;
  const names = kind === "pipeline" ? DEFAULT_STAGES : [kind];
  const current = job.stage?.toLowerCase();
  const currentIndex = current
    ? Math.max(0, names.findIndex((name) => current.includes(name)))
    : job.status === "completed" || job.status === "succeeded"
      ? names.length
      : 0;
  return names.map((name, index) => ({
    name,
    status:
      job.status === "failed" && index === currentIndex
        ? "failed"
        : job.status === "completed" ||
            job.status === "succeeded" ||
            index < currentIndex
          ? "completed"
          : index === currentIndex && job.status === "running"
            ? "running"
            : "waiting",
  }));
}

export function JobMonitor({
  initialJob,
  kind,
}: {
  initialJob: Job;
  kind: RunKind;
}) {
  const [job, setJob] = useState(initialJob);
  const [streaming, setStreaming] = useState(true);
  const [polling, setPolling] = useState(false);
  const [monitorError, setMonitorError] = useState("");

  useEffect(() => {
    if (TERMINAL.has(job.status)) return;
    let active = true;
    let pollTimer: number | undefined;

    const poll = async () => {
      try {
        const latest = await api.job(initialJob.id);
        if (!active) return;
        setJob(latest);
        setMonitorError("");
        if (!TERMINAL.has(latest.status)) {
          pollTimer = window.setTimeout(poll, 2_500);
        }
      } catch (error) {
        if (!active) return;
        setMonitorError(errorMessage(error));
        pollTimer = window.setTimeout(poll, 5_000);
      }
    };

    const unsubscribe = subscribeToJob(initialJob.id, {
      onEvent: (event) => {
        if (!active) return;
        setJob((current) => mergeEvent(current, event));
        const eventType = event.event_type ?? event.type;
        if (eventType === "succeeded" || eventType === "failed") {
          void api
            .job(initialJob.id)
            .then((latest) => {
              if (active) setJob(latest);
            })
            .catch(() => {
              if (active) {
                setStreaming(false);
                setPolling(true);
                void poll();
              }
            });
        }
      },
      onError: () => {
        if (!active) return;
        setStreaming(false);
        setPolling(true);
        unsubscribe();
        void poll();
      },
    });

    return () => {
      active = false;
      unsubscribe();
      if (pollTimer) window.clearTimeout(pollTimer);
    };
  }, [initialJob.id, job.status]);

  const stages = useMemo(() => stageStates(job, kind), [job, kind]);
  const result = job.result ?? job.partial_result;
  const partial = !job.result && Boolean(job.partial_result);

  return (
    <div className="stack">
      <Panel>
        <SectionHeading
          eyebrow={`Job ${job.id}`}
          title="Live run monitor"
          detail={`Started ${shortDate(job.created_at)} · ${titleCase(kind)}`}
          action={<StatusBadge status={job.status} />}
        />
        <div className="toolbar" style={{ marginBottom: 18 }}>
          <Badge tone={streaming ? "teal" : "amber"}>
            {streaming ? <RadioTower size={12} /> : <RotateCw size={12} />}
            {streaming ? "SSE connected" : polling ? "Polling fallback" : "Reconnecting"}
          </Badge>
          {job.progress !== undefined && (
            <span className="muted mono">{Math.round(job.progress)}%</span>
          )}
        </div>
        {job.progress !== undefined && (
          <div className="progress-track" style={{ marginBottom: 22 }}>
            <div
              className="progress-fill"
              style={{ width: `${Math.min(100, Math.max(0, job.progress))}%` }}
            />
          </div>
        )}
        <div className="timeline">
          {stages.map((stage) => (
            <div
              className={`timeline-item timeline-${stage.status}`}
              key={stage.name}
            >
              <div className="timeline-dot">
                {stage.status === "completed" ? (
                  <Check size={15} />
                ) : stage.status === "failed" ? (
                  <X size={15} />
                ) : stage.status === "running" ? (
                  <LoaderCircle className="animate-spin" size={15} />
                ) : (
                  <Clock3 size={14} />
                )}
              </div>
              <strong>{titleCase(stage.name)}</strong>
              <span>{stage.message ?? titleCase(stage.status)}</span>
            </div>
          ))}
        </div>
        {monitorError && <ErrorState message={monitorError} />}
        {job.error && (
          <ErrorState
            message={
              typeof job.error === "string"
                ? job.error
                : JSON.stringify(job.error)
            }
          />
        )}
      </Panel>
      {result && <ResultView result={result} kind={kind} partial={partial} />}
    </div>
  );
}
