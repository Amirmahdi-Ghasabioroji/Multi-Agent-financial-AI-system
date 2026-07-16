"use client";

import {
  Badge,
  ErrorState,
  Field,
  LoadingState,
  Metric,
  PageHeader,
  Panel,
  SectionHeading,
  StatusBadge,
} from "@/components/ui";
import { api, API_BASE, errorMessage } from "@/lib/api";
import { compactNumber, shortDate, titleCase } from "@/lib/format";
import type { AppConfig, Health } from "@/lib/types";
import {
  AlertTriangle,
  Archive,
  Database,
  FileStack,
  Radio,
  RefreshCw,
  Server,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

const RESET_PHRASE = "RESET FINANCIAL DOCS";

export default function DataPage() {
  const [health, setHealth] = useState<Health>();
  const [config, setConfig] = useState<AppConfig>();
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"refresh" | "reset" | "">("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    const [healthResult, configResult] = await Promise.allSettled([
      api.health(),
      api.config(),
    ]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (configResult.status === "fulfilled") setConfig(configResult.value);
    if (healthResult.status === "rejected" && configResult.status === "rejected") {
      setError(errorMessage(healthResult.reason));
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const refresh = async () => {
    if (
      !window.confirm(
        "Refresh the financial document corpus now? Existing records should remain available during the refresh.",
      )
    )
      return;
    setBusy("refresh");
    setError("");
    setNotice("");
    try {
      await api.refreshCorpus();
      setNotice("Corpus refresh was accepted. Freshness may update asynchronously.");
      window.setTimeout(() => void load(), 2_000);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setBusy("");
    }
  };

  const reset = async () => {
    if (confirmation !== RESET_PHRASE) return;
    setBusy("reset");
    setError("");
    setNotice("");
    try {
      await api.resetCorpus();
      setNotice("Corpus reset was accepted. Run a refresh before retrieval workloads.");
      setConfirmation("");
      await load();
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setBusy("");
    }
  };

  const details = useMemo(
    () =>
      Object.entries({ ...config, ...health }).filter(
        ([key, value]) =>
          value !== undefined &&
          value !== null &&
          typeof value !== "object" &&
          !["status", "collection_count", "document_count"].includes(key),
      ),
    [health, config],
  );
  const qdrant = health?.services?.qdrant ?? {};
  const ollama = health?.services?.ollama ?? {};
  const demo =
    config?.demo_mode ??
    !(qdrant.status === "ok" && ollama.status === "ok");

  return (
    <>
      <PageHeader
        eyebrow="Data operations"
        title="Data & corpus"
        description="Inspect source mode, service health and retrieval corpus freshness. Mutating operations require deliberate confirmation."
      >
        <Badge tone={demo ? "amber" : "teal"}>
          <Radio size={11} />
          {demo ? "Frozen / demo sources" : "Live sources configured"}
        </Badge>
        <button className="button button-secondary" onClick={load}>
          <RefreshCw size={15} />
          Check health
        </button>
      </PageHeader>

      {loading && !health ? (
        <LoadingState label="Reading service and corpus health…" />
      ) : (
        <>
          <div className="grid grid-4">
            <Metric
              label="Service"
              value={health?.status ?? "Unknown"}
              detail={health?.service ?? "MAFAS API"}
              icon={Server}
              tone={health?.status === "ok" ? "teal" : "amber"}
            />
            <Metric
              label="Documents"
              value={compactNumber(
                qdrant.count ??
                  health?.collection_count ??
                  health?.document_count,
              )}
              detail={config?.collection_name ?? "Financial document corpus"}
              icon={FileStack}
            />
            <Metric
              label="Corpus freshness"
              value={
                health?.corpus_updated_at
                  ? shortDate(health.corpus_updated_at)
                  : "Not reported"
              }
              detail="Verify source dates in every result"
              icon={Archive}
            />
            <Metric
              label="LLM runtime"
              value={
                ollama.status !== "ok" ||
                health?.llm_available === false ||
                config?.llm_enabled === false
                  ? "Unavailable"
                  : "Configured"
              }
              detail="Can be disabled per run"
              icon={Database}
            />
          </div>

          <div className="content-section grid grid-2">
            <Panel>
              <SectionHeading
                eyebrow="Source semantics"
                title="Live vs frozen data"
              />
              <div className="stack">
                <div className="callout">
                  <Radio size={19} />
                  <div>
                    <strong>Live mode</strong>
                    <p>
                      Requests current provider data where configured. A live
                      label does not guarantee every source is current—inspect
                      citation dates, provider status and market timestamps.
                    </p>
                  </div>
                </div>
                <div className="callout callout-amber">
                  <Archive size={19} />
                  <div>
                    <strong>Frozen / demo mode</strong>
                    <p>
                      Uses repeatable snapshots for development and evaluation.
                      Prices, filings, news and regimes may be historical and
                      must never be treated as current market state.
                    </p>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel>
              <SectionHeading
                eyebrow="Health detail"
                title="Reported configuration"
                action={<StatusBadge status={health?.status ?? "unknown"} />}
              />
              <div className="data-table-wrap">
                <table className="data-table">
                  <tbody>
                    <tr>
                      <td>API base</td>
                      <td className="mono">{API_BASE}</td>
                    </tr>
                    {details.slice(0, 12).map(([key, value]) => (
                      <tr key={key}>
                        <td>{titleCase(key)}</td>
                        <td className="mono">{String(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>

          <div className="content-section grid grid-2">
            <Panel>
              <SectionHeading
                eyebrow="Safe operation"
                title="Refresh corpus"
                detail="Request ingestion without intentionally deleting the current collection."
              />
              <p className="muted" style={{ lineHeight: 1.6 }}>
                A refresh may download filings, rebuild chunks and update vector
                indexes. Jobs can remain asynchronous after the API accepts the
                request.
              </p>
              <button
                className="button button-primary"
                style={{ marginTop: 18 }}
                disabled={Boolean(busy)}
                onClick={refresh}
              >
                <RefreshCw size={15} />
                {busy === "refresh" ? "Requesting refresh…" : "Refresh financial docs"}
              </button>
            </Panel>

            <Panel className="danger-zone">
              <SectionHeading
                eyebrow="Destructive operation"
                title="Reset corpus"
                detail="Removes indexed financial documents where supported by the backend."
              />
              <div className="callout callout-amber">
                <AlertTriangle size={19} />
                <div>
                  <strong>Retrieval will be unavailable after reset</strong>
                  <p>Only use this to recover from a corrupt or invalid corpus.</p>
                </div>
              </div>
              <Field label={`Type ${RESET_PHRASE} to continue`}>
                <input
                  className="input mono"
                  value={confirmation}
                  onChange={(event) => setConfirmation(event.target.value)}
                  autoComplete="off"
                />
              </Field>
              <button
                className="button button-danger"
                style={{ marginTop: 14 }}
                disabled={confirmation !== RESET_PHRASE || Boolean(busy)}
                onClick={reset}
              >
                <Trash2 size={15} />
                {busy === "reset" ? "Resetting corpus…" : "Reset financial docs"}
              </button>
            </Panel>
          </div>

          {error && (
            <div className="content-section">
              <ErrorState message={error} retry={load} />
            </div>
          )}
          {notice && (
            <div className="content-section callout" role="status">
              <RefreshCw size={18} />
              <div>
                <strong>Operation accepted</strong>
                <p>{notice}</p>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
