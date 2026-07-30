import { Metric, Panel, SectionHeading } from "@/components/ui";
import { compactNumber, percent } from "@/lib/format";
import type { JsonRecord } from "@/lib/types";
import { BarChart3, BrainCircuit, Gauge, ShieldCheck } from "lucide-react";

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" ? (value as JsonRecord) : {};
}

function formatMetricValue(value: unknown, unit?: string | null) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (unit === "ratio" && Math.abs(value) <= 1) return percent(value);
    if (unit === "cosine") return compactNumber(value);
    return compactNumber(value);
  }
  return String(value);
}

function suiteIcon(suite: string) {
  if (suite === "rag") return BrainCircuit;
  if (suite === "simulation") return BarChart3;
  if (suite === "risk") return ShieldCheck;
  return Gauge;
}

function MetricGrid({ metrics }: { metrics: JsonRecord[] }) {
  if (!metrics.length) return <p className="muted">No metrics recorded.</p>;
  return (
    <div className="grid grid-4">
      {metrics.map((metric) => (
        <Metric
          key={String(metric.name ?? metric.label)}
          label={String(metric.label ?? metric.name ?? "Metric")}
          value={formatMetricValue(metric.value, metric.unit as string | undefined)}
          icon={Gauge}
        />
      ))}
    </div>
  );
}

function SuitePanel({ suite }: { suite: JsonRecord }) {
  const Icon = suiteIcon(String(suite.suite ?? ""));
  const metrics = Array.isArray(suite.metrics)
    ? suite.metrics.map(asRecord)
    : [];
  const cases = Array.isArray(suite.cases) ? suite.cases.map(asRecord) : [];
  const status = String(suite.status ?? "completed");

  return (
    <Panel>
      <SectionHeading
        eyebrow="Evaluation suite"
        title={String(suite.label ?? suite.suite ?? "Suite")}
        detail={`Status: ${status}${suite.duration_ms ? ` · ${compactNumber(suite.duration_ms)} ms` : ""}`}
        action={
          <div className="agent-card-icon">
            <Icon size={18} />
          </div>
        }
      />
      {suite.error ? (
        <div className="error-box" style={{ marginBottom: 16 }}>
          <strong>Suite error</strong>
          <p>{String(suite.error)}</p>
        </div>
      ) : null}
      <MetricGrid metrics={metrics} />
      {cases.length > 0 && (
        <div className="trade-table-wrap" style={{ marginTop: 18 }}>
          <h3>Scenario breakdown</h3>
          <table className="trade-table">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Primary metric</th>
                <th>Value</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((item, index) => {
                const caseMetrics = Array.isArray(item.metrics)
                  ? item.metrics.map(asRecord)
                  : [];
                const primary =
                  caseMetrics.find((metric) =>
                    ["calibration_error", "top_similarity", "realised_vol"].includes(
                      String(metric.name),
                    ),
                  ) ?? caseMetrics[0];
                return (
                  <tr key={`${item.id}-${index}`}>
                    <td>{String(item.label ?? item.id ?? `Case ${index + 1}`)}</td>
                    <td>{String(primary?.label ?? "—")}</td>
                    <td>
                      {formatMetricValue(
                        primary?.value,
                        primary?.unit as string | undefined,
                      )}
                    </td>
                    <td>{String(item.notes ?? "")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function EvaluationReport({ report }: { report: JsonRecord }) {
  const summary = Array.isArray(report.summary_metrics)
    ? report.summary_metrics.map(asRecord)
    : [];
  const suites = Array.isArray(report.suites) ? report.suites.map(asRecord) : [];
  const notes = Array.isArray(report.notes)
    ? report.notes.map((item) => String(item))
    : [];

  return (
    <div className="result-shell evaluation-report">
      <div className="result-summary">
        <div>
          <p className="eyebrow">Evaluation report</p>
          <h2>System quality metrics</h2>
          <p>
            Live retrieval probes, Monte Carlo calibration checks, and risk metric
            completeness. Scores are informational — no pass/fail thresholds are applied.
          </p>
        </div>
      </div>

      {summary.length > 0 && (
        <section className="result-section">
          <SectionHeading eyebrow="Headline" title="Cross-suite summary" />
          <MetricGrid metrics={summary} />
        </section>
      )}

      {suites.map((suite, index) => (
        <section className="result-section" key={`${suite.suite}-${index}`}>
          <SuitePanel suite={suite} />
        </section>
      ))}

      {notes.length > 0 && (
        <section className="result-section">
          <Panel>
            <SectionHeading eyebrow="Methodology" title="How to read these scores" />
            <ul className="key-list">
              {notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </Panel>
        </section>
      )}
    </div>
  );
}
