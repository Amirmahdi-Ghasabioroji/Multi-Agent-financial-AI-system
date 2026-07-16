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
import { parseJsonObject, titleCase } from "@/lib/format";
import type { AgentKind, Job, RunRequest } from "@/lib/types";
import {
  Braces,
  BrainCircuit,
  ClipboardPaste,
  Database,
  Play,
  RotateCcw,
  ShieldCheck,
  Target,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { type FormEvent, useState } from "react";

const config: Record<
  AgentKind,
  {
    eyebrow: string;
    title: string;
    description: string;
    icon: LucideIcon;
    accent: string;
    queryLabel: string;
    queryPlaceholder: string;
  }
> = {
  analyst: {
    eyebrow: "Evidence specialist",
    title: "Analyst workbench",
    description:
      "Retrieve and synthesize source-grounded evidence with an explicit confidence breakdown.",
    icon: BrainCircuit,
    accent: "Query + instrument → evidence packet",
    queryLabel: "Research question",
    queryPlaceholder:
      "What evidence supports or challenges operating margin expansion over the next two quarters?",
  },
  risk: {
    eyebrow: "Risk specialist",
    title: "Risk workbench",
    description:
      "Evaluate regime, volatility, correlation, concentration and risk-aware position sizing.",
    icon: ShieldCheck,
    accent: "Analyst packet → risk constraints",
    queryLabel: "Risk objective",
    queryPlaceholder:
      "Evaluate portfolio and market risks relevant to this research packet.",
  },
  strategy: {
    eyebrow: "Strategy specialist",
    title: "Strategy workbench",
    description:
      "Score playbooks against the evidence and risk packet, with visible suppression guardrails.",
    icon: Target,
    accent: "Analysis + risk → scored setups",
    queryLabel: "Strategy objective",
    queryPlaceholder:
      "Generate and score conditional research playbooks for this context.",
  },
  execution: {
    eyebrow: "Execution specialist",
    title: "Execution workbench",
    description:
      "Stress-test a structured setup against risk constraints and estimate execution geometry.",
    icon: Zap,
    accent: "Setup + risk → simulated verdict",
    queryLabel: "Execution objective",
    queryPlaceholder:
      "Evaluate the supplied setup and return entry, stop, target and expected R.",
  },
};

function JsonInput({
  label,
  hint,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  const parsed = parseJsonObject(value);
  return (
    <Field label={label} hint={hint}>
      <textarea
        className="textarea textarea-code"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        spellCheck={false}
      />
      {value.trim() && (
        <Badge tone={parsed.error ? "red" : "teal"}>
          {parsed.error ? "Invalid JSON" : "Valid structured input"}
        </Badge>
      )}
    </Field>
  );
}

export function AgentWorkspace({ kind }: { kind: AgentKind }) {
  const details = config[kind];
  const Icon = details.icon;
  const [query, setQuery] = useState("");
  const [ticker, setTicker] = useState("AAPL");
  const [lookback, setLookback] = useState(90);
  const [demo, setDemo] = useState(true);
  const [useLlm, setUseLlm] = useState(true);
  const [primaryJson, setPrimaryJson] = useState("");
  const [secondaryJson, setSecondaryJson] = useState("");
  const [advancedJson, setAdvancedJson] = useState("");
  const [job, setJob] = useState<Job>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const primaryLabel =
    kind === "execution" ? "Strategy setup JSON" : "Analyst output JSON";
  const secondaryLabel =
    kind === "execution" ? "Risk constraints JSON" : "Risk output JSON";

  const reset = () => {
    setQuery("");
    setTicker("AAPL");
    setPrimaryJson("");
    setSecondaryJson("");
    setAdvancedJson("");
    setError("");
    setJob(undefined);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Add an objective for this specialist run.");
      return;
    }

    const primary = parseJsonObject(primaryJson);
    const secondary = parseJsonObject(secondaryJson);
    const advanced = parseJsonObject(advancedJson);
    const invalid = [primary.error, secondary.error, advanced.error].find(Boolean);
    if (invalid) {
      setError(invalid);
      return;
    }

    const symbol = ticker.trim().toUpperCase();
    const defaultRisk = {
      universe: symbol ? [symbol] : [],
      vol_regime: "medium",
      vix_level: 20,
      position_sizing: [],
      concentration: {},
    };
    const defaultSetup = {
      strategy: "trend_following",
      strategy_name: "Trend Following",
      instrument: symbol || "AAPL",
      direction: "long",
      rationale: query.trim(),
      confidence: 0.6,
      playbook_fit: 0.6,
      horizon: "swing",
      risk_note: "Apply the supplied risk constraints.",
    };
    let payload: RunRequest;
    if (kind === "analyst") {
      payload = {
        query: query.trim(),
        demo_mode: demo,
      };
    } else if (kind === "risk") {
      payload = {
        tickers: symbol ? [symbol] : [],
        lookback_days: lookback,
        use_llm: useLlm,
        demo_mode: demo,
        ...(primary.value ? { briefing: primary.value } : {}),
      };
    } else if (kind === "strategy") {
      if (!demo && !secondary.value) {
        setError("Paste a Risk Agent output before a live Strategy run.");
        return;
      }
      payload = {
        risk: secondary.value ?? defaultRisk,
        ...(primary.value ? { briefing: primary.value } : {}),
        use_llm: useLlm,
        demo_mode: demo,
      };
    } else {
      if (!demo && !primary.value) {
        setError("Paste a StrategySetup before a live Execution run.");
        return;
      }
      const setupRecord: Record<string, unknown> =
        primary.value ?? defaultSetup;
      const nestedSetups = setupRecord.setups;
      payload = {
        setups: Array.isArray(nestedSetups) ? nestedSetups : [setupRecord],
        ...(secondary.value ? { risk: secondary.value } : {}),
        use_llm: useLlm,
        demo_mode: demo,
      };
    }
    payload = {...payload, ...(advanced.value ?? {})};

    setSubmitting(true);
    setError("");
    try {
      setJob(await api.run(kind, payload));
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow={details.eyebrow}
        title={details.title}
        description={details.description}
      >
        <Badge tone={demo ? "amber" : "teal"}>
          {demo ? "Demo / frozen inputs" : "Live data requested"}
        </Badge>
      </PageHeader>

      <div className="grid grid-main">
        <form className="stack" onSubmit={submit}>
          <Panel>
            <SectionHeading
              eyebrow="Guided input"
              title="Specialist brief"
              detail={details.accent}
              action={
                <div className="agent-card-icon">
                  <Icon size={18} />
                </div>
              }
            />
            <div className="stack">
              <Field label={details.queryLabel}>
                <textarea
                  className="textarea"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder={details.queryPlaceholder}
                  required
                />
              </Field>
              <div className="grid grid-2">
                <Field label="Primary ticker / instrument">
                  <input
                    className="input"
                    value={ticker}
                    onChange={(event) => setTicker(event.target.value)}
                    placeholder="AAPL"
                  />
                </Field>
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
              </div>
            </div>
          </Panel>

          {kind !== "analyst" && (
            <Panel>
              <SectionHeading
                eyebrow="Upstream hand-off"
                title="Paste structured outputs"
                detail="Reuse a prior result or edit the JSON before running this agent."
                action={<ClipboardPaste size={18} />}
              />
              <div className="stack">
                <JsonInput
                  label={primaryLabel}
                  hint={
                    kind === "execution"
                      ? "Required setup fields may include ticker, direction, thesis and levels."
                      : "Paste the structured analyst result or a compatible subset."
                  }
                  value={primaryJson}
                  onChange={setPrimaryJson}
                  placeholder={
                    kind === "execution"
                      ? '{\n  "ticker": "AAPL",\n  "direction": "long",\n  "thesis": "..."\n}'
                      : '{\n  "confidence": 0.72,\n  "key_points": ["..."]\n}'
                  }
                />
                {(kind === "strategy" || kind === "execution") && (
                  <JsonInput
                    label={secondaryLabel}
                    hint="Paste the upstream risk output, including regime and sizing constraints where available."
                    value={secondaryJson}
                    onChange={setSecondaryJson}
                    placeholder={'{\n  "regime": "risk-on",\n  "position_size": 0.02\n}'}
                  />
                )}
              </div>
            </Panel>
          )}

          <Panel>
            <SectionHeading eyebrow="Runtime" title="Model & data controls" />
            <div className="grid grid-2">
              <label className="toggle-line">
                <span>
                  {kind === "analyst"
                    ? "LLM synthesis required in live mode"
                    : "LLM synthesis"}
                </span>
                <input
                  type="checkbox"
                  className="toggle"
                  checked={useLlm}
                  disabled={kind === "analyst"}
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
            <details className="advanced" style={{ marginTop: 16 }}>
              <summary>
                <Braces size={14} style={{ display: "inline", marginRight: 6 }} />
                Advanced request JSON
              </summary>
              <JsonInput
                label="Payload overrides"
                hint="Top-level fields merge into the request. Guided values remain the default."
                value={advancedJson}
                onChange={setAdvancedJson}
                placeholder={'{\n  "temperature": 0.2,\n  "metadata": {}\n}'}
              />
            </details>
            {error && <ErrorState message={error} />}
            <div className="button-row" style={{ marginTop: 18 }}>
              <button className="button button-primary" disabled={submitting}>
                <Play size={16} />
                {submitting ? "Starting agent…" : `Run ${titleCase(kind)} agent`}
              </button>
              <button
                className="button button-secondary"
                type="button"
                onClick={reset}
              >
                <RotateCcw size={15} />
                Reset
              </button>
            </div>
          </Panel>
        </form>

        <div className="stack">
          {job ? (
            <JobMonitor initialJob={job} kind={kind} />
          ) : (
            <>
              <Panel>
                <SectionHeading eyebrow="Output contract" title="What to expect" />
                <div className="timeline">
                  {kind === "analyst" && (
                    <>
                      <Preview label="Retrieve evidence" detail="Corpus search and citation scoring" />
                      <Preview label="Synthesize" detail="Key findings, risks and caveats" />
                      <Preview label="Calibrate" detail="Confidence component breakdown" />
                    </>
                  )}
                  {kind === "risk" && (
                    <>
                      <Preview label="Classify regime" detail="Market and volatility state" />
                      <Preview label="Map dependencies" detail="Full correlation structure" />
                      <Preview label="Constrain" detail="Concentration and sizing" />
                    </>
                  )}
                  {kind === "strategy" && (
                    <>
                      <Preview label="Set macro bias" detail="Evidence-adjusted stance" />
                      <Preview label="Score playbooks" detail="Comparable setup ranking" />
                      <Preview label="Suppress" detail="Guardrail-incompatible ideas" />
                    </>
                  )}
                  {kind === "execution" && (
                    <>
                      <Preview label="Validate setup" detail="Risk and data prerequisites" />
                      <Preview label="Build geometry" detail="Entry, stop and target" />
                      <Preview label="Estimate outcome" detail="Probability, R and MAE" />
                    </>
                  )}
                </div>
              </Panel>
              <div className="callout callout-amber">
                <Database size={20} />
                <div>
                  <strong>Research simulation boundary</strong>
                  <p>
                    Agent outputs may use stale, incomplete or model-generated
                    values. Validate against source data before any decision.
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

function Preview({ label, detail }: { label: string; detail: string }) {
  return (
    <div className="timeline-item">
      <div className="timeline-dot">
        <Braces size={14} />
      </div>
      <strong>{label}</strong>
      <span>{detail}</span>
    </div>
  );
}
