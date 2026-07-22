import { Badge, Metric, SectionHeading } from "@/components/ui";
import {
  asRecord,
  asStringList,
  compactNumber,
  getValue,
  percent,
  shortDate,
  titleCase,
} from "@/lib/format";
import type { Citation, JsonRecord, MAFASResult, RunKind } from "@/lib/types";
import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  CheckCircle2,
  CircleOff,
  Compass,
  Crosshair,
  ExternalLink,
  Gauge,
  Network,
  ShieldAlert,
  Target,
} from "lucide-react";

function numericEntries(value: unknown) {
  return Object.entries(asRecord(value)).filter(([, item]) =>
    Number.isFinite(Number(item)),
  );
}

function confidenceTone(value: number) {
  return value >= 0.7 || value >= 70
    ? ""
    : value >= 0.4 || value >= 40
      ? "bar-fill-amber"
      : "bar-fill-red";
}

function BarList({
  values,
  format = percent,
}: {
  values: Array<[string, unknown]>;
  format?: (value: unknown) => string;
}) {
  if (!values.length) return <p className="muted">No component scores returned.</p>;
  return (
    <div className="bar-list">
      {values.map(([label, raw]) => {
        const number = Number(raw);
        const normalized = Math.min(
          100,
          Math.max(0, Math.abs(number) <= 1 ? number * 100 : number),
        );
        return (
          <div className="bar-row" key={label}>
            <span>{titleCase(label)}</span>
            <div
              className="bar-track"
              role="meter"
              aria-label={titleCase(label)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={normalized}
            >
              <div
                className={`bar-fill ${confidenceTone(number)}`}
                style={{ width: `${normalized}%` }}
              />
            </div>
            <strong>{format(raw)}</strong>
          </div>
        );
      })}
    </div>
  );
}

function ListBlock({
  title,
  values,
  risk,
}: {
  title: string;
  values: string[];
  risk?: boolean;
}) {
  if (!values.length) return null;
  return (
    <div>
      <h3>{title}</h3>
      <ul className={`key-list ${risk ? "risk-list" : ""}`}>
        {values.map((value, index) => (
          <li key={`${value}-${index}`}>{value}</li>
        ))}
      </ul>
    </div>
  );
}

function CitationCards({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <section className="result-section">
      <SectionHeading
        eyebrow="Evidence trail"
        title="Sources & citations"
        detail={`${citations.length} retrieved evidence item${citations.length === 1 ? "" : "s"}`}
      />
      {citations.map((citation, index) => (
        <article className="citation-card" key={citation.id ?? `${citation.title}-${index}`}>
          <header>
            <strong>{citation.title ?? citation.source ?? `Source ${index + 1}`}</strong>
            {citation.url && (
              <a
                href={citation.url}
                target="_blank"
                rel="noreferrer"
                className="link"
                aria-label={`Open ${citation.title ?? "citation"}`}
              >
                <ExternalLink size={15} />
              </a>
            )}
          </header>
          {citation.excerpt && <p>{citation.excerpt}</p>}
          <div className="citation-meta">
            {citation.source && <span>{citation.source}</span>}
            {citation.date && <span>{shortDate(citation.date)}</span>}
            {citation.score !== undefined && (
              <span>Relevance {percent(citation.score)}</span>
            )}
          </div>
        </article>
      ))}
    </section>
  );
}

function AnalystResult({ data }: { data: JsonRecord }) {
  if (!Object.keys(data).length) return null;
  const confidence = getValue(data, ["confidence", "overall_confidence", "score"]);
  const summary = getValue(data, ["summary", "executive_summary", "overview"]);
  const breakdown = getValue(data, [
    "confidence_breakdown",
    "confidence_components",
    "scores",
  ]);
  const points = asStringList(
    Array.isArray(data.key_points)
      ? data.key_points.map((item) => {
          const point = asRecord(item);
          return String(point.point ?? item);
        })
      : getValue(data, ["key_findings", "findings", "thesis"]),
  );
  const risks = asStringList(
    getValue(data, ["risks", "risk_factors", "caveats", "limitations"]),
  );
  const rawCitations =
    getValue(data, ["citations", "sources", "evidence"]) ?? [];
  const citations = Array.isArray(rawCitations)
    ? rawCitations.map((item) =>
        typeof item === "string"
          ? ({ title: item } as Citation)
          : ({
              ...(item as Citation),
              title:
                String(asRecord(item).doc_type ?? "") ||
                (item as Citation).title,
              url:
                String((item as Citation).source ?? "").startsWith("http")
                  ? (item as Citation).source
                  : (item as Citation).url,
            } as Citation),
      )
    : [];

  return (
    <>
      <section className="result-section">
        <SectionHeading
          eyebrow="Analyst agent"
          title="Evidence synthesis"
          action={
            confidence !== undefined ? (
              <Badge tone="teal">Confidence {percent(confidence)}</Badge>
            ) : null
          }
        />
        {summary !== undefined && (
          <p className="muted" style={{ lineHeight: 1.65, marginBottom: 16 }}>
            {String(summary)}
          </p>
        )}
        <BarList values={numericEntries(breakdown)} />
      </section>
      {(points.length > 0 || risks.length > 0) && (
        <section className="result-section grid grid-2">
          <ListBlock title="Key points" values={points} />
          <ListBlock title="Risks & caveats" values={risks} risk />
        </section>
      )}
      <CitationCards citations={citations} />
    </>
  );
}

function correlationData(data: JsonRecord) {
  const raw = getValue(data, ["correlation_matrix", "correlations", "correlation"]);
  if (!raw) return { labels: [] as string[], matrix: [] as number[][] };
  if (Array.isArray(raw)) {
    const labels = asStringList(getValue(data, ["assets", "tickers", "symbols"]));
    return { labels, matrix: raw as number[][] };
  }
  const record = asRecord(raw);
  const labels = Object.keys(record);
  return {
    labels,
    matrix: labels.map((row) => {
      const values = asRecord(record[row]);
      return labels.map((column) => Number(values[column] ?? 0));
    }),
  };
}

function CorrelationHeatmap({ data }: { data: JsonRecord }) {
  const { labels, matrix } = correlationData(data);
  if (!labels.length || !matrix.length) return null;
  const columns = labels.length + 1;
  return (
    <div className="heatmap" aria-label="Asset correlation heatmap">
      <div
        className="heatmap-grid"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(54px, 1fr))` }}
      >
        <div className="heat-cell heat-label" />
        {labels.map((label) => (
          <div className="heat-cell heat-label" key={`column-${label}`}>
            {label}
          </div>
        ))}
        {matrix.map((row, rowIndex) => (
          <div style={{ display: "contents" }} key={`row-${labels[rowIndex]}`}>
            <div className="heat-cell heat-label">{labels[rowIndex]}</div>
            {row.map((value, columnIndex) => {
              const strength = Math.min(0.82, 0.13 + Math.abs(value) * 0.62);
              return (
                <div
                  className="heat-cell"
                  key={`${rowIndex}-${columnIndex}`}
                  style={{
                    background:
                      value < 0
                        ? `rgb(126 45 68 / ${strength})`
                        : `rgb(19 105 97 / ${strength})`,
                  }}
                  title={`${labels[rowIndex]} / ${labels[columnIndex]}: ${value.toFixed(2)}`}
                >
                  {value.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function RiskResult({ data }: { data: JsonRecord }) {
  if (!Object.keys(data).length) return null;
  const regime = getValue(data, [
    "vol_regime",
    "regime",
    "market_regime",
    "risk_regime",
  ]);
  const vix = getValue(data, ["vix", "vix_level", "volatility_index"]);
  const volatility = getValue(data, [
    "mean_realised_vol",
    "volatility",
    "portfolio_volatility",
    "annualized_volatility",
  ]);
  const concentration = asRecord(
    getValue(data, ["concentration", "concentration_risk"]),
  );
  const perAsset = Array.isArray(data.per_asset) ? data.per_asset : [];
  const positionSizing = Array.isArray(data.position_sizing)
    ? data.position_sizing
    : [];
  const riskScores = perAsset.length
    ? perAsset.map((item) => {
        const asset = asRecord(item);
        return [String(asset.ticker ?? "Asset"), asset.realised_vol] as [
          string,
          unknown,
        ];
      })
    : numericEntries(
        getValue(data, ["risk_scores", "volatility_breakdown", "metrics"]),
      );

  return (
    <>
      <section className="result-section">
        <SectionHeading eyebrow="Risk agent" title="Market & portfolio risk" />
        <div className="grid grid-4">
          <Metric
            label="Regime"
            value={titleCase(String(regime ?? "Unknown"))}
            icon={Gauge}
          />
          <Metric label="VIX" value={compactNumber(vix)} icon={BarChart3} tone="amber" />
          <Metric label="Volatility" value={percent(volatility)} icon={ShieldAlert} />
          <Metric
            label="Effective bets"
            value={compactNumber(concentration.effective_number_of_bets)}
            icon={Target}
            tone="teal"
          />
        </div>
      </section>
      {riskScores.length > 0 && (
        <section className="result-section">
          <h3>Volatility & exposure profile</h3>
          <BarList values={riskScores} />
        </section>
      )}
      <section className="result-section">
        <SectionHeading
          eyebrow="Cross-asset structure"
          title="Correlation matrix"
          action={
            Object.keys(concentration).length ? (
              <Badge tone={concentration.flagged ? "red" : "amber"}>
                Mean correlation{" "}
                {compactNumber(concentration.mean_pairwise_correlation)}
              </Badge>
            ) : null
          }
        />
        <CorrelationHeatmap data={data} />
      </section>
      {positionSizing.length > 0 && (
        <section className="result-section">
          <SectionHeading
            eyebrow="Portfolio constraints"
            title="Position sizing"
          />
          <div className="grid grid-3">
            {positionSizing.map((item, index) => {
              const sizing = asRecord(item);
              return (
                <article className="setup-card" key={index}>
                  <header>
                    <strong>{String(sizing.ticker ?? "Asset")}</strong>
                    <Badge tone="teal">
                      Max {percent(sizing.max_position_pct)}
                    </Badge>
                  </header>
                  <p>
                    Risk per trade {percent(sizing.risk_per_trade_pct)}.{" "}
                    {String(sizing.rationale ?? "")}
                  </p>
                </article>
              );
            })}
          </div>
        </section>
      )}
    </>
  );
}

function StrategyResult({ data }: { data: JsonRecord }) {
  if (!Object.keys(data).length) return null;
  const bias = asRecord(getValue(data, ["macro_bias", "bias", "market_bias"]));
  const candidates = Array.isArray(data.candidate_scores)
    ? data.candidate_scores
    : [];
  const scores = candidates.length
    ? candidates.map((item) => {
        const candidate = asRecord(item);
        return [
          String(candidate.name ?? candidate.key ?? "Playbook"),
          candidate.score,
        ] as [string, unknown];
      })
    : numericEntries(
        getValue(data, ["playbook_scores", "strategy_scores", "scores"]),
      );
  const setupsRaw = getValue(data, [
    "setups",
    "recommended_setups",
    "trade_setups",
  ]);
  const suppressedRaw = getValue(data, [
    "suppressed",
    "suppressed_setups",
    "rejected_setups",
  ]);
  const setups = Array.isArray(setupsRaw) ? setupsRaw : [];
  const suppressed = Array.isArray(suppressedRaw) ? suppressedRaw : [];

  return (
    <>
      <section className="result-section">
        <SectionHeading
          eyebrow="Strategy agent"
          title="Playbook selection"
          action={
            <Badge tone="blue">
              Macro bias: {String(bias.direction ?? "neutral")}{" "}
              {percent(bias.strength)}
            </Badge>
          }
        />
        <BarList values={scores} />
      </section>
      <div className="grid grid-2">
        <section className="result-section">
          <h3>Candidate setups</h3>
          {setups.length ? (
            setups.map((setup, index) => {
              const item = asRecord(setup);
              return (
                <article className="setup-card" key={index}>
                  <header>
                    <strong>{setupTitle(item, index, setups.length)}</strong>
                    <Badge tone="teal">
                      {percent(getValue(item, ["score", "confidence", "probability"]))}
                    </Badge>
                  </header>
                  <p>
                    <strong>
                      {String(item.instrument ?? "—")}{" "}
                      {String(item.direction ?? "").toUpperCase()}
                    </strong>
                    {" — "}
                    {String(
                      getValue(item, ["rationale", "thesis", "description"]) ??
                        "No rationale returned.",
                    )}
                  </p>
                </article>
              );
            })
          ) : (
            <p className="muted">No candidate setups returned.</p>
          )}
        </section>
        <section className="result-section">
          <h3>Suppressed by guardrails</h3>
          {suppressed.length ? (
            suppressed.map((setup, index) => {
              const item = asRecord(setup);
              if (typeof setup === "string") {
                return (
                  <article className="playbook-card" key={index}>
                    <header>
                      <strong>Suppressed playbook</strong>
                      <Badge tone="red">Suppressed</Badge>
                    </header>
                    <p>{setup}</p>
                  </article>
                );
              }
              return (
                <article className="playbook-card" key={index}>
                  <header>
                    <strong>
                      {String(
                        getValue(item, ["name", "strategy", "ticker"]) ??
                          `Suppressed ${index + 1}`,
                      )}
                    </strong>
                    <Badge tone="red">Suppressed</Badge>
                  </header>
                  <p>
                    {String(
                      getValue(item, ["reason", "rationale", "constraint"]) ??
                        "Guardrail threshold not met.",
                    )}
                  </p>
                </article>
              );
            })
          ) : (
            <p className="muted">No strategies were suppressed.</p>
          )}
        </section>
      </div>
    </>
  );
}

function setupTitle(item: JsonRecord, index: number, total: number): string {
  const name = String(
    getValue(item, [
      "strategy_name",
      "name",
      "strategy",
      "ticker",
      "setup",
    ]) ?? "",
  ).trim();
  const instrument = String(item.instrument ?? "").trim();
  const direction = String(item.direction ?? "").trim();
  const parts: string[] = [];
  if (total > 1) parts.push(`Setup ${index + 1}`);
  if (name) parts.push(name);
  if (instrument) parts.push(instrument);
  if (direction) parts.push(direction.toUpperCase());
  return parts.length ? parts.join(" · ") : `Setup ${index + 1}`;
}

function executionCardTitle(
  data: JsonRecord,
  index: number,
  total: number,
): string {
  const strategy = String(
    getValue(data, ["strategy_name", "strategy", "playbook"]) ?? "",
  ).trim();
  const instrument = String(data.instrument ?? "").trim();
  const direction = String(data.direction ?? "").trim();
  const parts: string[] = [];
  if (total > 1) parts.push(`Trade ${index + 1}`);
  if (strategy) parts.push(strategy);
  if (instrument) parts.push(instrument);
  if (direction) parts.push(direction.toUpperCase());
  return parts.length ? parts.join(" · ") : `Trade card ${index + 1}`;
}

function executionStatus(data: JsonRecord): {
  tone: "teal" | "amber" | "red";
  label: string;
  detail?: string;
} {
  const skipReason = getValue(data, [
    "skip_reason",
    "skipped_reason",
    "reason_skipped",
  ]);
  if (skipReason) {
    return { tone: "red", label: "Skipped", detail: String(skipReason) };
  }
  if (data.simulated === false) {
    return {
      tone: "amber",
      label: "Not simulated",
      detail: String(skipReason ?? "Simulation unavailable."),
    };
  }
  return { tone: "teal", label: "Simulated" };
}

function ExecutionResult({
  data,
  index = 0,
  total = 1,
}: {
  data: JsonRecord;
  index?: number;
  total?: number;
}) {
  if (!Object.keys(data).length) return null;
  const levels = asRecord(data.levels);
  const stats = asRecord(data.stats);
  const sizing = asRecord(data.sizing);
  const entry = getValue(
    { ...data, ...levels },
    ["entry", "entry_price", "suggested_entry"],
  );
  const stop = getValue(
    { ...data, ...levels },
    ["stop", "stop_price", "stop_loss"],
  );
  const target = getValue(
    { ...data, ...levels },
    ["target", "target_price", "take_profit"],
  );
  const probability = getValue(
    { ...data, ...stats },
    ["prob_tp_before_sl", "probability", "success_probability", "win_probability"],
  );
  const expectedR = getValue(
    { ...data, ...stats },
    ["expected_r", "expected_R", "reward_risk"],
  );
  const mae = getValue(
    { ...data, ...stats },
    ["mae_p95_r", "mae", "expected_mae", "max_adverse_excursion"],
  );
  const positionSize = getValue(
    { ...data, ...sizing },
    ["notional_pct", "position_size", "size"],
  );
  const verdict = getValue(data, ["verdict", "decision", "action"]);
  const status = executionStatus(data);
  const title = executionCardTitle(data, index, total);
  const subtitle = [
    data.horizon ? `${titleCase(String(data.horizon))} horizon` : "",
    data.data_source ? `Data: ${String(data.data_source)}` : "",
    data.bars_used ? `${String(data.bars_used)} bars` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="result-section execution-card">
      <SectionHeading
        eyebrow="Execution agent"
        title={title}
        detail={subtitle || "Monte Carlo trade geometry and sizing"}
        action={<Badge tone={status.tone}>{status.label}</Badge>}
      />
        <div className="trade-geometry">
          <div className="trade-level trade-stop">
            <span>Protective stop</span>
            <strong>{compactNumber(stop)}</strong>
          </div>
          <div className="trade-level trade-entry">
            <span>Entry</span>
            <strong>{compactNumber(entry)}</strong>
          </div>
          <div className="trade-level trade-target">
            <span>Target</span>
            <strong>{compactNumber(target)}</strong>
          </div>
        </div>
        <div className="grid grid-4" style={{ marginTop: 16 }}>
          <Metric label="Probability" value={percent(probability)} icon={Gauge} tone="teal" />
          <Metric label="Expected R" value={compactNumber(expectedR)} icon={Crosshair} />
          <Metric label="MAE p95" value={`${compactNumber(mae)}R`} icon={ShieldAlert} tone="amber" />
          <Metric label="Position size" value={percent(positionSize)} icon={Target} />
        </div>
        {probability !== undefined && (
          <div style={{ marginTop: 16 }}>
            <BarList
              values={[
                ["TP before SL", probability],
                ["SL before TP", stats.prob_sl_before_tp],
                ["Timeout", stats.prob_timeout],
              ]}
            />
          </div>
        )}
        {status.detail && status.label !== "Simulated" && (
          <div className="error-box" style={{ marginTop: 16 }}>
            <CircleOff size={18} aria-hidden />
            <div>
              <strong>{status.label}</strong>
              <p>{status.detail}</p>
            </div>
          </div>
        )}
        {verdict && (
          <div className="execution-verdict">
            <strong>Verdict</strong>
            <p>{String(verdict)}</p>
          </div>
        )}
      </section>
  );
}

function AggregateResult({ data }: { data: JsonRecord }) {
  if (!Object.keys(data).length) return null;
  const decision = getValue(data, [
    "decision",
    "verdict",
    "recommendation",
    "action",
  ]);
  const noTrade =
    String(decision ?? "").toLowerCase() === "no_trade" ||
    Boolean(getValue(data, ["no_trade", "noTrade"]));
  const noTradeReason = getValue(data, ["no_trade_reason", "reason"]);
  const route = getValue(data, ["route_log", "route", "agent_route", "path"]);
  const errors = getValue(data, ["errors", "agent_errors", "warnings"]);
  const errorList = asStringList(errors);

  return (
    <section className="result-section">
      <SectionHeading
        eyebrow="Orchestrator"
        title="Aggregate decision"
        action={
          <Badge tone={noTrade ? "amber" : "teal"}>
            {noTrade ? "No-trade guardrail" : String(decision ?? "Research complete")}
          </Badge>
        }
      />
      <div className="grid grid-2">
        <div className="callout">
          {noTrade ? <CircleOff size={20} /> : <CheckCircle2 size={20} />}
          <div>
            <strong>{String(decision ?? "No aggregate decision returned")}</strong>
            <p>
              {noTrade
                ? String(
                    noTradeReason ??
                      "At least one guardrail prevented a simulated trade setup.",
                  )
                : "This is a research output, not a trading recommendation."}
            </p>
          </div>
        </div>
        <div className="callout">
          <Compass size={20} />
          <div>
            <strong>Agent route</strong>
            <p>{Array.isArray(route) ? route.join(" → ") : String(route ?? "Standard pipeline")}</p>
          </div>
        </div>
      </div>
      {errorList.length > 0 && (
        <div className="error-box" style={{ marginTop: 14 }}>
          <AlertTriangle size={18} />
          <div>
            <strong>Pipeline warnings</strong>
            <p>{errorList.join(" · ")}</p>
          </div>
        </div>
      )}
    </section>
  );
}

export function ResultView({
  result,
  kind = "pipeline",
  partial = false,
}: {
  result: MAFASResult;
  kind?: RunKind;
  partial?: boolean;
}) {
  const root = asRecord(result);
  const analyst =
    Object.keys(asRecord(root.briefing ?? root.analyst)).length > 0
      ? asRecord(root.briefing ?? root.analyst)
      : kind === "analyst"
        ? root
        : {};
  const risk =
    Object.keys(asRecord(root.risk)).length > 0
      ? asRecord(root.risk)
      : kind === "risk"
        ? root
        : {};
  const strategy =
    Object.keys(asRecord(root.strategy)).length > 0
      ? asRecord(root.strategy)
      : kind === "strategy"
        ? root
        : {};
  const executionCards = Array.isArray(root.cards)
    ? root.cards.map(asRecord)
    : Array.isArray(asRecord(root.execution).cards)
      ? (asRecord(root.execution).cards as unknown[]).map(asRecord)
      : [];
  const execution =
    executionCards[0] ??
    (Object.keys(asRecord(root.execution)).length > 0
      ? asRecord(root.execution)
      : kind === "execution"
        ? root
        : {});
  const aggregate =
    Object.keys(asRecord(root.aggregate)).length > 0
      ? { ...root, ...asRecord(root.aggregate) }
      : root;

  return (
    <div className="result-shell">
      <div className="result-summary">
        <div>
          <p className="eyebrow">{partial ? "Live partial result" : "Completed research packet"}</p>
          <h2>{partial ? "Agents are still working" : "MAFAS analysis output"}</h2>
          <p>
            Structured model output for research and simulation. Verify every
            material fact against primary sources.
          </p>
        </div>
        <Badge tone={partial ? "amber" : "teal"}>
          <Network size={13} aria-hidden />
          {partial ? "Streaming" : "Final"}
        </Badge>
      </div>
      <AggregateResult data={aggregate} />
      <AnalystResult data={analyst} />
      <RiskResult data={risk} />
      <StrategyResult data={strategy} />
      {executionCards.length > 0 ? (
        executionCards.map((card, index) => (
          <ExecutionResult
            data={card}
            key={index}
            index={index}
            total={executionCards.length}
          />
        ))
      ) : (
        <ExecutionResult data={execution} />
      )}
      {!Object.keys(analyst).length &&
        !Object.keys(risk).length &&
        !Object.keys(strategy).length &&
        !Object.keys(execution).length && (
          <section className="result-section">
            <SectionHeading
              eyebrow="Raw output"
              title="Structured result"
              action={<BookOpen size={18} />}
            />
            <pre className="json-view">{JSON.stringify(result, null, 2)}</pre>
          </section>
        )}
    </div>
  );
}
