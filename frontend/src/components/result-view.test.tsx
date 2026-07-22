import { ResultView } from "@/components/result-view";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

describe("ResultView", () => {
  it("renders the canonical PipelineResult contract", () => {
    render(
      <ResultView
        kind="pipeline"
        result={{
          decision: "trade",
          route_log: ["analyst(attempt=1)", "risk", "strategy", "execution"],
          briefing: {
            confidence: 0.78,
            confidence_breakdown: { retrieval: 0.8, diversity: 0.7 },
            key_points: [{ point: "Inflation is moderating [1]." }],
            risks: ["Policy could remain restrictive."],
            citations: [
              {
                source: "https://example.com/fomc",
                doc_type: "fomc_minutes",
                date: "2026-01-01",
                score: 0.9,
                excerpt: "Participants discussed inflation.",
              },
            ],
          },
          risk: {
            vol_regime: "medium",
            vix_level: 18.4,
            mean_realised_vol: 0.24,
            correlation_matrix: {
              AAPL: { AAPL: 1, MSFT: 0.6 },
              MSFT: { AAPL: 0.6, MSFT: 1 },
            },
            concentration: {
              effective_number_of_bets: 1.6,
              mean_pairwise_correlation: 0.6,
            },
            position_sizing: [
              {
                ticker: "AAPL",
                max_position_pct: 0.2,
                risk_per_trade_pct: 0.01,
              },
            ],
          },
          strategy: {
            macro_bias: { direction: "bullish", strength: 0.7 },
            candidate_scores: [
              { name: "Trend Following", score: 0.8 },
            ],
            setups: [
              {
                strategy_name: "Trend Following",
                instrument: "AAPL",
                direction: "long",
                confidence: 0.72,
                rationale: "Constructive trend.",
              },
            ],
            suppressed: ["Carry: regime mismatch."],
          },
          cards: [
            {
              instrument: "AAPL",
              strategy_name: "Trend Following",
              direction: "long",
              levels: { entry: 185, stop_loss: 176, take_profit: 203 },
              stats: {
                prob_tp_before_sl: 0.55,
                prob_sl_before_tp: 0.33,
                prob_timeout: 0.12,
                expected_r: 0.4,
                mae_p95_r: 0.95,
              },
              sizing: { notional_pct: 0.2 },
              verdict: "Positive simulated expectancy.",
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("Inflation is moderating [1].")).toBeInTheDocument();
    expect(screen.getByText("Medium")).toBeInTheDocument();
    expect(screen.getByText("Trend Following · AAPL · LONG")).toBeInTheDocument();
    expect(screen.getByText(/Positive simulated expectancy/)).toBeInTheDocument();
  });

  it("labels multiple execution trade cards", () => {
    render(
      <ResultView
        kind="pipeline"
        result={{
          cards: [
            {
              strategy_name: "Trend Following",
              instrument: "AAPL",
              direction: "long",
              levels: { entry: 185, stop_loss: 176, take_profit: 203 },
              stats: { prob_tp_before_sl: 0.55, expected_r: 0.4, mae_p95_r: 0.95 },
              sizing: { notional_pct: 0.2 },
              verdict: "Positive simulated expectancy.",
            },
            {
              strategy_name: "Mean Reversion",
              instrument: "MSFT",
              direction: "short",
              levels: { entry: 410, stop_loss: 420, take_profit: 390 },
              stats: { prob_tp_before_sl: 0.45, expected_r: 0.2, mae_p95_r: 1.1 },
              sizing: { notional_pct: 0.15 },
              verdict: "Edge is marginal under current volatility.",
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("Trade 1 · Trend Following · AAPL · LONG")).toBeInTheDocument();
    expect(screen.getByText("Trade 2 · Mean Reversion · MSFT · SHORT")).toBeInTheDocument();
    expect(screen.getByText("Verdict")).toBeInTheDocument();
  });
});

describe("ResultView compatibility", () => {
  it("renders a complete pipeline packet", () => {
    render(
      <ResultView
        result={{
          decision: "Research candidate",
          route: ["analyst", "risk", "strategy", "execution"],
          analyst: {
            confidence: 0.82,
            confidence_breakdown: { retrieval: 0.9, agreement: 0.74 },
            key_points: ["Margins expanded year over year"],
            risks: ["Demand may normalize"],
            citations: [
              {
                title: "Quarterly filing",
                url: "https://example.com/filing",
                score: 0.91,
              },
            ],
          },
          risk: {
            regime: "risk-on",
            vix: 15.2,
            volatility: 0.22,
            assets: ["AAPL", "SPY"],
            correlation_matrix: [
              [1, 0.65],
              [0.65, 1],
            ],
          },
          strategy: {
            macro_bias: "constructive",
            playbook_scores: { momentum: 0.7, mean_reversion: 0.35 },
            setups: [{ name: "Conditional momentum", score: 0.7 }],
          },
          execution: {
            entry: 100,
            stop: 95,
            target: 112,
            probability: 0.61,
            expected_r: 1.4,
            verdict: "simulated",
          },
        }}
      />,
    );

    expect(screen.getByText("Aggregate decision")).toBeInTheDocument();
    expect(screen.getByText("Evidence synthesis")).toBeInTheDocument();
    expect(screen.getByText("Margins expanded year over year")).toBeInTheDocument();
    expect(screen.getByText("Quarterly filing")).toBeInTheDocument();
    expect(screen.getByText("Correlation matrix")).toBeInTheDocument();
    expect(screen.getByText("Conditional momentum")).toBeInTheDocument();
    expect(screen.getByText("Trade card 1")).toBeInTheDocument();
  });

  it("shows a raw fallback for unrecognized output", () => {
    render(<ResultView result={{ custom_backend_field: "preserved" }} />);
    expect(screen.getByText("Structured result")).toBeInTheDocument();
    expect(screen.getByText(/custom_backend_field/)).toBeInTheDocument();
  });
});
