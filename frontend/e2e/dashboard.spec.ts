import { expect, test } from "@playwright/test";

const API = "http://localhost:8000/api/v1";

const result = {
  demo_mode: true,
  query: "Fed outlook",
  original_query: "Fed outlook",
  tickers: ["AAPL"],
  decision: "trade",
  no_trade_reason: "",
  route_log: ["analyst(attempt=1)", "risk", "strategy", "execution"],
  errors: [],
  briefing: {
    query: "Fed outlook",
    summary: "Inflation is moderating while policy remains restrictive [1].",
    confidence: 0.82,
    confidence_breakdown: { retrieval: 0.88, diversity: 0.76 },
    key_points: [{ point: "Disinflation remains visible [1]." }],
    risks: ["A renewed inflation impulse could delay easing."],
    citations: [
      {
        index: 1,
        source: "https://example.com/fomc",
        doc_type: "fomc_minutes",
        date: "2026-01-01",
        score: 0.91,
        excerpt: "Participants noted progress toward price stability.",
      },
    ],
  },
  risk: {
    universe: ["AAPL"],
    vol_regime: "medium",
    vix_level: 18.4,
    mean_realised_vol: 0.24,
    correlation_matrix: { AAPL: { AAPL: 1 } },
    concentration: {
      effective_number_of_bets: 1,
      mean_pairwise_correlation: 0,
    },
    position_sizing: [
      { ticker: "AAPL", max_position_pct: 0.2, risk_per_trade_pct: 0.01 },
    ],
  },
  strategy: {
    macro_bias: { direction: "bullish", strength: 0.68 },
    candidate_scores: [{ name: "Trend Following", score: 0.81 }],
    setups: [
      {
        strategy_name: "Trend Following",
        instrument: "AAPL",
        direction: "long",
        confidence: 0.74,
        rationale: "Constructive macro bias.",
      },
    ],
    suppressed: [],
  },
  cards: [
    {
      instrument: "AAPL",
      strategy_name: "Trend Following",
      direction: "long",
      levels: { entry: 185, stop_loss: 176.6, take_profit: 201.8 },
      stats: {
        prob_tp_before_sl: 0.57,
        prob_sl_before_tp: 0.31,
        prob_timeout: 0.12,
        expected_r: 0.42,
        mae_p95_r: 0.94,
      },
      sizing: { notional_pct: 0.2 },
      verdict: "Positive simulated expectancy.",
    },
  ],
};

test.beforeEach(async ({ page }) => {
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace("/api/v1", "");
    const method = route.request().method();
    const json = (value: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value) });

    if (path === "/health") {
      return json({
        status: "ok",
        services: {
          api: { status: "ok" },
          database: { status: "ok" },
          qdrant: { status: "ok", count: 339 },
          ollama: { status: "ok", count: 1 },
        },
      });
    }
    if (path === "/config/defaults") {
      return json({ watchlist: ["AAPL", "MSFT"], playbooks: [] });
    }
    if (path === "/jobs" && method === "GET") return json([]);
    if (path === "/conversations" && method === "GET") return json([]);
    if (path === "/runs/pipeline" && method === "POST") {
      return json({ id: "demo-job", status: "queued" }, 202);
    }
    if (path === "/jobs/demo-job/events") {
      return route.fulfill({ status: 503, body: "stream unavailable" });
    }
    if (path === "/jobs/demo-job") {
      return json({
        id: "demo-job",
        kind: "pipeline",
        status: "succeeded",
        metadata: { demo_mode: true },
        result,
        created_at: "2026-01-01T12:00:00Z",
      });
    }
    if (path === "/corpus/reset" && method === "POST") {
      const body = route.request().postDataJSON();
      expect(body.confirmation).toBe("RESET FINANCIAL DOCS");
      return json({ id: "reset-job", status: "queued" }, 202);
    }
    return json({ detail: `Unhandled ${method} ${path}` }, 404);
  });
});

test("runs the demo pipeline and renders all agent results", async ({ page }) => {
  await page.goto("/workspace");
  await page.getByLabel("Research query").fill("Fed outlook");
  await page.getByRole("button", { name: "Start full pipeline" }).click();

  await expect(page.getByText("Inflation is moderating while policy remains restrictive [1].")).toBeVisible();
  await expect(page.getByText("Correlation matrix")).toBeVisible();
  await expect(page.getAllByText("Trend Following").first()).toBeVisible();
  await expect(page.getByText("Positive simulated expectancy.")).toBeVisible();
});

test("requires the exact destructive reset confirmation", async ({ page }) => {
  await page.goto("/data");
  const reset = page.getByRole("button", { name: "Reset financial docs" });
  await expect(reset).toBeDisabled();
  await page
    .getByLabel("Type RESET FINANCIAL DOCS to continue")
    .fill("RESET FINANCIAL DOCS");
  await expect(reset).toBeEnabled();
  await reset.click();
  await expect(page.getByText("Operation accepted")).toBeVisible();
});
