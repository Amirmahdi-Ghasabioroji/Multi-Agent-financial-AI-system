# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: dashboard.spec.ts >> runs the demo pipeline and renders all agent results
- Location: e2e/dashboard.spec.ts:128:5

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: getByText('Inflation is moderating while policy remains restrictive [1].')
Expected: visible
Timeout: 8000ms
Error: element(s) not found

Call log:
  - Expect "toBeVisible" with timeout 8000ms
  - waiting for getByText('Inflation is moderating while policy remains restrictive [1].')

```

```yaml
- complementary "Primary navigation":
  - strong: MAFAS
  - text: Research workstation
  - button "Close navigation"
  - navigation:
    - paragraph: Command centre
    - link "Overview":
      - /url: /
    - link "Pipeline workspace":
      - /url: /workspace
    - paragraph: Specialists
    - link "Analyst":
      - /url: /agents/analyst
    - link "Risk":
      - /url: /agents/risk
    - link "Strategy":
      - /url: /agents/strategy
    - link "Execution":
      - /url: /agents/execution
    - paragraph: Operations
    - link "Run history":
      - /url: /history
    - link "Data & corpus":
      - /url: /data
  - text: Pipeline topology 4-agent sequential route
  - button "Collapse sidebar": Collapse
- banner:
  - button "Open navigation"
  - text: Multi-Agent Financial Analysis System Demo / frozen data ok
- main:
  - paragraph: Orchestrated workflow
  - heading "Pipeline workspace" [level=1]
  - paragraph: Frame a research question, define its context, and follow the structured hand-off through all four specialist agents.
  - text: Demo / frozen inputs
  - paragraph: Research brief
  - heading "Question & instruments" [level=2]
  - paragraph: Specific questions produce more auditable outputs.
  - text: Research query
  - textbox "Research query Ask for evidence, conditions and risks—not a buy/sell instruction.":
    - /placeholder: Assess the evidence for margin expansion at NVIDIA over the next two reported quarters, including material downside risks.
    - text: Fed outlook
  - text: Ask for evidence, conditions and risks—not a buy/sell instruction. Ticker universe
  - button "Ticker universe Ticker selection" [pressed]: AAPL
  - button "MSFT"
  - button "NVDA"
  - button "AMZN"
  - button "GOOGL"
  - button "SPY"
  - button "QQQ"
  - textbox "Custom ticker":
    - /placeholder: Add ticker
  - button "Add"
  - paragraph: Context
  - heading "Conversation memory" [level=2]
  - paragraph: Keep the run isolated or attach it to reusable research context.
  - group "Conversation mode":
    - button "One-shot" [pressed]
    - button "Continue conversation"
    - button "New conversation"
  - paragraph: Runtime
  - heading "Run controls" [level=2]
  - text: Lookback window
  - combobox "Lookback window":
    - option "30 days"
    - option "90 days" [selected]
    - option "180 days"
    - option "1 year"
    - option "2 years"
  - text: LLM synthesis
  - checkbox "LLM synthesis" [checked]
  - text: Demo / frozen data
  - checkbox "Demo / frozen data" [checked]
  - button "Start full pipeline"
  - button "Reset"
  - paragraph: Job demo-job
  - heading "Live run monitor" [level=2]
  - paragraph: Started 01 Jan, 12:00 · Pipeline
  - text: succeeded Polling fallback
  - strong: Analyst
  - text: Completed
  - strong: Risk
  - text: Completed
  - strong: Strategy
  - text: Completed
  - strong: Execution
  - text: Completed
  - strong: Aggregate
  - text: Completed
  - paragraph: Completed research packet
  - heading "MAFAS analysis output" [level=2]
  - paragraph: Structured model output for research and simulation. Verify every material fact against primary sources.
  - text: Final
  - paragraph: Orchestrator
  - heading "Aggregate decision" [level=2]
  - text: trade
  - strong: trade
  - paragraph: This is a research output, not a trading recommendation.
  - strong: Agent route
  - paragraph: analyst(attempt=1) → risk → strategy → execution
  - paragraph: Analyst agent
  - heading "Evidence synthesis" [level=2]
  - text: Confidence 82% Retrieval
  - meter "Retrieval"
  - strong: 88%
  - text: Diversity
  - meter "Diversity"
  - strong: 76%
  - heading "Key points" [level=3]
  - list:
    - listitem: Disinflation remains visible [1].
  - heading "Risks & caveats" [level=3]
  - list:
    - listitem: A renewed inflation impulse could delay easing.
  - paragraph: Evidence trail
  - heading "Sources & citations" [level=2]
  - paragraph: 1 retrieved evidence item
  - article:
    - strong: fomc_minutes
    - link "Open fomc_minutes":
      - /url: https://example.com/fomc
    - paragraph: Participants noted progress toward price stability.
    - text: https://example.com/fomc 01 Jan, 00:00 Relevance 91%
  - paragraph: Risk agent
  - heading "Market & portfolio risk" [level=2]
  - text: Regime Medium VIX 18.4 Volatility 24% Effective bets 1
  - paragraph: Cross-asset structure
  - heading "Correlation matrix" [level=2]
  - text: Mean correlation 0 AAPL AAPL 1.00
  - paragraph: Portfolio constraints
  - heading "Position sizing" [level=2]
  - article:
    - strong: AAPL
    - text: Max 20%
    - paragraph: Risk per trade 1.0%.
  - paragraph: Strategy agent
  - heading "Playbook selection" [level=2]
  - text: "Macro bias: bullish 68% Trend Following"
  - meter "Trend Following"
  - strong: 81%
  - heading "Candidate setups" [level=3]
  - article:
    - strong: Trend Following
    - text: 74%
    - paragraph:
      - strong: AAPL LONG
      - text: — Constructive macro bias.
  - heading "Suppressed by guardrails" [level=3]
  - paragraph: No strategies were suppressed.
  - paragraph: Execution agent
  - heading "Execution geometry" [level=2]
  - text: Positive simulated expectancy. Protective stop
  - strong: "176.6"
  - text: Entry
  - strong: "185"
  - text: Target
  - strong: "201.8"
  - text: Probability 57% Expected R 0.42 MAE p95 0.94R Position size 20% TP Before SL
  - meter "TP Before SL"
  - strong: 57%
  - text: SL Before TP
  - meter "SL Before TP"
  - strong: 31%
  - text: Timeout
  - meter "Timeout"
  - strong: 12%
- contentinfo: Research and simulation only. Outputs are probabilistic, may be incomplete, and are not financial advice or a trading instruction.
- alert
```

# Test source

```ts
  33  |     universe: ["AAPL"],
  34  |     vol_regime: "medium",
  35  |     vix_level: 18.4,
  36  |     mean_realised_vol: 0.24,
  37  |     correlation_matrix: { AAPL: { AAPL: 1 } },
  38  |     concentration: {
  39  |       effective_number_of_bets: 1,
  40  |       mean_pairwise_correlation: 0,
  41  |     },
  42  |     position_sizing: [
  43  |       { ticker: "AAPL", max_position_pct: 0.2, risk_per_trade_pct: 0.01 },
  44  |     ],
  45  |   },
  46  |   strategy: {
  47  |     macro_bias: { direction: "bullish", strength: 0.68 },
  48  |     candidate_scores: [{ name: "Trend Following", score: 0.81 }],
  49  |     setups: [
  50  |       {
  51  |         strategy_name: "Trend Following",
  52  |         instrument: "AAPL",
  53  |         direction: "long",
  54  |         confidence: 0.74,
  55  |         rationale: "Constructive macro bias.",
  56  |       },
  57  |     ],
  58  |     suppressed: [],
  59  |   },
  60  |   cards: [
  61  |     {
  62  |       instrument: "AAPL",
  63  |       strategy_name: "Trend Following",
  64  |       direction: "long",
  65  |       levels: { entry: 185, stop_loss: 176.6, take_profit: 201.8 },
  66  |       stats: {
  67  |         prob_tp_before_sl: 0.57,
  68  |         prob_sl_before_tp: 0.31,
  69  |         prob_timeout: 0.12,
  70  |         expected_r: 0.42,
  71  |         mae_p95_r: 0.94,
  72  |       },
  73  |       sizing: { notional_pct: 0.2 },
  74  |       verdict: "Positive simulated expectancy.",
  75  |     },
  76  |   ],
  77  | };
  78  | 
  79  | test.beforeEach(async ({ page }) => {
  80  |   await page.route(`${API}/**`, async (route) => {
  81  |     const url = new URL(route.request().url());
  82  |     const path = url.pathname.replace("/api/v1", "");
  83  |     const method = route.request().method();
  84  |     const json = (value: unknown, status = 200) =>
  85  |       route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value) });
  86  | 
  87  |     if (path === "/health") {
  88  |       return json({
  89  |         status: "ok",
  90  |         services: {
  91  |           api: { status: "ok" },
  92  |           database: { status: "ok" },
  93  |           qdrant: { status: "ok", count: 339 },
  94  |           ollama: { status: "ok", count: 1 },
  95  |         },
  96  |       });
  97  |     }
  98  |     if (path === "/config/defaults") {
  99  |       return json({ watchlist: ["AAPL", "MSFT"], playbooks: [] });
  100 |     }
  101 |     if (path === "/jobs" && method === "GET") return json([]);
  102 |     if (path === "/conversations" && method === "GET") return json([]);
  103 |     if (path === "/runs/pipeline" && method === "POST") {
  104 |       return json({ id: "demo-job", status: "queued" }, 202);
  105 |     }
  106 |     if (path === "/jobs/demo-job/events") {
  107 |       return route.fulfill({ status: 503, body: "stream unavailable" });
  108 |     }
  109 |     if (path === "/jobs/demo-job") {
  110 |       return json({
  111 |         id: "demo-job",
  112 |         kind: "pipeline",
  113 |         status: "succeeded",
  114 |         metadata: { demo_mode: true },
  115 |         result,
  116 |         created_at: "2026-01-01T12:00:00Z",
  117 |       });
  118 |     }
  119 |     if (path === "/corpus/reset" && method === "POST") {
  120 |       const body = route.request().postDataJSON();
  121 |       expect(body.confirmation).toBe("RESET FINANCIAL DOCS");
  122 |       return json({ id: "reset-job", status: "queued" }, 202);
  123 |     }
  124 |     return json({ detail: `Unhandled ${method} ${path}` }, 404);
  125 |   });
  126 | });
  127 | 
  128 | test("runs the demo pipeline and renders all agent results", async ({ page }) => {
  129 |   await page.goto("/workspace");
  130 |   await page.getByLabel("Research query").fill("Fed outlook");
  131 |   await page.getByRole("button", { name: "Start full pipeline" }).click();
  132 | 
> 133 |   await expect(page.getByText("Inflation is moderating while policy remains restrictive [1].")).toBeVisible();
      |                                                                                                 ^ Error: expect(locator).toBeVisible() failed
  134 |   await expect(page.getByText("Correlation matrix")).toBeVisible();
  135 |   await expect(page.getAllByText("Trend Following").first()).toBeVisible();
  136 |   await expect(page.getByText("Positive simulated expectancy.")).toBeVisible();
  137 | });
  138 | 
  139 | test("requires the exact destructive reset confirmation", async ({ page }) => {
  140 |   await page.goto("/data");
  141 |   const reset = page.getByRole("button", { name: "Reset financial docs" });
  142 |   await expect(reset).toBeDisabled();
  143 |   await page
  144 |     .getByLabel("Type RESET FINANCIAL DOCS to continue")
  145 |     .fill("RESET FINANCIAL DOCS");
  146 |   await expect(reset).toBeEnabled();
  147 |   await reset.click();
  148 |   await expect(page.getByText("Operation accepted")).toBeVisible();
  149 | });
  150 | 
```