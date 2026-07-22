# MAFAS — Project Overview & Context Handoff

> Purpose of this file: a single, comprehensive reference so a new session (or a
> new engineer) can understand the entire project quickly. It documents the
> architecture, every module, key design decisions, conventions, gotchas, and how
> to run/test everything. **Keep it updated as the project evolves.**

Last updated: **July 2026** — covers all four agents, the **historical backtest +
forward Monte Carlo simulation stack**, LangGraph orchestration, cross-strategy
ranking, and the full Next.js/FastAPI/MySQL dashboard.

---

## 1. What this project is

**Multi-Agent Financial AI System (MAFAS)** — a portfolio/interview project that
chains four specialised agents behind a LangGraph orchestrator to go from a macro
question to **researched and stress-tested** trade ideas, entirely on
free/local infrastructure.

### End-to-end pipeline

```
User query
  → Analyst Agent   → MacroBriefing      (RAG over Qdrant + Ollama, sourced + confidence-scored)
  → Risk Agent      → RiskSummary        (live yfinance metrics: vol regime, correlations, sizing)
  → Strategy Agent  → StrategyReport     (deterministic playbook scoring + LLM → 2–3 setups)
  → Execution Agent → TradeCard[]          (historical backtest + forward MC + sizing + verdict)
                      ExecutionComparison  (ranked summary across the top setups)
  → Orchestrator    → PipelineResult       (LangGraph: broaden-retry loop + no-trade gate)
```

### Execution Agent — two simulation layers (read this carefully)

The Execution Agent is **not** a broker. It stress-tests each `StrategySetup`
from the Strategy Agent using **two complementary engines**:

| Layer | Module | Question it answers |
|-------|--------|-------------------|
| **Historical backtest** | `agents/simulation/historical.py` | “How would this playbook’s signals have performed on past bars?” |
| **Forward Monte Carlo** | `agents/simulation/barrier.py` | “From today’s entry, what is P(TP before SL) if history repeats randomly?” |

Per setup, `ExecutionAgent.simulate()`:

1. Loads OHLCV (Twelve Data → yfinance fallback; pairs load two tickers).
2. Runs a **playbook-driven historical backtest** → `TradeCard.backtest`.
3. Computes **current ATR bracket levels** from the latest bar.
4. Runs **bootstrap forward barrier MC** → `TradeCard.stats`.
5. Sizes the position from Risk Agent constraints → `TradeCard.sizing`.
6. Adds an LLM (or fallback) **verdict**.

`simulate_report()` returns `(cards, execution_comparison)` where
`execution_comparison` ranks all simulated cards by a transparent composite score
(Sharpe, P/L, drawdown, forward expected R).

### Design philosophy

**Hybrid** across all agents: deterministic, testable logic is the source of
truth; a local Ollama LLM (Mistral 7B) adds natural-language layers and **always
degrades gracefully** when unavailable. No paid LLM APIs. Optional Twelve Data key
for execution history only.

---

## 2. Environment & hard-won gotchas

### Python interpreter — use the venv

- **OS**: Windows, PowerShell. Repo root:
  `d:\QMUL\Summer project\Multi-Agent-financial-AI-system`. Agent code lives
  under `mafas/`.
- **ALWAYS use the project venv**: `mafas\.venv\Scripts\python` (or activate
  first). The **global/system Python** on this machine is incomplete and has
  caused `ModuleNotFoundError`, torch errors, and **NumPy/SciPy binary
  mismatches** (`AttributeError: _ARRAY_API not found`).
- After creating the venv:
  ```powershell
  cd mafas
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .
  pip install -r requirements.txt
  ```
- `requirements.txt` pins `numpy==2.2.6` and requires **`scipy>=1.14.0`** and
  **`scikit-learn>=1.5.0`** (NumPy 2–compatible wheels). If analyst tests fail
  at import with SciPy errors, upgrade: `pip install "scipy>=1.14.0"`.

### Imports & how to run modules

- Package imports: `from agents...`, `from data...`, `from rag...`.
- Editable install: `pip install -e .` from `mafas/` (`pyproject.toml` ships
  `data*`, `rag*`, `agents*`).
- Alternative: `cd mafas` + `$env:PYTHONPATH="."`.
- **Never** run agent files by path; use `python -m agents.<module>`.
- Run **agent CLIs/tests** from `mafas/`; run the **dashboard Compose stack**
  from the repository root.

### Lazy package imports (important for tests)

`agents/__init__.py` uses **lazy `__getattr__` exports** so lightweight imports
(e.g. `agents.execution_schemas`, `agents.simulation.*`) do **not** eagerly load
the Analyst → sentence-transformers → scipy chain. Heavy modules (`analyst`,
`orchestrator` graph build, `build_*_agent` factories) load only when accessed.

Similarly, `execution.py`, `orchestrator.py`, `risk.py`, and `strategy.py`
defer heavy third-party imports (`fredapi`, `langgraph`, data loaders) to
factory functions or `run()` time where possible.

### Infrastructure

- **Qdrant**: no account needed. `mafas/docker-compose.yml` runs Qdrant alone;
  root `docker-compose.yml` runs the full dashboard. **Do not run both** — same
  ports. Healthcheck: `/readyz` (not `/health`).
- **Ollama**: `ollama pull mistral` + `ollama serve` → `http://localhost:11434`.
- **HF Hub** `HF_TOKEN` warning on embedder load is harmless (anonymous download
  for `all-MiniLM-L6-v2`).

---

## 3. Repository layout

```
backend/
  app/                  FastAPI API, jobs, persistence, SSE, reports, health
  tests/                API/job/persistence tests
  Dockerfile
frontend/
  src/app/              Next.js App Router pages
  src/components/       shell, forms, timelines, result-view (incl. backtest UI)
  src/lib/              typed API client, schemas/helpers
  Dockerfile
docker-compose.yml      Full dashboard: frontend + backend + MySQL + Qdrant
.env.example            Full-stack local configuration

mafas/
  data/
    loaders/
      fomc.py           FOMCLoader        — Fed minutes PDFs (allowlisted host)
      edgar.py          EDGARLoader       — SEC 10-Q/10-K filings
      news.py           NewsLoader        — BBC/CNBC/MarketWatch RSS
      market.py         MarketDataLoader  — yfinance OHLCV/VIX + FRED
      twelvedata.py     TwelveDataLoader  — cached daily OHLCV for Execution
    processors/
      cleaner.py        TextCleaner
      metadata.py       MetadataExtractor
  rag/
    chunker.py          SemanticChunker
    embedder.py         TextEmbedder (all-MiniLM-L6-v2, dim=384)
    retriever.py        VectorRetriever (Qdrant, uuid5 IDs, date_ts filter)
    corpus_builder.py   manual ingestion entry point
  agents/
    llm.py              OllamaClient — chat/chat_json, is_available()
    schemas.py          MacroBriefing, KeyPoint, SourceCitation
    confidence.py       composite_confidence(...)
    analyst.py          AnalystAgent, build_analyst_agent()
    risk_metrics.py     pure vol/ATR/correlation/sizing maths
    risk_schemas.py     RiskSummary, AssetVolMetrics, ...
    risk.py             RiskAgent, build_risk_agent(), DEFAULT_WATCHLIST
    strategy_playbooks.py   8 playbooks, score_playbook(), rank_playbooks()
    strategy_schemas.py     MacroBias, PlaybookScore, StrategySetup, StrategyReport
    strategy.py         StrategyAgent, build_strategy_agent()
    simulation/         ★ backtest + forward MC engine (see §4)
      barrier.py        simulate_barrier_bootstrap(), walk_forward_barrier()
      historical.py     run_historical_backtest() — playbook signal replay
      metrics.py        Sharpe, Sortino, Calmar, drawdown, profit factor, MC bands
      levels.py         ATR bracket geometry at any bar
      sizing.py         compute_position_size()
      comparison.py     rank_trade_cards() → ExecutionComparison
      signals/
        base.py         SMA, RSI, ATR, spread z-score helpers
        playbooks.py    8 playbook entry signal functions
        registry.py     playbook key → signal, warmup_bars()
    backtest.py         thin re-export shim (backward compat for old imports)
    execution_schemas.py    TradeCard, BacktestResult, ExecutionComparison, ...
    execution.py        ExecutionAgent, build_execution_agent()
    pipeline_schemas.py PipelineState, PipelineResult (+ execution_comparison)
    orchestrator.py   RiskPipeline (LangGraph), build_pipeline()
    __init__.py         lazy public exports (do not add eager heavy imports)
  tests/
    test_prerequisites.py
    test_analyst.py
    test_risk.py
    test_strategy.py
    test_execution.py
    test_orchestrator.py
    test_signals.py           ★ playbook signal unit tests
    test_backtest_metrics.py  ★ performance metrics unit tests
    test_historical_backtest.py ★ historical engine unit tests
    test_corpus_dashboard.py
    conftest.py
  scripts/
    smoke_loaders.py
    smoke_pipeline.py       live 4-stage end-to-end check
  docker-compose.yml        Qdrant only
  pyproject.toml
  requirements.txt
  .env.example
```

---

## 4. Agent-by-agent detail

### Analyst Agent (`agents/analyst.py`)

- `brief(query, ...) -> MacroBriefing`. **Read-only** over Qdrant.
- Retrieves top_k=8 (score_threshold 0.30), builds numbered context with
  `<source_data>` tags (prompt-injection hardening), asks Mistral for JSON.
- **Composite confidence** (`confidence.py`): retrieval (0.35), diversity
  (0.25), recency (0.15), LLM self-report (0.25).
- Graceful empty briefing if no hits or LLM down.

### Risk Agent (`agents/risk.py`, `risk_metrics.py`, `risk_schemas.py`)

- `assess(universe=None, briefing=None) -> RiskSummary`.
- Universe = `DEFAULT_WATCHLIST` (AAPL, MSFT, NVDA, AMZN, GOOGL, JPM) + caller
  tickers.
- Deterministic metrics: realised vol, Wilder ATR, per-asset regime, VIX-blended
  market regime, correlation matrix + warnings (|ρ|≥0.8), concentration
  diagnostics, inverse-vol position sizing capped by `max_position_pct`.
- LLM narrative with deterministic fallback.
- Data: yfinance via `MarketDataLoader` (live each run).

### Strategy Agent (`agents/strategy.py`, `strategy_playbooks.py`, `strategy_schemas.py`)

- `decide(risk, briefing=None) -> StrategyReport`. Reasoning engine, **not** a
  signal generator — produces **suggestions** the Execution Agent stress-tests.
- Step 1: macro bias (LLM + keyword fallback).
- Step 2: deterministic playbook scoring —
  `score = 0.45·regime + 0.35·bias + 0.20·corr` across 8 playbooks.
- Step 3: LLM picks **2–3 setups** (`top_n_setups=3`) from top-5 candidates,
  each bound to instrument + direction + horizon.
- Final confidence = mean(LLM confidence, deterministic fit).
- `StrategySetup` fields: strategy (playbook key), strategy_name, instrument,
  direction, rationale, confidence, playbook_fit, horizon, risk_note.

### Execution Agent (`agents/execution.py`, `agents/simulation/`, `execution_schemas.py`)

**Public API**

- `simulate(setup, risk=None) -> TradeCard`
- `simulate_report(setups, risk=None) -> tuple[list[TradeCard], ExecutionComparison | None]`

**Data loading**

- Single ticker: `TwelveDataLoader.get_daily` (24h disk cache) → yfinance fallback.
- Pairs (`pairs_relative_value` with instrument `TICKER_A/TICKER_B`): loads both
  legs, aligns on dates, backtests the log-spread z-score.

**Historical backtest** (`run_historical_backtest`)

1. For each bar after playbook-specific warmup, compute entry signal from
   `simulation/signals/playbooks.py` via `registry.py`.
2. Only enter when signal matches `setup.direction` (long/short).
3. **Non-overlapping trades**: after entry, walk forward on real bars until
   TP / SL / horizon timeout (`walk_forward_barrier`). Stop wins if both
   barriers touched in one bar (conservative).
4. ATR brackets at entry: SL = 1.5×ATR, TP = 3.0×ATR (≈2:1 R:R).
5. Build equity curve, compute metrics, optional MC robustness on trade-R
   resampling.

**Playbook signal rules** (warmup typically 60 bars; carry uses 252):

| Playbook key | Long entry (summary) | Short entry (summary) |
|--------------|----------------------|------------------------|
| `trend_following` | close > SMA50, SMA20 > SMA50 | opposite |
| `ma_crossover` | SMA20 crosses above SMA50 | SMA20 crosses below SMA50 |
| `momentum_breakout` | close > 20-day high | close < 20-day low |
| `mean_reversion` | RSI(14) < 30 | RSI(14) > 70 |
| `range_support_resistance` | bounce off 20-day low | rejection at 20-day high |
| `volatility_based` | ATR > 75th pct of 60-day ATR window & close > SMA20 | vol filter & close < SMA20 |
| `carry` | low realised vol + close > SMA200 | opposite |
| `pairs_relative_value` | spread z-score < −2 | spread z-score > +2 |

Unknown playbook keys fall back to `trend_following` with a logged warning.

**Forward Monte Carlo** (`simulate_barrier_bootstrap`)

- Resamples historical daily returns into N=5000 forward paths from **latest**
  close (seeded, deterministic).
- Close-to-close; stop-first on dual breach; entry+exit slippage (5 bps default).
- Outputs: P(TP before SL), P(SL first), P(timeout), expected R, win rate, MAE.

**Sizing** (`compute_position_size`)

- Units = (equity × risk_per_trade_pct) / |entry − stop|, capped by
  `max_position_pct × equity`.
- Expected P/L = forward `expected_r × risk_amount`.

**Skips simulation when**

- `direction == "neutral"` or missing instrument.
- History cannot be loaded or ATR/price invalid.
- Pairs: second leg fails to load.

**Verdict**: LLM reads deterministic stats + backtest summary; fallback rule
based on forward edge and backtest P/L.

### Orchestrator (`agents/orchestrator.py`, `pipeline_schemas.py`)

- `RiskPipeline(analyst, risk, strategy, execution)` — injected agents for tests.
- `build_pipeline(with_llm=True)` wires real agents (lazy imports inside).
- LangGraph `StateGraph` compiled on first `run()` (not at `__init__`).
- Nodes: analyst → broaden (loop) → risk → strategy → execution | no_trade.
- **Thresholds**: analyst confidence 0.40, max retries 2, setup floor 0.45,
  high-vol floor 0.55.
- **Tradeable setups**: directional, confidence ≥ floor; single tickers OR
  `pairs_relative_value` with `A/B` instrument format.
- **Failure isolation**: per-node try/except → graceful no_trade.
- `PipelineResult` now includes `execution_comparison` when execution runs.

---

## 5. Data model summary (Pydantic outputs)

| Model | Key fields |
|-------|------------|
| `MacroBriefing` | query, summary, key_points[], risks[], citations[], confidence, breakdown |
| `RiskSummary` | universe, vol_regime, vix_level, per_asset[], correlation_warnings[], position_sizing[], narrative |
| `StrategyReport` | macro_bias, candidate_scores[], setups[] (2–3), suppressed[], narrative |
| `TradeCard` | instrument, strategy, direction, horizon, simulated, skip_reason, **levels**, **stats** (forward MC), **backtest**, sizing, expectancy_amount, verdict |
| `BacktestResult` | period_start/end, metrics, trades[], equity_curve[], drawdown_curve[], mc_robustness |
| `BacktestMetrics` | n_trades, win_rate, profit_factor, total_pnl, max_drawdown_pct/amount, sharpe, sortino, calmar, expectancy_r, low_sample |
| `ExecutionComparison` | ranked[], best_sharpe, best_pnl, lowest_drawdown |
| `PipelineResult` | all stage outputs + cards + **execution_comparison** + decision, route_log, errors |

All major models expose `render()` for CLI-friendly text output.

---

## 6. Data freshness

Two independent planes:

1. **RAG corpus (Qdrant)** — static. Written only by `python -m rag.corpus_builder`
   (manual). Analyst is read-only. Idempotent uuid5 IDs; `--reset` rebuilds.
2. **Live market data** — Risk fetches yfinance every run. Execution uses Twelve
   Data (24h cache) or yfinance fallback every run.

Corpus mix (approx): 97 SEC, 48 FOMC, 20 news. Fed-centric queries return mostly
FOMC chunks — expected retrieval behaviour.

---

## 7. How to run

```powershell
cd mafas
docker compose up -d          # Qdrant only (or use root compose for dashboard)
ollama serve                  # + ollama pull mistral (once)
.\.venv\Scripts\python -m rag.corpus_builder

# Individual agents
.\.venv\Scripts\python -m agents.analyst "Fed stance on inflation?"
.\.venv\Scripts\python -m agents.risk TSLA
.\.venv\Scripts\python -m agents.strategy "Fed outlook?" TSLA
.\.venv\Scripts\python -m agents.execution NVDA --strategy trend_following --direction long

# Full pipeline
.\.venv\Scripts\python -m agents.orchestrator "Fed stance on rates?" TSLA
.\.venv\Scripts\python -m agents.orchestrator "obscure query" --no-llm

# Tests
.\.venv\Scripts\python -m pytest tests/ -v

# Backtest / execution tests only (no langgraph corpus analyst needed for subset)
.\.venv\Scripts\python -m pytest tests/test_signals.py tests/test_backtest_metrics.py tests/test_historical_backtest.py tests/test_execution.py -v

# Live smoke
.\.venv\Scripts\python scripts/smoke_pipeline.py --tickers TSLA --show-reports
```

Every agent supports `--no-llm` / `with_llm=False` for deterministic operation.

### Dashboard (repository root)

```powershell
copy .env.example .env
docker compose up --build
```

Demo mode serves deterministic fixtures including backtest metrics and
`execution_comparison` without Ollama or live market data.

---

## 8. Conventions when extending

### New agent stage

Pure core module + `*_schemas.py` (Pydantic + `render()`) + agent class with
`build_*_agent()` + CLI `main()` + mocked unit tests + `smoke_pipeline.py` stage.

### Simulation / backtest changes

- Put **pure, testable logic** in `agents/simulation/` — no I/O in metrics,
  signals, or barrier code.
- Playbook entry rules live in `simulation/signals/playbooks.py`; register in
  `registry.py`.
- Extend `BacktestResult` / `TradeCard` schemas before touching the dashboard.
- Keep `backtest.py` as a re-export shim if renaming public functions.
- Do **not** add eager heavy imports to `agents/__init__.py`.

### Hybrid LLM pattern

Deterministic output is ground truth. LLM narrates only. Always set `llm_used`
and provide a deterministic fallback.

### Dependencies

Pin in `requirements.txt`. NumPy 2.x requires SciPy ≥ 1.14 and scikit-learn ≥
1.5. Use `loguru`, `pydantic`, `tenacity`, `httpx` per existing style.

---

## 9. Security notes

- FRED errors log exception type only (API key can appear in URLs).
- RAG context uses `<source_data>` tags; prompts state content is data, not
  instructions.
- SEC XML: `defusedxml` + 10 MB cap.
- FOMC loader allowlists `www.federalreserve.gov`.
- No authentication on localhost dashboard. Corpus reset requires typing
  `RESET FINANCIAL DOCS`. **No brokerage / order endpoints.**

---

## 10. Tests & troubleshooting

### Test files

| File | What it covers |
|------|----------------|
| `test_prerequisites.py` | cleaner, chunker, metadata, embedder |
| `test_analyst.py` | confidence, Analyst (mocked LLM/retriever) |
| `test_risk.py` | risk metrics + RiskAgent |
| `test_strategy.py` | playbook scoring, StrategyAgent |
| `test_signals.py` | 8 playbook signal generators |
| `test_backtest_metrics.py` | Sharpe, drawdown, profit factor, MC bands |
| `test_historical_backtest.py` | historical engine, barrier walk |
| `test_execution.py` | sizing, forward MC, ExecutionAgent, backtest on card |
| `test_orchestrator.py` | LangGraph paths with fake agents |
| `test_corpus_dashboard.py` | dashboard/corpus contracts |

All unit tests use synthetic data and mocks — **no network**.

### Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `AttributeError: _ARRAY_API not found` during collection | Old SciPy + NumPy 2.x in wrong Python | Use venv; `pip install "scipy>=1.14.0"` |
| `ModuleNotFoundError: langgraph` | Missing dep or wrong Python | `pip install -r requirements.txt` in venv |
| `ModuleNotFoundError: fredapi` | Same | `pip install -r requirements.txt` |
| Tests import analyst when testing execution | Fixed by lazy `agents/__init__.py` | Pull latest; import `agents.simulation` paths directly in new tests |
| All orchestrator tests fail, backtest tests pass | langgraph not installed | Install requirements or run backtest subset only |

---

## 11. Dashboard architecture

Thin UI over the Python domain layer — no duplicated agent maths.

```
Browser (Next.js)
  ↕ REST + SSE
FastAPI API
  ├─ bounded background job runner
  ├─ MySQL conversations, events, structured results
  ├─ LangGraph RiskPipeline / per-agent runners
  └─ health, corpus controls, report exports
       ↕
Qdrant + host Ollama + market/document APIs
```

### Result visualisation (`frontend/src/components/result-view.tsx`)

- **Analyst / Risk / Strategy** panels — existing structured views.
- **Execution comparison** — ranked table across top setups (Sharpe, P/L, DD,
  forward E[R]); highlights for best Sharpe, best P/L, lowest drawdown.
- **Per TradeCard**:
  - Forward MC: probability bar, expected R, MAE, sizing.
  - **Backtest panel**: total P/L, max DD, Sharpe/Sortino/Calmar, profit factor,
    win rate, equity + drawdown sparklines (SVG), recent trades table.
- Demo fixtures in `backend/app/runners.py` include sample `backtest` and
  `execution_comparison` blocks.

### Other dashboard behaviour

- Full workspace: one-shot + bounded contextual follow-ups (history is
  untrusted context, not RAG evidence).
- Individual agent workspaces with guided + advanced JSON inputs.
- SSE job events from orchestrator progress callbacks.
- MySQL stores conversations and JSON results — not Qdrant docs or PDFs.
- Ollama via `host.docker.internal` when backend runs in Docker.

---

## 12. Mental model for new agents

If you are picking up this codebase cold, follow this reading order:

1. **This file** — orientation.
2. `agents/orchestrator.py` — how the four agents connect.
3. `agents/strategy_schemas.py` + `strategy_playbooks.py` — what a “setup” is.
4. `agents/simulation/historical.py` + `signals/playbooks.py` — how setups are
   backtested.
5. `agents/execution.py` + `execution_schemas.py` — how results become
   `TradeCard` + `ExecutionComparison`.
6. `frontend/src/components/result-view.tsx` — how JSON reaches the UI.
7. `backend/app/runners.py` — how the API invokes agents and demo fixtures.

**Do not confuse:**

- **Strategy Agent** → picks *which* playbook and direction to try (reasoning).
- **Execution Agent** → measures *how that idea would have traded* (simulation).
- **Forward MC** (`stats`) → uncertainty from *today’s* entry forward.
- **Historical backtest** (`backtest`) → evidence from *past signal-fired trades*.

---

## 13. Possible next steps

- Reinforcement-learning forward scenario module (offline-trained, inference-only).
- Intrabar high/low modelling in forward MC (currently close-to-close bootstrap).
- Walk-forward train/test splits for backtest validation reporting.
- Scheduled corpus refresh (dashboard has manual refresh/reset today).
- Optional auth if exposed beyond localhost.
- Production job queue if moving off single-owner local Docker.

---

## 14. Quick reference — key defaults

| Parameter | Default | Location |
|-----------|---------|----------|
| Account equity | $100,000 | `EXECUTION_ACCOUNT_EQUITY` env / `ExecutionAgent` |
| SL / TP ATR mult | 1.5 / 3.0 | `ExecutionAgent` |
| Forward MC sims | 5,000 | `ExecutionAgent.n_sims` |
| Lookback bars | 504 | `ExecutionAgent.lookback_bars` |
| Horizon bars | intraday 5, swing 20, position 60 | `HORIZON_BARS` |
| Min trades for robust metrics | 5 | `BacktestConfig.min_trades_for_metrics` |
| Strategy setups per run | up to 3 | `StrategyAgent.top_n_setups` |
| Analyst confidence threshold | 0.40 | `orchestrator.py` |
| Setup confidence floor | 0.45 | `orchestrator.py` |
