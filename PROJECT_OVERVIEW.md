# MAFAS — Project Overview & Context Handoff

> Purpose of this file: a single, comprehensive reference so a new session (or a
> new engineer) can understand the entire project quickly. It documents the
> architecture, every module, key design decisions, conventions, gotchas, and how
> to run/test everything. Keep it updated as the project evolves.

Last updated: covers Analyst, Risk, Strategy, Execution agents + LangGraph orchestrator.

---

## 1. What this project is

**Multi-Agent Financial AI System (MAFAS)** — a portfolio/interview project that
chains four specialised agents behind a LangGraph orchestrator to go from a macro
question to simulated trade ideas, entirely on free/local infrastructure.

Pipeline:

```
User query
  → Analyst Agent   → MacroBriefing   (RAG over Qdrant + Ollama, sourced + confidence-scored)
  → Risk Agent      → RiskSummary     (live yfinance metrics: vol regime, correlations, sizing)
  → Strategy Agent  → StrategyReport  (deterministic playbook scoring + LLM reasoning → setups)
  → Execution Agent → TradeCard[]     (Monte Carlo simulation of each setup vs history)
  → Orchestrator    → PipelineResult  (LangGraph graph: broaden-retry loop + no-trade gate)
```

Design philosophy across all agents: **hybrid** — deterministic, testable logic
is the source of truth; a local Ollama LLM (Mistral 7B) adds a natural-language
layer and **always degrades gracefully** to a deterministic fallback if the LLM
is unavailable. No paid APIs anywhere.

---

## 2. Environment & hard-won gotchas

- **OS**: Windows, PowerShell. Repo root: `d:\QMUL\Summer project\Multi-Agent-financial-AI-system`. Code lives under `mafas/`.
- **ALWAYS use the venv interpreter**: `.\.venv\Scripts\python` (the venv is Python 3.10, torch 2.12, transformers 5.9, sentence-transformers 5.5). The **global** Python is broken/incomplete (old torch, missing deps) — running bare `python` causes `ModuleNotFoundError` and torch import errors. This has bitten us before.
- **Imports**: `from data...`, `from rag...`, `from agents...`. Run `pip install -e .` from `mafas/` (editable install via `pyproject.toml`, which includes `data*`, `rag*`, `agents*`). Or `cd mafas` + `$env:PYTHONPATH="."`. Never run module files by path; use `python -m package.module`.
- **Run everything from the `mafas/` directory.**
- Qdrant runs via `docker compose up -d` (no account needed). Healthcheck uses `/readyz` (not `/health`, which newer Qdrant removed).
- Ollama: `ollama pull mistral` + `ollama serve` → `http://localhost:11434`. Free, local.
- HF Hub warning about `HF_TOKEN` on embedder load is harmless (anonymous model download for `all-MiniLM-L6-v2`).

---

## 3. Repository layout

```
mafas/
  data/
    loaders/
      fomc.py        FOMCLoader        — Fed minutes PDFs (federalreserve.gov allowlist)
      edgar.py       EDGARLoader       — SEC 10-Q/10-K filings
      news.py        NewsLoader        — BBC/CNBC/MarketWatch RSS
      market.py      MarketDataLoader  — yfinance OHLCV/VIX + FRED (non-RAG)
      twelvedata.py  TwelveDataLoader  — cached daily OHLCV for Execution
    processors/
      cleaner.py     TextCleaner       — unicode/HTML/XML cleaning (defusedxml-guarded)
      metadata.py    MetadataExtractor — DocumentMetadata + date normalisation
  rag/
    chunker.py       SemanticChunker
    embedder.py      TextEmbedder      — sentence-transformers all-MiniLM-L6-v2 (dim=384)
    retriever.py     VectorRetriever   — Qdrant wrapper, deterministic IDs, date_ts filter
    corpus_builder.py                  — build_initial_corpus(); manual ingestion entry point
  agents/
    llm.py           OllamaClient      — chat/chat_json, is_available(); OllamaError
    schemas.py       MacroBriefing, KeyPoint, SourceCitation
    confidence.py    composite_confidence(...) and components
    analyst.py       AnalystAgent, build_analyst_agent()
    risk_metrics.py  pure vol/ATR/correlation/sizing functions
    risk_schemas.py  RiskSummary, AssetVolMetrics, CorrelationWarning, ConcentrationRisk, PositionSizingConstraint
    risk.py          RiskAgent, build_risk_agent(), DEFAULT_WATCHLIST
    strategy_playbooks.py  Playbook dataclass, PLAYBOOKS (8), score_playbook(), rank_playbooks()
    strategy_schemas.py    MacroBias, PlaybookScore, StrategySetup, StrategyReport
    strategy.py      StrategyAgent, build_strategy_agent()
    backtest.py      simulate_barrier_bootstrap(), compute_position_size(), HORIZON_BARS
    execution_schemas.py   TradeCard, TradeLevels, SimulationStats, SizingInfo
    execution.py     ExecutionAgent, build_execution_agent()
    pipeline_schemas.py    PipelineState (TypedDict), PipelineResult (Pydantic)
    orchestrator.py  RiskPipeline (LangGraph StateGraph), build_pipeline()
    __init__.py      exports all public classes/functions
  tests/
    test_prerequisites.py  cleaner, chunker, metadata, embedder, dedup, date-normalisation
    test_analyst.py        confidence + Analyst (mocked LLM/retriever)
    test_risk.py           risk metrics + RiskAgent (synthetic data, mocked LLM)
    test_strategy.py       playbook scoring, bias classify, StrategyAgent (mocked LLM)
    test_execution.py      simulation, sizing, ExecutionAgent (synthetic data, mocked LLM)
    test_orchestrator.py   LangGraph graph paths (fake agents)
    conftest.py            shared fixtures
  scripts/
    smoke_loaders.py       live loader checks
    smoke_pipeline.py      live 4-stage end-to-end check (path-robust; adds repo root to sys.path)
  docker-compose.yml       Qdrant service
  pyproject.toml           editable install config (data*, rag*, agents*), pytest config
  requirements.txt         pinned deps
  .env.example             config template
```

---

## 4. Agent-by-agent detail

### Analyst Agent (`agents/analyst.py`)
- `brief(query, doc_type=None, date_after=None) -> MacroBriefing`. **Read-only** over Qdrant.
- Retrieves top_k=8 (score_threshold 0.30), builds a numbered context with each source wrapped in `<source_data>` tags (prompt-injection hardening), asks Mistral for JSON, assembles a `MacroBriefing`.
- **Composite confidence** (`confidence.py`): weighted mean of retrieval similarity (0.35), source diversity (0.25), recency (0.15), LLM self-report (0.25).
- Empty/low-confidence briefing returned gracefully if no hits or LLM down.

### Risk Agent (`agents/risk.py`, `risk_metrics.py`, `risk_schemas.py`)
- `assess(universe=None, briefing=None) -> RiskSummary`. Universe = `DEFAULT_WATCHLIST` (AAPL, MSFT, NVDA, AMZN, GOOGL, JPM) + caller tickers.
- Deterministic metrics (source of truth): realised vol (annualised), Wilder ATR + ATR%, per-asset regime, VIX-blended overall regime, correlation matrix + warnings (|ρ|≥0.8), concentration (mean pairwise corr, effective number of bets = N/(1+(N-1)ρ̄), flagged if ρ̄≥0.5 or eff_bets < max(1.5, 0.4N)), inverse-vol position sizing scaled by regime & correlation, capped by max_position.
- LLM adds narrative; deterministic fallback narrative otherwise.
- Data: yfinance (`MarketDataLoader.get_ohlcv/get_vix`), fetched live each run.

### Strategy Agent (`agents/strategy.py`, `strategy_playbooks.py`, `strategy_schemas.py`)
- `decide(risk, briefing=None) -> StrategyReport`. Hybrid reasoning engine, NOT a signal generator.
- Step 1: **macro bias** classification (bullish/bearish/neutral + strength) — LLM with keyword fallback.
- Step 2: **deterministic playbook scoring** — `score = 0.45·regime_fit + 0.35·bias_fit + 0.20·corr_fit`; directional playbooks' bias weight scaled by conviction. 8 playbooks: trend_following, mean_reversion, momentum_breakout, volatility_based, ma_crossover, range_support_resistance, carry, pairs_relative_value.
- Step 3: **LLM selects 2–3 setups** from the top-5 candidates, bound to instrument + direction, with rationale. Validated (unknown playbooks dropped, off-universe instruments cleared). Final confidence = mean(LLM confidence, deterministic fit) to prevent LLM over-optimism.
- Deterministic fallback builds setups from top playbooks with rule-based instrument/direction selection.
- `StrategySetup` fields: strategy, strategy_name, instrument, direction, rationale, confidence, playbook_fit, horizon, risk_note.

### Execution Agent (`agents/execution.py`, `backtest.py`, `execution_schemas.py`)
- `simulate(setup, risk=None) -> TradeCard`; `simulate_report(setups, risk)` for a batch.
- Data: `TwelveDataLoader.get_daily` (24h disk cache) → yfinance fallback (`MarketDataLoader.get_ohlcv`) if no `TWELVE_DATA_API_KEY`.
- Levels: ATR(14)-based, SL=1.5×ATR, TP=3×ATR (≈2:1), direction-aware.
- **Monte Carlo bootstrap** (`simulate_barrier_bootstrap`): resamples historical daily returns into N=5000 forward paths (seeded, deterministic); close-to-close; on a bar breaching both barriers the STOP triggers first (conservative); entry+exit slippage (bps). Outputs P(TP before SL), P(SL first), P(timeout), expected R, win rate, avg bars to exit, MAE mean & p95 (in R).
- Sizing (`compute_position_size`): units = (equity × risk_per_trade_pct) / SL-distance, capped by max_position_pct × equity. Reports units, notional, notional %, risk amount/%, capped flag. Expected P/L = expected_R × risk_amount.
- **Skips** market-neutral / pairs setups (instrument None or contains "/") and direction=neutral with a `skip_reason` (single-instrument barrier sim not applicable).
- LLM verdict; deterministic fallback verdict otherwise.

### Orchestrator (`agents/orchestrator.py`, `pipeline_schemas.py`)
- `RiskPipeline(analyst, risk, strategy, execution)` — accepts injected agents (testable). `build_pipeline(with_llm=True)` wires real agents.
- `run(query, tickers=None, use_llm=True) -> PipelineResult`.
- LangGraph `StateGraph(PipelineState)` nodes: analyst, broaden, risk, strategy, execution, no_trade.
- Edges: START→analyst; analyst→(conditional) broaden|risk; broaden→analyst; risk→strategy; strategy→(conditional) execution|no_trade; execution→END; no_trade→END.
- **Thresholds**: ANALYST_CONFIDENCE_THRESHOLD=0.40, MAX_ANALYST_RETRIES=2, SETUP_CONFIDENCE_FLOOR=0.45, HIGH_VOL_CONFIDENCE_FLOOR=0.55.
- **Broaden loop**: low Analyst confidence → LLM query rewrite (deterministic fallback appends generic macro terms) → retry, bounded.
- **No-trade gate** (composite): no simulate-able setup, or none clears the floor, or high-vol regime with best setup < 0.55.
- **Failure isolation**: each node wrapped in try/except; errors recorded in state; degrades to no_trade instead of crashing.
- `PipelineResult`: query, original_query, tickers, decision (trade|no_trade), no_trade_reason, briefing, risk, strategy, cards, analyst_attempts, route_log, errors, `render()`.

---

## 5. Data model summary (Pydantic outputs)

- `MacroBriefing`: query, summary, key_points[], risks[], citations[], confidence, confidence_breakdown, llm_self_confidence, model.
- `RiskSummary`: universe, vol_regime, vix_level, mean_realised_vol, per_asset[], correlation_warnings[], concentration, position_sizing[], narrative, analyst_query/confidence.
- `StrategyReport`: macro_bias, vol_regime, universe, candidate_scores[], setups[], suppressed[], narrative, analyst_query/confidence, llm_used.
- `TradeCard`: instrument, strategy, direction, horizon, simulated, skip_reason, levels, stats, sizing, expectancy_amount, data_source, bars_used, verdict, llm_used.
- `PipelineResult`: see orchestrator above.

All have a `render()` method for human-readable CLI output.

---

## 6. Data freshness (important, commonly misunderstood)

Two independent planes:
1. **RAG corpus (Qdrant)** — static. Written ONLY by `python -m rag.corpus_builder` (manual). The Analyst is read-only and never ingests. No scheduler exists. Re-run the builder to refresh; ingestion is idempotent (deterministic uuid5 content-hash IDs → no duplicates); `--reset` drops & rebuilds. Old chunks are never auto-deleted (no TTL) without `--reset`.
2. **Live market data** — Risk (yfinance, no cache) and Execution (Twelve Data 24h cache → yfinance) fetch fresh every run.

Current corpus composition (approx): 97 SEC filings, 48 FOMC minutes, 20 news. Retrieval is relevance-based, so a Fed-centric query returns mostly FOMC chunks; company/market queries surface SEC/news. This is expected, not a bug.

---

## 7. How to run

```powershell
cd mafas
docker compose up -d
ollama serve                                   # + ollama pull mistral (once)
.\.venv\Scripts\python -m rag.corpus_builder   # build/refresh corpus (add --reset to rebuild)

# individual agents
.\.venv\Scripts\python -m agents.analyst "Fed stance on inflation?"
.\.venv\Scripts\python -m agents.risk TSLA
.\.venv\Scripts\python -m agents.strategy "Fed outlook?" TSLA
.\.venv\Scripts\python -m agents.execution NVDA --strategy trend_following --direction long

# full orchestrated pipeline
.\.venv\Scripts\python -m agents.orchestrator "Fed stance on rates?" TSLA
.\.venv\Scripts\python -m agents.orchestrator "obscure query" --no-llm

# tests + smoke
.\.venv\Scripts\pytest tests/ -v                # 97 tests
.\.venv\Scripts\python scripts/smoke_pipeline.py --tickers TSLA --show-reports
```

Every agent CLI and `build_*` factory supports `--no-llm` / `with_llm=False` for
deterministic, offline operation.

---

## 8. Conventions to follow when extending

- New agent = pure core module + `*_schemas.py` (Pydantic + `render()`) + agent class with `build_*_agent()` factory + CLI `main()` + mocked-LLM unit tests + a stage in `smoke_pipeline.py`.
- Hybrid pattern: deterministic result is ground truth; LLM is additive and must have a deterministic fallback (`llm_used` flag on outputs).
- Use `loguru` for logging, `pydantic` for schemas, `tenacity` for network retries, `httpx` for HTTP.
- Redact secrets in logs (e.g. FRED errors log only exception type — API key can appear in URLs).
- Pin new dependencies in `requirements.txt` and export new public symbols in `agents/__init__.py`.
- Keep the project structure and core concepts stable (explicit standing instruction from the project owner).

---

## 9. Security notes (already implemented)

- FRED exceptions log only the exception type (API key can leak via request URL).
- RAG context wraps sources in `<source_data>` tags; system prompts state the content is data, not instructions (prompt-injection hardening).
- XML parsing (SEC filings) guarded by `defusedxml` + a 10 MB cap (billion-laughs / XXE).
- FOMC loader allowlists `www.federalreserve.gov` (SSRF guard).
- Dependencies pinned to exact versions.

---

## 10. Test counts & status

- 97 unit tests passing (prerequisites, analyst, risk, strategy, execution, orchestrator), all mocked (no network/LLM).
- Live 4-stage `smoke_pipeline.py` and orchestrator verified end-to-end.

---

## 11. Possible next steps (not yet built)

- Corpus refresh mechanism: incremental `refresh_corpus()`, staleness warning, or scheduled ingestion (currently manual only).
- Richer simulation: intrabar high/low modelling (currently close-to-close), strategy-implied drift, pairs/spread backtest so market-neutral setups aren't skipped.
- Persist `PipelineResult` artifacts (JSON) for a demo UI or history.
- Architecture diagram / write-up for submission.
