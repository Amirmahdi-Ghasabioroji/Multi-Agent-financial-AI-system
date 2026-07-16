# Multi-Agent Financial AI System (MAFAS)

A four-agent financial analysis system coordinated by a LangGraph orchestrator.
It runs a RAG pipeline over a Qdrant corpus (FOMC minutes, SEC filings, news),
assesses the live risk environment, reasons over strategy playbooks, and
stress-tests the resulting ideas against historical price data — all using a
**local, free** Ollama LLM (Mistral 7B), so there are no API costs.

```
User query
   ↓
Analyst Agent   →  MacroBriefing    (sourced, confidence-scored)
   ↓
Risk Agent      →  RiskSummary      (vol regime, correlations, position sizing)
   ↓
Strategy Agent  →  StrategyReport   (playbook reasoning → setup suggestions)
   ↓
Execution Agent →  TradeCard[]      (Monte Carlo trade simulation)
   ↓
LangGraph orchestrator → PipelineResult (with broaden-retry loop + no-trade gate)
```

## Quick start

```bash
cd mafas
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # Windows
.\.venv\Scripts\pip install -e .                  # makes `data`, `rag`, `agents` importable
copy .env.example .env
```

Then run the infrastructure and a full pipeline:

```powershell
# 1. Vector DB
docker compose up -d

# 2. Local LLM (free)
ollama pull mistral
ollama serve                       # http://localhost:11434

# 3. Build the corpus (FOMC + SEC mega-caps + news)
.\.venv\Scripts\python -m rag.corpus_builder

# 4. Run the whole orchestrated pipeline
.\.venv\Scripts\python -m agents.orchestrator "What is the Fed's stance on rates and inflation?" TSLA
```

> **Always use the venv interpreter** (`.\.venv\Scripts\python`), not the global
> `python`. The global environment does not have the project's dependencies. In
> Cursor/VS Code, set the interpreter to `mafas\.venv\Scripts\python.exe` and the
> terminal cwd to `mafas`.

### `ModuleNotFoundError: No module named 'data' / 'agents'`

Imports use `from data...`, `from rag...`, `from agents...`. Those packages live
under `mafas/`. Fix by running `pip install -e .` from `mafas/` (recommended), or
`cd mafas` and set `$env:PYTHONPATH = "."`. Do not run module files by path
(e.g. `python agents/analyst.py`); use `python -m agents.analyst`.

## The agents

### 1. Analyst Agent — `agents/analyst.py`
RAG over Qdrant (FOMC minutes, SEC filings, news) + Ollama to produce a sourced
`MacroBriefing`. Every claim cites retrieved sources. Reports a **composite
confidence** score (retrieval similarity, source diversity, recency, LLM
self-report). Source dates are normalised to `YYYY-MM-DD` and stored as a
sortable `date_ts` so `--date-after` filters reliably.

```powershell
.\.venv\Scripts\python -m agents.analyst "What is the Fed's stance on inflation?"
.\.venv\Scripts\python -m agents.analyst "Apple revenue outlook" --doc-type sec_filing
.\.venv\Scripts\python -m agents.analyst "recent policy signals" --date-after 2024-01-01
```

### 2. Risk Agent — `agents/risk.py`
Consumes the briefing and pulls live market data (yfinance) to compute a
deterministic `RiskSummary`: per-asset ATR & realised vol, overall vol regime
(VIX-blended), cross-asset correlations, concentration (effective number of
bets), and inverse-vol **position-sizing constraints**. An Ollama narrative
interprets the numbers; deterministic maths is the source of truth.

```powershell
.\.venv\Scripts\python -m agents.risk TSLA --lookback 252
```

### 3. Strategy Agent — `agents/strategy.py`
Takes Analyst + Risk outputs and reasons over **8 strategy playbooks** (trend
following, mean reversion, momentum/breakout, volatility-based, MA crossover,
range/S-R, carry, pairs/relative-value). Deterministic suitability scoring
ranks playbooks against the regime + macro bias + correlations; the LLM then
selects 2–3 instrument-bound setups with rationale. A structured reasoning
engine, not a signal generator.

```powershell
.\.venv\Scripts\python -m agents.strategy "How is the Fed framing inflation?" TSLA
```

### 4. Execution Agent — `agents/execution.py`
Simulates each setup against historical data (Twelve Data, yfinance fallback).
ATR-based stop/target, Monte Carlo bootstrap of historical returns to estimate
**P(TP before SL)**, expected R, win rate, and max adverse excursion; sizes the
position from the Risk Agent's constraints on a configurable notional account.
Outputs a `TradeCard`. Does **not** place real trades.

```powershell
.\.venv\Scripts\python -m agents.execution NVDA --strategy trend_following --direction long
```

### Orchestrator — `agents/orchestrator.py`
A LangGraph `StateGraph` wiring the four agents with defensive edges:
- **Low-confidence loop**: if the Analyst confidence < 0.40, broaden the query
  (LLM rewrite, deterministic fallback) and retry (bounded to 2 retries).
- **No-trade gate**: emit a graceful "NO TRADE" (skipping Execution) when no
  simulate-able setup clears the confidence floor, or a high-vol regime has no
  high-conviction setup.
- **Failure isolation**: any agent exception degrades to NO-TRADE, never crashes.

```powershell
.\.venv\Scripts\python -m agents.orchestrator "Fed stance on rates?" TSLA
.\.venv\Scripts\python -m agents.orchestrator "obscure query" --no-llm   # exercises fallbacks
```

## Data freshness

Two independent data planes:

| Data | Source | Refresh | Freshness |
|---|---|---|---|
| FOMC / SEC / News (RAG corpus) | Qdrant | Manual `python -m rag.corpus_builder` | Frozen at last ingest |
| Vol / VIX / correlations | yfinance | Every Risk run | Live |
| Historical prices (simulation) | Twelve Data → yfinance | Every Execution run | Live (Twelve Data cached 24h) |

The Analyst is **read-only** — it never ingests. Re-run the corpus builder to
refresh the knowledge base. Ingestion is idempotent (deterministic content-hash
IDs, so no duplicates); add `--reset` to drop and rebuild the collection.

## Testing

```powershell
.\.venv\Scripts\pytest tests/ -v                       # 97 unit tests, mocked LLM/data
.\.venv\Scripts\python scripts/smoke_loaders.py        # live loader checks
.\.venv\Scripts\python scripts/smoke_pipeline.py --tickers TSLA --show-reports   # live 4-stage
```

## Layout

- `data/loaders/` — FOMC, EDGAR, news RSS, market (yfinance/FRED/VIX), Twelve Data
- `data/processors/` — text cleaning and metadata extraction
- `rag/` — chunking, embeddings, Qdrant retriever, corpus builder
- `agents/` — the four agents, LangGraph orchestrator, LLM client, playbooks,
  backtest engine, confidence scoring, and all Pydantic schemas
- `tests/` — unit tests per component
- `scripts/` — `smoke_loaders.py`, `smoke_pipeline.py`

## Configuration (`.env`)

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=financial_docs
FRED_API_KEY=...                    # optional (macro series)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral
CACHE_DIR=./data/cache
TWELVE_DATA_API_KEY=...             # optional (Execution; falls back to yfinance)
EXECUTION_ACCOUNT_EQUITY=100000
```

## Status

All four agents + the LangGraph orchestration layer are implemented, tested
(97 passing unit tests), and verified end-to-end live.

See `PROJECT_OVERVIEW.md` for a full architecture and implementation reference.
