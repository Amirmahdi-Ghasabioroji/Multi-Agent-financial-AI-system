# Multi-Agent Financial AI System (MAFAS)

MAFAS chains four Python agents behind a LangGraph orchestrator. You ask a macro
or markets question; the system retrieves evidence from a local document corpus,
scores the risk environment, picks a handful of strategy setups, and simulates
them against historical price data. Inference runs on a local Ollama model
(Mistral 7B by default). There are no paid LLM API calls.

A Next.js dashboard wraps the same agent code with conversation history, live
job progress, report export, and **on-demand evaluation reports** for RAG
retrieval quality, Monte Carlo calibration, and risk metric completeness. The
project is research and simulation only — it does not connect to a broker or
place orders.

For architecture detail, module maps, and design decisions, see
[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

---

## What runs where

```
User query
  → Analyst        MacroBriefing     (RAG over Qdrant + Ollama)
  → Risk           RiskSummary       (yfinance vol, correlations, sizing)
  → Strategy       StrategyReport    (8 playbooks → up to 3 setups)
  → Execution      TradeCard[]       (historical backtest + forward Monte Carlo)
                     + ranked comparison across setups
  → Orchestrator   PipelineResult    (broaden/retry loop, no-trade gate)
```

**Execution** does two separate jobs on each setup:

1. **Historical backtest** — replay playbook entry signals on past OHLCV bars;
   report P/L, drawdown, Sharpe, win rate, equity curve, and related stats.
2. **Forward simulation** — bootstrap daily returns from the latest price to
   estimate P(TP before SL), expected R, and max adverse excursion.

---

## Prerequisites

| Requirement | Used for |
|-------------|----------|
| Python 3.10+ | All agent code under `mafas/` |
| Docker Desktop | Qdrant (required); MySQL + dashboard (optional) |
| Ollama | Live LLM runs (`ollama pull mistral`) |
| Git | Clone this repo |

Optional API keys (free tiers exist):

- **FRED** — extra macro series in the market loader.
- **Twelve Data** — daily OHLCV for execution; without it, yfinance is used.

**Windows note:** the commands below use PowerShell. On macOS or Linux, swap
`.\.venv\Scripts\python` for `.venv/bin/python` and adjust path separators.

---

## Repository layout

```
Multi-Agent-financial-AI-system/
├── mafas/                  Python agents, RAG, data loaders, tests
│   ├── agents/             Four agents + orchestrator + simulation/
│   ├── eval/               On-demand RAG / MC / risk evaluation suites
│   ├── rag/                Chunking, embeddings, Qdrant retriever
│   ├── data/               Document and market loaders
│   ├── tests/
│   ├── scripts/            smoke_pipeline.py, smoke_loaders.py
│   ├── requirements.txt
│   └── docker-compose.yml  Qdrant only (CLI development)
├── backend/                FastAPI job runner and persistence
├── frontend/               Next.js dashboard
├── docker-compose.yml      Full stack: frontend + backend + MySQL + Qdrant
├── .env.example            Dashboard / Docker configuration
└── PROJECT_OVERVIEW.md     Deep reference for contributors
```

All agent work happens inside `mafas/`. The dashboard is started from the repo
root.

---

## Setup: agents and CLI (recommended first)

### 1. Clone and open a terminal in `mafas`

```powershell
cd path\to\Multi-Agent-financial-AI-system\mafas
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use this venv for every Python command in the project. The system-wide Python
on many machines is missing packages or has incompatible NumPy/SciPy wheels,
which shows up as import errors during `pytest` collection.

In VS Code or Cursor, set the interpreter to
`mafas\.venv\Scripts\python.exe` and open the terminal with `mafas` as the
working directory.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` registers the `data`, `rag`, and `agents` packages so imports
like `from agents.execution import ...` resolve correctly.

`requirements.txt` pins NumPy 2.x and expects SciPy ≥ 1.14 and scikit-learn ≥
1.5 (pulled in via sentence-transformers). If you see
`AttributeError: _ARRAY_API not found` when running tests, you are almost
certainly on the wrong interpreter or an old SciPy build:

```powershell
pip install "scipy>=1.14.0" "scikit-learn>=1.5.0"
```

### 4. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env` if you have FRED or Twelve Data keys. Everything else has sensible
defaults for local work.

### 5. Start Qdrant

```powershell
docker compose up -d
```

This uses `mafas/docker-compose.yml` and creates the shared volume
`mafas_qdrant_storage`. Qdrant listens on `http://localhost:6333`.

Do **not** run the root `docker-compose.yml` at the same time — both bind port
6333.

### 6. Start Ollama and pull the model

```powershell
ollama pull mistral
ollama serve
```

Ollama Desktop on Windows usually serves automatically at
`http://localhost:11434`. The agents call this URL unless you override
`OLLAMA_URL` in `.env`.

### 7. Build the document corpus

Ingestion is manual. The Analyst agent only reads from Qdrant; it never writes
documents itself.

```powershell
.\.venv\Scripts\python -m rag.corpus_builder
```

First run downloads the embedding model (`all-MiniLM-L6-v2`) and fetches FOMC
minutes, SEC filings for the default watchlist, and news RSS items. Ingestion
is idempotent — re-running adds nothing duplicate. To wipe and rebuild:

```powershell
.\.venv\Scripts\python -m rag.corpus_builder --reset
```

Expect roughly 97 SEC chunks, 48 FOMC, and 20 news items on a full build.

### 8. Run the full pipeline

```powershell
.\.venv\Scripts\python -m agents.orchestrator "What is the Fed's stance on rates and inflation?" TSLA
```

Add `--no-llm` to exercise deterministic fallbacks without Ollama.

Pass extra tickers after the query. The Risk agent merges them with the default
watchlist (AAPL, MSFT, NVDA, AMZN, GOOGL, JPM).

---

## Setup: full dashboard

The dashboard adds MySQL conversation storage, SSE job progress, per-agent
workspaces, backtest charts, evaluation reports, and demo mode (no Ollama or
corpus required).

### 1. Stop the CLI-only Qdrant container if it is running

```powershell
cd mafas
docker compose down
cd ..
```

This stops the container but keeps the `mafas_qdrant_storage` volume.

### 2. Configure root environment

```powershell
copy .env.example .env
```

Set `MYSQL_PASSWORD` and `MYSQL_ROOT_PASSWORD` to strong values before the
first `docker compose up`.

### 3. Start Ollama on the host

```powershell
ollama pull mistral
```

The backend container reaches the host LLM via `host.docker.internal` (Docker
Desktop). On Linux you may need to set `OLLAMA_URL` in `.env` to your machine's
LAN IP.

### 4. Build and run the stack

```powershell
docker compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| Evaluation reports | http://localhost:3000/evaluation |
| API docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333 |

The root Compose file reuses `mafas_qdrant_storage`, so a corpus built during
CLI setup is still there.

Populate or refresh the corpus from the **Data & Services** page in the UI, or
run `python -m rag.corpus_builder` from `mafas/` on the host (with Qdrant
reachable at `localhost:6333`).

Stop the stack:

```powershell
docker compose down
```

Add `-v` only if you intend to delete MySQL data and other named volumes.

**Demo mode** in the UI runs fixed fixtures (including sample backtest metrics)
without Ollama, market APIs, or an ingested corpus. Useful for checking the
interface before live setup is complete.

### Evaluation reports

The **Evaluation reports** page (`/evaluation`) runs on-demand quality checks
against the live stack. Results are scores only — no automated pass/fail
thresholds.

| Suite | What it measures | Requirements |
|-------|------------------|--------------|
| **RAG** | Corpus size, retrieval hit rate, top-k similarity, source/doc-type diversity on probe queries | Built Qdrant corpus |
| **Monte Carlo** | Calibration error: empirical TP rate vs MC `P(TP before SL)` on synthetic paths | None (in-process) |
| **Risk** | Live vol regime, VIX, correlations, sizing completeness, per-asset metrics | Network (yfinance) |

From the UI: open **Evaluation reports** in the sidebar → choose a suite →
**Run evaluation**. Past runs appear in the list and open as printable reports
at `/reports/[id]`.

API equivalent:

```powershell
curl -X POST http://localhost:8000/api/v1/evaluation/run `
  -H "Content-Type: application/json" `
  -d "{\"suites\": [\"all\"]}"
```

Poll `GET /api/v1/jobs/{id}` for the structured report in `result`. Filter run
history by workflow **Evaluation**.

**Note:** RAG evaluation reports operational retrieval quality (hit rate and
similarity), not labelled recall@k — that would require a golden relevance
dataset.

---

## Running agents individually

Always use `python -m` from `mafas/` with the venv active.

**Analyst** — RAG briefing with citations and confidence:

```powershell
.\.venv\Scripts\python -m agents.analyst "What is the Fed's stance on inflation?"
.\.venv\Scripts\python -m agents.analyst "Apple revenue outlook" --doc-type sec_filing
.\.venv\Scripts\python -m agents.analyst "recent policy signals" --date-after 2024-01-01
```

**Risk** — live vol regime, correlations, position sizing:

```powershell
.\.venv\Scripts\python -m agents.risk TSLA --lookback 252
```

**Strategy** — playbook scoring and 2–3 setup suggestions:

```powershell
.\.venv\Scripts\python -m agents.strategy "How is the Fed framing inflation?" TSLA
```

**Execution** — historical backtest + forward MC for one ticker:

```powershell
.\.venv\Scripts\python -m agents.execution NVDA --strategy trend_following --direction long
.\.venv\Scripts\python -m agents.execution NVDA --horizon swing --no-llm
```

**Orchestrator** — full graph with broaden loop and no-trade gate:

```powershell
.\.venv\Scripts\python -m agents.orchestrator "Fed stance on rates?" TSLA
.\.venv\Scripts\python -m agents.orchestrator "obscure query" --no-llm
```

Every agent honours `--no-llm` / `with_llm=False` for offline deterministic
output.

---

## Tests

From `mafas/` with the venv active:

```powershell
# Full agent test suite (mocked LLM and market data)
.\.venv\Scripts\python -m pytest tests/ -v

# Backtest and execution only (no langgraph corpus analyst required)
.\.venv\Scripts\python -m pytest tests/test_signals.py tests/test_backtest_metrics.py tests/test_historical_backtest.py tests/test_execution.py -v

# Evaluation harness (simulation suite is offline; RAG/risk need live services)
.\.venv\Scripts\python -m pytest tests/test_evaluation.py -v
```

Live checks (network required):

```powershell
.\.venv\Scripts\python scripts/smoke_loaders.py
.\.venv\Scripts\python scripts/smoke_pipeline.py --tickers TSLA --show-reports
```

API tests from the repo root:

```powershell
.\mafas\.venv\Scripts\python -m pytest mafas\tests backend\tests -q
```

Frontend (requires Node.js installed):

```powershell
cd frontend
npm install
npm test
npm run lint
```

---

## Configuration

### CLI (`mafas/.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Vector database |
| `QDRANT_COLLECTION` | `financial_docs` | Collection name |
| `OLLAMA_URL` | `http://localhost:11434` | Local LLM |
| `OLLAMA_MODEL` | `mistral` | Model tag |
| `CACHE_DIR` | `./data/cache` | Twelve Data disk cache |
| `FRED_API_KEY` | (empty) | Optional FRED access |
| `TWELVE_DATA_API_KEY` | (empty) | Optional; yfinance fallback if unset |
| `EXECUTION_ACCOUNT_EQUITY` | `100000` | Notional account for sizing sims |

### Dashboard (repo root `.env`)

Includes the above (with Docker-internal hostnames where needed), plus:

| Variable | Purpose |
|----------|---------|
| `MYSQL_*` | Database credentials (required before first start) |
| `NEXT_PUBLIC_API_URL` | Frontend → API URL |
| `CORS_ORIGINS` | Allowed browser origin |
| `JOB_MAX_WORKERS` | Background job concurrency |

---

## Data freshness

| Data | Source | When it updates |
|------|--------|-----------------|
| FOMC / SEC / news (RAG) | Qdrant | When you run `rag.corpus_builder` |
| Vol, VIX, correlations | yfinance | Every Risk agent run |
| Simulation prices | Twelve Data → yfinance | Every Execution run (Twelve Data cached 24h) |

The Analyst never ingests documents. Retrieval is relevance-based, so a Fed-focused
query returns mostly FOMC chunks even though the corpus also holds SEC and news
text. That is normal.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'data'` or `'agents'`**

Run `pip install -e .` from `mafas/`. Use `python -m agents.analyst`, not
`python agents/analyst.py`.

**Tests fail at collection with SciPy / `_ARRAY_API` errors**

Activate `mafas\.venv` and reinstall: `pip install -r requirements.txt`. Do not
use the global Python.

**`ModuleNotFoundError: langgraph`**

Install requirements in the venv. Orchestrator tests need langgraph; the
backtest test subset listed above does not.

**Port 6333 already in use**

Only one Compose project should own Qdrant. Run `docker compose down` in the
other directory first.

**Analyst returns empty or low-confidence briefings**

Check Qdrant is up and the corpus has been built. Run `corpus_builder` and
confirm chunks in the Qdrant UI at `http://localhost:6333/dashboard`.

**Ollama connection refused**

Confirm `ollama serve` is running and `OLLAMA_URL` matches. Use `--no-llm` to
verify the rest of the pipeline without the model.

**Dashboard backend cannot reach Ollama**

Ollama must run on the host, not inside Docker. On Linux, set `OLLAMA_URL` to
the host IP instead of `host.docker.internal`.

**Execution uses yfinance instead of Twelve Data**

Set `TWELVE_DATA_API_KEY` in `.env`. Without a key the agent falls back
automatically.

**Evaluation RAG suite fails or reports empty corpus**

Build or refresh the document index first (`rag.corpus_builder` or the **Data &
corpus** page). The RAG suite queries live Qdrant — it does not use demo
fixtures.

**Evaluation risk suite fails**

The backend container needs outbound network access for yfinance. Confirm market
data is reachable from inside Docker.

---

## Orchestrator behaviour (short version)

- Analyst confidence below 0.40 triggers a query-broadening loop (max 2 retries).
- Strategy must produce at least one setup above the 0.45 confidence floor.
- In a high-vol regime, the best setup must clear 0.55 or the pipeline returns
  no-trade.
- Agent exceptions are caught per node; the run degrades to no-trade instead of
  crashing.

Thresholds and routing live in `mafas/agents/orchestrator.py`.

---

## Licence and disclaimer

This software is for education and research. Outputs are not investment advice.
Verify material claims against primary sources before relying on them.
