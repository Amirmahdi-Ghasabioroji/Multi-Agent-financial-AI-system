# Multi-Agent Financial AI System (MAFAS)

Data foundation and RAG infrastructure for a multi-agent financial AI system.

## Quick start

```bash
cd mafas
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # Windows
.\.venv\Scripts\pip install -e .                  # makes `data` and `rag` importable
copy .env.example .env
```

### Fix: `ModuleNotFoundError: No module named 'data'`

Imports use `from data...` and `from rag...`. Those packages live under `mafas/`, not the repo root.

**Do one of the following:**

1. **Recommended:** from `mafas/`, run `pip install -e .` (see above), then run scripts from any directory.
2. **Or** always `cd mafas` before `python` / `pytest`, and set `PYTHONPATH=.`:
   ```powershell
   cd mafas
   $env:PYTHONPATH = "."
   .\.venv\Scripts\python -c "from data.loaders.news import NewsLoader; print(NewsLoader())"
   ```
3. **Do not** run loader files directly, e.g. `python data/loaders/edgar.py` — that bypasses the package layout.

In Cursor/VS Code: set the Python interpreter to `mafas\.venv\Scripts\python.exe` and the terminal **cwd** to the `mafas` folder.

```powershell
# Vector DB for corpus + retrieval
docker compose up -d

# Run all tests
.\.venv\Scripts\pytest tests/ -v

# Build the corpus: FOMC minutes + SEC filings (mega-caps) + news
.\.venv\Scripts\python -m rag.corpus_builder

# Smoke-test all loaders (live internet)
.\.venv\Scripts\python scripts/smoke_loaders.py
```

## Analyst Agent

The Analyst Agent runs a RAG pipeline over the Qdrant corpus (FOMC minutes,
SEC filings, financial news) and uses a **local, free** Ollama LLM (Mistral 7B)
to synthesise sourced macro briefings with **composite confidence scoring**.

### 1. Install Ollama (free, local, no API costs)

1. Download and install from <https://ollama.com/download>.
2. Pull the model and start the server:

```powershell
ollama pull mistral
ollama serve        # serves http://localhost:11434
```

Verify: `http://localhost:11434/api/tags` should list `mistral`.

### 2. Ask for a macro briefing

```powershell
# Qdrant must be running and the corpus built first
.\.venv\Scripts\python -m agents.analyst "What is the Fed's current stance on inflation and rate cuts?"

# Filter to a single source type or by date
.\.venv\Scripts\python -m agents.analyst "Apple revenue outlook" --doc-type sec_filing
.\.venv\Scripts\python -m agents.analyst "recent policy signals" --date-after 2024-01-01
```

The agent retrieves the most relevant chunks, asks Mistral to synthesise a
briefing that cites only those sources, and reports an overall confidence score
plus a per-component breakdown (retrieval similarity, source diversity,
recency, and the model's self-reported confidence).

All source dates are normalised to `YYYY-MM-DD` at ingestion (FOMC from the
meeting URL, SEC from the filing date, news from the RSS published date) and
stored as a sortable `date_ts`, so `--date-after YYYY-MM-DD` filters reliably
across every source type. Documents with no parseable date are excluded when a
date filter is applied.

## Layout

- `data/loaders/` — FOMC, EDGAR, news RSS, market data
- `data/processors/` — text cleaning and metadata extraction
- `rag/` — chunking, embeddings, Qdrant retriever, corpus builder
- `agents/` — Analyst Agent, Ollama LLM client, confidence scoring, schemas
- `tests/` — unit tests (`test_prerequisites.py`, `test_analyst.py`)
- `scripts/` — `smoke_loaders.py` live loader checks

## Roadmap

- **Analyst Agent** — sourced macro briefings (this phase). Done.
- **Risk Agent** — volatility/exposure assessment via `MarketDataLoader`.
- **Strategy Agent** — turns briefings + risk into positioning ideas.
- **Execution Agent** — sizing and order logic.
