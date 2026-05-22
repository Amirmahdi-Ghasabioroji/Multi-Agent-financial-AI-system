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
# Optional: vector DB for corpus ingestion
docker compose up -d

# Run prerequisite tests
.\.venv\Scripts\pytest tests/test_prerequisites.py -v

# Build initial corpus (requires Qdrant running)
.\.venv\Scripts\python -m rag.corpus_builder

# Smoke-test all loaders (live internet)
.\.venv\Scripts\python scripts/smoke_loaders.py
```

## Layout

- `data/loaders/` — FOMC, EDGAR, news RSS, market data
- `data/processors/` — text cleaning and metadata extraction
- `rag/` — chunking, embeddings, Qdrant retriever, corpus builder
- `tests/` — unit tests for core components

Agents will be added in a later phase.
