# Multi-Agent Financial AI System (MAFAS)

Data foundation and RAG infrastructure for a multi-agent financial AI system.

## Quick start

```bash
cd mafas
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # Windows
cp .env.example .env                                # or copy on Windows

# Optional: vector DB for corpus ingestion
docker compose up -d

# Run prerequisite tests
.\.venv\Scripts\pytest tests/test_prerequisites.py -v

# Build initial corpus (requires Qdrant running)
.\.venv\Scripts\python -m rag.corpus_builder
```

## Layout

- `data/loaders/` — FOMC, EDGAR, news RSS, market data
- `data/processors/` — text cleaning and metadata extraction
- `rag/` — chunking, embeddings, Qdrant retriever, corpus builder
- `tests/` — unit tests for core components

Agents will be added in a later phase.
