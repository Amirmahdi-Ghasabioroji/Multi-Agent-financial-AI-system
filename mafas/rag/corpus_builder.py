"""Orchestrates document ingestion into the vector store."""

import argparse
import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from data.loaders.edgar import EDGARLoader
from data.loaders.fomc import FOMCLoader
from data.loaders.news import NewsLoader
from rag.chunker import SemanticChunker
from rag.embedder import TextEmbedder
from rag.retriever import VectorRetriever

# Mega-cap watchlist for SEC filing ingestion.
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM"]


class CorpusBuildResult(BaseModel):
    """Structured corpus-refresh summary for CLIs and dashboard jobs."""

    fomc_documents: int
    sec_documents: int
    news_documents: int
    documents_ingested: int
    chunks_created: int
    vectors_in_collection: int
    reset: bool = False


def _emit(
    callback: Callable[[str, dict[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    """Emit progress without allowing UI observers to break ingestion."""
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Corpus progress callback failed ({})", type(exc).__name__)


class CorpusBuilder:
    """Chunks documents and upserts them into Qdrant via a VectorRetriever."""

    def __init__(
        self,
        retriever: VectorRetriever,
        chunker: SemanticChunker,
    ) -> None:
        self.retriever = retriever
        self.chunker = chunker

    def ingest_documents(self, documents: list[dict]) -> int:
        """Chunk and upsert a list of documents; returns total chunks created."""
        all_chunks: list[dict] = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc["text"], doc["metadata"])
            all_chunks.extend(chunks)

        if all_chunks:
            self.retriever.upsert(all_chunks)

        total_points = self.retriever.count()
        logger.info(
            "Ingested {} documents → {} chunks; collection now has {} points",
            len(documents),
            len(all_chunks),
            total_points,
        )
        return len(all_chunks)


def build_initial_corpus(
    n_fomc: int = 5,
    max_news_per_feed: int = 10,
    tickers: list[str] | None = None,
    filings_per_ticker: int = 1,
    reset: bool = False,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> CorpusBuildResult:
    """Load FOMC minutes, SEC filings, and news; ingest into Qdrant; print summary.

    Ingestion is idempotent: chunks use deterministic content-hash IDs, so
    re-running never creates duplicates. Pass reset=True to drop and rebuild
    the collection from scratch (useful to purge legacy duplicate points).
    """
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "financial_docs")
    cache_dir = os.getenv("CACHE_DIR", "./data/cache")
    tickers = tickers if tickers is not None else DEFAULT_TICKERS

    embedder = TextEmbedder()
    retriever = VectorRetriever(qdrant_url, collection, embedder, recreate=reset)
    chunker = SemanticChunker()
    corpus_builder = CorpusBuilder(retriever, chunker)

    fomc_loader = FOMCLoader(cache_dir=cache_dir)
    edgar_loader = EDGARLoader(cache_dir=cache_dir)
    news_loader = NewsLoader()

    logger.info("Loading FOMC minutes (n={})...", n_fomc)
    _emit(progress_callback, "stage_started", stage="fomc", requested=n_fomc)
    fomc_docs = fomc_loader.load_recent(n=n_fomc)
    _emit(progress_callback, "stage_completed", stage="fomc", loaded=len(fomc_docs))

    logger.info("Loading SEC filings for {} tickers...", len(tickers))
    _emit(progress_callback, "stage_started", stage="sec", tickers=tickers)
    sec_docs: list[dict] = []
    for ticker in tickers:
        docs = edgar_loader.load_recent_for_ticker(ticker, limit=filings_per_ticker)
        logger.info("  {}: {} filing(s)", ticker, len(docs))
        sec_docs.extend(docs)
        _emit(
            progress_callback,
            "stage_progress",
            stage="sec",
            ticker=ticker,
            loaded=len(docs),
        )
    _emit(progress_callback, "stage_completed", stage="sec", loaded=len(sec_docs))

    logger.info("Loading news articles...")
    _emit(progress_callback, "stage_started", stage="news")
    news_docs = news_loader.load_all(max_per_feed=max_news_per_feed)
    _emit(progress_callback, "stage_completed", stage="news", loaded=len(news_docs))

    all_documents = fomc_docs + sec_docs + news_docs
    _emit(
        progress_callback,
        "stage_started",
        stage="embedding",
        documents=len(all_documents),
    )
    total_chunks = corpus_builder.ingest_documents(all_documents)
    total_vectors = retriever.count()
    _emit(
        progress_callback,
        "stage_completed",
        stage="embedding",
        chunks=total_chunks,
        vectors=total_vectors,
    )

    print(
        f"\nCorpus build complete:\n"
        f"  FOMC minutes:       {len(fomc_docs)}\n"
        f"  SEC filings:        {len(sec_docs)}\n"
        f"  News articles:      {len(news_docs)}\n"
        f"  Documents ingested: {len(all_documents)}\n"
        f"  Chunks created:     {total_chunks}\n"
        f"  Vectors in Qdrant:  {total_vectors}\n"
    )
    result = CorpusBuildResult(
        fomc_documents=len(fomc_docs),
        sec_documents=len(sec_docs),
        news_documents=len(news_docs),
        documents_ingested=len(all_documents),
        chunks_created=total_chunks,
        vectors_in_collection=total_vectors,
        reset=reset,
    )
    _emit(progress_callback, "corpus_completed", **result.model_dump())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the MAFAS document corpus in Qdrant."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting (purges duplicates).",
    )
    args = parser.parse_args()
    build_initial_corpus(reset=args.reset)
