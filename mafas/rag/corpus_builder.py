"""Orchestrates document ingestion into the vector store."""

import os

from dotenv import load_dotenv
from loguru import logger

from data.loaders.edgar import EDGARLoader
from data.loaders.fomc import FOMCLoader
from data.loaders.news import NewsLoader
from rag.chunker import SemanticChunker
from rag.embedder import TextEmbedder
from rag.retriever import VectorRetriever

# Mega-cap watchlist for SEC filing ingestion.
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM"]


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
) -> None:
    """Load FOMC minutes, SEC filings, and news; ingest into Qdrant; print summary."""
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "financial_docs")
    cache_dir = os.getenv("CACHE_DIR", "./data/cache")
    tickers = tickers if tickers is not None else DEFAULT_TICKERS

    embedder = TextEmbedder()
    retriever = VectorRetriever(qdrant_url, collection, embedder)
    chunker = SemanticChunker()
    corpus_builder = CorpusBuilder(retriever, chunker)

    fomc_loader = FOMCLoader(cache_dir=cache_dir)
    edgar_loader = EDGARLoader(cache_dir=cache_dir)
    news_loader = NewsLoader()

    logger.info("Loading FOMC minutes (n={})...", n_fomc)
    fomc_docs = fomc_loader.load_recent(n=n_fomc)

    logger.info("Loading SEC filings for {} tickers...", len(tickers))
    sec_docs: list[dict] = []
    for ticker in tickers:
        docs = edgar_loader.load_recent_for_ticker(ticker, limit=filings_per_ticker)
        logger.info("  {}: {} filing(s)", ticker, len(docs))
        sec_docs.extend(docs)

    logger.info("Loading news articles...")
    news_docs = news_loader.load_all(max_per_feed=max_news_per_feed)

    all_documents = fomc_docs + sec_docs + news_docs
    total_chunks = corpus_builder.ingest_documents(all_documents)
    total_vectors = retriever.count()

    print(
        f"\nCorpus build complete:\n"
        f"  FOMC minutes:       {len(fomc_docs)}\n"
        f"  SEC filings:        {len(sec_docs)}\n"
        f"  News articles:      {len(news_docs)}\n"
        f"  Documents ingested: {len(all_documents)}\n"
        f"  Chunks created:     {total_chunks}\n"
        f"  Vectors in Qdrant:  {total_vectors}\n"
    )


if __name__ == "__main__":
    build_initial_corpus()
