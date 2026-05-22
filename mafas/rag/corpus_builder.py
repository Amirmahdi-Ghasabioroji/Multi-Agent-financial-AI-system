"""Orchestrates document ingestion into the vector store."""

import os

from dotenv import load_dotenv
from loguru import logger

from data.loaders.fomc import FOMCLoader
from data.loaders.news import NewsLoader
from rag.chunker import SemanticChunker
from rag.embedder import TextEmbedder
from rag.retriever import VectorRetriever


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


def build_initial_corpus() -> None:
    """Load FOMC minutes and news, ingest into Qdrant, print summary."""
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "financial_docs")
    cache_dir = os.getenv("CACHE_DIR", "./data/cache")

    embedder = TextEmbedder()
    retriever = VectorRetriever(qdrant_url, collection, embedder)
    chunker = SemanticChunker()
    corpus_builder = CorpusBuilder(retriever, chunker)

    fomc_loader = FOMCLoader(cache_dir=cache_dir)
    news_loader = NewsLoader()

    logger.info("Loading FOMC minutes...")
    fomc_docs = fomc_loader.load_recent(n=5)

    logger.info("Loading news articles...")
    news_docs = news_loader.load_all(max_per_feed=10)

    all_documents = fomc_docs + news_docs
    total_chunks = corpus_builder.ingest_documents(all_documents)
    total_vectors = retriever.count()

    print(
        f"\nCorpus build complete:\n"
        f"  Documents ingested: {len(all_documents)}\n"
        f"  Chunks created:     {total_chunks}\n"
        f"  Vectors in Qdrant:  {total_vectors}\n"
    )


if __name__ == "__main__":
    build_initial_corpus()
