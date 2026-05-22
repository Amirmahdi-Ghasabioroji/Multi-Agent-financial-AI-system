"""Qdrant vector store wrapper for semantic retrieval."""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from rag.embedder import TextEmbedder


class VectorRetriever:
    """Manages a Qdrant collection for document chunk storage and search."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        embedder: TextEmbedder,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedder = embedder
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = self.client.get_collections().collections
        names = {c.name for c in collections}
        if self.collection_name in names:
            logger.info("Using existing collection '{}'", self.collection_name)
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created collection '{}'", self.collection_name)

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def upsert(self, chunks: list[dict]) -> None:
        """Embed and upsert document chunks in batches of 100."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed_batch(texts)
        start_id = self.count()

        points: list[PointStruct] = []
        for offset, chunk in enumerate(chunks):
            payload = {**chunk["metadata"], "text": chunk["text"]}
            points.append(
                PointStruct(
                    id=start_id + offset,
                    vector=vectors[offset],
                    payload=payload,
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        logger.info("Upserted {} points to '{}'", len(points), self.collection_name)

    def search(
        self,
        query: str,
        top_k: int = 8,
        doc_type: str | None = None,
        date_after: str | None = None,
        score_threshold: float = 0.40,
    ) -> list[dict]:
        """Semantic search with optional metadata filters."""
        query_vector = self.embedder.embed(query)
        conditions = []
        if doc_type:
            conditions.append(
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type),
                )
            )
        if date_after:
            conditions.append(
                FieldCondition(key="date", range=Range(gte=date_after))
            )
        query_filter = Filter(must=conditions) if conditions else None

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: list[dict] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "score": hit.score,
                    "source": payload.get("source", ""),
                    "date": payload.get("date", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                }
            )
        return results
