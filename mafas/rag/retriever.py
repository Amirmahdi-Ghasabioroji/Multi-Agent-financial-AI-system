"""Qdrant vector store wrapper for semantic retrieval."""

import hashlib
import uuid

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

from data.processors.metadata import date_to_int
from rag.embedder import TextEmbedder

# Fixed namespace so the same chunk always maps to the same point ID,
# making re-ingestion idempotent (duplicates overwrite in place).
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def chunk_point_id(text: str, source: str, chunk_index: int) -> str:
    """Deterministic UUID for a chunk based on its content and origin.

    Identical content from the same source/position yields the same ID, so
    upserting it again overwrites the existing point instead of duplicating it.
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    key = f"{source}|{chunk_index}|{digest}"
    return str(uuid.uuid5(_POINT_NAMESPACE, key))


class VectorRetriever:
    """Manages a Qdrant collection for document chunk storage and search."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        embedder: TextEmbedder,
        recreate: bool = False,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedder = embedder
        self._ensure_collection(recreate=recreate)

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if needed; optionally drop and recreate it."""
        collections = self.client.get_collections().collections
        names = {c.name for c in collections}

        if self.collection_name in names:
            if not recreate:
                logger.info("Using existing collection '{}'", self.collection_name)
                return
            self.client.delete_collection(self.collection_name)
            logger.info("Dropped existing collection '{}'", self.collection_name)

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
        """Embed and upsert chunks with deterministic IDs (deduplicated)."""
        if not chunks:
            return

        # Collapse exact duplicate chunks within this batch up front.
        unique: dict[str, dict] = {}
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            point_id = chunk_point_id(
                chunk["text"],
                str(metadata.get("source", "")),
                int(metadata.get("chunk_index", 0)),
            )
            unique[point_id] = chunk

        before = self.count()
        point_ids = list(unique.keys())
        texts = [unique[pid]["text"] for pid in point_ids]
        vectors = self.embedder.embed_batch(texts)

        points: list[PointStruct] = []
        for pid, vector in zip(point_ids, vectors):
            chunk = unique[pid]
            payload = {**chunk["metadata"], "text": chunk["text"]}
            # Sortable integer date (YYYYMMDD) enables reliable range filtering.
            date_int = date_to_int(payload.get("date"))
            if date_int is not None:
                payload["date_ts"] = date_int
            points.append(PointStruct(id=pid, vector=vector, payload=payload))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        after = self.count()
        skipped = len(chunks) - len(points)
        logger.info(
            "Upsert: {} chunks in ({} intra-batch dupes), {} new, {} overwritten; "
            "collection {} -> {} points",
            len(chunks),
            skipped,
            after - before,
            len(points) - (after - before),
            before,
            after,
        )

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
            date_after_int = date_to_int(date_after)
            if date_after_int is not None:
                conditions.append(
                    FieldCondition(key="date_ts", range=Range(gte=date_after_int))
                )
            else:
                logger.warning(
                    "Ignoring date_after='{}' (expected YYYY-MM-DD)", date_after
                )
        query_filter = Filter(must=conditions) if conditions else None

        # qdrant-client >=1.10 uses query_points(); .search() was removed.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        hits = response.points

        results: list[dict] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "id": str(hit.id),
                    "text": payload.get("text", ""),
                    "score": hit.score,
                    "source": payload.get("source", ""),
                    "date": payload.get("date", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                }
            )
        return results

    def scroll_payloads(self, batch_size: int = 128) -> list[dict]:
        """Return every point payload (no vectors) for labelled recall denominators."""
        collected: list[dict] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                collected.append(
                    {
                        "id": str(point.id),
                        "text": payload.get("text", ""),
                        "source": payload.get("source", ""),
                        "date": payload.get("date", ""),
                        "doc_type": payload.get("doc_type", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                    }
                )
            if offset is None:
                break
        return collected
