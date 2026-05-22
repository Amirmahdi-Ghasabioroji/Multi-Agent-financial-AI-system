"""Semantic text chunking for RAG ingestion."""

import copy


class SemanticChunker:
    """Splits documents into overlapping chunks by paragraph boundaries."""

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _overlap_seed(self, chunk: str) -> str:
        words = chunk.split()
        if len(words) <= self.overlap_tokens:
            return chunk
        return " ".join(words[-self.overlap_tokens :])

    def chunk(self, text: str) -> list[str]:
        """Split text into paragraph-based chunks with token overlap."""
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if len(p.strip()) >= 30
        ]
        if not paragraphs:
            stripped = text.strip()
            return [stripped] if stripped else []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_words = 0

        for paragraph in paragraphs:
            para_words = self._word_count(paragraph)
            if (
                current_parts
                and current_words + para_words > self.max_tokens
            ):
                chunk_text = "\n\n".join(current_parts)
                chunks.append(chunk_text)
                seed = self._overlap_seed(chunk_text)
                current_parts = [seed, paragraph] if seed else [paragraph]
                current_words = self._word_count("\n\n".join(current_parts))
            else:
                current_parts.append(paragraph)
                current_words += para_words

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks if chunks else [text.strip()]

    def chunk_document(self, text: str, metadata: dict) -> list[dict]:
        """Chunk text and attach metadata with chunk_index."""
        raw_chunks = self.chunk(text)
        result: list[dict] = []
        for index, chunk_str in enumerate(raw_chunks):
            meta = copy.deepcopy(metadata)
            meta["chunk_index"] = index
            result.append({"text": chunk_str, "metadata": meta})
        return result
