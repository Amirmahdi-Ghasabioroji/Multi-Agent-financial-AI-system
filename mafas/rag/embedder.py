"""Text embedding via sentence-transformers."""

from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEmbedder:
    """Encodes text into dense vectors using a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedder '{}' (dim={})", model_name, self.dim)

    def embed(self, text: str) -> list[float]:
        """Encode a single string to a vector."""
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Encode multiple strings with a progress bar."""
        all_vectors: list[list[float]] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Embedding",
        ):
            batch = texts[start : start + batch_size]
            vectors = self.model.encode(batch, convert_to_numpy=True)
            all_vectors.extend(v.tolist() for v in vectors)
        return all_vectors
