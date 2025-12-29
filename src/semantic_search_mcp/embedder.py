# src/semantic_search_mcp/embedder.py
"""FastEmbed wrapper for code embeddings."""
import math
from typing import Optional

from fastembed import TextEmbedding


class Embedder:
    """Generate embeddings for code snippets using FastEmbed."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        embedding_dim: int = 768,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the embedding model.

        Args:
            model_name: Name of the embedding model to use
            embedding_dim: Expected embedding dimension
            cache_dir: Optional cache directory for model files
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model: Optional[TextEmbedding] = None
        self._cache_dir = cache_dir

    @property
    def model(self) -> TextEmbedding:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
            )
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of code snippets or queries to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # FastEmbed returns a generator, convert to list
        embeddings = list(self.model.embed(texts))
        return [list(emb) for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        return self.embed([query])[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity (0 to 1 for normalized vectors)
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
