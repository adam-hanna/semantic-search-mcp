# src/semantic_search_mcp/embedder.py
"""FastEmbed wrapper for code embeddings."""
import gc
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


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

    def _find_and_clear_model_cache(self) -> bool:
        """Find and clear the fastembed cache for this model.

        Returns:
            True if cache was found and cleared, False otherwise.
        """
        # fastembed uses temp directory by default
        cache_locations = [
            Path(tempfile.gettempdir()) / "fastembed_cache",
            Path.home() / ".cache" / "fastembed",
            Path.home() / ".cache" / "fastembed_cache",
        ]

        # Convert model name to cache directory name (e.g., "jinaai/jina-..." -> "models--jinaai--jina-...")
        model_cache_name = f"models--{self.model_name.replace('/', '--')}"

        cleared = False
        for cache_dir in cache_locations:
            model_path = cache_dir / model_cache_name
            if model_path.exists():
                logger.warning(f"Clearing incomplete model cache: {model_path}")
                try:
                    shutil.rmtree(model_path)
                    cleared = True
                except Exception as e:
                    logger.error(f"Failed to clear cache {model_path}: {e}")

        return cleared

    @property
    def model(self) -> TextEmbedding:
        """Lazy-load the embedding model with retry on incomplete download."""
        if self._model is None:
            try:
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=self._cache_dir,
                )
            except Exception as e:
                error_msg = str(e).lower()
                # Check for ONNX file not found errors (incomplete download)
                if "no_suchfile" in error_msg or "doesn't exist" in error_msg or "does not exist" in error_msg:
                    logger.warning(f"Model files incomplete, clearing cache and retrying: {e}")
                    if self._find_and_clear_model_cache():
                        # Retry after clearing cache
                        try:
                            self._model = TextEmbedding(
                                model_name=self.model_name,
                                cache_dir=self._cache_dir,
                            )
                        except Exception as retry_error:
                            raise RuntimeError(
                                f"Failed to download embedding model after cache clear. "
                                f"Check your internet connection and try again. "
                                f"Original error: {retry_error}"
                            ) from retry_error
                    else:
                        raise RuntimeError(
                            f"Embedding model files are incomplete but cache not found. "
                            f"Try manually clearing: rm -rf /tmp/fastembed_cache ~/.cache/fastembed*\n"
                            f"Original error: {e}"
                        ) from e
                else:
                    raise
        return self._model

    def embed(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of code snippets or queries to embed
            batch_size: Number of texts to embed at once (reduces memory)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        result = []

        # Process in small batches to prevent ONNX memory explosion
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # FastEmbed returns a generator - process one at a time
            for emb in self.model.embed(batch):
                result.append(list(emb))  # Convert numpy to list immediately
                del emb  # Explicitly free numpy array

            # Force cleanup between batches
            gc.collect()

        return result

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
