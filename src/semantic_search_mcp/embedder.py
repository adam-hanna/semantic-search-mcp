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


def _get_gpu_provider() -> tuple[bool, str | None]:
    """Check if GPU acceleration is available for ONNX runtime.

    Returns:
        Tuple of (is_available, provider_name)
        Provider can be: CUDAExecutionProvider (NVIDIA), CoreMLExecutionProvider (Apple Silicon), etc.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        # Check for GPU providers in order of preference
        gpu_providers = [
            "CUDAExecutionProvider",      # NVIDIA GPU
            "CoreMLExecutionProvider",    # Apple Silicon
            "ROCMExecutionProvider",      # AMD GPU
            "DmlExecutionProvider",       # Windows DirectML
        ]

        for provider in gpu_providers:
            if provider in available:
                return True, provider

        return False, None
    except Exception:
        return False, None

# Known model dimensions (fallback if not in fastembed metadata)
MODEL_DIMENSIONS = {
    "jinaai/jina-embeddings-v2-base-code": 768,
    "jinaai/jina-embeddings-v2-small-en": 512,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def get_model_dimension(model_name: str) -> int:
    """Get embedding dimension for a model.

    Looks up dimension from fastembed metadata, falls back to known dimensions.
    """
    # Try fastembed metadata first
    try:
        for m in TextEmbedding.list_supported_models():
            if m.get("model") == model_name:
                return m.get("dim", 768)
    except Exception:
        pass

    # Fallback to known dimensions
    return MODEL_DIMENSIONS.get(model_name, 768)


class Embedder:
    """Generate embeddings for code snippets using FastEmbed."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        embedding_dim: int = 768,
        cache_dir: Optional[str] = None,
        batch_size: int = 8,
        threads: int = 4,
        use_quantized: bool = False,
    ):
        """Initialize the embedding model.

        Args:
            model_name: Name of the embedding model to use
            embedding_dim: Expected embedding dimension
            cache_dir: Optional cache directory for model files
            batch_size: Texts per embedding call (prevents ONNX memory explosion)
            threads: Number of threads for ONNX runtime (higher = faster on multi-core)
            use_quantized: If True, use INT8 quantized model (30-40% faster, auto-quantizes on first run)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model: Optional[TextEmbedding] = None
        self._cache_dir = cache_dir
        self.batch_size = batch_size
        self.threads = threads
        self.use_quantized = use_quantized

    def _find_model_onnx_path(self) -> Optional[Path]:
        """Find the ONNX model file in fastembed cache.

        Returns:
            Path to model.onnx if found, None otherwise.
        """
        cache_locations = [
            Path(tempfile.gettempdir()) / "fastembed_cache",
            Path.home() / ".cache" / "fastembed",
            Path.home() / ".cache" / "fastembed_cache",
        ]

        model_cache_name = f"models--{self.model_name.replace('/', '--')}"

        for cache_dir in cache_locations:
            model_dir = cache_dir / model_cache_name
            if model_dir.exists():
                # Find the onnx directory in snapshots
                for onnx_path in model_dir.rglob("model.onnx"):
                    return onnx_path

        return None

    def _quantize_model(self, model_path: Path) -> Optional[Path]:
        """Quantize the ONNX model to INT8.

        Args:
            model_path: Path to the FP32 model.onnx

        Returns:
            Path to the quantized model, or None if quantization failed.
        """
        quantized_path = model_path.parent / "model_int8.onnx"

        # Already quantized
        if quantized_path.exists():
            return quantized_path

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            logger.info(f"Quantizing model to INT8 (this only happens once)...")
            logger.info(f"  Input: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

            # Resolve symlink to get actual file
            actual_model_path = model_path.resolve()

            quantize_dynamic(
                model_input=str(actual_model_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
            )

            logger.info(f"  Output: {quantized_path} ({quantized_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return quantized_path

        except ImportError:
            logger.warning("onnx package not installed, skipping quantization. Install with: pip install onnx")
            return None
        except Exception as e:
            logger.warning(f"Failed to quantize model: {e}")
            return None

    def _apply_quantized_model(self, model_path: Path, quantized_path: Path) -> bool:
        """Replace the model.onnx with the quantized version.

        Args:
            model_path: Path to the original model.onnx (may be symlink)
            quantized_path: Path to the quantized model

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Backup original if it's a real file (not symlink)
            backup_path = model_path.parent / "model_fp32.onnx"
            if model_path.is_symlink():
                # Just remove the symlink
                model_path.unlink()
            elif not backup_path.exists():
                # Move the original to backup
                shutil.move(str(model_path), str(backup_path))
            else:
                # Backup exists, just remove original
                model_path.unlink()

            # Copy quantized as the new model.onnx
            shutil.copy2(str(quantized_path), str(model_path))
            logger.info("Applied quantized model successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to apply quantized model: {e}")
            return False

    def _ensure_quantized(self) -> None:
        """Ensure the quantized model is in place if use_quantized is enabled."""
        if not self.use_quantized:
            return

        model_path = self._find_model_onnx_path()
        if model_path is None:
            logger.debug("Model not yet downloaded, will quantize after first load")
            return

        # Check if already using quantized (file size < 200MB indicates INT8)
        if model_path.stat().st_size < 200 * 1024 * 1024:
            logger.debug("Already using quantized model")
            return

        # Quantize and apply
        quantized_path = self._quantize_model(model_path)
        if quantized_path:
            self._apply_quantized_model(model_path, quantized_path)

    def _find_and_clear_model_cache(self) -> bool:
        """Find and clear the fastembed cache for this model.

        Returns:
            True if cache was found and cleared, False otherwise.
        """
        cache_locations = [
            Path(tempfile.gettempdir()) / "fastembed_cache",
            Path.home() / ".cache" / "fastembed",
            Path.home() / ".cache" / "fastembed_cache",
        ]

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
            # First, check if we need to apply quantization to an existing model
            self._ensure_quantized()

            # Auto-detect GPU availability
            gpu_available, gpu_provider = _get_gpu_provider()
            if gpu_available:
                logger.info(f"GPU detected ({gpu_provider}), using hardware acceleration")

            try:
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=self._cache_dir,
                    threads=self.threads,
                    cuda=gpu_available,  # FastEmbed uses 'cuda' but it works for other GPU providers too
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
                                threads=self.threads,
                                cuda=gpu_available,
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

            # If this was first load and quantization is enabled, quantize now
            if self.use_quantized:
                model_path = self._find_model_onnx_path()
                if model_path and model_path.stat().st_size > 200 * 1024 * 1024:
                    # Model was just downloaded, need to quantize and reload
                    quantized_path = self._quantize_model(model_path)
                    if quantized_path and self._apply_quantized_model(model_path, quantized_path):
                        # Reload with quantized model
                        self._model = None
                        self._model = TextEmbedding(
                            model_name=self.model_name,
                            cache_dir=self._cache_dir,
                            threads=self.threads,
                            cuda=gpu_available,
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

        result = []

        # Process in small batches to prevent ONNX memory explosion
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

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
