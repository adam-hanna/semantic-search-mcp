"""Configuration management for semantic-search-mcp."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for the semantic code search server."""

    # Embedding settings
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_dim: Optional[int] = None  # Auto-detected from model if not set

    # Database settings
    db_path: Path = field(default_factory=lambda: Path(".semantic-search/index.db"))

    # Chunking settings
    chunk_overlap_tokens: int = 50
    max_chunk_tokens: int = 2000

    # Search settings
    search_default_limit: int = 10
    search_min_score: float = 0.3
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # Watcher settings
    queue_max_size: int = 1000
    debounce_ms: int = 1000

    # Indexer settings
    index_batch_size: int = 50  # Files per batch for memory management
    max_file_size_kb: int = 512  # Skip files larger than this (KB)
    embedding_batch_size: int = 8  # Texts per embedding call (prevents ONNX memory explosion)
    embedding_threads: int = 4  # ONNX runtime threads (higher = faster on multi-core CPUs)
    use_quantized: bool = True  # Use INT8 quantized model (30-40% faster)

    def __post_init__(self):
        """Validate configuration values."""
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
        if not 0 <= self.search_min_score <= 1:
            raise ValueError(f"min_score must be between 0 and 1, got {self.search_min_score}")

        # Auto-detect embedding dimension from model if not set
        if self.embedding_dim is None:
            from semantic_search_mcp.embedder import get_model_dimension
            self.embedding_dim = get_model_dimension(self.embedding_model)


def load_config() -> Config:
    """Load configuration from environment variables."""
    # Get embedding dim - None means auto-detect from model
    embedding_dim_str = os.getenv("SEMANTIC_SEARCH_EMBEDDING_DIM")
    embedding_dim = int(embedding_dim_str) if embedding_dim_str else None

    # use_quantized defaults to True unless explicitly disabled
    use_quantized_str = os.getenv("SEMANTIC_SEARCH_USE_QUANTIZED", "true").lower()
    use_quantized = use_quantized_str not in ("0", "false", "no")

    return Config(
        embedding_model=os.getenv("SEMANTIC_SEARCH_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code"),
        embedding_dim=embedding_dim,
        db_path=Path(os.getenv("SEMANTIC_SEARCH_DB_PATH", ".semantic-search/index.db")),
        chunk_overlap_tokens=int(os.getenv("SEMANTIC_SEARCH_CHUNK_OVERLAP", "50")),
        max_chunk_tokens=int(os.getenv("SEMANTIC_SEARCH_MAX_CHUNK_TOKENS", "2000")),
        search_default_limit=int(os.getenv("SEMANTIC_SEARCH_LIMIT", "10")),
        search_min_score=float(os.getenv("SEMANTIC_SEARCH_MIN_SCORE", "0.3")),
        rrf_k=int(os.getenv("SEMANTIC_SEARCH_RRF_K", "60")),
        queue_max_size=int(os.getenv("SEMANTIC_SEARCH_QUEUE_MAX_SIZE", "1000")),
        debounce_ms=int(os.getenv("SEMANTIC_SEARCH_DEBOUNCE_MS", "1000")),
        index_batch_size=int(os.getenv("SEMANTIC_SEARCH_BATCH_SIZE", "50")),
        max_file_size_kb=int(os.getenv("SEMANTIC_SEARCH_MAX_FILE_SIZE_KB", "512")),
        embedding_batch_size=int(os.getenv("SEMANTIC_SEARCH_EMBEDDING_BATCH_SIZE", "8")),
        embedding_threads=int(os.getenv("SEMANTIC_SEARCH_EMBEDDING_THREADS", "4")),
        use_quantized=use_quantized,
    )
