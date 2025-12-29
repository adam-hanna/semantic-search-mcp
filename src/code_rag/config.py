"""Configuration management for code-rag."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for the semantic code search server."""

    # Embedding settings
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_dim: int = 768

    # Database settings
    db_path: Path = field(default_factory=lambda: Path(".code-rag/index.db"))

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

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.search_min_score <= 1:
            raise ValueError(f"min_score must be between 0 and 1, got {self.search_min_score}")


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        embedding_model=os.getenv("CODE_RAG_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code"),
        embedding_dim=int(os.getenv("CODE_RAG_EMBEDDING_DIM", "768")),
        db_path=Path(os.getenv("CODE_RAG_DB_PATH", ".code-rag/index.db")),
        chunk_overlap_tokens=int(os.getenv("CODE_RAG_CHUNK_OVERLAP", "50")),
        max_chunk_tokens=int(os.getenv("CODE_RAG_MAX_CHUNK_TOKENS", "2000")),
        search_default_limit=int(os.getenv("CODE_RAG_SEARCH_LIMIT", "10")),
        search_min_score=float(os.getenv("CODE_RAG_SEARCH_MIN_SCORE", "0.3")),
        rrf_k=int(os.getenv("CODE_RAG_RRF_K", "60")),
        queue_max_size=int(os.getenv("CODE_RAG_QUEUE_MAX_SIZE", "1000")),
        debounce_ms=int(os.getenv("CODE_RAG_DEBOUNCE_MS", "1000")),
    )
