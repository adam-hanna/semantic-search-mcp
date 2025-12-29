"""Configuration management for semantic-search-mcp."""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Configuration for the semantic code search server."""

    # Embedding settings
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_dim: int = 768

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

    def __post_init__(self):
        """Validate configuration values."""
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
        if not 0 <= self.search_min_score <= 1:
            raise ValueError(f"min_score must be between 0 and 1, got {self.search_min_score}")


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        embedding_model=os.getenv("SEMANTIC_SEARCH_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code"),
        embedding_dim=int(os.getenv("SEMANTIC_SEARCH_EMBEDDING_DIM", "768")),
        db_path=Path(os.getenv("SEMANTIC_SEARCH_DB_PATH", ".semantic-search/index.db")),
        chunk_overlap_tokens=int(os.getenv("SEMANTIC_SEARCH_CHUNK_OVERLAP", "50")),
        max_chunk_tokens=int(os.getenv("SEMANTIC_SEARCH_MAX_CHUNK_TOKENS", "2000")),
        search_default_limit=int(os.getenv("SEMANTIC_SEARCH_LIMIT", "10")),
        search_min_score=float(os.getenv("SEMANTIC_SEARCH_MIN_SCORE", "0.3")),
        rrf_k=int(os.getenv("SEMANTIC_SEARCH_RRF_K", "60")),
        queue_max_size=int(os.getenv("SEMANTIC_SEARCH_QUEUE_MAX_SIZE", "1000")),
        debounce_ms=int(os.getenv("SEMANTIC_SEARCH_DEBOUNCE_MS", "1000")),
    )
