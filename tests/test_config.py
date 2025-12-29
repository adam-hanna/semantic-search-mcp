"""Tests for configuration module."""
import os
from pathlib import Path

import pytest

from semantic_search_mcp.config import Config, load_config


def test_config_defaults():
    """Config should have sensible defaults."""
    config = Config()

    assert config.embedding_model == "jinaai/jina-embeddings-v2-base-code"
    assert config.embedding_dim == 768
    assert config.db_path == Path(".semantic-search/index.db")
    assert config.chunk_overlap_tokens == 50
    assert config.max_chunk_tokens == 2000
    assert config.search_default_limit == 10
    assert config.search_min_score == 0.3
    assert config.rrf_k == 60
    assert config.queue_max_size == 1000
    assert config.debounce_ms == 1000


def test_config_from_env(monkeypatch):
    """Config should read from environment variables."""
    monkeypatch.setenv("SEMANTIC_SEARCH_DB_PATH", "/custom/path/index.db")
    monkeypatch.setenv("SEMANTIC_SEARCH_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("SEMANTIC_SEARCH_MIN_SCORE", "0.5")

    config = load_config()

    assert config.db_path == Path("/custom/path/index.db")
    assert config.embedding_model == "custom-model"
    assert config.search_min_score == 0.5


def test_config_validates_min_score():
    """min_score must be between 0 and 1."""
    with pytest.raises(ValueError, match="min_score"):
        Config(search_min_score=1.5)

    with pytest.raises(ValueError, match="min_score"):
        Config(search_min_score=-0.1)
