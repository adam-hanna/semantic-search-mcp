# tests/test_embedder.py
"""Tests for embedder module."""
import pytest

from semantic_search_mcp.embedder import Embedder


@pytest.fixture
def embedder():
    """Create embedder with small model for fast tests."""
    # Use smaller model for tests
    return Embedder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=384)


def test_embedder_generates_correct_dimension(embedder):
    """Embeddings should have the configured dimension."""
    texts = ["def hello(): pass"]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


def test_embedder_batch_embedding(embedder):
    """Embedder should handle batch inputs."""
    texts = [
        "def add(a, b): return a + b",
        "class User: pass",
        "async def fetch_data(): pass",
    ]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 384


def test_embedder_similar_code_has_high_similarity(embedder):
    """Similar code snippets should have high cosine similarity."""
    code1 = "def binary_search(arr, target): left, right = 0, len(arr)"
    code2 = "def bsearch(array, value): lo, hi = 0, len(array)"
    code3 = "class DatabaseConnection: def connect(self): pass"

    emb1, emb2, emb3 = embedder.embed([code1, code2, code3])

    sim_12 = embedder.cosine_similarity(emb1, emb2)
    sim_13 = embedder.cosine_similarity(emb1, emb3)

    # Similar functions should be more similar than unrelated code
    assert sim_12 > sim_13


def test_embedder_empty_list_returns_empty(embedder):
    """Empty input should return empty output."""
    embeddings = embedder.embed([])
    assert embeddings == []


def test_embedder_handles_unicode(embedder):
    """Embedder should handle unicode in code."""
    texts = ["def greet(): return 'Hello, World!'"]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384
