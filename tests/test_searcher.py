# tests/test_searcher.py
"""Tests for hybrid search module."""
from pathlib import Path

import pytest

from semantic_search_mcp.database import Database
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.searcher import HybridSearcher, SearchResult


@pytest.fixture
def embedder():
    """Create embedder with small model for tests."""
    return Embedder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=384)


@pytest.fixture
def db_with_chunks(temp_dir: Path, embedder):
    """Create database with sample chunks."""
    db = Database(temp_dir / "test.db", embedding_dim=384)

    # Insert sample chunks
    chunks = [
        ("def binary_search(arr, target): ...", "function", "binary_search", "python"),
        ("class UserService: def get_user(self): ...", "class", "UserService", "python"),
        ("async def fetch_data(url): ...", "function", "fetch_data", "python"),
        ("def authenticate_user(username, password): ...", "function", "authenticate_user", "python"),
        ("class DatabaseConnection: def connect(self): ...", "class", "DatabaseConnection", "python"),
    ]

    file_id = db.upsert_file("/test/sample.py", "hash123", "python")

    for i, (content, chunk_type, name, language) in enumerate(chunks):
        chunk_id = db.insert_chunk(
            file_id=file_id,
            content=content,
            chunk_type=chunk_type,
            name=name,
            start_line=i * 10 + 1,
            end_line=i * 10 + 5,
        )

        embedding = embedder.embed([content])[0]
        db.insert_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
            language=language,
            chunk_type=chunk_type,
            file_path="/test/sample.py",
            name=name,
            preview=content[:100],
        )

    yield db
    db.close()


def test_searcher_vector_search(db_with_chunks, embedder):
    """Searcher should find results via vector similarity."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    results = searcher.search("find element in sorted array", max_results=3)

    assert len(results) > 0
    assert any("binary_search" in r.name for r in results if r.name)


def test_searcher_fts_search(db_with_chunks, embedder):
    """Searcher should find results via FTS keyword match."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    results = searcher.search("UserService", max_results=3)

    assert len(results) > 0
    assert any(r.name == "UserService" for r in results)


def test_searcher_hybrid_combines_results(db_with_chunks, embedder):
    """Hybrid search should combine vector and FTS results."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    # Query that matches both semantically and by keyword
    results = searcher.search("user authentication", max_results=5)

    assert len(results) > 0
    # Should find authenticate_user and possibly UserService
    names = [r.name for r in results if r.name]
    assert "authenticate_user" in names or "UserService" in names


def test_searcher_respects_min_score(db_with_chunks, embedder):
    """Searcher should filter out low-scoring results."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    # Very high min_score should return fewer results
    results = searcher.search("binary search", max_results=10, min_score=0.9)

    for result in results:
        assert result.score >= 0.9


def test_searcher_filters_by_language(db_with_chunks, embedder):
    """Searcher should filter by language."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    results = searcher.search("function", max_results=10, language="python")

    for result in results:
        assert result.language == "python"


def test_searcher_filters_by_file_pattern(db_with_chunks, embedder):
    """Searcher should filter by file pattern."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    results = searcher.search("function", max_results=10, file_pattern="**/sample.py")

    for result in results:
        assert "sample.py" in result.file_path


def test_searcher_returns_search_result_objects(db_with_chunks, embedder):
    """Results should be SearchResult objects with all fields."""
    searcher = HybridSearcher(db_with_chunks, embedder, rrf_k=60)

    results = searcher.search("search", max_results=1)

    assert len(results) >= 1
    result = results[0]

    assert isinstance(result, SearchResult)
    assert result.file_path is not None
    assert result.chunk_type is not None
    assert result.score >= 0
    assert result.start_line >= 1
