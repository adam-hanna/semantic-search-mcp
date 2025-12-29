"""Tests for code chunker module."""
from pathlib import Path

import pytest

from semantic_search_mcp.chunker import CodeChunker, Chunk


@pytest.fixture
def chunker():
    """Create a code chunker."""
    return CodeChunker(overlap_tokens=50, max_tokens=2000)


def test_chunker_extracts_python_functions(chunker, sample_python_file: Path):
    """Chunker should extract Python functions."""
    chunks = chunker.chunk_file(sample_python_file)

    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(function_chunks) >= 1

    # Should find binary_search
    names = [c.name for c in function_chunks]
    assert "binary_search" in names


def test_chunker_extracts_python_classes(chunker, sample_python_file: Path):
    """Chunker should extract Python classes."""
    chunks = chunker.chunk_file(sample_python_file)

    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1

    names = [c.name for c in class_chunks]
    assert "UserService" in names


def test_chunker_extracts_typescript_functions(chunker, sample_typescript_file: Path):
    """Chunker should extract TypeScript functions."""
    chunks = chunker.chunk_file(sample_typescript_file)

    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    names = [c.name for c in function_chunks]
    assert "fetchUser" in names


def test_chunker_extracts_typescript_classes(chunker, sample_typescript_file: Path):
    """Chunker should extract TypeScript classes."""
    chunks = chunker.chunk_file(sample_typescript_file)

    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    names = [c.name for c in class_chunks]
    assert "AuthService" in names


def test_chunker_includes_line_numbers(chunker, sample_python_file: Path):
    """Chunks should include start and end line numbers."""
    chunks = chunker.chunk_file(sample_python_file)

    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_chunker_detects_language(chunker, sample_python_file: Path, sample_typescript_file: Path):
    """Chunker should detect file language."""
    py_chunks = chunker.chunk_file(sample_python_file)
    ts_chunks = chunker.chunk_file(sample_typescript_file)

    assert all(c.language == "python" for c in py_chunks)
    assert all(c.language == "typescript" for c in ts_chunks)


def test_chunker_handles_empty_file(chunker, temp_dir: Path):
    """Chunker should handle empty files gracefully."""
    empty_file = temp_dir / "empty.py"
    empty_file.write_text("")

    chunks = chunker.chunk_file(empty_file)
    assert chunks == []


def test_chunker_handles_unsupported_extension(chunker, temp_dir: Path):
    """Chunker should handle unsupported file types."""
    txt_file = temp_dir / "readme.txt"
    txt_file.write_text("This is a readme file")

    # Should return empty or fall back to text chunking
    chunks = chunker.chunk_file(txt_file)
    # Either empty or chunked as plain text
    assert isinstance(chunks, list)
