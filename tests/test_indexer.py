# tests/test_indexer.py
"""Tests for file indexer module."""
import hashlib
from pathlib import Path

import pytest

from semantic_search_mcp.config import Config
from semantic_search_mcp.database import Database
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.indexer import FileIndexer


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )


@pytest.fixture
def indexer(temp_dir: Path, config):
    """Create file indexer."""
    db = Database(temp_dir / "test.db", embedding_dim=384)
    embedder = Embedder(model_name=config.embedding_model, embedding_dim=config.embedding_dim)
    return FileIndexer(db, embedder, temp_dir)


def test_indexer_indexes_single_file(indexer, sample_python_file: Path):
    """Indexer should index a single file."""
    result = indexer.index_file(sample_python_file)

    assert result["status"] == "indexed"
    assert result["chunks"] > 0


def test_indexer_stores_file_hash(indexer, sample_python_file: Path):
    """Indexer should store file content hash."""
    indexer.index_file(sample_python_file)

    file_record = indexer.db.get_file(str(sample_python_file))
    assert file_record is not None

    # Verify hash matches
    content = sample_python_file.read_text()
    expected_hash = hashlib.sha256(content.encode()).hexdigest()
    assert file_record["content_hash"] == expected_hash


def test_indexer_skips_unchanged_files(indexer, sample_python_file: Path):
    """Indexer should skip files that haven't changed."""
    # First index
    result1 = indexer.index_file(sample_python_file)
    assert result1["status"] == "indexed"

    # Second index (unchanged)
    result2 = indexer.index_file(sample_python_file)
    assert result2["status"] == "skipped"


def test_indexer_reindexes_changed_files(indexer, sample_python_file: Path):
    """Indexer should reindex files that have changed."""
    # First index
    indexer.index_file(sample_python_file)

    # Modify file
    original = sample_python_file.read_text()
    sample_python_file.write_text(original + "\n\ndef new_function(): pass\n")

    # Reindex
    result = indexer.index_file(sample_python_file)
    assert result["status"] == "indexed"


def test_indexer_force_reindex(indexer, sample_python_file: Path):
    """Indexer should reindex when force=True."""
    indexer.index_file(sample_python_file)

    result = indexer.index_file(sample_python_file, force=True)
    assert result["status"] == "indexed"


def test_indexer_handles_binary_files(indexer, temp_dir: Path):
    """Indexer should skip binary files gracefully."""
    binary_file = temp_dir / "binary.py"
    binary_file.write_bytes(b"\x00\x01\x02\x03")

    result = indexer.index_file(binary_file)
    assert result["status"] == "skipped"
    assert "binary" in result.get("reason", "").lower() or result["chunks"] == 0


def test_indexer_removes_deleted_file(indexer, sample_python_file: Path):
    """Indexer should remove file from index when deleted."""
    indexer.index_file(sample_python_file)

    # Delete file
    sample_python_file.unlink()

    # Remove from index
    indexer.remove_file(sample_python_file)

    file_record = indexer.db.get_file(str(sample_python_file))
    assert file_record is None


def test_indexer_full_index(indexer, temp_dir: Path):
    """Indexer should index all files in directory."""
    # Create multiple files
    (temp_dir / "a.py").write_text("def a(): pass")
    (temp_dir / "b.py").write_text("def b(): pass")
    (temp_dir / "sub").mkdir()
    (temp_dir / "sub" / "c.py").write_text("def c(): pass")

    stats = indexer.index_directory(temp_dir)

    assert stats["files_indexed"] >= 3
