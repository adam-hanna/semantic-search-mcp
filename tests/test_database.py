# tests/test_database.py
"""Tests for database module."""
import sqlite3
from pathlib import Path

import pytest

from semantic_search_mcp.database import Database


def test_database_creates_tables(temp_dir: Path):
    """Database should create all required tables on init."""
    db_path = temp_dir / "test.db"
    db = Database(db_path)

    # Check tables exist
    tables = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [t[0] for t in tables]

    assert "files" in table_names
    assert "chunks" in table_names
    assert "chunks_fts" in table_names
    assert "index_meta" in table_names

    db.close()


def test_database_creates_vec_table(temp_dir: Path):
    """Database should create vec_chunks virtual table."""
    db_path = temp_dir / "test.db"
    db = Database(db_path)

    # Check virtual table exists
    result = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
    ).fetchone()

    assert result is not None
    db.close()


def test_database_uses_wal_mode(temp_dir: Path):
    """Database should use WAL journal mode."""
    db_path = temp_dir / "test.db"
    db = Database(db_path)

    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"

    db.close()


def test_database_stores_and_retrieves_meta(temp_dir: Path):
    """Database should store and retrieve metadata."""
    db_path = temp_dir / "test.db"
    db = Database(db_path)

    db.set_meta("model_name", "test-model")
    db.set_meta("schema_version", "1")

    assert db.get_meta("model_name") == "test-model"
    assert db.get_meta("schema_version") == "1"
    assert db.get_meta("nonexistent") is None

    db.close()


def test_database_upserts_file(temp_dir: Path):
    """Database should insert and update file records."""
    db_path = temp_dir / "test.db"
    db = Database(db_path)

    file_id = db.upsert_file("/path/to/file.py", "abc123", "python")
    assert file_id == 1

    # Update same file
    file_id2 = db.upsert_file("/path/to/file.py", "def456", "python")
    assert file_id2 == 1  # Same ID

    # Check hash was updated
    row = db.conn.execute(
        "SELECT content_hash FROM files WHERE id = ?", (file_id,)
    ).fetchone()
    assert row[0] == "def456"

    db.close()


def test_database_deletes_file_cascades(temp_dir: Path):
    """Deleting a file should cascade to chunks and vec_chunks."""
    db_path = temp_dir / "test.db"
    db = Database(db_path)

    file_id = db.upsert_file("/path/to/file.py", "abc123", "python")
    chunk_id = db.insert_chunk(
        file_id=file_id,
        content="def foo(): pass",
        chunk_type="function",
        name="foo",
        start_line=1,
        end_line=1,
    )
    db.insert_embedding(
        chunk_id=chunk_id,
        embedding=[0.1] * 768,
        language="python",
        chunk_type="function",
        file_path="/path/to/file.py",
        name="foo",
        preview="def foo(): pass",
    )

    # Delete file
    db.delete_file("/path/to/file.py")

    # Check chunks deleted
    chunks = db.conn.execute("SELECT * FROM chunks WHERE file_id = ?", (file_id,)).fetchall()
    assert len(chunks) == 0

    # Check embeddings deleted
    embeddings = db.conn.execute("SELECT * FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)).fetchall()
    assert len(embeddings) == 0

    db.close()
