# Semantic Code Search MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MCP server that provides semantic code search using local embeddings, hybrid vector+FTS search, and incremental file watching.

**Architecture:** FastEmbed generates 768-dim embeddings from code chunks extracted via tree-sitter. sqlite-vec stores vectors alongside FTS5 for hybrid search with Reciprocal Rank Fusion. watchfiles monitors the codebase for incremental updates through a bounded async queue with single-writer pattern.

**Tech Stack:** Python 3.11+, FastEmbed (jina-embeddings-v2-base-code), sqlite-vec, FTS5, tree-sitter (py-tree-sitter-languages), watchfiles, pathspec, MCP Python SDK (FastMCP)

---

## Project Structure

```
code-rag/
├── pyproject.toml
├── src/
│   └── code_rag/
│       ├── __init__.py
│       ├── server.py          # MCP server entry point
│       ├── config.py          # Configuration management
│       ├── database.py        # SQLite + sqlite-vec + FTS5 schema
│       ├── embedder.py        # FastEmbed wrapper
│       ├── chunker.py         # Tree-sitter code chunking
│       ├── indexer.py         # File indexing orchestration
│       ├── searcher.py        # Hybrid search with RRF
│       ├── watcher.py         # File system monitoring
│       └── gitignore.py       # Gitignore filtering
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures
    ├── test_database.py
    ├── test_embedder.py
    ├── test_chunker.py
    ├── test_indexer.py
    ├── test_searcher.py
    ├── test_watcher.py
    └── test_server.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/code_rag/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml with all dependencies**

```toml
[project]
name = "code-rag"
version = "0.1.0"
description = "MCP semantic code search server"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.0.0",
    "fastembed>=0.4.0",
    "sqlite-vec>=0.1.6",
    "watchfiles>=1.0.0",
    "pathspec>=0.12.0",
    "tree-sitter>=0.23.0",
    "tree-sitter-languages>=1.10.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
]

[project.scripts]
code-rag = "code_rag.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create package init file**

```python
# src/code_rag/__init__.py
"""MCP semantic code search server."""
__version__ = "0.1.0"
```

**Step 3: Create test package init**

```python
# tests/__init__.py
"""Test suite for code-rag."""
```

**Step 4: Create conftest with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    code = '''
def binary_search(arr: list[int], target: int) -> int:
    """Find target in sorted array using binary search."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        """Fetch user by ID."""
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
'''
    file_path = temp_dir / "sample.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_typescript_file(temp_dir: Path) -> Path:
    """Create a sample TypeScript file for testing."""
    code = '''
interface User {
    id: number;
    name: string;
    email: string;
}

async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch user: ${response.status}`);
    }
    return response.json();
}

export class AuthService {
    private token: string | null = null;

    async login(username: string, password: string): Promise<boolean> {
        // Authentication logic here
        return true;
    }
}
'''
    file_path = temp_dir / "sample.ts"
    file_path.write_text(code)
    return file_path
```

**Step 5: Install dependencies**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pip install -e ".[dev]"`
Expected: Successfully installed all dependencies

**Step 6: Verify pytest runs**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest --collect-only`
Expected: "no tests ran" (collection succeeds with 0 tests)

**Step 7: Commit**

```bash
git init
git add pyproject.toml src/ tests/
git commit -m "chore: initialize project with dependencies and test fixtures"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `src/code_rag/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test for config loading**

```python
# tests/test_config.py
"""Tests for configuration module."""
import os
from pathlib import Path

import pytest

from code_rag.config import Config, load_config


def test_config_defaults():
    """Config should have sensible defaults."""
    config = Config()

    assert config.embedding_model == "jinaai/jina-embeddings-v2-base-code"
    assert config.embedding_dim == 768
    assert config.db_path == Path(".code-rag/index.db")
    assert config.chunk_overlap_tokens == 50
    assert config.max_chunk_tokens == 2000
    assert config.search_default_limit == 10
    assert config.search_min_score == 0.3
    assert config.rrf_k == 60
    assert config.queue_max_size == 1000
    assert config.debounce_ms == 1000


def test_config_from_env(monkeypatch):
    """Config should read from environment variables."""
    monkeypatch.setenv("CODE_RAG_DB_PATH", "/custom/path/index.db")
    monkeypatch.setenv("CODE_RAG_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("CODE_RAG_SEARCH_MIN_SCORE", "0.5")

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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.config'"

**Step 3: Write minimal implementation**

```python
# src/code_rag/config.py
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

        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)


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
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_config.py -v`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add src/code_rag/config.py tests/test_config.py
git commit -m "feat: add configuration module with env var support"
```

---

## Task 3: Database Module

**Files:**
- Create: `src/code_rag/database.py`
- Create: `tests/test_database.py`

**Step 1: Write the failing test for database initialization**

```python
# tests/test_database.py
"""Tests for database module."""
import sqlite3
from pathlib import Path

import pytest

from code_rag.database import Database


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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_database.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.database'"

**Step 3: Write implementation**

```python
# src/code_rag/database.py
"""SQLite database with sqlite-vec and FTS5 for hybrid search."""
import sqlite3
import struct
from pathlib import Path
from typing import Optional

import sqlite_vec


SCHEMA_VERSION = "1"


def serialize_embedding(embedding: list[float]) -> bytes:
    """Convert float list to bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    """Convert bytes back to float list."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


class Database:
    """SQLite database manager with vector and FTS support."""

    def __init__(self, db_path: Path, embedding_dim: int = 768):
        """Initialize database connection and create tables."""
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Configure for performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB

        self._create_tables()

    def _create_tables(self):
        """Create all required tables if they don't exist."""
        self.conn.executescript(f"""
            -- Metadata for versioning
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Track source files
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                language TEXT,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);

            -- Store code chunks with metadata
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                chunk_type TEXT,
                name TEXT,
                start_line INTEGER,
                end_line INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

            -- FTS5 for keyword search
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                name,
                file_path,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, name, file_path)
                SELECT NEW.id, NEW.content, NEW.name,
                       (SELECT path FROM files WHERE id = NEW.file_id);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, name, file_path)
                VALUES('delete', OLD.id, OLD.content, OLD.name,
                       (SELECT path FROM files WHERE id = OLD.file_id));
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, name, file_path)
                VALUES('delete', OLD.id, OLD.content, OLD.name,
                       (SELECT path FROM files WHERE id = OLD.file_id));
                INSERT INTO chunks_fts(rowid, content, name, file_path)
                SELECT NEW.id, NEW.content, NEW.name,
                       (SELECT path FROM files WHERE id = NEW.file_id);
            END;

            -- Vector embeddings
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{self.embedding_dim}] distance_metric=cosine,
                language TEXT,
                chunk_type TEXT,
                +file_path TEXT,
                +name TEXT,
                +preview TEXT
            );
        """)
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        row = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        """Set a metadata value."""
        self.conn.execute(
            """INSERT INTO index_meta (key, value, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
            (key, value)
        )
        self.conn.commit()

    def get_file(self, path: str) -> Optional[dict]:
        """Get file record by path."""
        row = self.conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_file(self, path: str, content_hash: str, language: str) -> int:
        """Insert or update a file record, returning its ID."""
        self.conn.execute(
            """INSERT INTO files (path, content_hash, language, last_indexed)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(path) DO UPDATE SET
                   content_hash=excluded.content_hash,
                   language=excluded.language,
                   last_indexed=CURRENT_TIMESTAMP""",
            (path, content_hash, language)
        )
        row = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (path,)
        ).fetchone()
        self.conn.commit()
        return row["id"]

    def delete_file(self, path: str):
        """Delete a file and its chunks (cascades via FK)."""
        # First delete vec_chunks manually (virtual tables don't cascade)
        self.conn.execute(
            """DELETE FROM vec_chunks WHERE chunk_id IN
               (SELECT c.id FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE f.path = ?)""",
            (path,)
        )
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()

    def delete_chunks_for_file(self, file_id: int):
        """Delete all chunks for a file."""
        self.conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
            (file_id,)
        )
        self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self.conn.commit()

    def insert_chunk(
        self,
        file_id: int,
        content: str,
        chunk_type: str,
        name: Optional[str],
        start_line: int,
        end_line: int,
    ) -> int:
        """Insert a code chunk, returning its ID."""
        cursor = self.conn.execute(
            """INSERT INTO chunks (file_id, content, chunk_type, name, start_line, end_line)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, content, chunk_type, name, start_line, end_line)
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_embedding(
        self,
        chunk_id: int,
        embedding: list[float],
        language: str,
        chunk_type: str,
        file_path: str,
        name: Optional[str],
        preview: str,
    ):
        """Insert a vector embedding for a chunk."""
        self.conn.execute(
            """INSERT INTO vec_chunks (chunk_id, embedding, language, chunk_type, file_path, name, preview)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, serialize_embedding(embedding), language, chunk_type, file_path, name, preview)
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Get index statistics."""
        files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "files": files,
            "chunks": chunks,
            "schema_version": self.get_meta("schema_version"),
            "model_name": self.get_meta("model_name"),
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_database.py -v`
Expected: PASS (6 passed)

**Step 5: Commit**

```bash
git add src/code_rag/database.py tests/test_database.py
git commit -m "feat: add database module with sqlite-vec and FTS5"
```

---

## Task 4: Embedder Module

**Files:**
- Create: `src/code_rag/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing test for embedder**

```python
# tests/test_embedder.py
"""Tests for embedder module."""
import pytest

from code_rag.embedder import Embedder


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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_embedder.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.embedder'"

**Step 3: Write implementation**

```python
# src/code_rag/embedder.py
"""FastEmbed wrapper for code embeddings."""
import math
from typing import Optional

from fastembed import TextEmbedding


class Embedder:
    """Generate embeddings for code snippets using FastEmbed."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        embedding_dim: int = 768,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the embedding model.

        Args:
            model_name: Name of the embedding model to use
            embedding_dim: Expected embedding dimension
            cache_dir: Optional cache directory for model files
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model: Optional[TextEmbedding] = None
        self._cache_dir = cache_dir

    @property
    def model(self) -> TextEmbedding:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
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

        # FastEmbed returns a generator, convert to list
        embeddings = list(self.model.embed(texts))
        return [list(emb) for emb in embeddings]

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
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_embedder.py -v`
Expected: PASS (5 passed)

**Step 5: Commit**

```bash
git add src/code_rag/embedder.py tests/test_embedder.py
git commit -m "feat: add embedder module with FastEmbed"
```

---

## Task 5: Code Chunker Module (Tree-sitter)

**Files:**
- Create: `src/code_rag/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write the failing test for code chunking**

```python
# tests/test_chunker.py
"""Tests for code chunker module."""
from pathlib import Path

import pytest

from code_rag.chunker import CodeChunker, Chunk


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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_chunker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.chunker'"

**Step 3: Write implementation**

```python
# src/code_rag/chunker.py
"""Tree-sitter based code chunking with overlap."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter_languages


@dataclass
class Chunk:
    """A chunk of code extracted from a file."""
    content: str
    chunk_type: str  # 'function', 'class', 'method', 'module'
    name: Optional[str]
    start_line: int
    end_line: int
    language: str


# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
}

# Node types to extract per language
EXTRACTABLE_NODES = {
    "python": ["function_definition", "class_definition", "async_function_definition"],
    "javascript": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
    "tsx": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
    "java": ["method_declaration", "class_declaration", "interface_declaration"],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item"],
    "c": ["function_definition", "struct_specifier"],
    "cpp": ["function_definition", "class_specifier", "struct_specifier"],
    "c_sharp": ["method_declaration", "class_declaration", "interface_declaration"],
    "ruby": ["method", "class", "module"],
    "php": ["function_definition", "class_declaration", "method_declaration"],
}


class CodeChunker:
    """Extract semantic code chunks using tree-sitter."""

    def __init__(self, overlap_tokens: int = 50, max_tokens: int = 2000):
        """Initialize chunker.

        Args:
            overlap_tokens: Number of tokens to overlap between chunks
            max_tokens: Maximum tokens per chunk
        """
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self._parsers: dict = {}

    def _get_parser(self, language: str):
        """Get or create a parser for the given language."""
        if language not in self._parsers:
            try:
                self._parsers[language] = tree_sitter_languages.get_parser(language)
            except Exception:
                return None
        return self._parsers[language]

    def _detect_language(self, filepath: Path) -> Optional[str]:
        """Detect language from file extension."""
        return EXTENSION_TO_LANGUAGE.get(filepath.suffix.lower())

    def _get_node_name(self, node, source_bytes: bytes, language: str) -> Optional[str]:
        """Extract the name from a function/class node."""
        # Look for identifier or name child
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
            # Python: function name is first identifier
            if language == "python" and child.type == "identifier":
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

        # For some languages, name is nested deeper
        for child in node.children:
            if child.type in ("declarator", "function_declarator"):
                return self._get_node_name(child, source_bytes, language)

        return None

    def _node_to_chunk_type(self, node_type: str) -> str:
        """Convert tree-sitter node type to our chunk type."""
        if "class" in node_type or "struct" in node_type or "interface" in node_type:
            return "class"
        if "method" in node_type:
            return "method"
        if "function" in node_type or "arrow" in node_type:
            return "function"
        if "impl" in node_type or "enum" in node_type or "type" in node_type:
            return "class"
        return "block"

    def _add_overlap(self, content: str, lines: list[str], start_line: int, end_line: int) -> tuple[str, int, int]:
        """Add overlap context before the chunk."""
        # Approximate tokens as words (rough estimate)
        words_per_line = 10  # Rough average
        overlap_lines = max(1, self.overlap_tokens // words_per_line)

        # Add lines before
        actual_start = max(0, start_line - 1 - overlap_lines)
        prefix_lines = lines[actual_start:start_line - 1]

        if prefix_lines:
            content = "\n".join(prefix_lines) + "\n" + content
            start_line = actual_start + 1

        return content, start_line, end_line

    def chunk_file(self, filepath: Path) -> list[Chunk]:
        """Extract semantic chunks from a source file.

        Args:
            filepath: Path to the source file

        Returns:
            List of Chunk objects
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return []

        language = self._detect_language(filepath)
        if not language:
            return []

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if not content.strip():
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        source_bytes = content.encode("utf-8")
        lines = content.split("\n")

        try:
            tree = parser.parse(source_bytes)
        except Exception:
            return []

        chunks = []
        extractable = EXTRACTABLE_NODES.get(language, [])

        def visit(node):
            """Recursively visit nodes and extract chunks."""
            if node.type in extractable:
                chunk_content = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Add overlap
                chunk_content, start_line, end_line = self._add_overlap(
                    chunk_content, lines, start_line, end_line
                )

                name = self._get_node_name(node, source_bytes, language)
                chunk_type = self._node_to_chunk_type(node.type)

                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=chunk_type,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                ))

            # Visit children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        # If no chunks extracted, treat whole file as one chunk
        if not chunks and len(content) > 50:
            chunks.append(Chunk(
                content=content[:self.max_tokens * 4],  # Rough char limit
                chunk_type="module",
                name=filepath.stem,
                start_line=1,
                end_line=len(lines),
                language=language,
            ))

        return chunks
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_chunker.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add src/code_rag/chunker.py tests/test_chunker.py
git commit -m "feat: add tree-sitter code chunker with overlap"
```

---

## Task 6: Gitignore Filter Module

**Files:**
- Create: `src/code_rag/gitignore.py`
- Create: `tests/test_gitignore.py`

**Step 1: Write the failing test**

```python
# tests/test_gitignore.py
"""Tests for gitignore filtering module."""
from pathlib import Path

import pytest

from code_rag.gitignore import GitignoreFilter


@pytest.fixture
def project_with_gitignore(temp_dir: Path) -> Path:
    """Create a project structure with .gitignore."""
    # Create .gitignore
    (temp_dir / ".gitignore").write_text("""
# Dependencies
node_modules/
__pycache__/
*.pyc

# Build output
dist/
build/

# IDE
.idea/
.vscode/

# Custom
secret.py
""")

    # Create some files
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "main.py").write_text("print('hello')")
    (temp_dir / "node_modules").mkdir()
    (temp_dir / "node_modules" / "package.json").write_text("{}")
    (temp_dir / "__pycache__").mkdir()
    (temp_dir / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"bytecode")
    (temp_dir / "secret.py").write_text("API_KEY = 'secret'")

    return temp_dir


def test_filter_allows_source_files(project_with_gitignore: Path):
    """Filter should allow regular source files."""
    filter = GitignoreFilter(project_with_gitignore)

    assert filter.should_index(project_with_gitignore / "src" / "main.py")


def test_filter_ignores_node_modules(project_with_gitignore: Path):
    """Filter should ignore node_modules."""
    filter = GitignoreFilter(project_with_gitignore)

    assert not filter.should_index(project_with_gitignore / "node_modules" / "package.json")


def test_filter_ignores_pycache(project_with_gitignore: Path):
    """Filter should ignore __pycache__."""
    filter = GitignoreFilter(project_with_gitignore)

    assert not filter.should_index(project_with_gitignore / "__pycache__" / "main.cpython-311.pyc")


def test_filter_ignores_gitignored_files(project_with_gitignore: Path):
    """Filter should ignore files matching .gitignore patterns."""
    filter = GitignoreFilter(project_with_gitignore)

    assert not filter.should_index(project_with_gitignore / "secret.py")


def test_filter_ignores_non_code_extensions(temp_dir: Path):
    """Filter should ignore non-code file extensions."""
    filter = GitignoreFilter(temp_dir)

    (temp_dir / "readme.md").write_text("# README")
    (temp_dir / "data.json").write_text("{}")
    (temp_dir / "image.png").write_bytes(b"PNG")

    assert not filter.should_index(temp_dir / "readme.md")
    assert not filter.should_index(temp_dir / "data.json")
    assert not filter.should_index(temp_dir / "image.png")


def test_filter_allows_code_extensions(temp_dir: Path):
    """Filter should allow code file extensions."""
    filter = GitignoreFilter(temp_dir)

    for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
        file = temp_dir / f"code{ext}"
        file.write_text("code")
        assert filter.should_index(file), f"Should index {ext} files"


def test_filter_always_ignores_git_directory(temp_dir: Path):
    """Filter should always ignore .git directory."""
    filter = GitignoreFilter(temp_dir)

    (temp_dir / ".git").mkdir()
    (temp_dir / ".git" / "config").write_text("[core]")

    assert not filter.should_index(temp_dir / ".git" / "config")


def test_filter_handles_nested_gitignore(temp_dir: Path):
    """Filter should respect nested .gitignore files."""
    # Root gitignore
    (temp_dir / ".gitignore").write_text("*.log")

    # Nested gitignore
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / ".gitignore").write_text("temp/")
    (temp_dir / "src" / "temp").mkdir()
    (temp_dir / "src" / "temp" / "cache.py").write_text("")
    (temp_dir / "src" / "main.py").write_text("")

    filter = GitignoreFilter(temp_dir)

    assert filter.should_index(temp_dir / "src" / "main.py")
    assert not filter.should_index(temp_dir / "src" / "temp" / "cache.py")
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_gitignore.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.gitignore'"

**Step 3: Write implementation**

```python
# src/code_rag/gitignore.py
"""Gitignore-aware file filtering using pathspec."""
from pathlib import Path
from typing import Optional

import pathspec


# Always ignore these directories regardless of .gitignore
ALWAYS_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "target",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage",
    ".coverage",
    "htmlcov",
    ".tox",
    ".nox",
    "*.egg-info",
}

# Code file extensions to index
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".lua",
    ".r",
    ".R",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".vim",
    ".el",
}


class GitignoreFilter:
    """Filter files based on .gitignore patterns and code extensions."""

    def __init__(self, root_dir: Path):
        """Initialize with root directory.

        Args:
            root_dir: Root directory to scan for .gitignore files
        """
        self.root = Path(root_dir).resolve()
        self.specs: dict[Path, pathspec.GitIgnoreSpec] = {}
        self._load_all_gitignores()

    def _load_all_gitignores(self):
        """Load all .gitignore files in the directory tree."""
        for gitignore in self.root.rglob(".gitignore"):
            try:
                with open(gitignore, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                self.specs[gitignore.parent] = pathspec.GitIgnoreSpec.from_lines(lines)
            except Exception:
                continue

    def _is_always_ignored(self, path: Path) -> bool:
        """Check if path is in an always-ignored directory."""
        for part in path.parts:
            if part in ALWAYS_IGNORE_DIRS:
                return True
            # Handle patterns like *.egg-info
            for pattern in ALWAYS_IGNORE_DIRS:
                if "*" in pattern:
                    import fnmatch
                    if fnmatch.fnmatch(part, pattern):
                        return True
        return False

    def _is_code_file(self, path: Path) -> bool:
        """Check if file has a code extension."""
        return path.suffix.lower() in CODE_EXTENSIONS

    def _matches_gitignore(self, path: Path) -> bool:
        """Check if path matches any gitignore pattern."""
        for gitignore_dir, spec in self.specs.items():
            try:
                # Get relative path from gitignore location
                rel_path = path.relative_to(gitignore_dir)
                if spec.match_file(str(rel_path)):
                    return True
            except ValueError:
                # Path is not relative to this gitignore
                continue
        return False

    def should_index(self, filepath: Path) -> bool:
        """Determine if a file should be indexed.

        Args:
            filepath: Path to check

        Returns:
            True if file should be indexed
        """
        path = Path(filepath).resolve()

        # Must be a file
        if not path.is_file():
            return False

        # Must have code extension
        if not self._is_code_file(path):
            return False

        # Check always-ignored directories
        if self._is_always_ignored(path):
            return False

        # Check gitignore patterns
        if self._matches_gitignore(path):
            return False

        return True

    def get_indexable_files(self) -> list[Path]:
        """Get all indexable files in the root directory.

        Returns:
            List of file paths that should be indexed
        """
        files = []
        for path in self.root.rglob("*"):
            if self.should_index(path):
                files.append(path)
        return files
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_gitignore.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add src/code_rag/gitignore.py tests/test_gitignore.py
git commit -m "feat: add gitignore filtering with pathspec"
```

---

## Task 7: Hybrid Searcher Module

**Files:**
- Create: `src/code_rag/searcher.py`
- Create: `tests/test_searcher.py`

**Step 1: Write the failing test**

```python
# tests/test_searcher.py
"""Tests for hybrid search module."""
from pathlib import Path

import pytest

from code_rag.database import Database
from code_rag.embedder import Embedder
from code_rag.searcher import HybridSearcher, SearchResult


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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_searcher.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.searcher'"

**Step 3: Write implementation**

```python
# src/code_rag/searcher.py
"""Hybrid vector + FTS search with Reciprocal Rank Fusion."""
import fnmatch
from dataclasses import dataclass
from typing import Optional

from code_rag.database import Database, serialize_embedding
from code_rag.embedder import Embedder


@dataclass
class SearchResult:
    """A search result with relevance score."""
    chunk_id: int
    file_path: str
    name: Optional[str]
    chunk_type: str
    content: str
    preview: str
    start_line: int
    end_line: int
    language: str
    score: float  # Combined RRF score


class HybridSearcher:
    """Combine vector similarity and FTS for hybrid search."""

    def __init__(self, db: Database, embedder: Embedder, rrf_k: int = 60):
        """Initialize hybrid searcher.

        Args:
            db: Database instance
            embedder: Embedder instance
            rrf_k: Reciprocal Rank Fusion constant (default 60)
        """
        self.db = db
        self.embedder = embedder
        self.rrf_k = rrf_k

    def _vector_search(
        self,
        query_embedding: list[float],
        max_results: int,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Perform vector similarity search.

        Returns:
            List of (chunk_id, distance) tuples
        """
        sql = """
            SELECT chunk_id, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
              AND k = ?
        """
        params = [serialize_embedding(query_embedding), max_results * 2]

        if language:
            sql += " AND language = ?"
            params.append(language)
        if chunk_type:
            sql += " AND chunk_type = ?"
            params.append(chunk_type)

        sql += " ORDER BY distance LIMIT ?"
        params.append(max_results * 2)

        rows = self.db.conn.execute(sql, params).fetchall()
        return [(row["chunk_id"], row["distance"]) for row in rows]

    def _fts_search(
        self,
        query: str,
        max_results: int,
    ) -> list[tuple[int, float]]:
        """Perform FTS5 keyword search.

        Returns:
            List of (chunk_id, bm25_score) tuples
        """
        # Escape FTS5 special characters
        safe_query = query.replace('"', '""')

        sql = """
            SELECT rowid, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """

        try:
            rows = self.db.conn.execute(sql, (safe_query, max_results * 2)).fetchall()
            return [(row["rowid"], row["score"]) for row in rows]
        except Exception:
            # FTS query failed, return empty
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[int, float]],
        fts_results: list[tuple[int, float]],
    ) -> dict[int, float]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list

        Args:
            vector_results: (chunk_id, distance) from vector search
            fts_results: (chunk_id, bm25_score) from FTS

        Returns:
            Dict mapping chunk_id to combined RRF score
        """
        scores: dict[int, float] = {}

        # Add vector search scores
        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)

        # Add FTS scores
        for rank, (chunk_id, _) in enumerate(fts_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)

        return scores

    def _get_chunk_details(self, chunk_id: int) -> Optional[dict]:
        """Get full chunk details from database."""
        row = self.db.conn.execute("""
            SELECT c.id, c.content, c.chunk_type, c.name, c.start_line, c.end_line,
                   f.path as file_path, f.language
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.id = ?
        """, (chunk_id,)).fetchone()

        return dict(row) if row else None

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        file_pattern: Optional[str] = None,
        recency_boost: bool = False,
    ) -> list[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Natural language search query
            max_results: Maximum results to return
            min_score: Minimum RRF score threshold (0-1 normalized)
            language: Filter by programming language
            chunk_type: Filter by chunk type
            file_pattern: Glob pattern to filter files
            recency_boost: Boost recent files (not yet implemented)

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Get results from both sources
        vector_results = self._vector_search(
            query_embedding, max_results * 2, language, chunk_type
        )
        fts_results = self._fts_search(query, max_results * 2)

        # Combine with RRF
        combined_scores = self._reciprocal_rank_fusion(vector_results, fts_results)

        # Normalize scores to 0-1 range
        if combined_scores:
            max_score = max(combined_scores.values())
            combined_scores = {k: v / max_score for k, v in combined_scores.items()}

        # Sort by score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Build results
        results = []
        for chunk_id in sorted_ids:
            if len(results) >= max_results:
                break

            score = combined_scores[chunk_id]
            if score < min_score:
                continue

            details = self._get_chunk_details(chunk_id)
            if not details:
                continue

            # Apply file pattern filter
            if file_pattern:
                if not fnmatch.fnmatch(details["file_path"], file_pattern):
                    continue

            results.append(SearchResult(
                chunk_id=chunk_id,
                file_path=details["file_path"],
                name=details["name"],
                chunk_type=details["chunk_type"],
                content=details["content"],
                preview=details["content"][:200],
                start_line=details["start_line"],
                end_line=details["end_line"],
                language=details["language"],
                score=score,
            ))

        return results
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_searcher.py -v`
Expected: PASS (7 passed)

**Step 5: Commit**

```bash
git add src/code_rag/searcher.py tests/test_searcher.py
git commit -m "feat: add hybrid searcher with RRF"
```

---

## Task 8: File Indexer Module

**Files:**
- Create: `src/code_rag/indexer.py`
- Create: `tests/test_indexer.py`

**Step 1: Write the failing test**

```python
# tests/test_indexer.py
"""Tests for file indexer module."""
import hashlib
from pathlib import Path

import pytest

from code_rag.config import Config
from code_rag.database import Database
from code_rag.embedder import Embedder
from code_rag.indexer import FileIndexer


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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_indexer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.indexer'"

**Step 3: Write implementation**

```python
# src/code_rag/indexer.py
"""File indexing orchestration."""
import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from code_rag.chunker import CodeChunker
from code_rag.database import Database
from code_rag.embedder import Embedder
from code_rag.gitignore import GitignoreFilter


logger = logging.getLogger(__name__)


class FileIndexer:
    """Orchestrates file indexing: chunking, embedding, and storage."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        root_dir: Path,
        chunk_overlap: int = 50,
        max_chunk_tokens: int = 2000,
    ):
        """Initialize indexer.

        Args:
            db: Database instance
            embedder: Embedder instance
            root_dir: Root directory for indexing
            chunk_overlap: Token overlap between chunks
            max_chunk_tokens: Maximum tokens per chunk
        """
        self.db = db
        self.embedder = embedder
        self.root_dir = Path(root_dir).resolve()
        self.chunker = CodeChunker(overlap_tokens=chunk_overlap, max_tokens=max_chunk_tokens)
        self.gitignore = GitignoreFilter(root_dir)

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_binary(self, content: bytes) -> bool:
        """Check if content appears to be binary."""
        # Check for null bytes (common in binary files)
        if b"\x00" in content[:8192]:
            return True
        # Check for high ratio of non-text bytes
        try:
            content[:8192].decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True

    def _read_file_safe(self, filepath: Path) -> Optional[str]:
        """Read file content, handling encoding and binary files."""
        try:
            content_bytes = filepath.read_bytes()
            if self._is_binary(content_bytes):
                return None
            return content_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return None

    def index_file(self, filepath: Path, force: bool = False) -> dict:
        """Index a single file.

        Args:
            filepath: Path to file
            force: Force reindex even if unchanged

        Returns:
            Dict with status and chunk count
        """
        filepath = Path(filepath).resolve()
        path_str = str(filepath)

        # Read content
        content = self._read_file_safe(filepath)
        if content is None:
            return {"status": "skipped", "reason": "binary or unreadable", "chunks": 0}

        if not content.strip():
            return {"status": "skipped", "reason": "empty", "chunks": 0}

        # Check if file needs reindexing
        content_hash = self._hash_content(content)
        existing = self.db.get_file(path_str)

        if existing and existing["content_hash"] == content_hash and not force:
            return {"status": "skipped", "reason": "unchanged", "chunks": 0}

        # Chunk the file
        chunks = self.chunker.chunk_file(filepath)
        if not chunks:
            return {"status": "skipped", "reason": "no chunks extracted", "chunks": 0}

        # Generate embeddings
        try:
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.error(f"Embedding failed for {filepath}: {e}")
            return {"status": "error", "reason": str(e), "chunks": 0}

        # Store in database (atomic transaction)
        language = chunks[0].language if chunks else "unknown"

        try:
            file_id = self.db.upsert_file(path_str, content_hash, language)

            # Clear old chunks
            self.db.delete_chunks_for_file(file_id)

            # Insert new chunks and embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = self.db.insert_chunk(
                    file_id=file_id,
                    content=chunk.content,
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )

                self.db.insert_embedding(
                    chunk_id=chunk_id,
                    embedding=embedding,
                    language=chunk.language,
                    chunk_type=chunk.chunk_type,
                    file_path=path_str,
                    name=chunk.name,
                    preview=chunk.content[:200],
                )

            return {"status": "indexed", "chunks": len(chunks)}

        except Exception as e:
            logger.error(f"Database error for {filepath}: {e}")
            return {"status": "error", "reason": str(e), "chunks": 0}

    def remove_file(self, filepath: Path):
        """Remove a file from the index.

        Args:
            filepath: Path to file
        """
        self.db.delete_file(str(Path(filepath).resolve()))

    def index_directory(
        self,
        directory: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Index all files in a directory.

        Args:
            directory: Directory to index (defaults to root_dir)
            progress_callback: Callback(current, total, message)

        Returns:
            Stats dict with files_indexed, files_skipped, etc.
        """
        directory = Path(directory or self.root_dir).resolve()

        # Get all indexable files
        files = [f for f in directory.rglob("*") if self.gitignore.should_index(f)]
        total = len(files)

        stats = {
            "files_indexed": 0,
            "files_skipped": 0,
            "files_error": 0,
            "total_chunks": 0,
        }

        for i, filepath in enumerate(files):
            if progress_callback:
                progress_callback(i, total, f"Indexing {filepath.name}")

            result = self.index_file(filepath)

            if result["status"] == "indexed":
                stats["files_indexed"] += 1
                stats["total_chunks"] += result["chunks"]
            elif result["status"] == "skipped":
                stats["files_skipped"] += 1
            else:
                stats["files_error"] += 1

        if progress_callback:
            progress_callback(total, total, "Complete")

        return stats

    def needs_reindex(self, filepath: Path) -> bool:
        """Check if a file needs reindexing.

        Args:
            filepath: Path to check

        Returns:
            True if file needs reindexing
        """
        filepath = Path(filepath).resolve()

        content = self._read_file_safe(filepath)
        if content is None:
            return False

        content_hash = self._hash_content(content)
        existing = self.db.get_file(str(filepath))

        if not existing:
            return True

        return existing["content_hash"] != content_hash
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_indexer.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add src/code_rag/indexer.py tests/test_indexer.py
git commit -m "feat: add file indexer with change detection"
```

---

## Task 9: File Watcher Module

**Files:**
- Create: `src/code_rag/watcher.py`
- Create: `tests/test_watcher.py`

**Step 1: Write the failing test**

```python
# tests/test_watcher.py
"""Tests for file watcher module."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.watcher import FileWatcher, ChangeEvent


@pytest.fixture
def mock_indexer():
    """Create mock indexer."""
    indexer = MagicMock()
    indexer.index_file = MagicMock(return_value={"status": "indexed", "chunks": 3})
    indexer.remove_file = MagicMock()
    indexer.gitignore = MagicMock()
    indexer.gitignore.should_index = MagicMock(return_value=True)
    return indexer


def test_watcher_creates_bounded_queue(mock_indexer, temp_dir):
    """Watcher should create a bounded queue."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    assert watcher.queue.maxsize == 100


def test_watcher_queue_event(mock_indexer, temp_dir):
    """Watcher should queue file change events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    event = ChangeEvent(type="modified", path=temp_dir / "test.py")
    watcher.queue_event(event)

    assert not watcher.queue.empty()
    queued = watcher.queue.get_nowait()
    assert queued.type == "modified"


def test_watcher_filters_non_indexable(mock_indexer, temp_dir):
    """Watcher should filter non-indexable files."""
    mock_indexer.gitignore.should_index = MagicMock(return_value=False)
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    event = ChangeEvent(type="modified", path=temp_dir / "node_modules" / "pkg.js")
    watcher.queue_event(event)

    # Event should be filtered out
    assert watcher.queue.empty()


@pytest.mark.asyncio
async def test_watcher_processes_modified_event(mock_indexer, temp_dir):
    """Watcher should call index_file for modified events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    test_file = temp_dir / "test.py"
    test_file.write_text("print('hello')")

    event = ChangeEvent(type="modified", path=test_file)
    await watcher._process_event(event)

    mock_indexer.index_file.assert_called_once_with(test_file)


@pytest.mark.asyncio
async def test_watcher_processes_deleted_event(mock_indexer, temp_dir):
    """Watcher should call remove_file for deleted events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    test_file = temp_dir / "test.py"
    event = ChangeEvent(type="deleted", path=test_file)
    await watcher._process_event(event)

    mock_indexer.remove_file.assert_called_once_with(test_file)


@pytest.mark.asyncio
async def test_watcher_handles_processing_error(mock_indexer, temp_dir):
    """Watcher should handle errors in event processing gracefully."""
    mock_indexer.index_file = MagicMock(side_effect=Exception("Test error"))
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    test_file = temp_dir / "test.py"
    test_file.write_text("print('hello')")

    event = ChangeEvent(type="modified", path=test_file)

    # Should not raise
    await watcher._process_event(event)


def test_watcher_queue_drops_on_full(mock_indexer, temp_dir):
    """When queue is full, oldest events should be dropped."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=2)

    # Fill queue
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "a.py"))
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "b.py"))

    # This should drop the oldest
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "c.py"))

    assert watcher.queue.qsize() == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_watcher.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.watcher'"

**Step 3: Write implementation**

```python
# src/code_rag/watcher.py
"""File system watcher for incremental indexing."""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Optional

from watchfiles import Change, awatch

from code_rag.indexer import FileIndexer


logger = logging.getLogger(__name__)


@dataclass
class ChangeEvent:
    """File change event."""
    type: str  # 'added', 'modified', 'deleted'
    path: Path


class FileWatcher:
    """Watch filesystem for changes and trigger incremental indexing."""

    def __init__(
        self,
        indexer: FileIndexer,
        watch_dir: Path,
        queue_max_size: int = 1000,
        debounce_ms: int = 1000,
    ):
        """Initialize file watcher.

        Args:
            indexer: FileIndexer instance
            watch_dir: Directory to watch
            queue_max_size: Maximum queue size (drops oldest on overflow)
            debounce_ms: Debounce interval in milliseconds
        """
        self.indexer = indexer
        self.watch_dir = Path(watch_dir).resolve()
        self.queue: Queue[ChangeEvent] = Queue(maxsize=queue_max_size)
        self.debounce_ms = debounce_ms
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None

    def queue_event(self, event: ChangeEvent):
        """Queue a change event for processing.

        If the queue is full, drops the oldest event.

        Args:
            event: Change event to queue
        """
        # Filter non-indexable files
        if not self.indexer.gitignore.should_index(event.path):
            return

        try:
            self.queue.put_nowait(event)
        except Full:
            # Drop oldest event
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(event)
            except Exception:
                pass

    async def _process_event(self, event: ChangeEvent):
        """Process a single change event.

        Args:
            event: Change event to process
        """
        try:
            if event.type == "deleted":
                self.indexer.remove_file(event.path)
                logger.info(f"Removed from index: {event.path}")
            else:
                result = self.indexer.index_file(event.path)
                if result["status"] == "indexed":
                    logger.info(f"Indexed: {event.path} ({result['chunks']} chunks)")
                elif result["status"] == "skipped":
                    logger.debug(f"Skipped: {event.path} ({result.get('reason', 'unknown')})")
        except Exception as e:
            logger.error(f"Error processing {event.path}: {e}")

    async def _process_loop(self):
        """Process events from the queue."""
        while self._running:
            try:
                # Non-blocking get with timeout
                event = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.queue.get(timeout=1.0)
                )
                await self._process_event(event)
            except Exception:
                # Queue.get timeout or other error
                continue

    async def _watch_loop(self):
        """Watch filesystem for changes."""
        try:
            async for changes in awatch(
                str(self.watch_dir),
                debounce=self.debounce_ms,
                recursive=True,
                force_polling=False,
            ):
                if not self._running:
                    break

                for change_type, path_str in changes:
                    path = Path(path_str)

                    if change_type == Change.added:
                        event_type = "added"
                    elif change_type == Change.modified:
                        event_type = "modified"
                    elif change_type == Change.deleted:
                        event_type = "deleted"
                    else:
                        continue

                    self.queue_event(ChangeEvent(type=event_type, path=path))

        except Exception as e:
            logger.error(f"Watch loop error: {e}")

    async def start(self):
        """Start watching for file changes."""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Started watching: {self.watch_dir}")

    async def stop(self):
        """Stop watching for file changes."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped watching")
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_watcher.py -v`
Expected: PASS (7 passed)

**Step 5: Commit**

```bash
git add src/code_rag/watcher.py tests/test_watcher.py
git commit -m "feat: add file watcher with bounded queue"
```

---

## Task 10: MCP Server Module

**Files:**
- Create: `src/code_rag/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing test**

```python
# tests/test_server.py
"""Tests for MCP server module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_rag.server import create_server


@pytest.fixture
def mock_components(temp_dir):
    """Create mock components for server testing."""
    with patch("code_rag.server.Database") as MockDB, \
         patch("code_rag.server.Embedder") as MockEmbed, \
         patch("code_rag.server.FileIndexer") as MockIndexer, \
         patch("code_rag.server.HybridSearcher") as MockSearcher:

        mock_db = MagicMock()
        mock_db.get_stats.return_value = {"files": 10, "chunks": 50}
        MockDB.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.is_loaded.return_value = False
        MockEmbed.return_value = mock_embedder

        mock_indexer = MagicMock()
        mock_indexer.index_directory.return_value = {"files_indexed": 5, "total_chunks": 25}
        MockIndexer.return_value = mock_indexer

        mock_searcher = MagicMock()
        MockSearcher.return_value = mock_searcher

        yield {
            "db": mock_db,
            "embedder": mock_embedder,
            "indexer": mock_indexer,
            "searcher": mock_searcher,
            "temp_dir": temp_dir,
        }


def test_server_creates_mcp_instance(mock_components):
    """Server should create an MCP instance."""
    mcp = create_server(mock_components["temp_dir"])

    assert mcp is not None
    assert mcp.name == "SemanticCodeSearch"


def test_server_has_initialize_tool(mock_components):
    """Server should have an initialize tool."""
    mcp = create_server(mock_components["temp_dir"])

    # FastMCP registers tools on the internal _tool_manager
    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "initialize" in tool_names


def test_server_has_search_tool(mock_components):
    """Server should have a search_code tool."""
    mcp = create_server(mock_components["temp_dir"])

    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "search_code" in tool_names


def test_server_has_reindex_tool(mock_components):
    """Server should have a reindex_file tool."""
    mcp = create_server(mock_components["temp_dir"])

    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "reindex_file" in tool_names


def test_server_has_status_resource(mock_components):
    """Server should have a status resource."""
    mcp = create_server(mock_components["temp_dir"])

    # Check resources
    resource_uris = [r.uri for r in mcp._resource_manager._resources.values()]
    assert any("status" in str(uri) for uri in resource_uris)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_server.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'code_rag.server'"

**Step 3: Write implementation**

```python
# src/code_rag/server.py
"""MCP Server for Semantic Code Search."""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from code_rag.config import Config, load_config
from code_rag.database import Database
from code_rag.embedder import Embedder
from code_rag.indexer import FileIndexer
from code_rag.searcher import HybridSearcher, SearchResult
from code_rag.watcher import FileWatcher


logger = logging.getLogger(__name__)


# Structured output models
class CodeMatch(BaseModel):
    """A matched code snippet."""
    file_path: str = Field(description="Path to source file")
    content: str = Field(description="Matched code snippet")
    name: Optional[str] = Field(default=None, description="Function/class name if available")
    chunk_type: str = Field(description="Type: function, class, method, module")
    language: str = Field(description="Programming language")
    start_line: int = Field(description="Starting line number")
    end_line: int = Field(description="Ending line number")
    score: float = Field(description="Relevance score 0-1")


class SearchResults(BaseModel):
    """Search results container."""
    query: str
    matches: list[CodeMatch]
    total_count: int
    search_time_ms: float


class InitializeResult(BaseModel):
    """Result of initialization."""
    status: str
    model: str
    files_indexed: int
    total_chunks: int


class IndexStats(BaseModel):
    """Index statistics."""
    files: int
    chunks: int
    model_name: Optional[str]
    schema_version: Optional[str]


def create_server(
    root_dir: Optional[Path] = None,
    config: Optional[Config] = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Args:
        root_dir: Root directory to index (defaults to cwd)
        config: Configuration (defaults to loading from env)

    Returns:
        Configured FastMCP instance
    """
    config = config or load_config()
    root_dir = Path(root_dir or os.getcwd()).resolve()

    # Initialize components (lazy-loaded where possible)
    db_path = root_dir / config.db_path
    db = Database(db_path, embedding_dim=config.embedding_dim)
    embedder = Embedder(model_name=config.embedding_model, embedding_dim=config.embedding_dim)
    indexer = FileIndexer(
        db, embedder, root_dir,
        chunk_overlap=config.chunk_overlap_tokens,
        max_chunk_tokens=config.max_chunk_tokens,
    )
    searcher = HybridSearcher(db, embedder, rrf_k=config.rrf_k)
    watcher: Optional[FileWatcher] = None

    # Store metadata
    db.set_meta("model_name", config.embedding_model)
    db.set_meta("schema_version", "1")

    mcp = FastMCP(
        name="SemanticCodeSearch",
        instructions="""
        Semantic code search for finding relevant code using natural language.

        First, call `initialize` to load the embedding model and build/update the index.
        Then use `search_code` with natural language queries like:
        - "function that handles user authentication"
        - "error handling for HTTP requests"
        - "database connection initialization"

        The search combines semantic similarity with keyword matching for best results.
        """
    )

    @mcp.tool()
    async def initialize(
        force_reindex: bool = Field(
            default=False,
            description="Force full reindex even if files haven't changed"
        ),
        ctx: Context = None,
    ) -> InitializeResult:
        """
        Initialize the semantic code search system.

        Loads the embedding model and builds or updates the code index.
        Call this once per session before searching.

        Progress will be reported during indexing.
        """
        start_time = time.time()

        # Report progress
        if ctx:
            await ctx.report_progress(0, 100, "Loading embedding model...")

        # Force model load
        _ = embedder.model

        if ctx:
            await ctx.report_progress(20, 100, "Scanning codebase...")

        # Index directory with progress
        def progress_callback(current: int, total: int, message: str):
            if ctx and total > 0:
                pct = 20 + int(70 * current / total)
                # Note: Can't await inside sync callback, so we skip reporting here

        if force_reindex:
            # Clear existing data
            db.conn.execute("DELETE FROM vec_chunks")
            db.conn.execute("DELETE FROM chunks")
            db.conn.execute("DELETE FROM files")
            db.conn.commit()

        stats = indexer.index_directory(root_dir, progress_callback)

        if ctx:
            await ctx.report_progress(90, 100, "Starting file watcher...")

        # Start file watcher
        nonlocal watcher
        if watcher is None:
            watcher = FileWatcher(
                indexer, root_dir,
                queue_max_size=config.queue_max_size,
                debounce_ms=config.debounce_ms,
            )
            asyncio.create_task(watcher.start())

        if ctx:
            await ctx.report_progress(100, 100, "Ready")

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Initialized in {elapsed:.0f}ms: {stats['files_indexed']} files, {stats['total_chunks']} chunks")

        return InitializeResult(
            status="initialized",
            model=config.embedding_model,
            files_indexed=stats["files_indexed"],
            total_chunks=stats["total_chunks"],
        )

    @mcp.tool()
    async def search_code(
        query: str = Field(description="Natural language search query"),
        file_pattern: Optional[str] = Field(
            default=None,
            description="Glob pattern to filter files, e.g., '**/*_test.py'"
        ),
        language: Optional[str] = Field(
            default=None,
            description="Filter by language: python, javascript, typescript, etc."
        ),
        chunk_type: Optional[str] = Field(
            default=None,
            description="Filter by type: function, class, method, module"
        ),
        max_results: int = Field(
            default=10,
            ge=1,
            le=50,
            description="Maximum results to return (1-50)"
        ),
        min_score: float = Field(
            default=0.3,
            ge=0,
            le=1,
            description="Minimum relevance score threshold (0-1)"
        ),
        ctx: Context = None,
    ) -> SearchResults:
        """
        Search the codebase using semantic similarity.

        Use natural language descriptions of code you're looking for:
        - "function that handles user authentication"
        - "error handling for HTTP requests"
        - "database connection initialization"
        - "unit tests for the payment service"

        Returns ranked code snippets with file locations and relevance scores.
        Combines vector similarity with keyword search for best results.
        """
        start_time = time.time()

        if ctx:
            await ctx.info(f"Searching: '{query}'")

        # Ensure model is loaded
        if not embedder.is_loaded():
            _ = embedder.model

        results = searcher.search(
            query=query,
            max_results=max_results,
            min_score=min_score,
            language=language,
            chunk_type=chunk_type,
            file_pattern=file_pattern,
        )

        elapsed = (time.time() - start_time) * 1000

        matches = [
            CodeMatch(
                file_path=r.file_path,
                content=r.content,
                name=r.name,
                chunk_type=r.chunk_type,
                language=r.language,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score,
            )
            for r in results
        ]

        return SearchResults(
            query=query,
            matches=matches,
            total_count=len(matches),
            search_time_ms=elapsed,
        )

    @mcp.tool()
    async def reindex_file(
        file_path: str = Field(description="Path to file to reindex"),
        force: bool = Field(default=False, description="Force reindex even if unchanged"),
    ) -> dict:
        """
        Re-index a specific file for search.

        Use when a file has been modified but not yet re-indexed,
        or when you want to force a refresh of a file's embeddings.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = root_dir / path

        result = indexer.index_file(path, force=force)
        return {
            "file": str(path),
            "status": result["status"],
            "chunks": result.get("chunks", 0),
            "reason": result.get("reason"),
        }

    @mcp.resource("search://status")
    def get_status() -> str:
        """Current index status and statistics."""
        stats = db.get_stats()
        return json.dumps(IndexStats(**stats).model_dump(), indent=2)

    return mcp


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_dir = Path(os.getenv("CODE_RAG_ROOT", os.getcwd()))
    mcp = create_server(root_dir)
    mcp.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_server.py -v`
Expected: PASS (5 passed)

**Step 5: Commit**

```bash
git add src/code_rag/server.py tests/test_server.py
git commit -m "feat: add MCP server with initialize, search, and reindex tools"
```

---

## Task 11: Integration Testing

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration tests**

```python
# tests/test_integration.py
"""Integration tests for the full pipeline."""
import asyncio
from pathlib import Path

import pytest

from code_rag.config import Config
from code_rag.database import Database
from code_rag.embedder import Embedder
from code_rag.indexer import FileIndexer
from code_rag.searcher import HybridSearcher


@pytest.fixture
def config():
    """Test config with small model."""
    return Config(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )


@pytest.fixture
def full_pipeline(temp_dir: Path, config):
    """Create full pipeline for integration testing."""
    db = Database(temp_dir / "test.db", embedding_dim=config.embedding_dim)
    embedder = Embedder(model_name=config.embedding_model, embedding_dim=config.embedding_dim)
    indexer = FileIndexer(db, embedder, temp_dir)
    searcher = HybridSearcher(db, embedder, rrf_k=config.rrf_k)

    yield {
        "db": db,
        "embedder": embedder,
        "indexer": indexer,
        "searcher": searcher,
        "root": temp_dir,
    }

    db.close()


def test_full_index_and_search_pipeline(full_pipeline):
    """Test complete index and search workflow."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    searcher = full_pipeline["searcher"]

    # Create test files
    (root / "auth.py").write_text('''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    # Hash password and check against database
    hashed = hash_password(password)
    user = db.get_user(username)
    return user and user.password_hash == hashed

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
''')

    (root / "search.py").write_text('''
def binary_search(arr: list[int], target: int) -> int:
    """Binary search to find target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def linear_search(arr: list, target) -> int:
    """Linear search through array."""
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
''')

    # Index files
    stats = indexer.index_directory(root)
    assert stats["files_indexed"] == 2
    assert stats["total_chunks"] > 0

    # Search for authentication code
    results = searcher.search("user login and password verification")
    assert len(results) > 0
    assert any("authenticate" in r.name.lower() for r in results if r.name)

    # Search for search algorithms
    results = searcher.search("find element in sorted array")
    assert len(results) > 0
    assert any("binary_search" in r.name for r in results if r.name)


def test_incremental_update(full_pipeline):
    """Test that file changes are detected and reindexed."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    db = full_pipeline["db"]

    # Create initial file
    test_file = root / "test.py"
    test_file.write_text("def foo(): pass")

    # First index
    result1 = indexer.index_file(test_file)
    assert result1["status"] == "indexed"

    initial_stats = db.get_stats()

    # Modify file
    test_file.write_text("def foo(): pass\ndef bar(): pass")

    # Reindex
    result2 = indexer.index_file(test_file)
    assert result2["status"] == "indexed"

    # Should have more chunks now
    updated_stats = db.get_stats()
    assert updated_stats["chunks"] >= initial_stats["chunks"]


def test_search_with_filters(full_pipeline):
    """Test search with language and type filters."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    searcher = full_pipeline["searcher"]

    # Create Python file
    (root / "app.py").write_text('''
class UserController:
    def get_user(self, id: int):
        return self.db.find(id)
''')

    # Create TypeScript file
    (root / "app.ts").write_text('''
class UserService {
    async getUser(id: number): Promise<User> {
        return await this.repo.findOne(id);
    }
}
''')

    indexer.index_directory(root)

    # Search with Python filter
    py_results = searcher.search("get user by id", language="python")
    for r in py_results:
        assert r.language == "python"

    # Search with class filter
    class_results = searcher.search("user", chunk_type="class")
    for r in class_results:
        assert r.chunk_type == "class"


def test_hybrid_search_finds_exact_matches(full_pipeline):
    """Hybrid search should find exact keyword matches."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    searcher = full_pipeline["searcher"]

    (root / "utils.py").write_text('''
def calculateTotalPrice(items: list) -> float:
    """Calculate the total price of all items."""
    return sum(item.price * item.quantity for item in items)
''')

    indexer.index_directory(root)

    # Search for exact function name
    results = searcher.search("calculateTotalPrice")

    assert len(results) > 0
    assert any("calculateTotalPrice" in r.name for r in results if r.name)
```

**Step 2: Run integration tests**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest tests/test_integration.py -v`
Expected: PASS (4 passed)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full pipeline"
```

---

## Task 12: Final Configuration and Documentation

**Files:**
- Create: `.mcp.json`
- Update: `pyproject.toml` (add script entry)

**Step 1: Create MCP configuration file**

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python",
      "args": ["-m", "code_rag.server"],
      "env": {
        "CODE_RAG_DB_PATH": ".code-rag/index.db",
        "CODE_RAG_EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-code"
      }
    }
  }
}
```

**Step 2: Run all tests to verify**

Run: `cd /home/adamhanna/apps/mcp/code-rag && pytest -v --cov=code_rag`
Expected: All tests pass with good coverage

**Step 3: Final commit**

```bash
git add .mcp.json
git commit -m "chore: add MCP server configuration"
```

---

## Execution Checklist

- [ ] Task 1: Project setup with dependencies
- [ ] Task 2: Configuration module
- [ ] Task 3: Database module with sqlite-vec and FTS5
- [ ] Task 4: Embedder module with FastEmbed
- [ ] Task 5: Code chunker with tree-sitter
- [ ] Task 6: Gitignore filter
- [ ] Task 7: Hybrid searcher with RRF
- [ ] Task 8: File indexer
- [ ] Task 9: File watcher
- [ ] Task 10: MCP server
- [ ] Task 11: Integration tests
- [ ] Task 12: Final configuration

---

**Plan complete and saved to `docs/plans/2025-12-28-semantic-code-search-mcp.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
