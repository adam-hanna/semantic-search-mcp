# src/semantic_search_mcp/database.py
"""SQLite database with sqlite-vec and FTS5 for hybrid search."""
import struct
from pathlib import Path
from typing import Optional

import apsw
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

        # Use apsw for extension loading support
        self.conn = apsw.Connection(str(self.db_path))

        # Load sqlite-vec extension
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        # Configure for performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB
        self.conn.execute("PRAGMA foreign_keys=ON")  # Enable FK constraints

        self._create_tables()

    def _create_tables(self):
        """Create all required tables if they don't exist."""
        # apsw doesn't have executescript, so we execute each statement
        statements = [
            # Metadata for versioning
            """CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            # Track source files
            """CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                language TEXT,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            "CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)",
            "CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash)",

            # Store code chunks with metadata
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                chunk_type TEXT,
                name TEXT,
                start_line INTEGER,
                end_line INTEGER
            )""",
            "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id)",

            # FTS5 for keyword search
            """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                name,
                file_path,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            )""",

            # Triggers to keep FTS in sync
            """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, name, file_path)
                SELECT NEW.id, NEW.content, NEW.name,
                       (SELECT path FROM files WHERE id = NEW.file_id);
            END""",

            """CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, name, file_path)
                VALUES('delete', OLD.id, OLD.content, OLD.name,
                       (SELECT path FROM files WHERE id = OLD.file_id));
            END""",

            """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, name, file_path)
                VALUES('delete', OLD.id, OLD.content, OLD.name,
                       (SELECT path FROM files WHERE id = OLD.file_id));
                INSERT INTO chunks_fts(rowid, content, name, file_path)
                SELECT NEW.id, NEW.content, NEW.name,
                       (SELECT path FROM files WHERE id = NEW.file_id);
            END""",

            # Vector embeddings
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{self.embedding_dim}] distance_metric=cosine,
                language TEXT,
                chunk_type TEXT,
                +file_path TEXT,
                +name TEXT,
                +preview TEXT
            )""",
        ]

        for stmt in statements:
            self.conn.execute(stmt)

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        cursor = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str):
        """Set a metadata value."""
        self.conn.execute(
            """INSERT INTO index_meta (key, value, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
            (key, value)
        )

    def get_file(self, path: str) -> Optional[dict]:
        """Get file record by path."""
        cursor = self.conn.execute(
            "SELECT id, path, content_hash, language, last_indexed FROM files WHERE path = ?", (path,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "path": row[1],
                "content_hash": row[2],
                "language": row[3],
                "last_indexed": row[4],
            }
        return None

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
        cursor = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (path,)
        )
        row = cursor.fetchone()
        return row[0]

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

    def delete_chunks_for_file(self, file_id: int):
        """Delete all chunks for a file."""
        self.conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
            (file_id,)
        )
        self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

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
        self.conn.execute(
            """INSERT INTO chunks (file_id, content, chunk_type, name, start_line, end_line)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, content, chunk_type, name, start_line, end_line)
        )
        return self.conn.last_insert_rowid()

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

    def get_stats(self) -> dict:
        """Get index statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM files")
        files = cursor.fetchone()[0]
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        chunks = cursor.fetchone()[0]
        return {
            "files": files,
            "chunks": chunks,
            "schema_version": self.get_meta("schema_version"),
            "model_name": self.get_meta("model_name"),
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()
