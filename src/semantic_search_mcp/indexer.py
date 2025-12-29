# src/semantic_search_mcp/indexer.py
"""File indexing orchestration."""
import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from semantic_search_mcp.chunker import CodeChunker
from semantic_search_mcp.database import Database
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.gitignore import GitignoreFilter


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
