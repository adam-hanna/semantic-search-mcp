# src/code_rag/server.py
"""MCP Server for Semantic Code Search."""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from code_rag.config import Config, load_config
from code_rag.database import Database
from code_rag.embedder import Embedder
from code_rag.indexer import FileIndexer
from code_rag.searcher import HybridSearcher
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
