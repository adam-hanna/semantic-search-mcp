# src/semantic_search_mcp/server.py
"""MCP Server for Semantic Code Search."""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from semantic_search_mcp.config import Config, load_config
from semantic_search_mcp.database import Database
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.indexer import FileIndexer
from semantic_search_mcp.searcher import HybridSearcher
from semantic_search_mcp.watcher import FileWatcher


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
    status: str = Field(default="ready", description="Server status: initializing, ready, or error")


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


class ServerState:
    """Tracks server initialization state."""
    def __init__(self):
        self.status = "pending"  # pending, initializing, ready, error
        self.error: Optional[str] = None
        self.files_indexed = 0
        self.total_chunks = 0
        self.model = ""
        self.init_time_ms = 0.0


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
    state = ServerState()

    # Store metadata
    db.set_meta("model_name", config.embedding_model)
    db.set_meta("schema_version", "1")

    async def background_init():
        """Run initialization in background."""
        nonlocal watcher

        try:
            state.status = "initializing"
            start_time = time.time()

            # Load embedding model
            logger.info("Loading embedding model...")
            _ = embedder.model

            # Index codebase
            logger.info(f"Indexing codebase at {root_dir}...")
            stats = indexer.index_directory(root_dir)

            # Start file watcher
            logger.info("Starting file watcher...")
            watcher = FileWatcher(
                indexer, root_dir,
                queue_max_size=config.queue_max_size,
                debounce_ms=config.debounce_ms,
            )
            asyncio.create_task(watcher.start())

            elapsed = (time.time() - start_time) * 1000
            state.status = "ready"
            state.files_indexed = stats["files_indexed"]
            state.total_chunks = stats["total_chunks"]
            state.model = config.embedding_model
            state.init_time_ms = elapsed

            logger.info(f"Ready in {elapsed:.0f}ms: {stats['files_indexed']} files, {stats['total_chunks']} chunks")

        except Exception as e:
            state.status = "error"
            state.error = str(e)
            logger.error(f"Initialization failed: {e}")

    @asynccontextmanager
    async def lifespan(app):
        """Start server immediately, initialize in background."""
        nonlocal watcher

        logger.info("Starting semantic code search server...")

        # Start initialization in background (non-blocking)
        init_task = asyncio.create_task(background_init())

        yield  # Server starts accepting connections immediately

        # Cleanup on shutdown
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            pass
        if watcher:
            await watcher.stop()
        logger.info("Server stopped.")

    mcp = FastMCP(
        name="SemanticCodeSearch",
        instructions="""
        Semantic code search for finding relevant code using natural language.

        The server auto-initializes in the background. Use `search_code` with natural language queries like:
        - "function that handles user authentication"
        - "error handling for HTTP requests"
        - "database connection initialization"

        The search combines semantic similarity with keyword matching for best results.
        If the server is still initializing, search will wait briefly or return a status message.
        """,
        lifespan=lifespan,
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
        Initialize or re-initialize the semantic code search system.

        Loads the embedding model and builds or updates the code index.
        Use force_reindex=True to rebuild the entire index.

        Progress will be reported during indexing.
        """
        nonlocal watcher
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

        # Start file watcher if not already running
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
        state.status = "ready"
        state.files_indexed = stats["files_indexed"]
        state.total_chunks = stats["total_chunks"]

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

        # Wait for initialization if still in progress (with timeout)
        if state.status == "initializing":
            if ctx:
                await ctx.info("Server is initializing, please wait...")
            # Wait up to 60 seconds for initialization
            for _ in range(120):
                if state.status != "initializing":
                    break
                await asyncio.sleep(0.5)

        if state.status == "error":
            return SearchResults(
                query=query,
                matches=[],
                total_count=0,
                search_time_ms=0,
                status=f"error: {state.error}",
            )

        if state.status not in ("ready", "initializing"):
            # Still pending - initialization hasn't started yet
            return SearchResults(
                query=query,
                matches=[],
                total_count=0,
                search_time_ms=0,
                status="initializing",
            )

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
            status="ready",
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
        result = IndexStats(**stats).model_dump()
        result["server_status"] = state.status
        if state.error:
            result["error"] = state.error
        return json.dumps(result, indent=2)

    return mcp


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_dir = Path(os.getenv("SEMANTIC_SEARCH_ROOT", os.getcwd()))
    mcp = create_server(root_dir)
    mcp.run()


if __name__ == "__main__":
    main()
