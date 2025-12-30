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
    """Tracks server initialization and runtime state."""
    def __init__(self):
        # Initialization state
        self.status = "pending"  # pending, initializing, ready, error
        self.error: Optional[str] = None
        self.files_indexed = 0
        self.total_chunks = 0
        self.model = ""
        self.init_time_ms = 0.0

        # Watcher state
        self.watcher_status = "stopped"  # running, paused, stopped

        # Indexing state
        self.indexing_in_progress = False
        self.indexing_cancelled = False
        self.indexing_progress = {"current": 0, "total": 0, "current_file": ""}

        # Last indexed timestamp
        self.last_indexed_at: Optional[str] = None


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
    db_path = root_dir / config.db_path

    # Components are created lazily in background_init to avoid blocking
    # These are wrapped in a container so tools can access them after init
    class Components:
        db: Optional[Database] = None
        embedder: Optional[Embedder] = None
        indexer: Optional[FileIndexer] = None
        searcher: Optional[HybridSearcher] = None
        watcher: Optional[FileWatcher] = None

    components = Components()
    state = ServerState()

    async def background_init():
        """Run ALL initialization in background - no blocking before server starts."""
        try:
            state.status = "initializing"
            start_time = time.time()

            # Create database (this loads sqlite-vec extension, creates tables)
            logger.info("Initializing database...")
            components.db = await asyncio.to_thread(
                Database, db_path, config.embedding_dim
            )

            # Store metadata
            await asyncio.to_thread(components.db.set_meta, "model_name", config.embedding_model)
            await asyncio.to_thread(components.db.set_meta, "schema_version", "1")

            # Create embedder (model loads lazily on first use)
            components.embedder = Embedder(
                model_name=config.embedding_model,
                embedding_dim=config.embedding_dim,
                batch_size=config.embedding_batch_size,
                threads=config.embedding_threads,
                use_quantized=config.use_quantized,
            )

            # Create indexer and searcher
            components.indexer = FileIndexer(
                components.db, components.embedder, root_dir,
                chunk_overlap=config.chunk_overlap_tokens,
                max_chunk_tokens=config.max_chunk_tokens,
                max_file_size_kb=config.max_file_size_kb,
            )
            components.searcher = HybridSearcher(
                components.db, components.embedder, rrf_k=config.rrf_k
            )

            # Check if we need to do initial indexing
            existing_stats = components.db.get_stats()
            if existing_stats.get("files", 0) == 0:
                # No existing index - load model and index codebase
                logger.info("Loading embedding model...")
                await asyncio.to_thread(lambda: components.embedder.model)

                logger.info(f"Indexing codebase at {root_dir}...")
                stats = await asyncio.to_thread(
                    components.indexer.index_directory, root_dir, None, config.index_batch_size
                )
                state.files_indexed = stats["files_indexed"]
                state.total_chunks = stats["total_chunks"]
            else:
                logger.info(
                    f"Found existing index ({existing_stats['files']} files, "
                    f"{existing_stats['chunks']} chunks)"
                )
                state.files_indexed = existing_stats["files"]
                state.total_chunks = existing_stats["chunks"]

            # Start file watcher for incremental updates
            logger.info("Starting file watcher...")
            components.watcher = FileWatcher(
                components.indexer, root_dir,
                queue_max_size=config.queue_max_size,
                debounce_ms=config.debounce_ms,
            )
            asyncio.create_task(components.watcher.start())

            elapsed = (time.time() - start_time) * 1000
            state.status = "ready"
            state.model = config.embedding_model
            state.init_time_ms = elapsed

            logger.info(f"Ready in {elapsed:.0f}ms: {state.files_indexed} files, {state.total_chunks} chunks")

        except Exception as e:
            state.status = "error"
            state.error = str(e)
            logger.error(f"Initialization failed: {e}")

    @asynccontextmanager
    async def lifespan(app):
        """Start server immediately, initialize in background."""
        logger.info("Starting semantic code search server...")

        # Start initialization in background (non-blocking)
        # Server accepts MCP connections immediately while this runs
        init_task = asyncio.create_task(background_init())

        logger.info("Server ready to accept connections (init running in background)")
        yield  # Server starts accepting connections immediately

        # Cleanup on shutdown
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            pass
        if components.watcher:
            await components.watcher.stop()
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
        start_time = time.time()

        # Wait for background init to complete first
        if state.status == "initializing":
            if ctx:
                await ctx.info("Waiting for background initialization...")
            for _ in range(120):  # 60 second timeout
                if state.status != "initializing":
                    break
                await asyncio.sleep(0.5)

        if state.status == "error":
            return InitializeResult(
                status=f"error: {state.error}",
                model=config.embedding_model,
                files_indexed=0,
                total_chunks=0,
            )

        if components.db is None or components.embedder is None or components.indexer is None:
            return InitializeResult(
                status="error: components not initialized",
                model=config.embedding_model,
                files_indexed=0,
                total_chunks=0,
            )

        # Report progress
        if ctx:
            await ctx.report_progress(0, 100, "Loading embedding model...")

        # Force model load (run in thread to avoid blocking event loop)
        await asyncio.to_thread(lambda: components.embedder.model)

        if ctx:
            await ctx.report_progress(20, 100, "Scanning codebase...")

        if force_reindex:
            # Clear existing data
            components.db.conn.execute("DELETE FROM vec_chunks")
            components.db.conn.execute("DELETE FROM chunks")
            components.db.conn.execute("DELETE FROM files")
            components.db.conn.commit()

        # Index directory (run in thread to avoid blocking event loop)
        stats = await asyncio.to_thread(
            components.indexer.index_directory, root_dir, None, config.index_batch_size
        )

        if ctx:
            await ctx.report_progress(90, 100, "Starting file watcher...")

        # Start file watcher if not already running
        if components.watcher is None:
            components.watcher = FileWatcher(
                components.indexer, root_dir,
                queue_max_size=config.queue_max_size,
                debounce_ms=config.debounce_ms,
            )
            asyncio.create_task(components.watcher.start())

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

        if components.embedder is None or components.searcher is None:
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
        if not components.embedder.is_loaded():
            _ = components.embedder.model

        results = components.searcher.search(
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
        if components.indexer is None:
            return {
                "file": file_path,
                "status": "error",
                "chunks": 0,
                "reason": "Server still initializing",
            }

        path = Path(file_path)
        if not path.is_absolute():
            path = root_dir / path

        result = components.indexer.index_file(path, force=force)
        return {
            "file": str(path),
            "status": result["status"],
            "chunks": result.get("chunks", 0),
            "reason": result.get("reason"),
        }

    @mcp.tool()
    async def pause_watcher() -> dict:
        """
        Pause the file watcher.

        Events that occur while paused are discarded.
        Use resume_watcher to start watching again.
        """
        if components.watcher is None:
            return {"status": "error", "reason": "Watcher not initialized"}

        if state.watcher_status == "paused":
            return {"status": "already_paused", "events_discarded": 0}

        discarded = await components.watcher.pause()
        state.watcher_status = "paused"

        return {"status": "paused", "events_discarded": discarded}

    @mcp.tool()
    async def resume_watcher() -> dict:
        """
        Resume the file watcher after pausing.

        Starts watching for file changes again.
        """
        if components.watcher is None:
            return {"status": "error", "reason": "Watcher not initialized"}

        if state.watcher_status == "running":
            return {"status": "already_running"}

        await components.watcher.resume()
        state.watcher_status = "running"

        return {"status": "running"}

    @mcp.tool()
    async def cancel_indexing() -> dict:
        """
        Cancel any running indexing job.

        The indexing will stop after the current file completes.
        Partial results are kept in the index.
        """
        if not state.indexing_in_progress:
            return {"status": "not_running"}

        state.indexing_cancelled = True

        return {
            "status": "cancelling",
            "progress": state.indexing_progress.copy(),
        }

    @mcp.tool()
    async def clear_index() -> dict:
        """
        Clear all indexed data.

        Removes all files and chunks from the index.
        The index will be empty until reindex is called.
        """
        if components.db is None:
            return {"status": "error", "reason": "Database not initialized"}

        # Cancel any running indexing first
        if state.indexing_in_progress:
            state.indexing_cancelled = True
            # Wait briefly for cancellation
            for _ in range(10):
                if not state.indexing_in_progress:
                    break
                await asyncio.sleep(0.1)

        # Get current counts
        stats = components.db.get_stats()
        files_removed = stats.get("files", 0)
        chunks_removed = stats.get("chunks", 0)

        # Clear the database
        components.db.conn.execute("DELETE FROM vec_chunks")
        components.db.conn.execute("DELETE FROM chunks")
        components.db.conn.execute("DELETE FROM files")
        components.db.conn.commit()

        # Update state
        state.files_indexed = 0
        state.total_chunks = 0

        logger.info(f"Cleared index: {files_removed} files, {chunks_removed} chunks")

        return {
            "status": "cleared",
            "files_removed": files_removed,
            "chunks_removed": chunks_removed,
        }

    @mcp.tool()
    async def exclude_paths(
        patterns: list[str] = Field(
            description="Glob patterns to exclude, e.g. ['node_modules', '*.test.py']"
        ),
    ) -> dict:
        """
        Add paths to exclude from indexing (session-only).

        Patterns use glob syntax. Examples:
        - "node_modules" - exclude any path containing node_modules
        - "*.test.py" - exclude files ending in .test.py
        - "vendor/**" - exclude everything under vendor/

        Exclusions reset when the server restarts.
        """
        if components.indexer is None:
            return {"status": "error", "reason": "Indexer not initialized"}

        components.indexer.gitignore.add_exclusions(patterns)
        current = components.indexer.gitignore.get_exclusions()

        return {
            "status": "updated",
            "excluded_patterns": current,
        }

    @mcp.tool()
    async def include_paths(
        patterns: list[str] = Field(
            description="Glob patterns to remove from exclusion list"
        ),
    ) -> dict:
        """
        Remove paths from the exclusion list.

        Reverses the effect of exclude_paths for the specified patterns.
        """
        if components.indexer is None:
            return {"status": "error", "reason": "Indexer not initialized"}

        components.indexer.gitignore.remove_exclusions(patterns)
        current = components.indexer.gitignore.get_exclusions()

        return {
            "status": "updated",
            "excluded_patterns": current,
        }

    @mcp.tool()
    async def get_status() -> dict:
        """
        Get comprehensive server status.

        Returns server state, watcher status, indexing progress,
        index statistics, and current exclusion patterns.
        """
        # Get database stats if available
        if components.db is not None:
            db_stats = components.db.get_stats()
        else:
            db_stats = {"files": 0, "chunks": 0}

        # Get exclusion patterns if available
        if components.indexer is not None:
            excluded = components.indexer.gitignore.get_exclusions()
        else:
            excluded = []

        return {
            "server_status": state.status,
            "watcher_status": state.watcher_status,
            "indexing": {
                "in_progress": state.indexing_in_progress,
                "current_file": state.indexing_progress.get("current_file", ""),
                "progress": {
                    "current": state.indexing_progress.get("current", 0),
                    "total": state.indexing_progress.get("total", 0),
                },
            },
            "index": {
                "files": db_stats.get("files", state.files_indexed),
                "chunks": db_stats.get("chunks", state.total_chunks),
                "last_indexed": state.last_indexed_at,
            },
            "excluded_patterns": excluded,
            "model": state.model,
            "error": state.error,
        }

    @mcp.resource("search://status")
    def get_status_resource() -> str:
        """Current index status and statistics."""
        if components.db is None:
            result = {
                "files": 0,
                "chunks": 0,
                "model_name": config.embedding_model,
                "schema_version": None,
                "server_status": state.status,
            }
            if state.error:
                result["error"] = state.error
            return json.dumps(result, indent=2)

        stats = components.db.get_stats()
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
