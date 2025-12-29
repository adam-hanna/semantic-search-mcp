# Building an MCP semantic code search server for Claude Code

A semantic code search engine combining vector embeddings with SQLite storage and the Model Context Protocol enables Claude Code to intelligently navigate large codebases using natural language queries. The optimal architecture uses **FastEmbed with Jina code embeddings**, **sqlite-vec** for vector storage, **Python's MCP SDK with FastMCP**, and **watchfiles** for incremental indexing—delivering sub-100ms search across tens of thousands of code chunks without external API dependencies.

## The recommended technology stack

After evaluating available options, this combination delivers the best balance of performance, simplicity, and code-search quality:

| Component | Recommendation | Why |
|-----------|----------------|-----|
| **Embeddings** | FastEmbed + `jina-embeddings-v2-base-code` | 768-dim, 8192 token context, 30 languages, ONNX-optimized |
| **Vector Store** | sqlite-vec v0.1.6+ | Pure C, zero dependencies, cross-platform, MIT licensed |
| **MCP Framework** | Python `mcp` SDK with FastMCP | Type-safe tools, auto schema generation, stdio transport |
| **File Watching** | watchfiles | Rust-based, built-in debouncing, native async |
| **Gitignore** | pathspec | Full gitignore spec compliance including negation patterns |

**Installation baseline:**
```bash
pip install "mcp[cli]" fastembed sqlite-vec watchfiles pathspec
```

## Embedding libraries compared

Local embedding libraries eliminate external API costs and latency while maintaining search quality. The choice significantly impacts both indexing speed and search relevance.

### FastEmbed emerges as the production choice

FastEmbed, from the Qdrant team, uses ONNX Runtime instead of PyTorch, reducing installation size from ~500MB to ~50MB while achieving **1.5-2x faster CPU inference**. It supports the critical `jina-embeddings-v2-base-code` model, which was trained specifically for code search across 30 programming languages with an **8192-token context window**—16x longer than CodeBERT's 512 tokens.

```python
from fastembed import TextEmbedding
import numpy as np

# Initialize code-specific embedding model
model = TextEmbedding("jinaai/jina-embeddings-v2-base-code")

def embed_code(texts: list[str]) -> np.ndarray:
    """Generate embeddings for code snippets or queries."""
    return np.array(list(model.embed(texts)))

# Example usage
code_embedding = embed_code(["def binary_search(arr, target): ..."])
query_embedding = embed_code(["find element in sorted array"])
```

For memory-constrained environments, `BAAI/bge-small-en-v1.5` at 67MB offers 384-dimensional embeddings with strong general-purpose performance. For maximum quality without code-specific training, `mixedbread-ai/mxbai-embed-large-v1` provides 1024-dimensional embeddings ranking near state-of-the-art on MTEB benchmarks.

### Model selection by use case

| Model | Dimensions | Size | Context | Best For |
|-------|------------|------|---------|----------|
| `jina-embeddings-v2-base-code` | 768 | 640MB | 8192 | **Code search (production)** |
| `all-MiniLM-L6-v2` | 384 | 90MB | 256 | Fast prototyping |
| `bge-base-en-v1.5` | 768 | 210MB | 512 | Balanced general-purpose |
| `mxbai-embed-large-v1` | 1024 | 640MB | 512 | Maximum quality |

Ollama provides the simplest setup (`ollama pull nomic-embed-text`) but adds HTTP overhead and lacks code-specific models. Reserve it for prototyping rather than production systems.

## SQLite vector storage with sqlite-vec

**sqlite-vec** is the actively maintained successor to sqlite-vss, written in pure C with zero dependencies. It runs anywhere SQLite runs—Linux, macOS, Windows, WebAssembly, and mobile devices—making it ideal for portable developer tools.

### Database schema for code search

The schema separates file metadata from code chunks and vector embeddings, enabling efficient updates when files change:

```python
import sqlite3
import sqlite_vec
import struct
from pathlib import Path

def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with vector extension."""
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    
    db.executescript("""
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
            chunk_type TEXT,  -- 'function', 'class', 'method', 'block'
            name TEXT,        -- function/class name
            start_line INTEGER,
            end_line INTEGER,
            signature TEXT    -- for display
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
        
        -- Vector embeddings (768-dim for Jina code model)
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[768] distance_metric=cosine,
            language TEXT,
            chunk_type TEXT,
            +file_path TEXT,
            +name TEXT,
            +preview TEXT
        );
    """)
    return db

def serialize_embedding(embedding) -> bytes:
    """Convert numpy array to SQLite-compatible bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)
```

### Performing KNN vector search

sqlite-vec uses the `MATCH` operator with a `k` parameter for approximate nearest neighbor queries:

```python
def search_code(
    db: sqlite3.Connection,
    query_embedding: list[float],
    max_results: int = 10,
    language: str = None,
    chunk_type: str = None
) -> list[dict]:
    """Search for semantically similar code chunks."""
    sql = """
        SELECT chunk_id, file_path, name, chunk_type, preview, distance
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
    params.append(max_results)
    
    rows = db.execute(sql, params).fetchall()
    return [
        {"chunk_id": r[0], "file_path": r[1], "name": r[2], 
         "type": r[3], "preview": r[4], "distance": r[5]}
        for r in rows
    ]
```

### Performance at scale

sqlite-vec uses brute-force search, which determines practical scale limits:

| Vector Count | Query Time (768-dim) | Recommendation |
|--------------|---------------------|----------------|
| 1K - 10K | <10ms | Excellent for most projects |
| 10K - 100K | 10-100ms | Works well, consider quantization |
| 100K - 250K | 100-500ms | Use binary quantization |
| 250K+ | >500ms | Consider vectorlite (HNSW) or partitioning |

**Binary quantization** reduces storage 32x and speeds queries significantly with ~95% accuracy retention:

```sql
-- Create table with binary vectors
CREATE VIRTUAL TABLE vec_chunks_binary USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding bit[768] distance_metric=hamming
);

-- Insert quantized vectors
INSERT INTO vec_chunks_binary (chunk_id, embedding)
SELECT chunk_id, vec_quantize_binary(embedding) FROM vec_chunks;
```

## MCP server architecture with FastMCP

The Model Context Protocol (MCP) standardizes how AI assistants interact with external tools. FastMCP, included in the Python SDK, provides a decorator-based API that automatically generates JSON schemas from type hints.

### Core server structure

```python
#!/usr/bin/env python3
"""MCP Server for Semantic Code Search"""

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from typing import Optional
import os

# Configuration via environment variables
DB_PATH = os.getenv("CODE_SEARCH_DB", "./code_search.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")

mcp = FastMCP(
    name="SemanticCodeSearch",
    instructions="""
    Semantic code search for finding relevant code using natural language.
    Use queries like "authentication middleware" or "database connection pooling".
    """
)

# Structured output models
class CodeMatch(BaseModel):
    file_path: str = Field(description="Path to source file")
    content: str = Field(description="Matched code snippet")
    language: str = Field(description="Programming language")
    start_line: int = Field(description="Starting line number")
    end_line: int = Field(description="Ending line number")
    score: float = Field(description="Relevance score 0-1")

class SearchResults(BaseModel):
    query: str
    matches: list[CodeMatch]
    total_count: int
    search_time_ms: float
```

### Defining the search tool

The tool's docstring becomes its description in Claude Code, so write it to help the AI understand when and how to use the tool:

```python
@mcp.tool()
async def search_code(
    query: str,
    file_extensions: Optional[list[str]] = Field(
        default=None,
        description="Filter by extensions, e.g., ['py', 'ts', 'js']"
    ),
    directory: Optional[str] = Field(
        default=None,
        description="Limit search to specific directory"
    ),
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return (1-50)"
    ),
    ctx: Context[ServerSession, None] = None
) -> SearchResults:
    """
    Search the codebase using semantic similarity.
    
    Use natural language descriptions of code you're looking for:
    - "function that handles user authentication"
    - "error handling for HTTP requests"
    - "database connection initialization"
    - "unit tests for the payment service"
    
    Returns ranked code snippets with file locations and relevance scores.
    """
    import time
    start = time.time()
    
    if ctx:
        await ctx.info(f"Searching: '{query}'")
    
    # Generate query embedding
    query_embedding = embed_code([query])[0]
    
    # Perform vector search
    results = search_database(
        query_embedding, 
        max_results=max_results,
        extensions=file_extensions,
        directory=directory
    )
    
    return SearchResults(
        query=query,
        matches=[CodeMatch(**r) for r in results],
        total_count=len(results),
        search_time_ms=(time.time() - start) * 1000
    )

@mcp.tool()
async def reindex_file(
    file_path: str,
    force: bool = False
) -> dict:
    """
    Re-index a specific file for search.
    Use when a file has been modified but not yet re-indexed.
    """
    result = await index_single_file(file_path, force_reindex=force)
    return {"status": "indexed", "file": file_path, "chunks": result}

@mcp.resource("search://status")
def get_index_status() -> str:
    """Current indexing statistics."""
    import json
    stats = get_database_stats()
    return json.dumps(stats, indent=2)

if __name__ == "__main__":
    mcp.run()  # Runs with stdio transport
```

### Configuring Claude Code

Add the server using the CLI or a project configuration file:

```bash
# Add via CLI
claude mcp add semantic-search \
  -e CODE_SEARCH_DB="./code_search.db" \
  -- python /path/to/semantic_search_server.py
```

Or create `.mcp.json` in your project root for team-shared configuration:

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python",
      "args": ["./tools/semantic_search_server.py"],
      "env": {
        "CODE_SEARCH_DB": "./.code-search/index.db",
        "EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-code"
      }
    }
  }
}
```

Verify the connection with `/mcp` in Claude Code to see available tools.

## File system watching for incremental indexing

Efficient incremental indexing requires watching for file changes, filtering against gitignore patterns, debouncing rapid events, and tracking file state to avoid re-embedding unchanged content.

### watchfiles for high-performance monitoring

The Rust-based watchfiles library provides built-in debouncing and native async support with roughly **10x lower CPU usage** than pure-Python alternatives:

```python
import asyncio
from watchfiles import awatch, Change
from pathlib import Path

async def watch_codebase(root_dir: str, on_change):
    """Watch directory for code changes with built-in debouncing."""
    async for changes in awatch(
        root_dir,
        debounce=1000,  # 1 second debounce built into Rust layer
        recursive=True,
        force_polling=False  # Use native OS events
    ):
        for change_type, path in changes:
            if change_type == Change.added:
                await on_change("created", path)
            elif change_type == Change.modified:
                await on_change("modified", path)
            elif change_type == Change.deleted:
                await on_change("deleted", path)
```

### Gitignore filtering with pathspec

The pathspec library handles the full gitignore specification including negation patterns and nested `.gitignore` files:

```python
import pathspec
from pathlib import Path

class GitignoreManager:
    ALWAYS_IGNORE = {
        '.git', 'node_modules', '__pycache__', '.venv', 'venv',
        'build', 'dist', '.idea', '.vscode', 'target', 'vendor',
        '.mypy_cache', '.pytest_cache', 'coverage', '*.egg-info'
    }
    
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
        '.c', '.cpp', '.h', '.cs', '.rb', '.php', '.swift', '.kt'
    }
    
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.specs: dict[Path, pathspec.GitIgnoreSpec] = {}
        self._load_all_gitignores()
    
    def _load_all_gitignores(self):
        for gitignore in self.root.rglob('.gitignore'):
            with open(gitignore) as f:
                self.specs[gitignore.parent] = pathspec.GitIgnoreSpec.from_lines(f)
    
    def should_index(self, filepath: str) -> bool:
        path = Path(filepath)
        
        # Check extension
        if path.suffix.lower() not in self.CODE_EXTENSIONS:
            return False
        
        # Check always-ignored directories
        if any(part in self.ALWAYS_IGNORE for part in path.parts):
            return False
        
        # Check gitignore patterns (most specific wins)
        for gitignore_dir, spec in self.specs.items():
            try:
                rel_path = path.relative_to(gitignore_dir)
                if spec.match_file(str(rel_path)):
                    return False
            except ValueError:
                continue
        
        return True
```

### File state tracking for content-based updates

Track file hashes to avoid re-embedding files that haven't actually changed (mtime can update without content changes):

```python
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class FileState:
    path: str
    content_hash: str
    mtime: float
    size: int

class FileStateTracker:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.states: dict[str, FileState] = {}
        self._load()
    
    def _load(self):
        if Path(self.state_file).exists():
            with open(self.state_file) as f:
                data = json.load(f)
                self.states = {k: FileState(**v) for k, v in data.items()}
    
    def _save(self):
        with open(self.state_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.states.items()}, f)
    
    def _hash_file(self, filepath: str) -> str:
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def needs_reindex(self, filepath: str) -> bool:
        filepath = str(Path(filepath).resolve())
        
        if filepath not in self.states:
            return True
        
        stat = Path(filepath).stat()
        old = self.states[filepath]
        
        # Quick check: unchanged mtime and size
        if stat.st_mtime == old.mtime and stat.st_size == old.size:
            return False
        
        # Content verification
        return self._hash_file(filepath) != old.content_hash
    
    def mark_indexed(self, filepath: str):
        filepath = str(Path(filepath).resolve())
        stat = Path(filepath).stat()
        self.states[filepath] = FileState(
            path=filepath,
            content_hash=self._hash_file(filepath),
            mtime=stat.st_mtime,
            size=stat.st_size
        )
        self._save()
```

## Integration architecture and initialization

The complete system coordinates embedding generation, vector storage, file watching, and MCP communication.

### Initial index building

On first run or when the index is stale, perform a full scan:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class CodeSearchIndexer:
    def __init__(self, root_dir: str, db_path: str):
        self.root = Path(root_dir).resolve()
        self.db = init_database(db_path)
        self.embedder = TextEmbedding("jinaai/jina-embeddings-v2-base-code")
        self.gitignore = GitignoreManager(root_dir)
        self.state_tracker = FileStateTracker(str(self.root / '.index_state.json'))
    
    async def build_full_index(self, progress_callback=None):
        """Build or rebuild the complete index."""
        files = [
            f for f in self.root.rglob('*')
            if f.is_file() and self.gitignore.should_index(str(f))
        ]
        
        total = len(files)
        for i, filepath in enumerate(files):
            if self.state_tracker.needs_reindex(str(filepath)):
                await self._index_file(filepath)
            
            if progress_callback and i % 10 == 0:
                await progress_callback(i / total, f"Indexed {i}/{total} files")
        
        return {"files_indexed": total}
    
    async def _index_file(self, filepath: Path):
        """Index a single file by chunking and embedding."""
        content = filepath.read_text(errors='ignore')
        chunks = self._chunk_code(content, filepath)
        
        if not chunks:
            return
        
        # Batch embed all chunks
        texts = [c['content'] for c in chunks]
        embeddings = list(self.embedder.embed(texts))
        
        # Store in database
        with self.db:
            # Upsert file record
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            self.db.execute("""
                INSERT INTO files (path, content_hash, language)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    last_indexed = CURRENT_TIMESTAMP
            """, [str(filepath), file_hash, self._detect_language(filepath)])
            
            file_id = self.db.execute(
                "SELECT id FROM files WHERE path = ?", [str(filepath)]
            ).fetchone()[0]
            
            # Clear old chunks
            self.db.execute("""
                DELETE FROM vec_chunks 
                WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)
            """, [file_id])
            self.db.execute("DELETE FROM chunks WHERE file_id = ?", [file_id])
            
            # Insert new chunks and embeddings
            for chunk, embedding in zip(chunks, embeddings):
                self.db.execute("""
                    INSERT INTO chunks (file_id, content, chunk_type, name, start_line, end_line)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [file_id, chunk['content'], chunk['type'], 
                      chunk.get('name'), chunk['start_line'], chunk['end_line']])
                
                chunk_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
                
                self.db.execute("""
                    INSERT INTO vec_chunks (chunk_id, embedding, language, chunk_type, file_path, name, preview)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [chunk_id, serialize_embedding(embedding),
                      self._detect_language(filepath), chunk['type'],
                      str(filepath), chunk.get('name'), chunk['content'][:200]])
        
        self.state_tracker.mark_indexed(str(filepath))
```

### Code chunking strategies

AST-based chunking preserves semantic boundaries. For a lightweight approach without full AST parsing, use regex-based function/class extraction:

```python
import re

def chunk_python_code(content: str, filepath: Path) -> list[dict]:
    """Extract functions and classes from Python code."""
    chunks = []
    lines = content.split('\n')
    
    # Pattern for top-level definitions
    def_pattern = re.compile(r'^(async\s+)?def\s+(\w+)|^class\s+(\w+)')
    
    current_chunk = None
    current_start = 0
    base_indent = 0
    
    for i, line in enumerate(lines):
        match = def_pattern.match(line)
        if match:
            # Save previous chunk
            if current_chunk:
                current_chunk['content'] = '\n'.join(lines[current_start:i])
                current_chunk['end_line'] = i
                chunks.append(current_chunk)
            
            # Start new chunk
            is_async = match.group(1) is not None
            func_name = match.group(2)
            class_name = match.group(3)
            
            current_chunk = {
                'type': 'class' if class_name else 'function',
                'name': class_name or func_name,
                'start_line': i + 1,
                'is_async': is_async
            }
            current_start = i
    
    # Don't forget last chunk
    if current_chunk:
        current_chunk['content'] = '\n'.join(lines[current_start:])
        current_chunk['end_line'] = len(lines)
        chunks.append(current_chunk)
    
    # If no functions/classes found, chunk by size
    if not chunks and len(content) > 100:
        chunks.append({
            'type': 'module',
            'name': filepath.stem,
            'content': content[:2000],  # First 2000 chars
            'start_line': 1,
            'end_line': min(50, len(lines))
        })
    
    return chunks
```

For production systems, **tree-sitter** provides language-agnostic AST parsing with excellent performance.

### Background indexing with async coordination

Run indexing in the background while the MCP server handles queries:

```python
class BackgroundIndexer:
    def __init__(self, indexer: CodeSearchIndexer):
        self.indexer = indexer
        self.queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """Start background processing loop."""
        self.running = True
        asyncio.create_task(self._process_loop())
        asyncio.create_task(self._watch_loop())
    
    async def _process_loop(self):
        """Process file events from queue."""
        while self.running:
            try:
                event_type, path = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                await self._handle_event(event_type, path)
            except asyncio.TimeoutError:
                continue
    
    async def _watch_loop(self):
        """Watch filesystem and queue events."""
        async for changes in awatch(str(self.indexer.root), debounce=1000):
            for change_type, path in changes:
                if self.indexer.gitignore.should_index(path):
                    await self.queue.put((change_type.name, path))
    
    async def _handle_event(self, event_type: str, path: str):
        if event_type == 'added' or event_type == 'modified':
            if self.indexer.state_tracker.needs_reindex(path):
                await self.indexer._index_file(Path(path))
        elif event_type == 'deleted':
            await self.indexer.remove_file(path)
```

## Memory and performance optimization

For large codebases exceeding **50,000 chunks**, apply these optimizations:

- **Binary quantization**: 32x storage reduction with Hamming distance search
- **Partition keys**: Shard by project or directory for faster filtered queries
- **Matryoshka truncation**: Many models support dimension reduction (768→256) with minimal quality loss
- **Memory mapping**: `PRAGMA mmap_size = 268435456` for 256MB of memory-mapped access
- **Batch embedding**: Process 32-64 texts per embedding call for GPU efficiency

**Linux inotify limits** may need adjustment for large codebases:
```bash
# Increase to 500K watches (default is 8192)
echo "fs.inotify.max_user_watches=524288" | sudo tee /etc/sysctl.d/60-inotify.conf
sudo sysctl -p /etc/sysctl.d/60-inotify.conf
```

## Conclusion

This architecture delivers **sub-100ms semantic search** across codebases up to 100,000 chunks while running entirely locally without API dependencies. The key architectural decisions—Jina code embeddings for quality, sqlite-vec for portability, FastMCP for type safety, and watchfiles for efficient monitoring—create a system that's both powerful and maintainable.

For immediate implementation, start with the FastEmbed + sqlite-vec combination for core search functionality, then add file watching for incremental updates. The MCP layer integrates naturally with Claude Code's tool system, making semantic search available through natural language queries like "find the authentication middleware" or "show me error handling for database connections."
