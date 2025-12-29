# Semantic Search MCP Server

An MCP server providing semantic code search for Claude Code using local embeddings.

## Project Overview

This server enables natural language code search across codebases. It combines vector similarity search with traditional full-text search using Reciprocal Rank Fusion for optimal results.

**Key capabilities:**
- Search code with queries like "authentication middleware" or "database connection pooling"
- 30+ language support via Tree-sitter parsing
- Automatic incremental indexing via file watcher
- Zero external API dependencies - all embeddings generated locally

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Server                              │
│  (server.py - FastMCP with lifespan for auto-init)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Indexer    │    │   Searcher   │    │   Watcher    │     │
│  │ (indexer.py) │    │(searcher.py) │    │ (watcher.py) │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                   │                   │              │
│  ┌──────▼───────┐    ┌──────▼───────┐          │              │
│  │   Chunker    │    │   Embedder   │◄─────────┘              │
│  │ (chunker.py) │    │(embedder.py) │                         │
│  └──────┬───────┘    └──────┬───────┘                         │
│         │                   │                                  │
│  ┌──────▼───────┐    ┌──────▼───────┐                         │
│  │  Gitignore   │    │   Database   │                         │
│  │(gitignore.py)│    │(database.py) │                         │
│  └──────────────┘    └──────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `server.py` | MCP server with tools: `search_code`, `initialize`, `reindex_file` |
| `config.py` | Configuration from env vars with `SEMANTIC_SEARCH_*` prefix |
| `database.py` | SQLite with sqlite-vec (vectors) and FTS5 (keywords) via APSW |
| `embedder.py` | FastEmbed wrapper for Jina code embeddings (768-dim) |
| `chunker.py` | Tree-sitter AST parsing to extract functions/classes/methods |
| `searcher.py` | Hybrid search with Reciprocal Rank Fusion (k=60) |
| `indexer.py` | File indexing with content-hash change detection |
| `watcher.py` | Async file watcher with bounded queue (watchfiles) |
| `gitignore.py` | Gitignore pattern matching (pathspec library) |

## Database Schema

```sql
-- Source files with content hash for change detection
files (id, path, content_hash, indexed_at)

-- Code chunks (functions, classes, methods, modules)
chunks (id, file_id, content, name, chunk_type, language, start_line, end_line)

-- FTS5 virtual table for keyword search (auto-synced via triggers)
chunks_fts (content, name, chunk_type)

-- Vector embeddings for semantic search
vec_chunks (rowid, embedding) -- 768-dim float32

-- Metadata storage
index_meta (key, value)
```

## Key Technical Decisions

### 1. APSW instead of sqlite3
Standard Python sqlite3 lacks `ENABLE_LOAD_EXTENSION` on many systems. APSW provides reliable extension loading for sqlite-vec.

### 2. Tree-sitter version pinning
```toml
tree-sitter>=0.21.0,<0.22.0
tree-sitter-languages>=1.7.0,<1.8.0
```
Version 0.23+ has breaking API changes incompatible with tree-sitter-languages.

### 3. Hybrid search with RRF
Combines vector similarity (semantic) with FTS5 BM25 (keywords) using Reciprocal Rank Fusion:
```python
score = 1 / (k + rank_vector) + 1 / (k + rank_fts)  # k=60
```

### 4. sqlite-vec query constraints
sqlite-vec doesn't allow both `k=?` and `LIMIT` in the same query. Use only `k=?` parameter.

### 5. Auto-initialization
Server uses FastMCP's lifespan context manager to automatically load the model and index the codebase on startup. No explicit `initialize` call required.

## Development

### Setup
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest tests/ -v
```

### Test coverage
```bash
pytest tests/ --cov=semantic_search_mcp
```

Current: 61 tests, ~75% coverage

### Run server directly
```bash
python -m semantic_search_mcp.server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_SEARCH_DB_PATH` | `.semantic-search/index.db` | Database location |
| `SEMANTIC_SEARCH_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Embedding model |
| `SEMANTIC_SEARCH_EMBEDDING_DIM` | `768` | Embedding dimensions |
| `SEMANTIC_SEARCH_MIN_SCORE` | `0.3` | Minimum relevance threshold |
| `SEMANTIC_SEARCH_RRF_K` | `60` | RRF constant |
| `SEMANTIC_SEARCH_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `SEMANTIC_SEARCH_MAX_CHUNK_TOKENS` | `2000` | Maximum chunk size |
| `SEMANTIC_SEARCH_QUEUE_MAX_SIZE` | `1000` | File watcher queue size |
| `SEMANTIC_SEARCH_DEBOUNCE_MS` | `1000` | Watcher debounce delay |

## MCP Tools

### search_code
```python
search_code(
    query: str,              # Natural language query
    file_pattern: str = None,  # Glob pattern e.g. "**/*_test.py"
    language: str = None,      # Filter: python, typescript, etc.
    chunk_type: str = None,    # Filter: function, class, method, module
    max_results: int = 10,     # 1-50
    min_score: float = 0.3,    # 0-1
)
```

### initialize
Force reindex with `force_reindex=True`. Normally not needed due to auto-init.

### reindex_file
Manually reindex a specific file path.

## File Structure

```
semantic-search-mcp/
├── src/semantic_search_mcp/
│   ├── __init__.py
│   ├── server.py       # MCP server entry point
│   ├── config.py       # Configuration management
│   ├── database.py     # SQLite + sqlite-vec + FTS5
│   ├── embedder.py     # FastEmbed wrapper
│   ├── chunker.py      # Tree-sitter code parsing
│   ├── searcher.py     # Hybrid search with RRF
│   ├── indexer.py      # File indexing logic
│   ├── watcher.py      # Async file watching
│   └── gitignore.py    # Gitignore filtering
├── tests/
│   ├── test_*.py       # Unit and integration tests
│   └── conftest.py     # Shared fixtures
├── docs/
│   ├── design.md       # Original design document
│   └── plans/          # Implementation plans
├── pyproject.toml
├── .mcp.json           # Example MCP config
├── README.md
└── CLAUDE.md           # This file
```

## Common Tasks

### Adding language support
Tree-sitter-languages supports 30+ languages automatically. To check supported languages:
```python
from tree_sitter_languages import get_language
lang = get_language('rust')  # Returns language if supported
```

### Debugging search results
The searcher logs detailed scoring information. Enable debug logging:
```python
import logging
logging.getLogger('semantic_search_mcp.searcher').setLevel(logging.DEBUG)
```

### Force full reindex
Call the `initialize` tool with `force_reindex=True`, or delete the `.semantic-search/` directory.

### Testing changes
Always run the full test suite after changes:
```bash
pytest tests/ -v --tb=short
```

## Known Limitations

1. **First startup is slow** (~10-20s) - embedding model download (~700MB) and initial indexing
2. **Memory usage** - embedding model requires ~1GB RAM
3. **Binary files skipped** - only text files with recognized extensions are indexed
4. **Large files** - files are chunked but very large single constructs may be truncated

## Publishing to PyPI

### Prerequisites
```bash
pip install build twine
```

### Build the package
```bash
python -m build
```

This creates `dist/semantic_search_mcp-X.Y.Z-py3-none-any.whl` and `.tar.gz`.

### Upload to PyPI

**Test PyPI (recommended first):**
```bash
twine upload --repository testpypi dist/*
```

**Production PyPI:**
```bash
twine upload dist/*
```

### Version bumping
Update version in `pyproject.toml`:
```toml
version = "0.2.0"
```

### Release checklist
1. Update version in `pyproject.toml`
2. Run tests: `pytest tests/ -v`
3. Build: `python -m build`
4. Upload to Test PyPI and verify: `twine upload --repository testpypi dist/*`
5. Test install: `uvx install --index-url https://test.pypi.org/simple/ semantic-search-mcp`
6. Upload to PyPI: `twine upload dist/*`
7. Tag release: `git tag v0.2.0 && git push --tags`

## Git Workflow

- `main` - stable releases only
- `develop` - active development branch

All work should be done on `develop` and merged to `main` for releases.
