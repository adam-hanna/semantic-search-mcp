# Semantic Search MCP Server

An MCP server providing semantic code search for Claude Code using local embeddings.

## Project Overview

This server enables natural language code search across codebases. It combines vector similarity search with traditional full-text search using Reciprocal Rank Fusion for optimal results.

**Key capabilities:**
- Search code with queries like "authentication middleware" or "database connection pooling"
- 165+ language support via Tree-sitter parsing
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
| `embedder.py` | FastEmbed wrapper with INT8 quantization and GPU auto-detection |
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

### 2. Tree-sitter language pack
Uses `tree-sitter-language-pack` (actively maintained) instead of the deprecated `tree-sitter-languages`.
Supports Python 3.9-3.13 and 165+ languages.

### 3. Hybrid search with RRF
Combines vector similarity (semantic) with FTS5 BM25 (keywords) using Reciprocal Rank Fusion:
```python
score = 1 / (k + rank_vector) + 1 / (k + rank_fts)  # k=60
```

### 4. sqlite-vec query constraints
sqlite-vec doesn't allow both `k=?` and `LIMIT` in the same query. Use only `k=?` parameter.

### 5. Auto-initialization
Server uses FastMCP's lifespan context manager to automatically load the model and index the codebase on startup. No explicit `initialize` call required.

### 6. Chunk size limits
Large functions/classes are automatically split into smaller chunks (~8000 chars max) to prevent slow embeddings. Embedding time scales with input size - a 10KB function would take 5+ seconds vs ~200ms for a 2KB chunk.

### 7. INT8 quantization
Models are automatically quantized to INT8 on first run, reducing size by 75% and improving inference speed by 30-40%. The quantized model is cached for subsequent runs.

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
| `SEMANTIC_SEARCH_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Embedding model (see below for alternatives) |
| `SEMANTIC_SEARCH_EMBEDDING_DIM` | auto-detected | Embedding dimensions (auto-detected from model) |
| `SEMANTIC_SEARCH_MIN_SCORE` | `0.3` | Minimum relevance threshold |
| `SEMANTIC_SEARCH_RRF_K` | `60` | RRF constant |
| `SEMANTIC_SEARCH_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `SEMANTIC_SEARCH_MAX_CHUNK_TOKENS` | `2000` | Maximum chunk size |
| `SEMANTIC_SEARCH_QUEUE_MAX_SIZE` | `1000` | File watcher queue size |
| `SEMANTIC_SEARCH_DEBOUNCE_MS` | `1000` | Watcher debounce delay |
| `SEMANTIC_SEARCH_BATCH_SIZE` | `50` | Files per batch (memory management) |
| `SEMANTIC_SEARCH_MAX_FILE_SIZE_KB` | `512` | Skip files larger than this (KB) |
| `SEMANTIC_SEARCH_EMBEDDING_BATCH_SIZE` | `8` | Texts per embedding call (prevents ONNX memory explosion) |
| `SEMANTIC_SEARCH_EMBEDDING_THREADS` | `4` | ONNX runtime threads (higher = faster, try 16 on multi-core CPUs) |
| `SEMANTIC_SEARCH_USE_QUANTIZED` | `true` | Use INT8 quantized model (30-40% faster, auto-quantizes on first run) |

### Alternative Embedding Models

| Model | Dims | Size | Speed | Quality |
|-------|------|------|-------|---------|
| `jinaai/jina-embeddings-v2-base-code` | 768 | 640MB | 1x (baseline) | Excellent (code-specific) |
| `BAAI/bge-base-en-v1.5` | 768 | 210MB | ~2x faster | Good |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 274MB | ~1.1x faster | Good (8192 context) |
| `BAAI/bge-small-en-v1.5` | 384 | 67MB | ~4x faster | Decent |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90MB | **~32x faster** | Decent |

To switch models:
```bash
export SEMANTIC_SEARCH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
rm -rf .semantic-search/  # Must reindex with new model
```
Dimension is auto-detected. **Note:** Switching models requires a full reindex.

### UniXcoder (Experimental, Code-Specific)

[Microsoft UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) is pre-trained on code + AST + comments for deeper code understanding. It may provide better semantic matching but is **~20x slower** than Jina.

| Model | Dims | Speed | Languages |
|-------|------|-------|-----------|
| `microsoft/unixcoder-base` | 768 | ~20x slower | java, ruby, python, php, js, go |
| `microsoft/unixcoder-base-nine` | 768 | ~20x slower | + c, c++, c# |

**Installation:**
```bash
pip install semantic-search-mcp[unixcoder]  # Adds torch + transformers (~2GB)
```

**Usage:**
```bash
export SEMANTIC_SEARCH_EMBEDDING_MODEL=microsoft/unixcoder-base-nine
```

**Architecture:** Uses `transformers` library with PyTorch backend. Supports GPU acceleration:
- NVIDIA: CUDA (auto-detected)
- Apple Silicon: MPS (auto-detected)
- CPU: Fallback (slow)

**Benchmarks (CPU):**
| Model | ms/embedding | 1000 chunks |
|-------|--------------|-------------|
| Jina Code | 47ms | ~47 seconds |
| UniXcoder | 919ms | ~15 minutes |

**When to use:** Small codebases, quality-focused, GPU available.
**When to avoid:** Large codebases, fast indexing needed, CPU-only.

## Performance

### Optimizations Applied

The embedder includes several performance optimizations enabled by default:

| Optimization | Impact | Default |
|--------------|--------|---------|
| **INT8 Quantization** | 30-40% faster, 75% smaller model | Enabled |
| **Chunk Splitting** | Prevents slow embeddings for huge functions | Enabled |
| **GPU Auto-detection** | 10-50x faster if available | Auto |

### Recommended Settings for Multi-core CPUs

```bash
# For systems with 8+ cores (e.g., Threadripper, Ryzen 9, Xeon)
export SEMANTIC_SEARCH_EMBEDDING_THREADS=16
```

### Benchmarks (10 Python files, 91 chunks)

| Configuration | Embedding Time | Total Time |
|---------------|----------------|------------|
| Original (no optimizations) | 108s | 113s |
| + INT8 quantization | 74s | 75s |
| + threads=16 | 68s | 69s |
| **MiniLM model** | **1.7s** | **2.3s** |

### GPU Acceleration

GPU is auto-detected and used when available. Supported platforms:

| Platform | Provider | Installation |
|----------|----------|--------------|
| **NVIDIA GPU** | CUDAExecutionProvider | `pip install onnxruntime-gpu` |
| **Apple Silicon** | CoreMLExecutionProvider | Built into `onnxruntime` |
| **AMD GPU** | ROCMExecutionProvider | `pip install onnxruntime-rocm` |
| **Windows DirectML** | DmlExecutionProvider | `pip install onnxruntime-directml` |

When GPU is detected, you'll see:
```
INFO: GPU detected (CUDAExecutionProvider), using hardware acceleration
```

For NVIDIA GPUs, install with:
```bash
pip install semantic-search-mcp[gpu]
# or manually:
pip uninstall onnxruntime && pip install onnxruntime-gpu
```

### Speed vs Quality Tradeoffs

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Best code understanding | `jinaai/jina-embeddings-v2-base-code` | Code-specific training |
| Fast indexing, large codebases | `sentence-transformers/all-MiniLM-L6-v2` | 32x faster |
| Balanced | `BAAI/bge-base-en-v1.5` | 2x faster, good quality |

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
Tree-sitter-language-pack supports 165+ languages automatically. To check supported languages:
```python
from tree_sitter_language_pack import get_parser
parser = get_parser('rust')  # Returns parser if supported
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

1. **First startup includes one-time setup** - Model download (~150MB quantized) and initial indexing. Subsequent starts are fast (~1s model load).
2. **Memory usage** - Embedding model requires ~500MB RAM (quantized) or ~1GB (FP32)
3. **Binary files skipped** - Only text files with recognized extensions are indexed
4. **Model switching requires reindex** - Different models produce incompatible embeddings
5. **CPU-bound without GPU** - On CPU, indexing large codebases can take minutes. Consider using MiniLM for 32x faster indexing, or enable GPU acceleration.

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
