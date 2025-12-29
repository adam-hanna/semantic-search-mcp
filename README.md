# Semantic Search MCP Server

An MCP server that provides semantic code search using local embeddings. Search your codebase with natural language queries like "authentication middleware" or "database connection pooling".

## Features

- **Hybrid search**: Combines vector similarity (Jina code embeddings) with FTS5 keyword matching using Reciprocal Rank Fusion
- **30+ languages**: Tree-sitter parsing for Python, TypeScript, JavaScript, Go, Rust, Java, C/C++, Ruby, PHP, and more
- **Incremental indexing**: File watcher automatically detects additions, modifications, and deletions
- **Respects .gitignore**: Honors your project's `.gitignore` files (including nested ones)
- **Auto-initialization**: Model loads and codebase indexes in the background on server startup
- **Zero external APIs**: All embeddings generated locally with FastEmbed

## Installation

```bash
uvx install semantic-search-mcp
```

Or with pip:
```bash
pip install semantic-search-mcp
```

## Quick Start

### Add to Claude Code

**Option A: Project-level config**

Create `.mcp.json` in your project root:
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "uvx",
      "args": ["semantic-search-mcp"]
    }
  }
}
```

**Option B: CLI**
```bash
claude mcp add semantic-search -- uvx semantic-search-mcp
```

### Use

The server auto-initializes on startup. Available tools:

- `search_code` - Search with natural language queries
- `initialize` - Force re-index if needed
- `reindex_file` - Manually reindex a specific file

## How It Works

### Indexing

On startup, the server:
1. Scans your codebase for supported file types
2. Parses code into semantic chunks (functions, classes, methods) using Tree-sitter
3. Generates embeddings for each chunk using Jina's code embedding model
4. Stores everything in a local SQLite database with vector search support

### File Watching

The server monitors your codebase for changes in real-time:

| Event | Action |
|-------|--------|
| File created | Parsed, embedded, and added to index |
| File modified | Re-indexed if content hash changed |
| File deleted | Removed from index |

Changes are debounced (default 1s) to batch rapid modifications.

### What Gets Indexed

**Included:**
- Files with code extensions: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, and more

**Excluded:**
- Files matching `.gitignore` patterns (all `.gitignore` files in your project are respected)
- Common non-code directories: `node_modules`, `__pycache__`, `.venv`, `build`, `dist`, `.git`, `vendor`, etc.
- Binary files and non-code file types

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_SEARCH_DB_PATH` | `.semantic-search/index.db` | Index database location |
| `SEMANTIC_SEARCH_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Embedding model |
| `SEMANTIC_SEARCH_MIN_SCORE` | `0.3` | Minimum relevance threshold (0-1) |
| `SEMANTIC_SEARCH_DEBOUNCE_MS` | `1000` | File watcher debounce in milliseconds |

## Requirements

- Python 3.11+
- ~700MB disk for embedding model (downloaded on first run)
- ~1GB RAM for embedding model

## License

[MIT](LICENSE)
