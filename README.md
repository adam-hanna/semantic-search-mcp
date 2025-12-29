# Semantic Search MCP Server

An MCP server that provides semantic code search using local embeddings. Search your codebase with natural language queries like "authentication middleware" or "database connection pooling".

## Features

- **Hybrid search**: Combines vector similarity (Jina code embeddings) with FTS5 keyword matching
- **30+ languages**: Tree-sitter parsing for Python, TypeScript, JavaScript, Go, Rust, Java, and more
- **Incremental indexing**: File watcher detects changes automatically
- **Auto-initialization**: Model loads and codebase indexes on server startup
- **Zero external APIs**: All embeddings generated locally

## Quick Start

### 1. Install

```bash
cd /path/to/semantic-search-mcp
pip install -e .
```

### 2. Add to Claude Code

**Option A: Project-level config**

Copy `.mcp.json` to your project root:
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python",
      "args": ["-m", "semantic_search_mcp.server"],
      "env": {
        "SEMANTIC_SEARCH_DB_PATH": ".semantic-search/index.db"
      }
    }
  }
}
```

**Option B: CLI**
```bash
claude mcp add semantic-search -- python -m semantic_search_mcp.server
```

### 3. Use

The server auto-initializes on startup. Available tools:

- `search_code` - Search with natural language queries
- `initialize` - Force re-index if needed
- `reindex_file` - Manually reindex a specific file

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_SEARCH_DB_PATH` | `.semantic-search/index.db` | Index database location |
| `SEMANTIC_SEARCH_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Embedding model |
| `SEMANTIC_SEARCH_MIN_SCORE` | `0.3` | Minimum relevance threshold |

## Requirements

- Python 3.11+
- ~700MB disk for embedding model (downloaded on first run)
