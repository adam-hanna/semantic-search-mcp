# Semantic Code Search MCP Server

An MCP server that provides semantic code search using local embeddings. Search your codebase with natural language queries like "authentication middleware" or "database connection pooling".

## Features

- **Hybrid search**: Combines vector similarity (Jina code embeddings) with FTS5 keyword matching
- **30+ languages**: Tree-sitter parsing for Python, TypeScript, JavaScript, Go, Rust, Java, and more
- **Incremental indexing**: File watcher detects changes automatically
- **Zero external APIs**: All embeddings generated locally

## Quick Start

### 1. Install

```bash
cd /path/to/code-rag
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
      "args": ["-m", "code_rag.server"],
      "env": {
        "CODE_RAG_DB_PATH": ".code-rag/index.db"
      }
    }
  }
}
```

**Option B: CLI**
```bash
claude mcp add semantic-search -- python -m code_rag.server
```

### 3. Initialize

In Claude Code, the server provides three tools:

- `initialize` - Load model and index codebase (call once per session)
- `search_code` - Search with natural language queries
- `reindex_file` - Manually reindex a specific file

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_RAG_DB_PATH` | `.code-rag/index.db` | Index database location |
| `CODE_RAG_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Embedding model |
| `CODE_RAG_SEARCH_MIN_SCORE` | `0.3` | Minimum relevance threshold |

## Requirements

- Python 3.11+
- ~700MB disk for embedding model (downloaded on first run)
