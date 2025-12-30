# Semantic Search MCP Server

An MCP server that provides semantic code search using local embeddings. Search your codebase with natural language queries like "authentication middleware" or "database connection pooling".

## Features

- **Hybrid search**: Combines vector similarity (Jina code embeddings) with FTS5 keyword matching using Reciprocal Rank Fusion
- **165+ languages**: Tree-sitter parsing for Python, TypeScript, JavaScript, Go, Rust, Java, C/C++, Ruby, PHP, and more
- **Incremental indexing**: File watcher automatically detects additions, modifications, and deletions
- **Respects .gitignore**: Honors your project's `.gitignore` files (including nested ones)
- **Auto-initialization**: Model loads and codebase indexes in the background on server startup
- **Zero external APIs**: All embeddings generated locally with FastEmbed

## Installation

```bash
uv tool install semantic-search-mcp
```

Or with pip:
```bash
pip install semantic-search-mcp
```

Or run directly without installing:
```bash
uvx semantic-search-mcp
```

## Quick Start

### Add to Claude Code

**Option A: Project-level config (recommended)**

After installing with `uv tool install` or `pip install`, create `.mcp.json` in your project root:
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "semantic-search-mcp"
    }
  }
}
```

**Option B: CLI**
```bash
claude mcp add semantic-search -- semantic-search-mcp
```

**Option C: Without installing (ephemeral)**

If you prefer not to install, use `uvx` to run in an ephemeral environment:
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

### Use

The server auto-initializes on startup.

### Available Tools

| Tool | Description |
|------|-------------|
| `search_code` | Search codebase with natural language |
| `get_status` | Get server state, progress, and statistics |
| `pause_watcher` | Pause file watching (events discarded) |
| `resume_watcher` | Resume file watching |
| `reindex` | Start full reindex (runs in background) |
| `cancel_indexing` | Cancel running indexing job |
| `clear_index` | Wipe all indexed data |
| `exclude_paths` | Add paths to ignore (session-only) |
| `include_paths` | Remove paths from exclusion list |

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
| `SEMANTIC_SEARCH_BATCH_SIZE` | `50` | Files per batch (reduce if running out of memory) |
| `SEMANTIC_SEARCH_MAX_FILE_SIZE_KB` | `512` | Skip files larger than this (KB) |
| `SEMANTIC_SEARCH_EMBEDDING_BATCH_SIZE` | `8` | Texts per embedding call (reduce if OOM) |
| `SEMANTIC_SEARCH_EMBEDDING_THREADS` | `4` | ONNX runtime threads (higher = faster on multi-core) |
| `SEMANTIC_SEARCH_USE_QUANTIZED` | `true` | Use INT8 quantized model (30-40% faster) |

## Performance

### GPU Acceleration

GPU acceleration is auto-detected and used when available:

| Platform | Provider | Installation |
|----------|----------|--------------|
| NVIDIA | CUDA | `pip install semantic-search-mcp[gpu]` |
| Apple Silicon | CoreML | Automatic (M1/M2/M3) |
| AMD | ROCm | Install ROCm-enabled onnxruntime |
| Windows | DirectML | Install DirectML-enabled onnxruntime |

### Alternative Models

For faster indexing (with quality tradeoffs), you can use a lighter model:

| Model | Dimensions | Speed | Best For |
|-------|------------|-------|----------|
| `jinaai/jina-embeddings-v2-base-code` | 768 | Baseline | Code search (default) |
| `BAAI/bge-small-en-v1.5` | 384 | ~10x faster | General text |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~32x faster | Speed priority |

To use an alternative model:
```bash
export SEMANTIC_SEARCH_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

Note: Changing models requires a full reindex (delete `.semantic-search/` directory).

### UniXcoder (Experimental)

[Microsoft UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) is a code-specific model pre-trained on code + AST + comments. It may provide better semantic understanding of code structure, but is **substantially slower** (~20x slower than Jina).

| Model | Dimensions | Speed | Languages |
|-------|------------|-------|-----------|
| `microsoft/unixcoder-base` | 768 | ~20x slower | 6 (java, ruby, python, php, js, go) |
| `microsoft/unixcoder-base-nine` | 768 | ~20x slower | 9 (+ c, c++, c#) |

**Installation** (requires additional dependencies):
```bash
pip install semantic-search-mcp[unixcoder]
```

**Usage:**
```bash
export SEMANTIC_SEARCH_EMBEDDING_MODEL="microsoft/unixcoder-base-nine"
```

**When to use UniXcoder:**
- You prioritize search quality over indexing speed
- Your codebase is small to medium sized
- You have GPU acceleration (CUDA or Apple Silicon MPS)

**When to avoid UniXcoder:**
- Large codebases (10,000+ files) - indexing will take hours
- You need fast initial indexing
- Running on CPU without GPU acceleration

## Claude Code Integration

Skills and commands are **automatically installed** when the MCP server first starts:
- **Skills** → `~/.claude/skills/` (AI auto-discovery)
- **Commands** → `~/.claude/commands/` (user-invocable slash commands)

To manually reinstall or update:
```bash
semantic-search-mcp-install-skills
```

### Available Slash Commands

| Command | Description |
|---------|-------------|
| `/semantic-search-search <query>` | Search codebase with natural language |
| `/semantic-search-status` | Check server status and index stats |
| `/semantic-search-reindex` | Trigger full codebase reindex |
| `/semantic-search-cancel` | Cancel running indexing job |
| `/semantic-search-clear` | Wipe all indexed data |
| `/semantic-search-pause` | Pause file watcher |
| `/semantic-search-resume` | Resume file watcher |

## Requirements

- Python 3.11+
- ~700MB disk for embedding model (downloaded on first run, ~150MB with INT8 quantization)
- ~1GB RAM for embedding model

## License

[MIT](LICENSE)
