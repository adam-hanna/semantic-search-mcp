# Server Control Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add MCP tools to control the semantic-search-mcp server: pause/resume watcher, reindex, cancel indexing, clear index, exclude paths, and get status.

**Architecture:** Extend ServerState to track watcher/indexing state. Add pause/resume methods to FileWatcher. Add cancellation flag to FileIndexer. Add runtime exclusions to GitignoreFilter. Expose all controls as MCP tools in server.py.

**Tech Stack:** Python, FastMCP, asyncio, pathspec (for glob patterns)

---

### Task 1: Add Pause/Resume to FileWatcher

**Files:**
- Modify: `src/semantic_search_mcp/watcher.py:35-75`
- Test: `tests/test_watcher.py`

**Step 1: Write the failing test for pause**

Add to `tests/test_watcher.py`:

```python
@pytest.mark.asyncio
async def test_pause_clears_pending_and_discards_events(tmp_path, mock_indexer):
    """Pausing should clear pending events and discard new ones."""
    watcher = FileWatcher(mock_indexer, tmp_path)

    # Add some events
    event1 = ChangeEvent(type="added", path=tmp_path / "file1.py")
    event2 = ChangeEvent(type="modified", path=tmp_path / "file2.py")
    await watcher._add_event(event1)
    await watcher._add_event(event2)

    # Pause should clear pending and return count
    discarded = await watcher.pause()
    assert discarded == 2
    assert watcher.is_paused
    assert len(watcher._pending) == 0

    # New events should be discarded when paused
    event3 = ChangeEvent(type="added", path=tmp_path / "file3.py")
    await watcher._add_event(event3)
    assert len(watcher._pending) == 0


@pytest.mark.asyncio
async def test_resume_allows_events(tmp_path, mock_indexer):
    """Resuming should allow events to be queued again."""
    watcher = FileWatcher(mock_indexer, tmp_path)

    await watcher.pause()
    assert watcher.is_paused

    await watcher.resume()
    assert not watcher.is_paused

    # Events should now be accepted
    event = ChangeEvent(type="added", path=tmp_path / "file1.py")
    await watcher._add_event(event)
    assert len(watcher._pending) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_watcher.py::test_pause_clears_pending_and_discards_events -v`
Expected: FAIL with "AttributeError: 'FileWatcher' object has no attribute 'pause'"

**Step 3: Implement pause/resume in FileWatcher**

Modify `src/semantic_search_mcp/watcher.py`. Add `_paused` field and `is_paused` property in `__init__`:

```python
def __init__(
    self,
    indexer: FileIndexer,
    watch_dir: Path,
    queue_max_size: int = 1000,
    debounce_ms: int = 1000,
):
    """Initialize file watcher.

    Args:
        indexer: FileIndexer instance
        watch_dir: Directory to watch
        queue_max_size: Maximum pending events (oldest dropped on overflow)
        debounce_ms: Debounce interval in milliseconds
    """
    self.indexer = indexer
    self.watch_dir = Path(watch_dir).resolve()
    self.queue_max_size = queue_max_size
    self.debounce_ms = debounce_ms
    self._running = False
    self._paused = False
    self._watch_task: Optional[asyncio.Task] = None
    self._process_task: Optional[asyncio.Task] = None

    # Deduplication: path -> (event_type, timestamp)
    # Only keeps the latest event per path
    self._pending: dict[Path, tuple[str, float]] = {}
    self._pending_lock = asyncio.Lock()

    # Burst detection
    self._recent_events: list[float] = []  # Timestamps of recent events
    self._last_event_time: float = 0.0
    self._in_burst_mode: bool = False

@property
def is_paused(self) -> bool:
    """Return whether the watcher is paused."""
    return self._paused
```

Add `pause` and `resume` methods after `stop`:

```python
async def pause(self) -> int:
    """Pause watching and discard pending events.

    Returns:
        Number of events discarded
    """
    async with self._pending_lock:
        discarded = len(self._pending)
        self._pending.clear()
        self._paused = True
        logger.info(f"Watcher paused, {discarded} events discarded")
        return discarded

async def resume(self) -> None:
    """Resume watching for file changes."""
    self._paused = False
    logger.info("Watcher resumed")
```

Modify `_add_event` to check pause state at the start:

```python
async def _add_event(self, event: ChangeEvent) -> None:
    """Add event to pending dict with deduplication.

    Args:
        event: Change event to add
    """
    # Discard events when paused
    if self._paused:
        return

    # Filter non-indexable files
    if not self.indexer.gitignore.should_index(event.path):
        return
    # ... rest of method unchanged
```

Modify `_process_loop` to skip when paused, after the `while self._running:` line:

```python
async def _process_loop(self) -> None:
    """Process events from the pending dict."""
    while self._running:
        try:
            # Skip processing when paused
            if self._paused:
                await asyncio.sleep(0.5)
                continue

            # Check if we should wait for burst to settle
            # ... rest of method unchanged
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_watcher.py::test_pause_clears_pending_and_discards_events tests/test_watcher.py::test_resume_allows_events -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/watcher.py tests/test_watcher.py
git commit -m "feat(watcher): add pause/resume methods"
```

---

### Task 2: Add Cancellation to FileIndexer

**Files:**
- Modify: `src/semantic_search_mcp/indexer.py:182-247`
- Test: `tests/test_indexer.py`

**Step 1: Write the failing test for cancellation**

Add to `tests/test_indexer.py`:

```python
def test_index_directory_respects_cancel_flag(tmp_path, db, embedder):
    """Indexing should stop when cancel flag returns True."""
    # Create multiple files
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")

    indexer = FileIndexer(db, embedder, tmp_path)

    # Cancel after 3 files
    call_count = 0
    def cancel_after_3():
        nonlocal call_count
        call_count += 1
        return call_count > 3

    stats = indexer.index_directory(cancel_flag=cancel_after_3)

    assert stats.get("cancelled") is True
    assert stats["files_indexed"] + stats["files_skipped"] + stats["files_error"] <= 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_indexer.py::test_index_directory_respects_cancel_flag -v`
Expected: FAIL with "TypeError: index_directory() got an unexpected keyword argument 'cancel_flag'"

**Step 3: Implement cancellation in FileIndexer**

Modify `index_directory` method signature in `src/semantic_search_mcp/indexer.py`:

```python
def index_directory(
    self,
    directory: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cancel_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """Index all files in a directory with memory-efficient batching.

    Args:
        directory: Directory to index (defaults to root_dir)
        progress_callback: Callback(current, total, message)
        batch_size: Number of files to process before garbage collection
        cancel_flag: Callable returning True to cancel indexing

    Returns:
        Stats dict with files_indexed, files_skipped, cancelled, etc.
    """
    directory = Path(directory or self.root_dir).resolve()

    # Get all indexable files
    files = [f for f in directory.rglob("*") if self.gitignore.should_index(f)]
    total = len(files)

    stats = {
        "files_indexed": 0,
        "files_skipped": 0,
        "files_error": 0,
        "total_chunks": 0,
        "cancelled": False,
    }

    logger.info(f"Found {total} files to index (batch size: {batch_size})")

    for i, filepath in enumerate(files):
        # Check cancellation flag
        if cancel_flag and cancel_flag():
            stats["cancelled"] = True
            logger.info(f"Indexing cancelled at {i}/{total} files")
            break

        if progress_callback:
            progress_callback(i, total, f"Indexing {filepath.name}")

        result = self.index_file(filepath)

        if result["status"] == "indexed":
            stats["files_indexed"] += 1
            stats["total_chunks"] += result["chunks"]
        elif result["status"] == "skipped":
            stats["files_skipped"] += 1
        else:
            stats["files_error"] += 1

        # Memory management: run garbage collection after each batch
        if (i + 1) % batch_size == 0:
            gc.collect()
            logger.info(
                f"Progress: {i + 1}/{total} files "
                f"({stats['files_indexed']} indexed, {stats['files_skipped']} skipped)"
            )

    # Final garbage collection
    gc.collect()

    if progress_callback and not stats["cancelled"]:
        progress_callback(total, total, "Complete")

    logger.info(
        f"Indexing {'cancelled' if stats['cancelled'] else 'complete'}: "
        f"{stats['files_indexed']} indexed, "
        f"{stats['files_skipped']} skipped, {stats['files_error']} errors, "
        f"{stats['total_chunks']} chunks"
    )

    return stats
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_indexer.py::test_index_directory_respects_cancel_flag -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/indexer.py tests/test_indexer.py
git commit -m "feat(indexer): add cancel_flag to index_directory"
```

---

### Task 3: Add Runtime Exclusions to GitignoreFilter

**Files:**
- Modify: `src/semantic_search_mcp/gitignore.py`
- Test: `tests/test_gitignore.py`

**Step 1: Write the failing test for runtime exclusions**

Add to `tests/test_gitignore.py`:

```python
def test_runtime_exclusions(tmp_path):
    """Runtime exclusions should be respected by should_index."""
    # Create a Python file
    py_file = tmp_path / "src" / "main.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("print('hello')")

    gf = GitignoreFilter(tmp_path)

    # File should be indexable initially
    assert gf.should_index(py_file) is True

    # Add runtime exclusion
    gf.add_exclusions(["src"])
    assert gf.should_index(py_file) is False
    assert "src" in gf.get_exclusions()

    # Remove exclusion
    gf.remove_exclusions(["src"])
    assert gf.should_index(py_file) is True
    assert "src" not in gf.get_exclusions()


def test_runtime_exclusions_glob_patterns(tmp_path):
    """Runtime exclusions should support glob patterns."""
    test_file = tmp_path / "test_main.py"
    test_file.write_text("def test(): pass")

    src_file = tmp_path / "main.py"
    src_file.write_text("def main(): pass")

    gf = GitignoreFilter(tmp_path)

    # Both should be indexable initially
    assert gf.should_index(test_file) is True
    assert gf.should_index(src_file) is True

    # Exclude test files
    gf.add_exclusions(["test_*.py"])
    assert gf.should_index(test_file) is False
    assert gf.should_index(src_file) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_gitignore.py::test_runtime_exclusions -v`
Expected: FAIL with "AttributeError: 'GitignoreFilter' object has no attribute 'add_exclusions'"

**Step 3: Read GitignoreFilter to understand structure**

Run: `head -50 src/semantic_search_mcp/gitignore.py`

**Step 4: Implement runtime exclusions in GitignoreFilter**

Add to `src/semantic_search_mcp/gitignore.py` in `__init__`:

```python
def __init__(self, root_dir: Path):
    """Initialize gitignore filter.

    Args:
        root_dir: Root directory for the project
    """
    self.root_dir = Path(root_dir).resolve()
    self._specs: dict[Path, pathspec.PathSpec] = {}
    self._runtime_exclusions: set[str] = set()
    self._load_gitignores()
```

Add new methods after `should_index`:

```python
def add_exclusions(self, patterns: list[str]) -> None:
    """Add runtime exclusion patterns (session-only).

    Args:
        patterns: List of glob patterns to exclude
    """
    for pattern in patterns:
        self._runtime_exclusions.add(pattern)
    logger.info(f"Added exclusions: {patterns}")

def remove_exclusions(self, patterns: list[str]) -> None:
    """Remove runtime exclusion patterns.

    Args:
        patterns: List of patterns to remove
    """
    for pattern in patterns:
        self._runtime_exclusions.discard(pattern)
    logger.info(f"Removed exclusions: {patterns}")

def get_exclusions(self) -> list[str]:
    """Get current runtime exclusion patterns.

    Returns:
        List of active exclusion patterns
    """
    return list(self._runtime_exclusions)

def clear_exclusions(self) -> None:
    """Clear all runtime exclusions."""
    self._runtime_exclusions.clear()
    logger.info("Cleared all runtime exclusions")

def _matches_runtime_exclusion(self, path: Path) -> bool:
    """Check if path matches any runtime exclusion pattern.

    Args:
        path: Path to check

    Returns:
        True if path should be excluded
    """
    if not self._runtime_exclusions:
        return False

    try:
        rel_path = path.relative_to(self.root_dir)
    except ValueError:
        rel_path = path

    rel_str = str(rel_path)

    for pattern in self._runtime_exclusions:
        # Check if pattern matches the path or any parent
        spec = pathspec.PathSpec.from_lines('gitwildmatch', [pattern])
        if spec.match_file(rel_str):
            return True
        # Also check each path component
        for part in rel_path.parts:
            if spec.match_file(part):
                return True

    return False
```

Modify `should_index` to check runtime exclusions:

```python
def should_index(self, filepath: Path) -> bool:
    """Check if a file should be indexed.

    Args:
        filepath: Path to check

    Returns:
        True if the file should be indexed
    """
    filepath = Path(filepath).resolve()

    # Check runtime exclusions first
    if self._matches_runtime_exclusion(filepath):
        return False

    # ... rest of existing method
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gitignore.py::test_runtime_exclusions tests/test_gitignore.py::test_runtime_exclusions_glob_patterns -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/semantic_search_mcp/gitignore.py tests/test_gitignore.py
git commit -m "feat(gitignore): add runtime exclusion patterns"
```

---

### Task 4: Expand ServerState

**Files:**
- Modify: `src/semantic_search_mcp/server.py:64-73`

**Step 1: Expand ServerState class**

Modify `ServerState` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 2: Commit**

```bash
git add src/semantic_search_mcp/server.py
git commit -m "feat(server): expand ServerState with watcher and indexing fields"
```

---

### Task 5: Add pause_watcher Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_pause_watcher_tool(server_with_watcher):
    """pause_watcher should pause the watcher and return status."""
    mcp, components, state = server_with_watcher

    # Start the watcher
    await components.watcher.start()
    state.watcher_status = "running"

    # Get the tool
    tools = {t.name: t for t in await mcp.list_tools()}
    assert "pause_watcher" in tools

    # Call pause_watcher
    result = await mcp.call_tool("pause_watcher", {})

    assert result["status"] == "paused"
    assert "events_discarded" in result
    assert components.watcher.is_paused
    assert state.watcher_status == "paused"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_pause_watcher_tool -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement pause_watcher tool**

Add after the `reindex_file` tool in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_pause_watcher_tool -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add pause_watcher tool"
```

---

### Task 6: Add resume_watcher Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_resume_watcher_tool(server_with_watcher):
    """resume_watcher should resume a paused watcher."""
    mcp, components, state = server_with_watcher

    # Start and pause the watcher
    await components.watcher.start()
    await components.watcher.pause()
    state.watcher_status = "paused"

    # Call resume_watcher
    result = await mcp.call_tool("resume_watcher", {})

    assert result["status"] == "running"
    assert not components.watcher.is_paused
    assert state.watcher_status == "running"


@pytest.mark.asyncio
async def test_resume_watcher_when_not_paused(server_with_watcher):
    """resume_watcher when already running should be a no-op."""
    mcp, components, state = server_with_watcher

    await components.watcher.start()
    state.watcher_status = "running"

    result = await mcp.call_tool("resume_watcher", {})

    assert result["status"] == "already_running"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_resume_watcher_tool -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement resume_watcher tool**

Add after `pause_watcher` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server.py::test_resume_watcher_tool tests/test_server.py::test_resume_watcher_when_not_paused -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add resume_watcher tool"
```

---

### Task 7: Add cancel_indexing Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_cancel_indexing_when_not_running(server_with_components):
    """cancel_indexing when not indexing should return not_running."""
    mcp, components, state = server_with_components

    state.indexing_in_progress = False

    result = await mcp.call_tool("cancel_indexing", {})

    assert result["status"] == "not_running"


@pytest.mark.asyncio
async def test_cancel_indexing_sets_flag(server_with_components):
    """cancel_indexing should set the cancelled flag."""
    mcp, components, state = server_with_components

    state.indexing_in_progress = True
    state.indexing_cancelled = False

    result = await mcp.call_tool("cancel_indexing", {})

    assert result["status"] == "cancelling"
    assert state.indexing_cancelled is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_cancel_indexing_when_not_running -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement cancel_indexing tool**

Add after `resume_watcher` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server.py::test_cancel_indexing_when_not_running tests/test_server.py::test_cancel_indexing_sets_flag -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add cancel_indexing tool"
```

---

### Task 8: Add clear_index Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_clear_index_tool(server_with_components):
    """clear_index should wipe all indexed data."""
    mcp, components, state = server_with_components

    # Add some data to the index
    components.db.upsert_file("/test/file.py", "hash123", "python")

    result = await mcp.call_tool("clear_index", {})

    assert result["status"] == "cleared"
    assert result["files_removed"] >= 0
    assert result["chunks_removed"] >= 0

    # Verify data is gone
    stats = components.db.get_stats()
    assert stats["files"] == 0
    assert stats["chunks"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_clear_index_tool -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement clear_index tool**

Add after `cancel_indexing` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_clear_index_tool -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add clear_index tool"
```

---

### Task 9: Add exclude_paths Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_exclude_paths_tool(server_with_components):
    """exclude_paths should add patterns to gitignore filter."""
    mcp, components, state = server_with_components

    result = await mcp.call_tool("exclude_paths", {
        "patterns": ["node_modules", "*.test.py"]
    })

    assert result["status"] == "updated"
    assert "node_modules" in result["excluded_patterns"]
    assert "*.test.py" in result["excluded_patterns"]

    # Verify patterns are in the filter
    exclusions = components.indexer.gitignore.get_exclusions()
    assert "node_modules" in exclusions
    assert "*.test.py" in exclusions
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_exclude_paths_tool -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement exclude_paths tool**

Add after `clear_index` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_exclude_paths_tool -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add exclude_paths tool"
```

---

### Task 10: Add include_paths Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_include_paths_tool(server_with_components):
    """include_paths should remove patterns from exclusion list."""
    mcp, components, state = server_with_components

    # First exclude some patterns
    components.indexer.gitignore.add_exclusions(["node_modules", "vendor", "*.test.py"])

    # Remove one pattern
    result = await mcp.call_tool("include_paths", {
        "patterns": ["vendor"]
    })

    assert result["status"] == "updated"
    assert "vendor" not in result["excluded_patterns"]
    assert "node_modules" in result["excluded_patterns"]
    assert "*.test.py" in result["excluded_patterns"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_include_paths_tool -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement include_paths tool**

Add after `exclude_paths` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_include_paths_tool -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add include_paths tool"
```

---

### Task 11: Add get_status Tool

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_get_status_tool(server_with_components):
    """get_status should return comprehensive server state."""
    mcp, components, state = server_with_components

    state.status = "ready"
    state.watcher_status = "running"
    state.files_indexed = 10
    state.total_chunks = 50
    state.model = "test-model"

    result = await mcp.call_tool("get_status", {})

    assert result["server_status"] == "ready"
    assert result["watcher_status"] == "running"
    assert result["index"]["files"] == 10
    assert result["index"]["chunks"] == 50
    assert result["model"] == "test-model"
    assert "indexing" in result
    assert "excluded_patterns" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_get_status_tool -v`
Expected: FAIL (tool doesn't exist yet)

**Step 3: Implement get_status tool**

Add after `include_paths` in `src/semantic_search_mcp/server.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_get_status_tool -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add get_status tool"
```

---

### Task 12: Refactor reindex Tool (rename from initialize)

**Files:**
- Modify: `src/semantic_search_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_reindex_tool_runs_in_background(server_with_components):
    """reindex should start indexing in background and return immediately."""
    mcp, components, state = server_with_components

    state.status = "ready"
    state.indexing_in_progress = False

    result = await mcp.call_tool("reindex", {"force": True})

    assert result["status"] == "started"
    assert "files_found" in result


@pytest.mark.asyncio
async def test_reindex_fails_when_already_indexing(server_with_components):
    """reindex should fail if indexing is already in progress."""
    mcp, components, state = server_with_components

    state.indexing_in_progress = True

    result = await mcp.call_tool("reindex", {})

    assert result["status"] == "error"
    assert "already" in result["reason"].lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_reindex_tool_runs_in_background -v`
Expected: FAIL (tool doesn't exist or has wrong behavior)

**Step 3: Implement reindex tool**

Add new `reindex` tool (keep `initialize` for backwards compatibility) in `src/semantic_search_mcp/server.py`:

```python
@mcp.tool()
async def reindex(
    force: bool = Field(
        default=True,
        description="Force reindex even if files haven't changed"
    ),
    clear_first: bool = Field(
        default=False,
        description="Clear all existing index data before reindexing"
    ),
) -> dict:
    """
    Start a full reindex of the codebase.

    Runs in the background - use get_status to monitor progress.
    Use cancel_indexing to abort if needed.
    """
    if state.status not in ("ready", "error"):
        return {"status": "error", "reason": "Server not ready"}

    if state.indexing_in_progress:
        return {"status": "error", "reason": "Indexing already in progress"}

    if components.db is None or components.indexer is None:
        return {"status": "error", "reason": "Components not initialized"}

    # Clear if requested
    if clear_first:
        components.db.conn.execute("DELETE FROM vec_chunks")
        components.db.conn.execute("DELETE FROM chunks")
        components.db.conn.execute("DELETE FROM files")
        components.db.conn.commit()

    # Count files to index
    files = [f for f in root_dir.rglob("*") if components.indexer.gitignore.should_index(f)]
    files_found = len(files)

    # Reset cancellation flag
    state.indexing_cancelled = False
    state.indexing_in_progress = True
    state.indexing_progress = {"current": 0, "total": files_found, "current_file": ""}

    async def run_indexing():
        try:
            def progress_callback(current, total, message):
                state.indexing_progress = {
                    "current": current,
                    "total": total,
                    "current_file": message.replace("Indexing ", ""),
                }

            def cancel_flag():
                return state.indexing_cancelled

            if force:
                # Clear for force reindex
                components.db.conn.execute("DELETE FROM vec_chunks")
                components.db.conn.execute("DELETE FROM chunks")
                components.db.conn.execute("DELETE FROM files")
                components.db.conn.commit()

            stats = await asyncio.to_thread(
                components.indexer.index_directory,
                root_dir,
                progress_callback,
                config.index_batch_size,
                cancel_flag,
            )

            state.files_indexed = stats["files_indexed"]
            state.total_chunks = stats["total_chunks"]
            state.last_indexed_at = datetime.now(timezone.utc).isoformat()

            if stats.get("cancelled"):
                logger.info("Reindex cancelled")
            else:
                logger.info(f"Reindex complete: {stats['files_indexed']} files, {stats['total_chunks']} chunks")

        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            state.error = str(e)
        finally:
            state.indexing_in_progress = False
            state.indexing_cancelled = False

    # Start in background
    asyncio.create_task(run_indexing())

    return {
        "status": "started",
        "files_found": files_found,
        "clear_first": clear_first,
        "force": force,
    }
```

Add import at top of file:

```python
from datetime import datetime, timezone
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server.py::test_reindex_tool_runs_in_background tests/test_server.py::test_reindex_fails_when_already_indexing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/semantic_search_mcp/server.py tests/test_server.py
git commit -m "feat(server): add reindex tool with background execution"
```

---

### Task 13: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md with new tools**

Add new section after "Use" in README.md:

```markdown
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
```

**Step 2: Update CLAUDE.md MCP Tools section**

Update the MCP Tools section in CLAUDE.md:

```markdown
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

### get_status
Returns comprehensive server state including watcher status, indexing progress, and statistics.

### pause_watcher / resume_watcher
Control the file watcher. Events during pause are discarded.

### reindex
Start a full reindex in the background. Use `get_status` to monitor progress.
```python
reindex(
    force: bool = True,       # Reindex even unchanged files
    clear_first: bool = False # Wipe index before starting
)
```

### cancel_indexing
Cancel any running indexing job. Partial results are kept.

### clear_index
Wipe all indexed data from the database.

### exclude_paths / include_paths
Manage runtime path exclusions (session-only, reset on restart).
```python
exclude_paths(patterns: ["node_modules", "*.test.py"])
include_paths(patterns: ["node_modules"])  # Remove from exclusions
```
```

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add new control tools to documentation"
```

---

### Task 14: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run with coverage**

Run: `pytest tests/ --cov=semantic_search_mcp --cov-report=term-missing`
Expected: Coverage maintained or improved

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address test failures" # if needed
```

---

### Task 15: Integration Test

**Step 1: Run the server manually**

Run: `timeout 10 python -m semantic_search_mcp.server 2>&1 || true`
Expected: Server starts, shows "Server ready to accept connections"

**Step 2: Verify no import errors or startup crashes**

Check output for errors. If any, fix and commit.
