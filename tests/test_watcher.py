# tests/test_watcher.py
"""Tests for file watcher module."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search_mcp.watcher import FileWatcher, ChangeEvent


@pytest.fixture
def mock_indexer():
    """Create mock indexer."""
    indexer = MagicMock()
    indexer.index_file = MagicMock(return_value={"status": "indexed", "chunks": 3})
    indexer.remove_file = MagicMock()
    indexer.gitignore = MagicMock()
    indexer.gitignore.should_index = MagicMock(return_value=True)
    return indexer


def test_watcher_creates_bounded_queue(mock_indexer, temp_dir):
    """Watcher should create a bounded queue."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    assert watcher.queue.maxsize == 100


def test_watcher_queue_event(mock_indexer, temp_dir):
    """Watcher should queue file change events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    event = ChangeEvent(type="modified", path=temp_dir / "test.py")
    watcher.queue_event(event)

    assert not watcher.queue.empty()
    queued = watcher.queue.get_nowait()
    assert queued.type == "modified"


def test_watcher_filters_non_indexable(mock_indexer, temp_dir):
    """Watcher should filter non-indexable files."""
    mock_indexer.gitignore.should_index = MagicMock(return_value=False)
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    event = ChangeEvent(type="modified", path=temp_dir / "node_modules" / "pkg.js")
    watcher.queue_event(event)

    # Event should be filtered out
    assert watcher.queue.empty()


@pytest.mark.asyncio
async def test_watcher_processes_modified_event(mock_indexer, temp_dir):
    """Watcher should call index_file for modified events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    test_file = temp_dir / "test.py"
    test_file.write_text("print('hello')")

    event = ChangeEvent(type="modified", path=test_file)
    await watcher._process_event(event)

    mock_indexer.index_file.assert_called_once_with(test_file)


@pytest.mark.asyncio
async def test_watcher_processes_deleted_event(mock_indexer, temp_dir):
    """Watcher should call remove_file for deleted events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    test_file = temp_dir / "test.py"
    event = ChangeEvent(type="deleted", path=test_file)
    await watcher._process_event(event)

    mock_indexer.remove_file.assert_called_once_with(test_file)


@pytest.mark.asyncio
async def test_watcher_handles_processing_error(mock_indexer, temp_dir):
    """Watcher should handle errors in event processing gracefully."""
    mock_indexer.index_file = MagicMock(side_effect=Exception("Test error"))
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    test_file = temp_dir / "test.py"
    test_file.write_text("print('hello')")

    event = ChangeEvent(type="modified", path=test_file)

    # Should not raise
    await watcher._process_event(event)


def test_watcher_queue_drops_on_full(mock_indexer, temp_dir):
    """When queue is full, oldest events should be dropped."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=2)

    # Fill queue
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "a.py"))
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "b.py"))

    # This should drop the oldest
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "c.py"))

    assert watcher.queue.qsize() == 2
