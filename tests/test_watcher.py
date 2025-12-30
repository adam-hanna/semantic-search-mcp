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
    """Watcher should respect max queue size setting."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    assert watcher.queue_max_size == 100


def test_watcher_queue_event(mock_indexer, temp_dir):
    """Watcher should queue file change events."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    event = ChangeEvent(type="modified", path=temp_dir / "test.py")
    watcher.queue_event(event)

    assert len(watcher._pending) == 1
    assert temp_dir / "test.py" in watcher._pending
    event_type, _ = watcher._pending[temp_dir / "test.py"]
    assert event_type == "modified"


def test_watcher_filters_non_indexable(mock_indexer, temp_dir):
    """Watcher should filter non-indexable files."""
    mock_indexer.gitignore.should_index = MagicMock(return_value=False)
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    event = ChangeEvent(type="modified", path=temp_dir / "node_modules" / "pkg.js")
    watcher.queue_event(event)

    # Event should be filtered out
    assert len(watcher._pending) == 0


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

    assert len(watcher._pending) == 2
    # c.py should be there, a.py should have been dropped
    assert temp_dir / "c.py" in watcher._pending


def test_watcher_deduplicates_events(mock_indexer, temp_dir):
    """Watcher should deduplicate events for the same file."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    # Queue multiple events for the same file
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "test.py"))
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "test.py"))
    watcher.queue_event(ChangeEvent(type="modified", path=temp_dir / "test.py"))

    # Should only have one entry
    assert len(watcher._pending) == 1


@pytest.mark.asyncio
async def test_watcher_batch_processing(mock_indexer, temp_dir):
    """Watcher should process events in batches."""
    watcher = FileWatcher(mock_indexer, temp_dir, queue_max_size=100)

    # Add multiple events
    await watcher._add_event(ChangeEvent(type="modified", path=temp_dir / "a.py"))
    await watcher._add_event(ChangeEvent(type="modified", path=temp_dir / "b.py"))
    await watcher._add_event(ChangeEvent(type="added", path=temp_dir / "c.py"))

    assert len(watcher._pending) == 3

    # Get batch should return all and clear pending
    batch = await watcher._get_pending_batch()
    assert len(batch) == 3
    assert len(watcher._pending) == 0


@pytest.mark.asyncio
async def test_pause_clears_pending_and_discards_events(temp_dir, mock_indexer):
    """Pausing should clear pending events and discard new ones."""
    watcher = FileWatcher(mock_indexer, temp_dir)

    # Add some events
    event1 = ChangeEvent(type="added", path=temp_dir / "file1.py")
    event2 = ChangeEvent(type="modified", path=temp_dir / "file2.py")
    await watcher._add_event(event1)
    await watcher._add_event(event2)

    # Pause should clear pending and return count
    discarded = await watcher.pause()
    assert discarded == 2
    assert watcher.is_paused
    assert len(watcher._pending) == 0

    # New events should be discarded when paused
    event3 = ChangeEvent(type="added", path=temp_dir / "file3.py")
    await watcher._add_event(event3)
    assert len(watcher._pending) == 0


@pytest.mark.asyncio
async def test_resume_allows_events(temp_dir, mock_indexer):
    """Resuming should allow events to be queued again."""
    watcher = FileWatcher(mock_indexer, temp_dir)

    await watcher.pause()
    assert watcher.is_paused

    await watcher.resume()
    assert not watcher.is_paused

    # Events should now be accepted
    event = ChangeEvent(type="added", path=temp_dir / "file1.py")
    await watcher._add_event(event)
    assert len(watcher._pending) == 1
