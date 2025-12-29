# src/semantic_search_mcp/watcher.py
"""File system watcher for incremental indexing."""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Optional

from watchfiles import Change, awatch

from semantic_search_mcp.indexer import FileIndexer


logger = logging.getLogger(__name__)


@dataclass
class ChangeEvent:
    """File change event."""
    type: str  # 'added', 'modified', 'deleted'
    path: Path


class FileWatcher:
    """Watch filesystem for changes and trigger incremental indexing."""

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
            queue_max_size: Maximum queue size (drops oldest on overflow)
            debounce_ms: Debounce interval in milliseconds
        """
        self.indexer = indexer
        self.watch_dir = Path(watch_dir).resolve()
        self.queue: Queue[ChangeEvent] = Queue(maxsize=queue_max_size)
        self.debounce_ms = debounce_ms
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None

    def queue_event(self, event: ChangeEvent):
        """Queue a change event for processing.

        If the queue is full, drops the oldest event.

        Args:
            event: Change event to queue
        """
        # Filter non-indexable files
        if not self.indexer.gitignore.should_index(event.path):
            return

        try:
            self.queue.put_nowait(event)
        except Full:
            # Drop oldest event
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(event)
            except Exception:
                pass

    async def _process_event(self, event: ChangeEvent):
        """Process a single change event.

        Args:
            event: Change event to process
        """
        try:
            if event.type == "deleted":
                self.indexer.remove_file(event.path)
                logger.info(f"Removed from index: {event.path}")
            else:
                result = self.indexer.index_file(event.path)
                if result["status"] == "indexed":
                    logger.info(f"Indexed: {event.path} ({result['chunks']} chunks)")
                elif result["status"] == "skipped":
                    logger.debug(f"Skipped: {event.path} ({result.get('reason', 'unknown')})")
        except Exception as e:
            logger.error(f"Error processing {event.path}: {e}")

    async def _process_loop(self):
        """Process events from the queue."""
        while self._running:
            try:
                # Non-blocking get with timeout
                event = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.queue.get(timeout=1.0)
                )
                await self._process_event(event)
            except Exception:
                # Queue.get timeout or other error
                continue

    async def _watch_loop(self):
        """Watch filesystem for changes."""
        try:
            async for changes in awatch(
                str(self.watch_dir),
                debounce=self.debounce_ms,
                recursive=True,
                force_polling=False,
            ):
                if not self._running:
                    break

                for change_type, path_str in changes:
                    path = Path(path_str)

                    if change_type == Change.added:
                        event_type = "added"
                    elif change_type == Change.modified:
                        event_type = "modified"
                    elif change_type == Change.deleted:
                        event_type = "deleted"
                    else:
                        continue

                    self.queue_event(ChangeEvent(type=event_type, path=path))

        except Exception as e:
            logger.error(f"Watch loop error: {e}")

    async def start(self):
        """Start watching for file changes."""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Started watching: {self.watch_dir}")

    async def stop(self):
        """Stop watching for file changes."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped watching")
