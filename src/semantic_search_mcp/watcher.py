# src/semantic_search_mcp/watcher.py
"""File system watcher for incremental indexing."""
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from watchfiles import Change, awatch

from semantic_search_mcp.indexer import FileIndexer


logger = logging.getLogger(__name__)

# Burst detection thresholds (e.g., git checkout, branch switch)
BURST_THRESHOLD = 20  # Events in window to trigger burst mode
BURST_WINDOW_SEC = 2.0  # Time window to detect burst
BURST_SETTLE_SEC = 2.0  # Wait time after last event before processing


@dataclass
class ChangeEvent:
    """File change event."""
    type: str  # 'added', 'modified', 'deleted'
    path: Path
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FileWatcher:
    """Watch filesystem for changes and trigger incremental indexing.

    Features:
    - Event deduplication: Same file only processed once per batch
    - Burst detection: Detects git checkout/branch switch and waits for settle
    - Batched processing: Processes all pending events together after burst
    """

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

    async def _add_event(self, event: ChangeEvent) -> None:
        """Add event to pending dict with deduplication.

        Args:
            event: Change event to add
        """
        # Filter non-indexable files
        if not self.indexer.gitignore.should_index(event.path):
            return

        async with self._pending_lock:
            # Enforce max size by removing oldest if needed
            if len(self._pending) >= self.queue_max_size and event.path not in self._pending:
                # Find and remove oldest
                oldest_path = min(self._pending, key=lambda p: self._pending[p][1])
                del self._pending[oldest_path]

            # Add/update event (deduplicates automatically)
            self._pending[event.path] = (event.type, event.timestamp)
            self._last_event_time = event.timestamp

            # Track for burst detection
            now = time.time()
            self._recent_events = [t for t in self._recent_events if now - t < BURST_WINDOW_SEC]
            self._recent_events.append(now)

            # Check if we're in burst mode
            if len(self._recent_events) >= BURST_THRESHOLD:
                if not self._in_burst_mode:
                    logger.info(f"Burst detected ({len(self._recent_events)} events) - waiting for settle...")
                self._in_burst_mode = True

    async def _get_pending_batch(self) -> list[ChangeEvent]:
        """Get all pending events as a batch, clearing the pending dict.

        Returns:
            List of events to process
        """
        async with self._pending_lock:
            if not self._pending:
                return []

            events = [
                ChangeEvent(type=event_type, path=path, timestamp=timestamp)
                for path, (event_type, timestamp) in self._pending.items()
            ]
            self._pending.clear()
            self._in_burst_mode = False
            return events

    async def _process_event(self, event: ChangeEvent) -> None:
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

    async def _process_loop(self) -> None:
        """Process events from the pending dict."""
        while self._running:
            try:
                # Check if we should wait for burst to settle
                now = time.time()
                time_since_last = now - self._last_event_time if self._last_event_time > 0 else float('inf')

                if self._in_burst_mode and time_since_last < BURST_SETTLE_SEC:
                    # Still in burst mode, wait for settle
                    await asyncio.sleep(0.5)
                    continue

                # Get batch of events
                events = await self._get_pending_batch()

                if not events:
                    await asyncio.sleep(0.5)
                    continue

                if len(events) > 1:
                    logger.info(f"Processing batch of {len(events)} file changes...")

                # Process all events
                for event in events:
                    if not self._running:
                        break
                    await self._process_event(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                await asyncio.sleep(1.0)

    async def _watch_loop(self) -> None:
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

                    await self._add_event(ChangeEvent(type=event_type, path=path))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Watch loop error: {e}")

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Started watching: {self.watch_dir}")

    async def stop(self) -> None:
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

    # Legacy method for backwards compatibility with tests
    def queue_event(self, event: ChangeEvent) -> None:
        """Queue a change event for processing (sync wrapper).

        Deprecated: Use _add_event() for async code.
        """
        # Filter non-indexable files
        if not self.indexer.gitignore.should_index(event.path):
            return

        # Run in event loop if available
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._add_event(event))
        except RuntimeError:
            # No running loop - add directly (for tests)
            # Enforce max size by removing oldest if needed
            if len(self._pending) >= self.queue_max_size and event.path not in self._pending:
                oldest_path = min(self._pending, key=lambda p: self._pending[p][1])
                del self._pending[oldest_path]

            self._pending[event.path] = (event.type, event.timestamp or time.time())
