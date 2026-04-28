"""Filesystem watcher with 500 ms debounce.

The watchdog observer fires for every individual file event. Most
``primus-turbo-optimize`` actions write multiple files per phase
(``cost.md`` + ``run.json`` + a transcript line), so without
debouncing we'd queue half a dozen rebuilds for one logical change.

We collapse a burst into a single rebuild + a single ``reload`` SSE
event by setting a timer that resets on every event.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger(__name__)


class _DebouncedTrigger:
    def __init__(self, callback: Callable[[], None], delay_s: float) -> None:
        self._cb = callback
        self._delay = delay_s
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def fire(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._cb)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


# Only treat events that actually change file contents as "the
# campaign moved on". Watchdog also reports IN_OPEN / IN_CLOSE_NOWRITE
# for read-only access, and on a 100-round campaign our own render
# pass opens hundreds of files (summary.md, benchmark.csv, ...) -- a
# wide-open ``on_any_event`` then re-fires the debounce timer for
# every single open/close, so the renderer chains into a permanent
# rebuild loop. White-listing real mutations breaks that cycle.
_MUTATING_EVENT_TYPES = frozenset({
    "created",
    "modified",
    "deleted",
    "moved",
    "closed",  # IN_CLOSE_WRITE: write finished, contents committed
})


class _ChangeHandler(FileSystemEventHandler):
    def __init__(self, trigger: _DebouncedTrigger) -> None:
        self._trigger = trigger

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if event.event_type not in _MUTATING_EVENT_TYPES:
            return
        path = event.src_path or ""
        # Ignore our own output to avoid feedback loops; the path is
        # always inside ``<campaign>/view/`` for the static dashboard
        # and ``out_dir`` for watch mode.
        if "/view/" in path or path.endswith(".swp"):
            return
        self._trigger.fire()


class CampaignWatcher:
    """Wrap a single ``Observer`` on a campaign directory.

    The user-supplied ``on_change`` is called from the timer thread,
    so it MUST be cheap and thread-safe (the typical pattern is to
    rebuild the payload + ``broker.publish('reload')``).
    """

    def __init__(
        self,
        campaign_dir: Path,
        on_change: Callable[[], None],
        debounce_s: float = 0.5,
    ) -> None:
        self._dir = campaign_dir.resolve()
        self._trigger = _DebouncedTrigger(on_change, debounce_s)
        self._handler = _ChangeHandler(self._trigger)
        self._observer = Observer()
        self._started = False

    def start(self) -> None:
        if not self._dir.is_dir():
            log.warning("watcher: not a directory: %s", self._dir)
            return
        self._observer.schedule(self._handler, str(self._dir), recursive=True)
        self._observer.start()
        self._started = True
        log.info("watching %s", self._dir)

    def stop(self) -> None:
        self._trigger.cancel()
        if not self._started:
            return
        try:
            self._observer.stop()
        except RuntimeError:
            pass
        self._observer.join(timeout=2.0)


def tail_lines(path: Path, n: int = 50) -> list[str]:
    """Return the last ``n`` non-empty lines of ``path``.

    Used by panel 9 (live tail). Reads from disk on every poll — fine
    because the file is small (one phase's transcript).
    """
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-n:]
