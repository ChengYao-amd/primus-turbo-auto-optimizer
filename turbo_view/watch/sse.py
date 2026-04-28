"""In-process Server-Sent-Events broker.

A single ``SSEBroker`` instance is shared between the HTTP request
handler and the filesystem watcher. The handler subscribes a per-
connection queue; the watcher publishes events that fan out to every
queue.

Why a hand-rolled broker rather than ``asyncio``: the HTTP server is
a stdlib ``ThreadingHTTPServer``, so threads + queues map naturally
without dragging in an event loop. Each subscriber blocks on its
queue with a short timeout; the broker tracks subscribers in a set
under a lock so unsubscribe is O(1).
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Iterator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSEEvent:
    name: str
    data: str


class SSEBroker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: set[queue.Queue[SSEEvent]] = set()
        self._closed = False

    # ---- producer -------------------------------------------------

    def publish(self, name: str, data: str = "") -> None:
        if self._closed:
            return
        ev = SSEEvent(name=name, data=data)
        with self._lock:
            for q in tuple(self._subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    log.warning("SSE queue full; dropping subscriber")
                    self._subs.discard(q)

    # ---- consumer -------------------------------------------------

    def subscribe(self, maxsize: int = 32) -> queue.Queue[SSEEvent]:
        q: queue.Queue[SSEEvent] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[SSEEvent]) -> None:
        with self._lock:
            self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def close(self) -> None:
        self._closed = True
        with self._lock:
            self._subs.clear()


def format_sse(ev: SSEEvent) -> bytes:
    """Render an event in the SSE wire format (``event:`` + ``data:``)."""
    lines = [f"event: {ev.name}".encode("utf-8")]
    for line in ev.data.splitlines() or [""]:
        lines.append(f"data: {line}".encode("utf-8"))
    return b"\n".join(lines) + b"\n\n"


def stream(q: queue.Queue[SSEEvent], heartbeat_s: float = 15.0) -> Iterator[bytes]:
    """Block on ``q`` and yield SSE frames; emit a comment heartbeat
    every ``heartbeat_s`` seconds so idle proxies don't kill the
    connection.
    """
    while True:
        try:
            ev = q.get(timeout=heartbeat_s)
        except queue.Empty:
            yield b": keepalive\n\n"
            continue
        yield format_sse(ev)
