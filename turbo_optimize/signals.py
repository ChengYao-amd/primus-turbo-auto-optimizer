"""SIGINT handling for unattended, long-running campaigns.

Semantics (per design):

- first `Ctrl+C` → set an `asyncio.Event` stop flag. The current phase
  checks the flag on each message, calls `client.interrupt()` to cut off
  the Claude request, and raises :class:`GracefulStop`. The orchestrator
  catches it, jumps to REPORT, writes the final log, then exits.
- second `Ctrl+C` → handler restores the default behaviour and re-raises
  `KeyboardInterrupt`, giving the user a hard exit.

`SIGTERM` is *not* caught: container stop goes through the default path
and the append-only logs remain consistent.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys


log = logging.getLogger(__name__)


class GracefulStop(Exception):
    """Raised inside a phase runner when SIGINT was observed."""


_stop_event: asyncio.Event | None = None


def get_stop_event() -> asyncio.Event:
    global _stop_event
    if _stop_event is None:
        _stop_event = asyncio.Event()
    return _stop_event


def stop_requested() -> bool:
    event = _stop_event
    return event is not None and event.is_set()


def install_sigint_handler() -> None:
    """Register a two-stage SIGINT handler.

    Must be called after `asyncio.run()` has entered its event loop,
    typically from the top of `run_campaign`.
    """
    event = get_stop_event()
    default_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):  # noqa: ARG001
        if event.is_set():
            log.warning("second SIGINT: restoring default handler and exiting")
            signal.signal(signal.SIGINT, default_handler)
            raise KeyboardInterrupt
        log.warning(
            "SIGINT received: requesting graceful shutdown; "
            "current phase will finalize, then jump to REPORT. "
            "Press Ctrl+C again to exit immediately."
        )
        event.set()

    if sys.platform == "win32":
        signal.signal(signal.SIGINT, handler)
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        signal.signal(signal.SIGINT, handler)
        return
    loop.add_signal_handler(signal.SIGINT, lambda: handler(signal.SIGINT, None))


def reset_stop_event() -> None:
    """Test-only helper. Clears the module-level event."""
    global _stop_event
    _stop_event = None
