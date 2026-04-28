"""HTTP server for ``primus-turbo-view --watch``.

Design choice: the server does not template anything itself. On
startup (and again on every debounced filesystem change) we run the
exact same renderer that the offline build uses, writing fresh HTML
and assets into ``out_dir``. The HTTP server then only does two
things beyond serving static files:

* ``GET /events`` -- a long-lived SSE stream that emits ``reload``
  whenever the watcher fires.
* ``GET /tail?phase=<P>&n=<N>`` -- last ``N`` non-empty lines of the
  matching ``_transcript_<P>.jsonl`` (used by the live-tail panel).

This keeps the rendering pipeline single-sourced and the watch path
diff-free relative to the offline ``build`` command.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from turbo_view.io.loader import load_campaign
from turbo_view.render.build import write_detail
from turbo_view.render.overview import write_overview
from turbo_view.watch.sse import SSEBroker, stream
from turbo_view.watch.watcher import CampaignWatcher, tail_lines

log = logging.getLogger(__name__)


class WatchSession:
    """Hold the mutable state shared between watcher and server.

    Per spec §8 we re-run the static renderer on every change; the
    session object owns ``out_dir`` and the renderer callable so the
    request handler does not need to know which mode we are in.
    """

    def __init__(
        self,
        *,
        mode: str,
        root: Path,
        out_dir: Path,
        broker: SSEBroker,
    ) -> None:
        if mode not in ("single", "multi"):
            raise ValueError(f"invalid mode: {mode!r}")
        self.mode = mode
        self.root = root.resolve()
        self.out_dir = out_dir.resolve()
        self.broker = broker
        self._render_lock = threading.Lock()
        self._render_count = 0

    # ---- public ---------------------------------------------------

    def render(self) -> Path:
        with self._render_lock:
            if self.mode == "single":
                bundle = load_campaign(self.root)
                index = write_detail(
                    bundle, self.out_dir, watch_mode=True
                )
            else:
                index = write_overview(
                    self.root, self.out_dir, watch_mode=True
                )
            self._render_count += 1
            log.info("render #%d -> %s", self._render_count, index)
            return index

    def on_change(self) -> None:
        try:
            self.render()
        except Exception as exc:
            log.exception("render failed: %s", exc)
            self.broker.publish("error", str(exc))
            return
        self.broker.publish("reload", "")

    @property
    def render_count(self) -> int:
        return self._render_count


def _make_handler(session: WatchSession) -> type[SimpleHTTPRequestHandler]:
    out_dir = str(session.out_dir)

    class _Handler(SimpleHTTPRequestHandler):
        # SimpleHTTPRequestHandler resolves files relative to
        # ``directory``; passing it as a default keeps the handler
        # class pickle-safe and avoids reaching into the server
        # instance for it.
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, directory=out_dir, **kwargs)

        # ---- routing --------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 stdlib API
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/events":
                    self._serve_sse()
                    return
                if parsed.path == "/tail":
                    self._serve_tail(parsed.query)
                    return
                if parsed.path == "/phases":
                    self._serve_phases(parsed.query)
                    return
                super().do_GET()
            except (BrokenPipeError, ConnectionResetError) as exc:
                # Common when the browser aborts an inflight fetch
                # (live-tail polling overlaps with an SSE reload, or
                # the user navigates away). Drop to debug; the
                # default ThreadingHTTPServer would otherwise dump
                # the traceback to stderr.
                log.debug("client disconnected mid-response: %s", exc)

        # Stdlib BaseHTTPRequestHandler.handle_one_request swallows
        # most errors but escalates write failures (BrokenPipeError /
        # ConnectionResetError) to ``handle_error``, which prints to
        # stderr. Override so those two stay quiet without hiding
        # programming bugs.
        def handle_one_request(self) -> None:  # noqa: D401 stdlib name
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError) as exc:
                log.debug("client disconnected: %s", exc)
                self.close_connection = True

        # SSE inherits cleanly from base streaming logic
        def _serve_sse(self) -> None:
            q = session.broker.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                # Tell the client the current generation so it can
                # tell whether reload events apply to its render.
                first = f": connected gen={session.render_count}\n\n"
                self.wfile.write(first.encode("utf-8"))
                self.wfile.flush()
                for chunk in stream(q):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                session.broker.unsubscribe(q)

        def _serve_tail(self, query: str) -> None:
            params = parse_qs(query)
            phase = (params.get("phase") or [""])[0]
            campaign = (params.get("campaign") or [""])[0]
            try:
                n = int((params.get("n") or ["50"])[0])
            except ValueError:
                n = 50
            n = max(1, min(500, n))

            search_root = self._tail_root(campaign)
            path = self._find_transcript(search_root, phase) if search_root else None
            lines = tail_lines(path, n) if path else []

            body = json.dumps(
                {"phase": phase, "path": str(path) if path else None, "lines": lines}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _tail_root(self, campaign_id: str) -> Path | None:
            # Single mode: always the campaign dir. Multi mode + a
            # ``campaign=<id>`` hint: locate that campaign's directory
            # via discover_campaigns; without the hint we fall back
            # to the workspace root and let rglob find anything.
            if session.mode == "single":
                return session.root
            if not campaign_id:
                return session.root
            try:
                from turbo_view.discover import discover_campaigns

                for h in discover_campaigns(session.root):
                    if h.campaign_id == campaign_id:
                        return h.campaign_dir
            except Exception:
                log.exception("discover failed during tail lookup")
            return session.root

        def _find_transcript(self, root: Path, phase: str) -> Path | None:
            if not phase:
                return None
            # Match phase case-insensitively: real workspaces use
            # lowercase (``_transcript_analyze.jsonl``) while older
            # fixtures use uppercase. We pick the most recently
            # modified file when several variants coexist.
            phase_l = phase.lower()
            best: Path | None = None
            best_mtime = -1.0
            for hit in root.rglob("_transcript_*.jsonl"):
                stem = hit.name[len("_transcript_"):-len(".jsonl")]
                if stem.lower() != phase_l:
                    continue
                try:
                    mtime = hit.stat().st_mtime
                except OSError:
                    continue
                if mtime > best_mtime:
                    best_mtime = mtime
                    best = hit
            return best

        def _serve_phases(self, query: str) -> None:
            params = parse_qs(query)
            campaign = (params.get("campaign") or [""])[0]
            root = self._tail_root(campaign) or session.root
            phases: dict[str, dict[str, object]] = {}
            for hit in root.rglob("_transcript_*.jsonl"):
                stem = hit.name[len("_transcript_"):-len(".jsonl")]
                try:
                    mtime = hit.stat().st_mtime
                    size = hit.stat().st_size
                except OSError:
                    continue
                key = stem.lower()
                # Keep the newest variant per case-folded name, but
                # retain the on-disk casing for display.
                if key not in phases or mtime > phases[key]["mtime"]:  # type: ignore[operator]
                    phases[key] = {"phase": stem, "mtime": mtime, "size": size}
            ordered = sorted(
                phases.values(),
                key=lambda p: p["mtime"],  # type: ignore[arg-type]
                reverse=True,
            )
            body = json.dumps({"phases": ordered}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # quiet the default access log; we have our own
        def log_message(self, format: str, *args) -> None:  # type: ignore[override]
            log.debug("%s - " + format, self.address_string(), *args)

    return _Handler


class WatchServer:
    """Bind a ``ThreadingHTTPServer`` to ``host:port`` and start the
    watcher.

    The two threads are intentionally daemonic so that ``Ctrl+C`` in
    the foreground tears the whole thing down promptly without us
    having to wire signal handlers.
    """

    def __init__(
        self,
        *,
        session: WatchSession,
        host: str = "127.0.0.1",
        port: int = 8765,
        watch_dir: Path | None = None,
    ) -> None:
        self.session = session
        self.host = host
        self.port = port
        self._watch_dir = (watch_dir or session.root).resolve()
        self._httpd: ThreadingHTTPServer | None = None
        self._watcher: CampaignWatcher | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle ------------------------------------------------

    def start(self) -> str:
        # Render once before binding so a race-y first request
        # never sees a missing index.html.
        self.session.render()

        handler_cls = _make_handler(self.session)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)
        # Honour the OS-chosen port when the user passed 0.
        self.host, self.port = self._httpd.server_address[:2]

        self._watcher = CampaignWatcher(
            self._watch_dir, on_change=self.session.on_change
        )
        self._watcher.start()

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="turbo-view-http",
            daemon=True,
        )
        self._thread.start()
        url = f"http://{self.host}:{self.port}/"
        log.info("serving %s", url)
        return url

    def stop(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.session.broker.close()

    # ---- helpers --------------------------------------------------

    def __enter__(self) -> "WatchServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:  # type: ignore[no-untyped-def]
        self.stop()


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return an OS-assigned free TCP port. Used as a fallback when
    ``--port`` is omitted and the default 8765 is busy.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def serve(
    *,
    mode: str,
    root: Path,
    out_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    watch_dir: Path | None = None,
    open_browser: bool = True,
    on_started: Callable[[str], None] | None = None,
) -> WatchServer:
    """Start a watch server and (optionally) open the browser.

    Returns the running ``WatchServer``; the caller is expected to
    block (e.g. on a ``threading.Event`` or ``signal.pause``) and
    call :meth:`WatchServer.stop` on shutdown.
    """
    broker = SSEBroker()
    session = WatchSession(
        mode=mode, root=root, out_dir=out_dir, broker=broker
    )
    server = WatchServer(
        session=session, host=host, port=port, watch_dir=watch_dir
    )
    url = server.start()
    if on_started is not None:
        on_started(url)
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # pragma: no cover -- best effort
            log.warning("could not open browser for %s", url)
    return server
