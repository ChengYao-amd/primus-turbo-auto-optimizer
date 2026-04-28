"""End-to-end tests of the watch HTTP server.

We bind the server to ``127.0.0.1:0`` (OS-chosen port), drive it via
``urllib.request``, and shut it down per-test. SSE is exercised by
publishing directly through the broker rather than mutating the
filesystem, which keeps the test deterministic; a separate test
covers the watchdog -> render path end-to-end.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from turbo_view.watch.server import (
    WatchServer,
    WatchSession,
    find_free_port,
)
from turbo_view.watch.sse import SSEBroker

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def _start_server(
    tmp_path: Path, *, mode: str = "single", root: Path | None = None
) -> tuple[WatchServer, str]:
    out_dir = tmp_path / "out"
    broker = SSEBroker()
    session = WatchSession(
        mode=mode,
        root=(root or FIXTURE).resolve(),
        out_dir=out_dir,
        broker=broker,
    )
    server = WatchServer(
        session=session,
        host="127.0.0.1",
        port=0,
        watch_dir=tmp_path / "_unused_watch_dir",  # avoid filesystem fan-out
    )
    # ``watch_dir`` doesn't exist on disk; CampaignWatcher.start logs
    # a warning and skips scheduling, which is exactly what we want
    # in unit tests that don't exercise the filesystem path.
    url = server.start()
    return server, url


def _get(url: str, *, timeout: float = 5.0) -> tuple[int, bytes, dict[str, str]]:
    with urlopen(url, timeout=timeout) as resp:
        body = resp.read()
        # Stdlib BaseHTTPRequestHandler folds header names with mixed
        # case (``Content-type``); normalise so tests don't care.
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status, body, headers


def test_server_serves_index_and_data(tmp_path: Path):
    server, url = _start_server(tmp_path)
    try:
        status, body, _ = _get(url)
        assert status == 200
        assert b"primus-turbo-view" in body
        assert b'name="turbo-view-watch"' in body, "watch meta tag missing"

        status, body, headers = _get(url + "data.json")
        assert status == 200
        assert headers.get("content-type", "").startswith("application/json")
        payload = json.loads(body)
        assert payload["state"]["campaign_id"] == "campaign_mini"
    finally:
        server.stop()


def test_server_serves_assets(tmp_path: Path):
    server, url = _start_server(tmp_path)
    try:
        for name in ("app.js", "app.css", "chart.umd.min.js"):
            status, body, _ = _get(url + "assets/" + name)
            assert status == 200, name
            assert len(body) > 100, name
    finally:
        server.stop()


def test_server_streams_reload_event(tmp_path: Path):
    server, url = _start_server(tmp_path)
    try:
        # Open the SSE channel in a background thread; the server
        # writes the connect comment immediately so we know we're
        # subscribed before publishing.
        chunks: list[bytes] = []
        ready = threading.Event()
        done = threading.Event()

        def consume():
            with urlopen(url + "events", timeout=5.0) as resp:
                while not done.is_set():
                    chunk = resp.fp.readline()
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"connected" in chunk:
                        ready.set()
                    if b"event: reload" in chunk:
                        done.set()
                        return

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        assert ready.wait(timeout=3.0), "subscriber never connected"

        server.session.broker.publish("reload", "1")
        assert done.wait(timeout=3.0), "reload event not received"
        assert any(b"event: reload" in c for c in chunks)
    finally:
        server.stop()


def test_server_tail_endpoint_returns_lines(tmp_path: Path):
    # Copy fixture so we can write into it without touching the
    # repository copy.
    workdir = tmp_path / "campaign"
    shutil.copytree(FIXTURE, workdir)
    transcript = workdir / "profiles" / "_transcript_ANALYZE.jsonl"
    transcript.write_text("\n".join(['{"i":1}', '{"i":2}', '{"i":3}']) + "\n",
                          encoding="utf-8")

    server, url = _start_server(tmp_path / "out", mode="single", root=workdir)
    try:
        status, body, _ = _get(url + "tail?phase=ANALYZE&n=10")
        assert status == 200
        data = json.loads(body)
        assert data["phase"] == "ANALYZE"
        assert data["lines"] == ['{"i":1}', '{"i":2}', '{"i":3}']
        assert data["path"] is not None
    finally:
        server.stop()


def test_server_tail_returns_empty_for_unknown_phase(tmp_path: Path):
    server, url = _start_server(tmp_path)
    try:
        status, body, _ = _get(url + "tail?phase=NOPE&n=5")
        assert status == 200
        data = json.loads(body)
        assert data["lines"] == []
        assert data["path"] is None
    finally:
        server.stop()


def test_server_tail_is_case_insensitive(tmp_path: Path):
    """Real workspaces emit ``_transcript_analyze.jsonl`` (lowercase);
    older fixtures emit ``_transcript_ANALYZE.jsonl``. Both must hit.
    """
    workdir = tmp_path / "campaign"
    shutil.copytree(FIXTURE, workdir)
    (workdir / "profiles" / "_transcript_analyze.jsonl").write_text(
        '{"i":1}\n{"i":2}\n', encoding="utf-8"
    )

    server, url = _start_server(tmp_path / "out", mode="single", root=workdir)
    try:
        status, body, _ = _get(url + "tail?phase=ANALYZE&n=10")
        assert status == 200
        data = json.loads(body)
        assert data["lines"]  # any non-empty result is acceptable
        assert data["path"] is not None
    finally:
        server.stop()


def test_server_phases_lists_transcripts(tmp_path: Path):
    workdir = tmp_path / "campaign"
    shutil.copytree(FIXTURE, workdir)
    (workdir / "profiles" / "_transcript_analyze.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (workdir / "profiles" / "_transcript_optimize.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    server, url = _start_server(tmp_path / "out", mode="single", root=workdir)
    try:
        status, body, _ = _get(url + "phases")
        assert status == 200
        data = json.loads(body)
        names = {p["phase"].lower() for p in data["phases"]}
        # at minimum the two we just wrote
        assert "analyze" in names
        assert "optimize" in names
    finally:
        server.stop()


def test_render_count_increases_after_on_change(tmp_path: Path):
    """``WatchSession.on_change`` is the timer's callback; calling it
    directly is the deterministic path through the rebuild + SSE
    publish without depending on inotify timing.
    """
    server, _ = _start_server(tmp_path)
    try:
        before = server.session.render_count
        server.session.on_change()
        assert server.session.render_count == before + 1
    finally:
        server.stop()


def test_find_free_port_returns_usable_port():
    p = find_free_port()
    assert 1024 < p < 65536


def test_server_swallows_client_disconnect(tmp_path: Path, capsys):
    """Aborting an inflight fetch (a common live-tail pattern) used
    to print a BrokenPipe traceback to stderr. The handler now logs
    at DEBUG instead.
    """
    import socket as _socket
    import time as _time

    server, url = _start_server(tmp_path)
    try:
        host, port = server._httpd.server_address[:2]
        # Send a half-formed GET, then close before reading the body.
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(b"GET /data.json HTTP/1.1\r\nHost: x\r\n\r\n")
            # Close immediately; the server is mid-write.
            s.shutdown(_socket.SHUT_RDWR)
        # Give the server thread a moment to surface any traceback.
        _time.sleep(0.2)
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "BrokenPipeError" not in err
    finally:
        server.stop()
