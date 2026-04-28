"""Unit tests for the SSE broker (no sockets involved)."""

from __future__ import annotations

import threading
import time

from turbo_view.watch.sse import SSEBroker, SSEEvent, format_sse, stream


def test_publish_dispatches_to_all_subscribers():
    broker = SSEBroker()
    a = broker.subscribe()
    b = broker.subscribe()
    broker.publish("reload", "1")

    ev_a = a.get(timeout=1.0)
    ev_b = b.get(timeout=1.0)
    assert ev_a == ev_b == SSEEvent("reload", "1")
    assert broker.subscriber_count == 2


def test_unsubscribe_removes_queue():
    broker = SSEBroker()
    q = broker.subscribe()
    assert broker.subscriber_count == 1
    broker.unsubscribe(q)
    assert broker.subscriber_count == 0
    broker.publish("reload")
    assert q.empty()


def test_publish_after_close_is_noop():
    broker = SSEBroker()
    q = broker.subscribe()
    broker.close()
    broker.publish("reload")
    assert q.empty()


def test_format_sse_handles_multiline_data():
    out = format_sse(SSEEvent("status", "line1\nline2"))
    assert b"event: status" in out
    assert b"data: line1" in out
    assert b"data: line2" in out
    assert out.endswith(b"\n\n")


def test_format_sse_emits_empty_data_field_for_empty_payload():
    out = format_sse(SSEEvent("reload", ""))
    assert out == b"event: reload\ndata: \n\n"


def test_stream_yields_keepalive_when_idle():
    broker = SSEBroker()
    q = broker.subscribe()
    chunks: list[bytes] = []
    done = threading.Event()

    def consume():
        for chunk in stream(q, heartbeat_s=0.05):
            chunks.append(chunk)
            if len(chunks) >= 2:
                done.set()
                return

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    assert done.wait(timeout=2.0)
    assert any(b"keepalive" in c for c in chunks)
