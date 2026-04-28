"""Tests for the debounce-and-watchdog wrapper.

Filesystem-watch behaviour on Linux is straightforward (inotify), so
we exercise the real ``watchdog.Observer`` rather than mocking it.
A 0.2 s debounce keeps the test fast without making it flaky.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from turbo_view.watch.watcher import CampaignWatcher, _DebouncedTrigger, tail_lines


def test_debounce_collapses_burst():
    counter = {"n": 0}
    fired = threading.Event()

    def cb():
        counter["n"] += 1
        fired.set()

    trig = _DebouncedTrigger(cb, delay_s=0.1)
    for _ in range(20):
        trig.fire()
        time.sleep(0.005)

    assert fired.wait(timeout=1.0)
    time.sleep(0.2)
    assert counter["n"] == 1


def test_debounce_cancel_drops_pending():
    counter = {"n": 0}

    def cb():
        counter["n"] += 1

    trig = _DebouncedTrigger(cb, delay_s=0.1)
    trig.fire()
    trig.cancel()
    time.sleep(0.2)
    assert counter["n"] == 0


def test_campaign_watcher_fires_on_file_create(tmp_path: Path):
    triggers = []
    done = threading.Event()

    def on_change():
        triggers.append(time.time())
        done.set()

    watcher = CampaignWatcher(tmp_path, on_change=on_change, debounce_s=0.1)
    watcher.start()
    try:
        time.sleep(0.1)
        (tmp_path / "run.json").write_text("{}", encoding="utf-8")
        assert done.wait(timeout=3.0)
    finally:
        watcher.stop()
    assert len(triggers) == 1


def test_campaign_watcher_ignores_read_only_access(tmp_path: Path):
    """Reading a file (open + close-no-write) must not retrigger.

    When the renderer itself opens 100+ summary.md / benchmark.csv
    files during a single rebuild, watchdog reports IN_OPEN /
    IN_CLOSE_NOWRITE for each one. If those events fired the
    debounce timer, we'd land in an infinite render loop -- which
    is what bit us on a real 100-round campaign.
    """
    triggers = []
    done = threading.Event()

    def on_change():
        triggers.append(time.time())
        done.set()

    target = tmp_path / "summary.md"
    target.write_text("seed", encoding="utf-8")

    watcher = CampaignWatcher(tmp_path, on_change=on_change, debounce_s=0.1)
    watcher.start()
    try:
        # Drain the initial create event so the timer is idle.
        done.wait(timeout=1.0)
        done.clear()
        triggers.clear()
        time.sleep(0.2)

        # Pure read access: open + close, no write.
        for _ in range(50):
            with target.open("r", encoding="utf-8") as fh:
                fh.read()
        time.sleep(0.4)
        assert not done.is_set(), (
            f"read-only access retriggered watcher: {triggers}"
        )
    finally:
        watcher.stop()
    assert triggers == []


def test_campaign_watcher_skips_view_subdir(tmp_path: Path):
    """Writing inside ``<dir>/view/`` must not retrigger.

    Otherwise the watch server's own re-render would feed back into
    the watcher and rebuild forever.
    """
    triggers = []
    done = threading.Event()

    def on_change():
        triggers.append(time.time())
        done.set()

    (tmp_path / "view").mkdir()
    watcher = CampaignWatcher(tmp_path, on_change=on_change, debounce_s=0.1)
    watcher.start()
    try:
        time.sleep(0.1)
        (tmp_path / "view" / "index.html").write_text("ignored", encoding="utf-8")
        time.sleep(0.4)
        assert not done.is_set()
    finally:
        watcher.stop()
    assert triggers == []


def test_tail_lines_returns_last_n(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(str(i) for i in range(20)) + "\n", encoding="utf-8")
    assert tail_lines(p, n=5) == ["15", "16", "17", "18", "19"]
    assert tail_lines(p, n=999)[-1] == "19"
    assert tail_lines(tmp_path / "missing", n=5) == []


def test_tail_lines_drops_blank_lines(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text("a\n\n\nb\n\nc\n", encoding="utf-8")
    assert tail_lines(p, n=10) == ["a", "b", "c"]
