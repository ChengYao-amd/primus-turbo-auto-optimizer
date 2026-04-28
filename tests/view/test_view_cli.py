"""End-to-end smoke tests for ``primus-turbo-view``.

We invoke ``main()`` directly (subprocess avoids interpreter startup
cost; in-process is enough since ``main`` returns an int exit code).
The tests pin the file-system contract: ``index.html`` plus
``data.json`` plus the vendored assets are all present, and the CLI
exits 0 on a normal mini-campaign render and 2 on a missing path.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from turbo_view.cli import main as cli_main

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def _multi_workspace(tmp_path: Path) -> Path:
    a = tmp_path / "campaign-a"
    b = tmp_path / "campaign-b"
    shutil.copytree(FIXTURE, a)
    shutil.copytree(FIXTURE, b)
    (a / "state" / "campaign_mini").rename(a / "state" / "campaign-a")
    (b / "state" / "campaign_mini").rename(b / "state" / "campaign-b")
    for d, name in ((a, "campaign-a"), (b, "campaign-b")):
        rj = d / "state" / name / "run.json"
        data = json.loads(rj.read_text())
        data["campaign_id"] = name
        rj.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_cli_renders_to_explicit_out_dir(tmp_path: Path, capsys):
    out = tmp_path / "view"
    rc = cli_main([str(FIXTURE), "-o", str(out)])
    assert rc == 0
    assert (out / "index.html").is_file()
    assert (out / "index.html").stat().st_size > 5_000
    assert (out / "data.json").is_file()
    assert (out / "assets" / "chart.umd.min.js").is_file()

    payload = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert payload["state"]["campaign_id"] == "campaign_mini"

    out_msg = capsys.readouterr().out
    assert "wrote" in out_msg
    assert "index.html" in out_msg


def test_cli_defaults_out_dir_to_campaign_view(tmp_path: Path):
    """Without ``-o``, the CLI writes to ``<campaign>/view/``.

    We mirror the fixture into ``tmp_path`` so the real fixture tree
    stays untouched.
    """
    import shutil
    work = tmp_path / "work"
    shutil.copytree(FIXTURE, work)
    rc = cli_main([str(work)])
    assert rc == 0
    assert (work / "view" / "index.html").is_file()
    assert (work / "view" / "data.json").is_file()


def test_cli_returns_2_on_missing_path(tmp_path: Path, caplog):
    rc = cli_main([str(tmp_path / "does-not-exist")])
    assert rc == 2


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "primus-turbo-view" in out
    assert "campaign directory" in out.lower()


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "primus-turbo-view" in out


def test_cli_auto_detects_multi_mode(tmp_path: Path):
    workspace = _multi_workspace(tmp_path)
    out = tmp_path / "view"
    rc = cli_main([str(workspace), "-o", str(out)])
    assert rc == 0
    assert (out / "c" / "campaign-a" / "index.html").is_file()
    assert (out / "c" / "campaign-b" / "index.html").is_file()
    assert (out / "assets" / "app.js").is_file()


def test_cli_force_single_when_no_state_returns_warning(tmp_path: Path, caplog):
    rc = cli_main([str(tmp_path), "--single", "-o", str(tmp_path / "view")])
    assert rc == 0
    assert (tmp_path / "view" / "index.html").is_file()


def test_cli_watch_rejects_when_single_lacks_state(tmp_path: Path, caplog):
    # ``--watch --single`` against a directory with NOTHING (no state,
    # no manifest.yaml, no rounds/) is fatal: there's nothing to watch.
    rc = cli_main([str(tmp_path), "--watch", "--single", "--no-open",
                   "-o", str(tmp_path / "view")])
    assert rc == 2


def test_cli_watch_accepts_manifest_only_campaign(tmp_path: Path, monkeypatch):
    """Real workspaces often have ``manifest.yaml`` + ``rounds/`` but
    no ``run.json`` yet; watch mode must still accept them.

    We monkeypatch ``serve`` so the test never opens a real socket.
    """
    (tmp_path / "manifest.yaml").write_text("target_op: gemm\n", encoding="utf-8")
    (tmp_path / "rounds").mkdir()
    (tmp_path / "out").mkdir()

    captured: dict = {}

    class _FakeServer:
        def __init__(self):
            self.session = type("S", (), {"render_count": 1})()
        def stop(self):
            pass

    def _fake_serve(**kwargs):
        captured.update(kwargs)
        on = kwargs.get("on_started")
        if on:
            on("http://127.0.0.1:0/")
        # Don't block; signal handler path runs elsewhere.
        return _FakeServer()

    # The CLI uses signal.SIGINT/SIGTERM + threading.Event.wait.
    # Skip the wait by replacing Event.wait with an immediate return.
    import threading
    real_wait = threading.Event.wait
    monkeypatch.setattr(threading.Event, "wait",
                        lambda self, timeout=None: True)

    monkeypatch.setattr("turbo_view.watch.server.serve", _fake_serve)

    rc = cli_main([str(tmp_path), "--watch", "--single", "--no-open",
                   "-o", str(tmp_path / "out")])
    assert rc == 0
    assert captured.get("mode") == "single"
    monkeypatch.setattr(threading.Event, "wait", real_wait)
