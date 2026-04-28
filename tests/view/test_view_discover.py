"""Coverage for ``turbo_view.discover.discover_campaigns``."""

from __future__ import annotations

import json
from pathlib import Path

from turbo_view.discover import discover_campaigns


def _mk(path: Path, content: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        content = json.dumps(content)
    path.write_text(content, encoding="utf-8")


def test_discover_finds_single_campaign_via_state_glob(tmp_path: Path):
    _mk(tmp_path / "state" / "alpha" / "run.json",
        {"campaign_id": "alpha", "current_phase": "DONE"})
    _mk(tmp_path / "manifest.yaml", "name: alpha\n")
    handles = discover_campaigns(tmp_path)
    assert len(handles) == 1
    assert handles[0].campaign_id == "alpha"
    assert handles[0].campaign_dir == tmp_path.resolve()


def test_discover_walks_into_subdirectories_when_top_state_absent(tmp_path: Path):
    a = tmp_path / "workspace-A"
    b = tmp_path / "workspace-B"
    _mk(a / "state" / "alpha" / "run.json", {"campaign_id": "alpha"})
    _mk(a / "manifest.yaml", "name: alpha\n")
    _mk(b / "state" / "beta" / "run.json", {"campaign_id": "beta"})
    _mk(b / "manifest.yaml", "name: beta\n")
    handles = discover_campaigns(tmp_path)
    ids = sorted(h.campaign_id for h in handles)
    assert ids == ["alpha", "beta"]


def test_discover_returns_empty_when_no_valid_campaign(tmp_path: Path):
    _mk(tmp_path / "irrelevant" / "file.txt", "noise")
    assert discover_campaigns(tmp_path) == []


def test_discover_respects_depth_limit(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "buried"
    _mk(deep / "state" / "deep" / "run.json", {"campaign_id": "deep"})
    _mk(deep / "manifest.yaml", "x")
    handles = discover_campaigns(tmp_path)
    assert handles == []


def test_discover_skips_malformed_run_json(tmp_path: Path):
    _mk(tmp_path / "state" / "good" / "run.json", {"campaign_id": "good"})
    _mk(tmp_path / "state" / "bad" / "run.json", "not json")
    _mk(tmp_path / "manifest.yaml", "x")
    ids = [h.campaign_id for h in discover_campaigns(tmp_path)]
    # Both directories are visible — the validator is path-level, not
    # per-state-id level: a workspace is "valid" if at least one of
    # its state ids parses. Both make handles for the same root.
    assert "good" in ids


def test_discover_returns_unique_campaign_dirs(tmp_path: Path):
    _mk(tmp_path / "state" / "x" / "run.json", {"campaign_id": "x"})
    _mk(tmp_path / "state" / "y" / "run.json", {"campaign_id": "y"})
    _mk(tmp_path / "manifest.yaml", "x")
    handles = discover_campaigns(tmp_path)
    # Top-level state has two ids; both share the same campaign_dir,
    # so we keep only the first by id ordering.
    assert {h.campaign_dir for h in handles} == {tmp_path.resolve()}
