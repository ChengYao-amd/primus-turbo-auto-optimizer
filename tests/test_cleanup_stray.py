"""Unit tests for ``turbo_optimize.orchestrator.cleanup``.

Scope:

1. ``discover_stray_files`` only reports *top-level* untracked files
   and ignores directories and nested untracked paths.
2. ``cleanup_stray_files`` dry-run mode never mutates the filesystem.
3. ``cleanup_stray_files(apply=True)`` moves each stray into
   ``<campaign_dir>/_stray/<ts>/`` and clears the source.
4. Name collisions (rare, but possible on re-run) are resolved by
   suffixing the destination so no data is silently overwritten.
5. Non-git worktrees raise a clear error instead of guessing.

The tests shell out to the real ``git`` binary via ``subprocess``; the
fixture builds a throwaway repo under ``tmp_path`` so no test depends on
the host repo's state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from turbo_optimize.orchestrator.cleanup import (
    cleanup_stray_files,
    discover_stray_files,
    format_report,
)


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(ws), check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=str(ws), check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "tester"], cwd=str(ws), check=True
    )
    (ws / "README").write_text("readme\n")
    subprocess.run(["git", "add", "README"], cwd=str(ws), check=True)
    subprocess.run(
        ["git", "commit", "-qm", "init"], cwd=str(ws), check=True
    )
    return ws


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_returns_only_top_level_files(git_workspace):
    _write(git_workspace / "stray1.csv")
    _write(git_workspace / "stray2.py", "print('x')")
    _write(git_workspace / "subdir" / "nested_untracked.txt")
    (git_workspace / "untracked_dir").mkdir()
    (git_workspace / "untracked_dir" / "x.txt").write_text("x")

    stray = discover_stray_files(git_workspace)
    names = [p.name for p in stray]
    assert "stray1.csv" in names
    assert "stray2.py" in names
    assert not any("nested_untracked" in n for n in names)
    assert not any("untracked_dir" in n for n in names)


def test_discover_returns_empty_for_clean_workspace(git_workspace):
    assert discover_stray_files(git_workspace) == []


def test_dry_run_does_not_move_anything(git_workspace, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    stray_file = git_workspace / "stray.csv"
    _write(stray_file, "csv-content")

    report = cleanup_stray_files(campaign, git_workspace, apply=False)

    assert stray_file.exists()
    assert not (campaign / "_stray").exists()
    assert report.moved == []
    assert report.dest_dir is None
    assert stray_file in report.stray_files


def test_apply_moves_files_under_stray_subdir(git_workspace, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    stray = git_workspace / "stray.csv"
    _write(stray, "payload")

    report = cleanup_stray_files(
        campaign, git_workspace, apply=True, timestamp="20260421_200000"
    )

    assert not stray.exists()
    moved = campaign / "_stray" / "20260421_200000" / "stray.csv"
    assert moved.exists()
    assert moved.read_text(encoding="utf-8") == "payload"
    assert report.dest_dir == campaign / "_stray" / "20260421_200000"
    assert report.moved == [moved]


def test_apply_handles_multiple_files(git_workspace, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write(git_workspace / "a.csv", "a")
    _write(git_workspace / "b.py", "b")
    _write(git_workspace / "c.log", "c")

    report = cleanup_stray_files(
        campaign, git_workspace, apply=True, timestamp="ts"
    )
    assert len(report.moved) == 3
    names = {p.name for p in report.moved}
    assert names == {"a.csv", "b.py", "c.log"}
    assert discover_stray_files(git_workspace) == []


def test_apply_collision_suffixes_filename(git_workspace, tmp_path):
    campaign = tmp_path / "campaign"
    existing_ts_dir = campaign / "_stray" / "ts"
    existing_ts_dir.mkdir(parents=True)
    (existing_ts_dir / "collide.csv").write_text("old")
    _write(git_workspace / "collide.csv", "new")

    report = cleanup_stray_files(
        campaign, git_workspace, apply=True, timestamp="ts"
    )

    assert (existing_ts_dir / "collide.csv").read_text() == "old"
    fresh = next(
        p for p in report.moved if p.name != "collide.csv"
    )
    assert fresh.name.startswith("collide.") and fresh.suffix == ".csv"
    assert fresh.read_text() == "new"


def test_non_git_workspace_raises(tmp_path):
    ws = tmp_path / "not_a_repo"
    ws.mkdir()
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    with pytest.raises(RuntimeError, match="not a git worktree"):
        cleanup_stray_files(campaign, ws, apply=False)


def test_format_report_clean_and_dirty(git_workspace, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    clean = cleanup_stray_files(campaign, git_workspace, apply=False)
    assert "is clean" in format_report(clean)

    _write(git_workspace / "stray.csv")
    dirty = cleanup_stray_files(campaign, git_workspace, apply=False)
    text = format_report(dirty)
    assert "stray.csv" in text
    assert "--apply" in text

    applied = cleanup_stray_files(
        campaign, git_workspace, apply=True, timestamp="T"
    )
    text2 = format_report(applied)
    assert "Moved 1 file" in text2
    assert "_stray/T" in text2
