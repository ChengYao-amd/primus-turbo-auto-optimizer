"""Tests for orchestrator path normalisation and stray-file rescue.

These regressions come from the real crash

    [PREPARE_ENVIRONMENT] phase end status=ok ...
    FileNotFoundError: expected phase output missing at
      state/phase_result/prepare_environment.json

The root cause was that Claude's ``cwd`` (``workspace_root``) and Python's
shell cwd disagreed on what the relative path ``state/...`` meant. The
orchestrator now resolves every filesystem path to absolute form at startup
and migrates any stray phase_result files that older runs parked under
``workspace_root/state/phase_result``.
"""

from __future__ import annotations

import json
from pathlib import Path

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator.campaign import (
    _absolutize_paths,
    _migrate_stray_phase_results,
)


def _make_params(
    tmp_path: Path,
    *,
    rel_workspace: str = "ws",
    rel_state: str = "state",
    rel_skills: str = "ws/agent",
    rel_campaign: str | None = None,
    monkeypatch=None,
) -> CampaignParams:
    (tmp_path / rel_workspace).mkdir(parents=True, exist_ok=True)
    (tmp_path / rel_skills).mkdir(parents=True, exist_ok=True)
    (tmp_path / rel_state).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return CampaignParams(
        workspace_root=Path(rel_workspace),
        skills_root=Path(rel_skills),
        state_dir=Path(rel_state),
        campaign_dir=(Path(rel_campaign) if rel_campaign else None),
    )


def test_absolutize_paths_sets_absolute_paths(tmp_path, monkeypatch):
    (tmp_path / "ws" / "agent" / "workspace" / "camp").mkdir(parents=True)
    params = _make_params(
        tmp_path,
        rel_campaign="ws/agent/workspace/camp",
        monkeypatch=monkeypatch,
    )
    assert not params.workspace_root.is_absolute()
    assert not params.state_dir.is_absolute()

    _absolutize_paths(params)

    assert params.workspace_root.is_absolute()
    assert params.skills_root.is_absolute()
    assert params.state_dir.is_absolute()
    assert params.campaign_dir is not None and params.campaign_dir.is_absolute()


def test_absolutize_preserves_already_absolute_paths(tmp_path, monkeypatch):
    params = _make_params(tmp_path, monkeypatch=monkeypatch)
    absolute_state = (tmp_path / "state").resolve()
    params.state_dir = absolute_state

    _absolutize_paths(params)
    assert params.state_dir == absolute_state


def test_migrate_stray_phase_results_moves_files(tmp_path, monkeypatch):
    params = _make_params(tmp_path, monkeypatch=monkeypatch)
    _absolutize_paths(params)

    stray = params.workspace_root / "state" / "phase_result"
    stray.mkdir(parents=True)
    payload = {"hello": "world"}
    (stray / "prepare_environment.json").write_text(json.dumps(payload))
    (stray / "baseline.json").write_text(json.dumps({"k": 1}))

    _migrate_stray_phase_results(params)

    dest = params.state_dir / "phase_result"
    assert (dest / "prepare_environment.json").exists()
    assert json.loads((dest / "prepare_environment.json").read_text()) == payload
    assert (dest / "baseline.json").exists()
    assert not stray.exists() or not any(stray.iterdir())


def test_migrate_stray_phase_results_skips_conflicts(tmp_path, monkeypatch):
    """If state_dir already has the file (e.g. a valid fresh run), do not
    clobber it with the stray copy."""
    params = _make_params(tmp_path, monkeypatch=monkeypatch)
    _absolutize_paths(params)

    stray = params.workspace_root / "state" / "phase_result"
    stray.mkdir(parents=True)
    (stray / "baseline.json").write_text(json.dumps({"from": "stray"}))

    dest = params.state_dir / "phase_result"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "baseline.json").write_text(json.dumps({"from": "canonical"}))

    _migrate_stray_phase_results(params)

    assert (stray / "baseline.json").exists(), "stray should remain untouched"
    assert json.loads((dest / "baseline.json").read_text()) == {"from": "canonical"}


def test_migrate_is_noop_when_no_stray_dir(tmp_path, monkeypatch):
    params = _make_params(tmp_path, monkeypatch=monkeypatch)
    _absolutize_paths(params)
    _migrate_stray_phase_results(params)
    _migrate_stray_phase_results(params)
