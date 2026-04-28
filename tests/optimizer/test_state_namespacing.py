"""Tests for state-dir namespacing (``state/<campaign_id>/...``).

Before the namespacing change every campaign shared ``state/run.json``
and ``state/phase_result/*``, so running two campaigns in sequence
clobbered each other's resume state. The fix nests state under
``campaign_id``; these tests cover the three moving parts:

* ``_resolve_campaign_id`` fills in a deterministic id for ``-p`` runs
* ``_namespace_state_dir`` is idempotent (``warm_restart.sh`` re-passes
  the already-namespaced dir)
* ``_migrate_legacy_state`` relocates pre-namespace files exactly once,
  and only when the legacy ``run.json`` actually belongs to the current
  campaign
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator.campaign import (
    _absolutize_paths,
    _migrate_legacy_state,
    _namespace_state_dir,
    _resolve_campaign_id,
)
from turbo_optimize.state import phase_result_path, run_json_path


def _fresh_params(tmp_path: Path, **overrides) -> CampaignParams:
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "agent").mkdir(exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    base = dict(
        prompt="optimize gemm fp8",
        workspace_root=tmp_path / "ws",
        skills_root=tmp_path / "ws" / "agent",
        state_dir=tmp_path / "state",
    )
    base.update(overrides)
    return CampaignParams(**base)


def test_resolve_campaign_id_fills_from_prompt(tmp_path):
    params = _fresh_params(tmp_path)
    assert params.campaign_id is None
    _resolve_campaign_id(params)
    assert params.campaign_id
    slug = params.campaign_id.split("_")[0]
    assert slug.startswith("optimize")


def test_resolve_campaign_id_respects_existing(tmp_path):
    params = _fresh_params(tmp_path, campaign_id="given_id_12345")
    _resolve_campaign_id(params)
    assert params.campaign_id == "given_id_12345"


def test_namespace_state_dir_appends_campaign_id(tmp_path):
    params = _fresh_params(tmp_path, campaign_id="camp_a")
    original = params.state_dir
    _namespace_state_dir(params)
    assert params.state_dir == original / "camp_a"


def test_namespace_state_dir_is_idempotent(tmp_path):
    """warm_restart.sh writes the already-nested path back into the CLI;
    nesting twice would create ``state/camp_a/camp_a/``."""
    params = _fresh_params(tmp_path, campaign_id="camp_a")
    _namespace_state_dir(params)
    once = params.state_dir
    _namespace_state_dir(params)
    _namespace_state_dir(params)
    assert params.state_dir == once


def test_two_campaigns_keep_state_isolated(tmp_path):
    """Same ``--state-dir``, different campaign ids → different on-disk
    paths for ``run.json`` and per-phase JSONs."""
    a = _fresh_params(tmp_path, campaign_id="camp_a")
    b = _fresh_params(tmp_path, campaign_id="camp_b")
    _namespace_state_dir(a)
    _namespace_state_dir(b)

    run_a = run_json_path(a.state_dir)
    run_b = run_json_path(b.state_dir)
    assert run_a != run_b
    assert run_a.parent.name == "camp_a"
    assert run_b.parent.name == "camp_b"

    phase_a = phase_result_path(a.state_dir, "BASELINE")
    phase_b = phase_result_path(b.state_dir, "BASELINE")
    assert phase_a != phase_b
    assert "camp_a" in str(phase_a)
    assert "camp_b" in str(phase_b)


def test_migrate_legacy_state_moves_matching_campaign(tmp_path):
    params = _fresh_params(tmp_path, campaign_id="camp_legacy")
    legacy_root = params.state_dir
    (legacy_root / "run.json").write_text(
        json.dumps({"campaign_id": "camp_legacy", "current_phase": "BASELINE"}),
        encoding="utf-8",
    )
    legacy_phase = legacy_root / "phase_result"
    legacy_phase.mkdir(parents=True)
    (legacy_phase / "define_target.json").write_text(
        json.dumps({"target_op": "gemm"}), encoding="utf-8"
    )
    (legacy_phase / "baseline.json").write_text(
        json.dumps({"test_pass": True}), encoding="utf-8"
    )

    _migrate_legacy_state(params)

    dest = legacy_root / "camp_legacy"
    assert (dest / "run.json").exists()
    assert json.loads((dest / "run.json").read_text())["campaign_id"] == "camp_legacy"
    assert (dest / "phase_result" / "define_target.json").exists()
    assert (dest / "phase_result" / "baseline.json").exists()
    assert not (legacy_root / "run.json").exists()


def test_migrate_legacy_state_skips_unrelated_campaign(tmp_path):
    """Legacy run.json for campaign X must not be swallowed when we're
    starting / resuming campaign Y."""
    params = _fresh_params(tmp_path, campaign_id="camp_current")
    legacy_root = params.state_dir
    (legacy_root / "run.json").write_text(
        json.dumps({"campaign_id": "camp_other"}), encoding="utf-8"
    )

    _migrate_legacy_state(params)

    assert (legacy_root / "run.json").exists(), (
        "unrelated legacy run.json must stay in place"
    )
    assert not (legacy_root / "camp_current").exists()


def test_migrate_legacy_state_is_noop_without_legacy(tmp_path):
    params = _fresh_params(tmp_path, campaign_id="camp_a")
    _migrate_legacy_state(params)
    _migrate_legacy_state(params)
    assert params.state_dir.exists()


def test_migrate_legacy_state_handles_conflict(tmp_path):
    """If ``state/<id>/run.json`` already exists (e.g. a partial earlier
    migration), leave the legacy file in place rather than overwriting."""
    params = _fresh_params(tmp_path, campaign_id="camp_a")
    legacy_root = params.state_dir
    (legacy_root / "run.json").write_text(
        json.dumps({"campaign_id": "camp_a", "from": "legacy"}),
        encoding="utf-8",
    )
    dest = legacy_root / "camp_a"
    dest.mkdir()
    (dest / "run.json").write_text(
        json.dumps({"campaign_id": "camp_a", "from": "canonical"}),
        encoding="utf-8",
    )

    _migrate_legacy_state(params)

    assert (
        json.loads((dest / "run.json").read_text())["from"] == "canonical"
    ), "canonical run.json must win over the legacy duplicate"


def test_run_campaign_dry_run_creates_namespaced_state(tmp_path, monkeypatch):
    """End-to-end: dry-run goes through the full namespacing pipeline
    without ever booting Claude, so the resulting state_dir must point at
    ``state/<campaign_id>/`` and writes must land there."""
    from turbo_optimize.orchestrator import campaign as campaign_module

    workspace = tmp_path / "ws"
    (workspace / "agent").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    params = CampaignParams(
        prompt="namespace dry run",
        workspace_root=workspace,
        skills_root=workspace / "agent",
        state_dir=state_dir,
        dry_run=True,
    )
    monkeypatch.chdir(tmp_path)

    rc = asyncio.run(campaign_module.run_campaign(params))
    assert rc == 0
    assert params.campaign_id is not None
    assert params.state_dir == (state_dir / params.campaign_id).resolve()
    assert params.state_dir.is_dir()
