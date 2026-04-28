"""Coverage for ``turbo_view.io.state.load_run_state``.

The state file is the ground truth for sticky-bar fields. We pin:

* The full happy path against ``tests/view/fixtures/campaign_mini``.
* History entries are reified into ``HistoryEntry`` dataclasses.
* Missing run.json returns ``None`` (graceful-degrade per spec §3).
* ``campaign_dir`` is set to the campaign root, not the state dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_view.io.state import load_run_state
from turbo_view.model import HistoryEntry, RunState

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_load_run_state_happy_path():
    state = load_run_state(FIXTURE)

    assert isinstance(state, RunState)
    assert state.campaign_id == "campaign_mini"
    assert state.campaign_dir == FIXTURE
    assert state.current_phase == "ANALYZE"
    assert state.current_round == 4
    assert state.best_round == 2
    assert state.best_score["step_geomean"] == pytest.approx(1314.927)
    assert state.rollback_streak == 1
    assert state.params["target_op"] == "grouped_gemm_fp8_tensorwise"
    assert len(state.history) == 3
    assert all(isinstance(h, HistoryEntry) for h in state.history)
    assert state.history[0].decision == "BASELINE"
    assert state.history[1].decision == "ACCEPTED"
    assert state.history[2].decision == "ROLLBACK"


def test_load_run_state_returns_none_when_missing(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    assert load_run_state(tmp_path) is None


def test_load_run_state_handles_malformed_json(tmp_path: Path):
    state_dir = tmp_path / "state" / "broken"
    state_dir.mkdir(parents=True)
    (state_dir / "run.json").write_text("{not json", encoding="utf-8")
    assert load_run_state(tmp_path) is None
