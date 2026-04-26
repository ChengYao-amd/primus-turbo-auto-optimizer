"""Tests for :func:`_rewind_if_needed` resume-time state corrections.

Two scenarios are exercised:

* Mid-round crash recovery: a campaign that died inside ``OPTIMIZE`` /
  ``VALIDATE`` / ``DECIDE`` / ``PROFILE`` / ``REVIEW`` is rewound to
  ``ANALYZE`` for the same ``round_n``.
* DONE-extension warm restart: a campaign that already terminated with
  ``current_phase == DONE`` is rewound to ``TERMINATION_CHECK`` iff the
  user passed a new ``--max-iterations`` / ``--max-duration`` budget
  that leaves room for at least one more round.  Without a new budget
  (or with one that re-triggers T3 / T4 immediately) the state stays
  ``DONE`` so the orchestrator returns quickly instead of silently
  looping.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator.campaign import (
    _can_extend_after_done,
    _rewind_if_needed,
)
from turbo_optimize.state import RunState


def _params(tmp_path: Path, **overrides) -> CampaignParams:
    base = dict(
        prompt="optimize gemm fp8",
        workspace_root=tmp_path / "ws",
        skills_root=tmp_path / "ws" / "agent",
        state_dir=tmp_path / "state",
    )
    base.update(overrides)
    return CampaignParams(**base)


def _state(**overrides) -> RunState:
    base = dict(
        campaign_id="test_campaign",
        current_phase="ANALYZE",
        current_round=10,
        best_round=8,
        best_score={"Combined Step TFLOPS": 1000.0},
    )
    base.update(overrides)
    return RunState(**base)


@pytest.mark.parametrize("phase", ["OPTIMIZE", "VALIDATE", "DECIDE", "PROFILE", "REVIEW"])
def test_mid_round_phase_rewinds_to_analyze(tmp_path, phase):
    state = _state(current_phase=phase, current_round=7)
    params = _params(tmp_path)
    _rewind_if_needed(state, params)
    assert state.current_phase == "ANALYZE"
    assert state.current_round == 7  # round number must stay stable


def test_mid_round_rewind_skipped_when_round_zero(tmp_path):
    """Round 0 is the BASELINE round; rewinding it to ANALYZE is wrong."""
    state = _state(current_phase="OPTIMIZE", current_round=0)
    params = _params(tmp_path)
    _rewind_if_needed(state, params)
    assert state.current_phase == "OPTIMIZE"


def test_done_with_higher_max_iterations_rewinds_to_termination_check(tmp_path):
    state = _state(current_phase="DONE", current_round=50)
    params = _params(tmp_path, max_iterations=100)
    _rewind_if_needed(state, params)
    assert state.current_phase == "TERMINATION_CHECK"
    assert state.current_round == 50


def test_done_without_new_budget_stays_done(tmp_path):
    state = _state(current_phase="DONE", current_round=50)
    params = _params(tmp_path)  # no max_iterations, no max_duration
    _rewind_if_needed(state, params)
    assert state.current_phase == "DONE"


def test_done_with_equal_max_iterations_stays_done(tmp_path):
    """``-i 50`` after terminating at round 50 must NOT silently rewind."""
    state = _state(current_phase="DONE", current_round=50)
    params = _params(tmp_path, max_iterations=50)
    _rewind_if_needed(state, params)
    assert state.current_phase == "DONE"


def test_done_with_lower_max_iterations_stays_done(tmp_path):
    state = _state(current_phase="DONE", current_round=50)
    params = _params(tmp_path, max_iterations=30)
    _rewind_if_needed(state, params)
    assert state.current_phase == "DONE"


def test_done_with_fresh_max_duration_rewinds(tmp_path):
    """``-d 4h`` past a 50-round terminator: T4 elapsed=0 < 4h → rewind."""
    state = _state(
        current_phase="DONE",
        current_round=50,
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    params = _params(tmp_path, max_duration="4h")
    _rewind_if_needed(state, params)
    assert state.current_phase == "TERMINATION_CHECK"


def test_done_with_already_elapsed_duration_stays_done(tmp_path):
    """Started 5h ago, user passes -d 1h: T4 would re-fire immediately."""
    long_ago = (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
    state = _state(current_phase="DONE", current_round=50, started_at=long_ago)
    params = _params(tmp_path, max_duration="1h")
    _rewind_if_needed(state, params)
    assert state.current_phase == "DONE"


def test_can_extend_after_done_pure_predicate(tmp_path):
    """Direct unit test for the predicate, decoupled from the rewind side effect."""
    state_at_50 = _state(current_phase="DONE", current_round=50)

    assert _can_extend_after_done(state_at_50, _params(tmp_path, max_iterations=100))
    assert not _can_extend_after_done(state_at_50, _params(tmp_path, max_iterations=50))
    assert not _can_extend_after_done(state_at_50, _params(tmp_path, max_iterations=10))
    assert not _can_extend_after_done(state_at_50, _params(tmp_path))  # no budget

    fresh_start = _state(
        current_phase="DONE",
        current_round=50,
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    assert _can_extend_after_done(fresh_start, _params(tmp_path, max_duration="2h"))


def test_done_rewind_does_not_touch_round_zero_baseline(tmp_path):
    """Edge case: a DONE state at round=0 (BASELINE-only campaign) is unusual
    but must not crash; the rewind still applies if the user supplies a new
    iteration budget.
    """
    state = _state(current_phase="DONE", current_round=0, best_round=None)
    params = _params(tmp_path, max_iterations=10)
    _rewind_if_needed(state, params)
    assert state.current_phase == "TERMINATION_CHECK"
