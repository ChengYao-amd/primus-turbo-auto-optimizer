"""Sanity checks for ``turbo_view.model`` dataclasses.

Pure-data layer: instances must construct from kwargs, ``slots=True``
must reject unknown attribute writes, and the one computed property
(``KernelDispatch.dur_us``) must be correct.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from turbo_view.model import (
    CampaignBundle,
    CostRow,
    HistoryEntry,
    KernelDispatch,
    PerfRow,
    ProfileBundle,
    RoundBundle,
    RunState,
    ShapeRow,
    TranscriptEvent,
)


def test_runstate_constructs_with_required_fields():
    state = RunState(
        campaign_id="c1",
        campaign_dir=Path("/tmp/c1"),
        current_phase="OPTIMIZE",
        current_round=3,
        best_round=2,
        best_score={"step_geomean": 1314.93},
        rollback_streak=0,
        started_at="2026-04-23 15:20:25",
        last_update="2026-04-23 18:01:11",
        params={"target_op": "grouped_gemm_fp8_tensorwise"},
        history=[],
    )
    assert state.campaign_id == "c1"
    assert state.history == []


def test_runstate_rejects_unknown_attribute():
    state = RunState(
        campaign_id="c1", campaign_dir=Path("/tmp"), current_phase="X",
        current_round=0, best_round=None, best_score={}, rollback_streak=0,
        started_at="", last_update="", params={}, history=[],
    )
    with pytest.raises(AttributeError):
        state.unknown_field = 1  # type: ignore[attr-defined]


def test_kernel_dispatch_dur_us():
    kd = KernelDispatch(
        name="kernel_a", start_ns=1_000_000, end_ns=1_002_500,
        vgpr=128, sgpr=80, lds_bytes=0, scratch_bytes=0,
        wg_x=512, grid_x=131072,
    )
    assert kd.dur_us == pytest.approx(2.5)


def test_history_perf_cost_shape_profile_transcript_round_bundle_construct():
    HistoryEntry(round=1, decision="BASELINE", score={"x": 1.0},
                 description="baseline", at="2026-04-23 15:20:25")
    PerfRow(round=1, status="BASELINE", description="d",
            fwd_avg=1.0, fwd_peak=2.0, bwd_avg=3.0, bwd_peak=4.0,
            step_geomean=2.5, vs_baseline="—", key_finding="start")
    CostRow(ts=datetime(2026, 4, 23, 15, 20, 25),
            phase="DEFINE_TARGET", round=None, status="ok",
            wall_s=65.8, sdk_s=65.3, turns=13,
            cost_usd=0.8695, cumulative_usd=0.8695)
    ShapeRow(label="B32_M64_N5760_K2880", B=32, M=64, N=5760, K=2880,
             fwd_tflops=1308.0, bwd_tflops=1309.0,
             fwd_std=None, bwd_std=None, check=None)
    pb = ProfileBundle(round=2, flavor="post_accept", summary_md_html=None,
                       dispatches=[], perfetto_json_path=None)
    rb = RoundBundle(n=1, summary_md_html=None, bench_shapes=[],
                     artifacts=[], kernel_snapshot_dir=None)
    TranscriptEvent(phase="ANALYZE", ts=None, kind="phase_begin", fields={})

    state = RunState(
        campaign_id="c1", campaign_dir=Path("/tmp"), current_phase="OPTIMIZE",
        current_round=2, best_round=2, best_score={}, rollback_streak=0,
        started_at="", last_update="", params={}, history=[],
    )
    bundle = CampaignBundle(
        state=state, cost=[], perf=[], rounds={1: rb},
        profiles={2: pb}, ineffective=[], transcripts={},
        optimize_md_sections={},
    )
    assert bundle.rounds[1].n == 1
    assert bundle.profiles[2].flavor == "post_accept"
