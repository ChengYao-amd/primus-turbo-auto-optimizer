"""Coverage for ``turbo_view.analytics.{cost,gantt,heatmap}``.

We exercise each public entry against the mini fixture and assert
the chart-ready dict's invariants (sums, ordering, derived deltas).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_view.analytics.cost import (
    cost_panel,
    cost_per_improvement,
    cumulative_series,
    per_phase_breakdown,
    per_round_series,
    token_turn_wall_panel,
)
from turbo_view.analytics.gantt import gantt_panel
from turbo_view.analytics.heatmap import heatmap_panel
from turbo_view.io.loader import load_campaign

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def _bundle():
    return load_campaign(FIXTURE)


# --- cost ----------------------------------------------------------


def test_cumulative_series_is_monotonic():
    series = cumulative_series(_bundle().cost)
    ys = [p["y"] for p in series]
    assert ys == sorted(ys)
    assert ys[-1] == pytest.approx(7.6112)


def test_per_phase_strips_variant_and_sums_correctly():
    rows = per_phase_breakdown(_bundle().cost)
    by_phase = {r["phase"]: r for r in rows}
    assert "VALIDATE" in by_phase
    assert by_phase["ANALYZE"]["count"] == 3
    total = sum(r["cost_usd"] for r in rows)
    assert total == pytest.approx(7.6112, rel=1e-3)


def test_per_round_aggregation_includes_round_zero_for_no_round_phases():
    rows = per_round_series(_bundle().cost)
    by_round = {r["round"]: r for r in rows}
    assert 0 in by_round
    assert by_round[0]["cost_usd"] == pytest.approx(0.8695 + 1.3337)
    assert by_round[2]["turns"] > 0


def test_cost_per_improvement_only_on_accept():
    bundle = _bundle()
    points = cost_per_improvement(bundle.cost, bundle.state.history)
    decisions = [p["decision"] for p in points]
    assert decisions == ["BASELINE", "ACCEPTED"]
    assert points[0]["delta_pct"] == pytest.approx(0.0)
    assert points[1]["delta_pct"] > 0.0


def test_cost_panel_has_all_keys():
    panel = cost_panel(_bundle().cost, _bundle().state)
    assert {"cumulative", "per_phase", "per_round", "cost_per_improvement",
            "total_usd", "total_wall_s", "total_turns"} <= set(panel)
    assert panel["total_usd"] == pytest.approx(7.6112)


def test_token_turn_wall_arrays_aligned():
    panel = token_turn_wall_panel(_bundle().cost)
    n = len(panel["rounds"])
    assert n > 0
    assert len(panel["wall_s"]) == n
    assert len(panel["sdk_s"]) == n
    assert len(panel["turns"]) == n


# --- gantt ---------------------------------------------------------


def test_gantt_blocks_have_start_end_and_abnormal_flag():
    panel = gantt_panel(_bundle().cost)
    assert panel["events"] == []
    blocks = panel["blocks"]
    assert len(blocks) == 11
    assert all("start_ts" in b and "end_ts" in b for b in blocks)
    abnormal = [b for b in blocks if b["abnormal"]]
    assert len(abnormal) == 1
    assert abnormal[0]["status"] == "idle_timeout_compose"


# --- heatmap -------------------------------------------------------


def test_heatmap_rounds_and_shape_labels_present():
    panel = heatmap_panel(_bundle().rounds)
    assert panel["rounds"] == [1, 2, 3]
    assert panel["baseline_round"] == 1
    assert panel["shape_labels"][0] == "B32_M64_N5760_K2880"
    assert len(panel["shape_labels"]) == 5


def test_heatmap_delta_pct_is_zero_on_baseline_row_and_positive_on_accepted():
    panel = heatmap_panel(_bundle().rounds)
    by_round = {r["round"]: r for r in panel["rows"]}
    assert by_round[1]["delta_fwd"][0] == pytest.approx(0.0)
    assert by_round[2]["delta_fwd"][0] > 0.0


def test_heatmap_returns_empty_when_no_bench():
    from turbo_view.model import RoundBundle
    panel = heatmap_panel({1: RoundBundle(n=1, summary_md_html=None,
                                          bench_shapes=[], artifacts=[],
                                          kernel_snapshot_dir=None)})
    assert panel["rounds"] == []
    assert panel["shape_labels"] == []
    assert panel["baseline_round"] is None
