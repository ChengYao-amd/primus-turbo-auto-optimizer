"""Markdown-table parsers for ``logs/cost.md``, ``performance_trend.md``,
and ``optimize.md`` section split.

Hard requirements:

* Cost rows preserve the literal ``Status`` string (``ok`` /
  ``idle_timeout_compose`` / ``cached`` / etc.) — downstream gantt
  panel keys colour off this.
* Cost ``round`` is None for ``-`` cells.
* Cost ``cumulative_usd`` is monotonically non-decreasing.
* Perf rows survive ``→`` arrow notation (always take the post-arrow
  value) — exercised by the synthetic row in this test, since the mini
  fixture doesn't have one.
* ``optimize.md`` splits into a dict keyed by ``## `` heading.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from turbo_view.io.logs import (
    parse_cost_md,
    parse_optimize_md_sections,
    parse_perf_trend_md,
)

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_parse_cost_md_yields_all_rows():
    rows = parse_cost_md(FIXTURE / "logs" / "cost.md")
    assert len(rows) == 11
    assert rows[0].phase == "DEFINE_TARGET"
    assert rows[0].round is None
    assert rows[0].status == "ok"
    assert rows[0].ts == datetime(2026, 4, 23, 15, 20, 25)
    assert rows[0].wall_s == pytest.approx(65.8)
    assert rows[0].turns == 13
    assert rows[0].cost_usd == pytest.approx(0.8695)
    assert rows[0].cumulative_usd == pytest.approx(0.8695)


def test_parse_cost_md_picks_up_phase_variant_and_idle_timeout():
    rows = parse_cost_md(FIXTURE / "logs" / "cost.md")
    by_phase = {(r.phase, r.round): r for r in rows}
    assert ("VALIDATE (quick)", 2) in by_phase
    assert by_phase[("OPTIMIZE", 3)].status == "idle_timeout_compose"
    assert by_phase[("OPTIMIZE", 3)].cost_usd == pytest.approx(0.0)


def test_parse_cost_md_cumulative_is_monotonic():
    rows = parse_cost_md(FIXTURE / "logs" / "cost.md")
    cums = [r.cumulative_usd for r in rows]
    assert cums == sorted(cums)


def test_parse_cost_md_returns_empty_for_missing_file(tmp_path: Path):
    assert parse_cost_md(tmp_path / "nope.md") == []


def test_parse_perf_trend_md_yields_all_rows():
    rows = parse_perf_trend_md(FIXTURE / "logs" / "performance_trend.md")
    assert [r.round for r in rows] == [1, 2, 3]
    assert rows[0].status == "BASELINE"
    assert rows[0].step_geomean == pytest.approx(1309.062)
    assert rows[0].vs_baseline == "—"
    assert rows[1].status == "ACCEPTED"
    assert rows[1].vs_baseline.startswith("step +0.448%")
    assert rows[2].status == "ROLLBACK"


def test_parse_perf_trend_md_resolves_arrow_notation(tmp_path: Path):
    src = tmp_path / "perf.md"
    src.write_text(
        "# Performance Trend\n\n"
        "| Round | Status | Description | Fwd Avg TFLOPS | Fwd Peak TFLOPS | "
        "Bwd Avg TFLOPS | Bwd Peak TFLOPS | Step Geomean TFLOPS | vs Baseline | "
        "Key Finding |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        "| 5 | ACCEPTED | tweak | 1.0 → 1.5 | 2.0 | 3.0 | 4.0 | "
        "2.5 → 2.7 | step +5% | x |\n",
        encoding="utf-8",
    )
    rows = parse_perf_trend_md(src)
    assert len(rows) == 1
    assert rows[0].fwd_avg == pytest.approx(1.5)
    assert rows[0].step_geomean == pytest.approx(2.7)


def test_parse_optimize_md_sections_splits_by_h2():
    sections = parse_optimize_md_sections(FIXTURE / "logs" / "optimize.md")
    assert "Baseline" in sections
    assert "Current Best" in sections
    assert "Optimization History" in sections
    assert "Verified Ineffective Directions" in sections
    assert "TFLOPS" in sections["Baseline"]
    assert "round-2" in sections["Current Best"]
    assert "round-2" in sections["Optimization History"]
