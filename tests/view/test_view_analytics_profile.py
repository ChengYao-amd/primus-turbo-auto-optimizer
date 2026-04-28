"""Coverage for ``turbo_view.analytics.{profile,diff}``."""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_view.analytics.diff import all_round_pairs, round_diff
from turbo_view.analytics.profile import (
    clean_name,
    family_for,
    family_rollup,
    gpu_resource_trends,
    profile_panel_for_round,
    round_over_round_topn,
    top_n_kernels,
    treemap_layout,
)
from turbo_view.io.profiles import load_profiles

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def _profiles():
    return load_profiles(FIXTURE)


def test_family_for_matches_design_spec_classes():
    assert family_for("_grouped_fp8_persistent_gemm_kernel") == "fwd_dgrad"
    assert family_for("_grouped_variable_k_gemm_kernel") == "wgrad"
    assert family_for("void primus_turbo::unary_kernel<bf16, fp8>") == "quant"
    assert family_for("void primus_turbo::reduce_row_kernel<float, AbsMax>") == "amax"
    assert family_for("void at::native::vectorized_elementwise_kernel<float>") == "elementwise"
    assert family_for("void at::native::indexSelectLargeIndex<float>") == "other"


def test_clean_name_collapses_template_args():
    assert clean_name("void primus_turbo::unary_kernel<bf16, fp8>") == "primus_turbo::unary_kernel<…>"
    assert clean_name("void at::native::vectorized_elementwise_kernel<float>") == "at::native::vectorized_elementwise_kernel<…>"
    assert clean_name("_grouped_fp8_persistent_gemm_kernel") == "_grouped_fp8_persistent_gemm_kernel"


def test_top_n_kernels_orders_by_total_time():
    profiles = _profiles()
    rows = top_n_kernels(profiles[1].dispatches, n=3)
    assert rows[0]["family"] == "fwd_dgrad"
    assert rows[0]["total_us"] >= rows[1]["total_us"]
    assert rows[0]["count"] == 3


def test_round_over_round_topn_aligns_arrays_with_rounds():
    out = round_over_round_topn(_profiles(), n=3)
    assert out["rounds"] == [1, 2]
    for s in out["series"]:
        assert len(s["totals_us"]) == 2


def test_family_rollup_arrays_aligned_per_round():
    out = family_rollup(_profiles())
    assert out["rounds"] == [1, 2]
    for fam in out["families"]:
        assert len(out["series"][fam]) == 2


def test_gpu_resource_trends_picks_default_target():
    out = gpu_resource_trends(_profiles())
    assert out["target"] == "_grouped_fp8_persistent_gemm_kernel"
    assert out["vgpr"] == [256, 256]


def test_treemap_layout_unit_square_total_area():
    rows = top_n_kernels(_profiles()[1].dispatches, n=10)
    cells = treemap_layout(rows)
    total = sum(c["w"] * c["h"] for c in cells)
    assert total == pytest.approx(1.0, abs=1e-3)
    assert all(0 <= c["x"] <= 1 and 0 <= c["y"] <= 1 for c in cells)


def test_round_diff_marks_delta_when_round2_faster():
    profiles = _profiles()
    diff = round_diff(profiles[1], profiles[2])
    by_name = {r["name_raw"]: r for r in diff["rows"]}
    fwd = by_name["_grouped_fp8_persistent_gemm_kernel"]
    assert fwd["delta_pct"] is not None
    assert fwd["delta_pct"] < 0


def test_all_round_pairs_returns_consecutive_diffs():
    out = all_round_pairs(_profiles())
    assert out["rounds"] == [1, 2]
    assert len(out["pairs"]) == 1
    assert out["pairs"][0]["left_round"] == 1
    assert out["pairs"][0]["right_round"] == 2


def test_profile_panel_for_round_carries_round_local_subfields():
    profiles = _profiles()
    panel = profile_panel_for_round(profiles[1])
    keys = {"round", "flavor", "summary_md_html", "perfetto_results_path",
            "top_n", "treemap"}
    assert keys <= set(panel)
    assert panel["round"] == 1
    assert panel["flavor"] == "baseline"
    assert "<table" in (panel["summary_md_html"] or "")
