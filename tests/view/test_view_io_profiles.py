"""Coverage for ``turbo_view.io.profiles.load_profiles``."""

from __future__ import annotations

from pathlib import Path

from turbo_view.io.profiles import load_profiles

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_load_profiles_discovers_both_rounds():
    profiles = load_profiles(FIXTURE)
    assert sorted(profiles) == [1, 2]
    assert profiles[1].flavor == "baseline"
    assert profiles[2].flavor == "post_accept"


def test_load_profiles_renders_summary_to_html():
    profiles = load_profiles(FIXTURE)
    html = profiles[1].summary_md_html or ""
    assert "Profile summary" in html
    assert "<table" in html


def test_load_profiles_parses_kernel_trace_dispatches():
    profiles = load_profiles(FIXTURE)
    dispatches = profiles[1].dispatches
    assert len(dispatches) == 10
    first = dispatches[0]
    assert first.name == "_grouped_fp8_persistent_gemm_kernel"
    assert first.start_ns < first.end_ns
    assert first.dur_us > 0
    assert first.vgpr == 256


def test_load_profiles_returns_empty_when_no_dir(tmp_path: Path):
    assert load_profiles(tmp_path) == {}


def test_load_profiles_skips_dispatches_with_invalid_timestamps(tmp_path: Path):
    p = tmp_path / "profiles" / "round-1_baseline" / "rocprofv3" / "h" / "1_kernel_trace.csv"
    p.parent.mkdir(parents=True)
    p.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,VGPR_Count,SGPR_Count,LDS_Block_Size_Bytes,Scratch_Size,Workgroup_Size_X,Grid_Size_X\n"
        "good,1000,2000,1,1,1,0,256,1\n"
        "bad,abc,xyz,1,1,1,0,256,1\n"
        "zero,1000,1000,1,1,1,0,256,1\n",
        encoding="utf-8",
    )
    bundles = load_profiles(tmp_path)
    assert [d.name for d in bundles[1].dispatches] == ["good"]
