"""Coverage for ``turbo_view.io.bench.parse_benchmark_csv``.

Both bench schemas (288-shape primus-turbo full, 5-shape
quick_test_bench) must normalise into the same ``ShapeRow`` shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_view.io.bench import parse_benchmark_csv

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_parse_full_bench_schema():
    rows = parse_benchmark_csv(FIXTURE / "rounds" / "round-1" / "artifacts" / "benchmark.csv")
    assert len(rows) == 5
    r0 = rows[0]
    assert r0.label == "B32_M64_N5760_K2880"
    assert r0.B == 32 and r0.M == 64 and r0.N == 5760 and r0.K == 2880
    assert r0.fwd_tflops == pytest.approx(1308.688)
    assert r0.bwd_tflops == pytest.approx(1309.436)
    assert r0.fwd_std is None and r0.bwd_std is None
    assert r0.check is None


def test_parse_quick_bench_schema_with_std_and_check():
    rows = parse_benchmark_csv(FIXTURE / "rounds" / "round-2" / "artifacts" / "benchmark.csv")
    assert len(rows) == 5
    r0 = rows[0]
    assert r0.fwd_tflops == pytest.approx(1318.307)
    assert r0.fwd_std == pytest.approx(2.1)
    assert r0.bwd_std == pytest.approx(2.3)
    assert r0.check == "PASS"


def test_parse_quick_bench_records_check_fail():
    rows = parse_benchmark_csv(FIXTURE / "rounds" / "round-3" / "artifacts" / "benchmark.csv")
    checks = [r.check for r in rows]
    assert "FAIL" in checks


def test_parse_returns_empty_when_missing(tmp_path: Path):
    assert parse_benchmark_csv(tmp_path / "nope.csv") == []


def test_parse_skips_rows_without_any_tflops(tmp_path: Path):
    src = tmp_path / "b.csv"
    src.write_text(
        "label,B,M,N,K,Forward TFLOPS,Backward TFLOPS\n"
        "ok,1,2,3,4,100,200\n"
        "bad,5,6,7,8,,\n",
        encoding="utf-8",
    )
    rows = parse_benchmark_csv(src)
    assert [r.label for r in rows] == ["ok"]
