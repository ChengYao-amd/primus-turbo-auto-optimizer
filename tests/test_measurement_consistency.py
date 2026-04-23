"""Tests for the BASELINE / VALIDATE measurement-consistency contract.

These tests cover the P0 fix described in
``docs/performance-measurement-confidence.md``:

* ``parse_bench_csv`` normalises the ``quick_test_bench.py`` CSV
  schema (``correct`` / ``fwd_tflops_mean`` / ``fwd_tflops_std``) to
  the canonical Primus-Turbo schema (``Check`` / ``Forward TFLOPS`` /
  ``Forward TFLOPS_stddev``).
* :func:`verify_shape_consistency` reports the exact set of shapes
  missing from either side without tripping on trivial bookkeeping
  columns (``Case`` / ``label`` / ``TestID``).
* ``_history_best_score_vector`` parses round-1's CSV into a proper
  list of ShapeResult-like dicts rather than collapsing every row into
  an empty shape key (which was the latent bug that made the
  per-shape regression gate a no-op).

The tests exercise the public scorer API + the campaign helper with
on-disk fixtures so a regression in the wiring surfaces here rather
than only inside a live campaign.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator import campaign as campaign_mod
from turbo_optimize.scoring import (
    ScoreVector,
    ShapeResult,
    parse_bench_csv,
    verify_shape_consistency,
)


# ---------------------------------------------------------------------
# parse_bench_csv schema normalisation
# ---------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_bench_csv_accepts_quick_harness_schema(tmp_path: Path):
    csv = _write(
        tmp_path / "bench.csv",
        "label,B,M,N,K,correct,out_snr,da_snr,db_snr,"
        "fwd_ms_mean,fwd_ms_std,bwd_ms_mean,bwd_ms_std,"
        "fwd_tflops_mean,fwd_tflops_std,bwd_tflops_mean,bwd_tflops_std\n"
        "DSV2L-Down-small,2,512,2048,1408,True,28.4,28.4,28.4,"
        "0.15,0.01,0.16,0.01,38.67,0.74,74.00,0.20\n"
        "Kimi-K2-GateUP-lg,12,4096,4096,7168,True,28.5,28.5,28.5,"
        "2.00,0.01,3.90,0.02,1500.00,3.00,1515.98,2.80\n",
    )

    parse = parse_bench_csv(csv, "Forward TFLOPS,Backward TFLOPS")

    assert len(parse.rows) == 2
    row = parse.rows[0]
    assert row.check == "PASS"  # correct=True normalised
    assert row.metrics["Forward TFLOPS"] == pytest.approx(38.67)
    assert row.metrics["Backward TFLOPS"] == pytest.approx(74.00)
    # `fwd_tflops_std` is absolute stddev; normaliser converts to %.
    assert row.metrics_stddev_pct["Forward TFLOPS"] == pytest.approx(
        0.74 / 38.67 * 100.0
    )
    # shape dict should be geometry axes (B/M/N/K + label) — no SNR noise.
    assert "out_snr" not in row.shape
    assert "da_snr" not in row.shape
    assert row.shape["B"] == "2" and row.shape["M"] == "512"


def test_parse_bench_csv_preserves_canonical_schema_unchanged(tmp_path: Path):
    csv = _write(
        tmp_path / "bench.csv",
        "TestID,Platform,GPU,Case,B,M,N,K,Dtype,Granularity,Check,"
        "Forward Time (ms),Forward TFLOPS,Backward Time (ms),Backward TFLOPS\n"
        "1,ROCm,MI355X,DeepSeek-V3-GateUP,8,512,4096,7168,fp8,tensorwise,PASS,"
        "0.37,653.91,0.72,668.29\n",
    )

    parse = parse_bench_csv(csv, "Forward TFLOPS,Backward TFLOPS")

    row = parse.rows[0]
    assert row.check == "PASS"
    assert row.metrics["Forward TFLOPS"] == pytest.approx(653.91)
    assert row.metrics["Backward TFLOPS"] == pytest.approx(668.29)
    # Canonical CSVs have no stddev columns — scorer falls back to the
    # static threshold, not an observed value.
    assert row.metrics_stddev_pct == {}


def test_parse_bench_csv_normalises_failed_quick_row(tmp_path: Path):
    csv = _write(
        tmp_path / "bench.csv",
        "label,B,M,N,K,correct,fwd_tflops_mean,bwd_tflops_mean\n"
        "shape-ok,1,128,128,128,True,10.0,20.0\n"
        "shape-bad,1,256,256,256,False,0.0,0.0\n",
    )

    parse = parse_bench_csv(csv, "Forward TFLOPS")

    assert parse.rows[0].check == "PASS"
    assert parse.rows[1].check == "FAIL"
    assert parse.all_pass is False


# ---------------------------------------------------------------------
# Shape consistency checker
# ---------------------------------------------------------------------


def _vec(shapes: list[dict]) -> ScoreVector:
    rows = [
        ShapeResult(shape=s, check="PASS", metrics={"Forward TFLOPS": 100.0})
        for s in shapes
    ]
    return ScoreVector(per_shape=rows, aggregate={"Forward TFLOPS": 100.0})


def test_shape_consistency_matches_identical_sets():
    base = _vec([{"B": 2, "M": 512, "N": 2048, "K": 1408}])
    cand = _vec([{"B": 2, "M": 512, "N": 2048, "K": 1408}])
    report = verify_shape_consistency(cand, base)
    assert report.consistent is True
    assert report.candidate_only == []
    assert report.mismatch_reason == ""


def test_shape_consistency_allows_baseline_superset():
    base = _vec(
        [
            {"B": 2, "M": 512, "N": 2048, "K": 1408},
            {"B": 8, "M": 1024, "N": 4096, "K": 7168},
            {"B": 12, "M": 4096, "N": 4096, "K": 7168},
        ]
    )
    cand = _vec(
        [
            {"B": 2, "M": 512, "N": 2048, "K": 1408},
        ]
    )
    report = verify_shape_consistency(cand, base)
    assert report.consistent is True, (
        "baseline may measure more shapes than the candidate — this is "
        "the BASELINE-full vs. VALIDATE-quick relationship and should be "
        "considered consistent, not flagged."
    )
    assert len(report.baseline_only) == 2


def test_shape_consistency_detects_candidate_drift():
    base = _vec([{"B": 2, "M": 512, "N": 2048, "K": 1408}])
    cand = _vec([{"B": 12, "M": 4096, "N": 4096, "K": 7168}])
    report = verify_shape_consistency(cand, base)
    assert report.consistent is False
    assert report.candidate_only  # drifted shape surfaces here
    assert "absent from the baseline" in report.mismatch_reason


def test_shape_consistency_ignores_bookkeeping_columns():
    # BASELINE full sweep uses ``Case`` column; quick bench uses
    # ``label``. As long as the numeric geometry axes match the two
    # should compare equal.
    base = _vec(
        [
            {
                "Case": "DeepSeek-V3-GateUP",
                "B": 8,
                "M": 1024,
                "N": 4096,
                "K": 7168,
                "TestID": "47",
            }
        ]
    )
    cand = _vec(
        [
            {
                "label": "DSV3-GateUP-sm",
                "B": 8,
                "M": 1024,
                "N": 4096,
                "K": 7168,
            }
        ]
    )
    report = verify_shape_consistency(cand, base)
    assert report.consistent is True


def test_shape_consistency_missing_side_is_inconclusive():
    cand = _vec([{"B": 1, "M": 1, "N": 1, "K": 1}])
    assert verify_shape_consistency(cand, None).consistent is True
    assert verify_shape_consistency(None, cand).consistent is True


# ---------------------------------------------------------------------
# _history_best_score_vector wiring
# ---------------------------------------------------------------------


def _baseline_csv(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "label,B,M,N,K,Check,"
        "Forward TFLOPS,Forward TFLOPS_stddev,"
        "Backward TFLOPS,Backward TFLOPS_stddev,"
        "Forward Time (ms),Backward Time (ms),out_snr,da_snr,db_snr\n"
    )
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def test_history_best_score_vector_parses_canonical_csv(tmp_path: Path):
    params = CampaignParams(
        campaign_dir=tmp_path,
        primary_metric="Forward TFLOPS,Backward TFLOPS",
    )
    _baseline_csv(
        tmp_path / "rounds" / "round-1" / "artifacts" / "benchmark.csv",
        [
            "s1,2,512,2048,1408,PASS,38.67,0.74,74.00,0.19,0.15,0.16,28,28,28\n",
            "s2,12,4096,4096,7168,PASS,1500.0,3.0,1515.98,2.8,2.0,3.9,28,28,28\n",
        ],
    )

    rows = campaign_mod._history_best_score_vector(params)

    assert len(rows) == 2
    first = rows[0]
    # Shape dict must carry geometry axes so later shape-matching works.
    assert first["shape"]["B"] == "2" and first["shape"]["M"] == "512"
    assert first["check"] == "PASS"
    # Metrics must contain numeric Forward/Backward TFLOPS, NOT the raw
    # row (strings / shape columns mixed in).
    assert set(first["metrics"]) == {"Forward TFLOPS", "Backward TFLOPS"}
    assert first["metrics"]["Forward TFLOPS"] == pytest.approx(38.67)
    assert first["metrics"]["Backward TFLOPS"] == pytest.approx(74.00)


def test_history_best_score_vector_parses_quick_schema_csv(tmp_path: Path):
    # Older campaigns recorded round-1 with the quick-bench schema.
    # Regression test: the alias-aware path must pick those up
    # identically to the canonical schema so resuming a campaign
    # against a round-1 written before this fix keeps working.
    params = CampaignParams(
        campaign_dir=tmp_path,
        primary_metric="Forward TFLOPS,Backward TFLOPS",
    )
    csv = tmp_path / "rounds" / "round-1" / "artifacts" / "benchmark.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text(
        "label,B,M,N,K,correct,out_snr,da_snr,db_snr,"
        "fwd_ms_mean,fwd_ms_std,bwd_ms_mean,bwd_ms_std,"
        "fwd_tflops_mean,fwd_tflops_std,bwd_tflops_mean,bwd_tflops_std\n"
        "s1,2,512,2048,1408,True,28,28,28,"
        "0.15,0.01,0.16,0.01,38.67,0.74,74.00,0.20\n"
        "s2,12,4096,4096,7168,True,28,28,28,"
        "2.00,0.01,3.90,0.02,1500.0,3.0,1515.98,2.8\n",
        encoding="utf-8",
    )

    rows = campaign_mod._history_best_score_vector(params)

    assert len(rows) == 2
    assert rows[0]["check"] == "PASS"
    assert rows[0]["metrics"]["Forward TFLOPS"] == pytest.approx(38.67)
    assert rows[1]["metrics"]["Backward TFLOPS"] == pytest.approx(1515.98)
    assert rows[0]["shape"]["M"] == "512"


def test_history_best_score_vector_absent_csv_returns_empty(tmp_path: Path):
    params = CampaignParams(campaign_dir=tmp_path, primary_metric="Forward TFLOPS")
    assert campaign_mod._history_best_score_vector(params) == []
