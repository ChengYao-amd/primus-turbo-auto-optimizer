"""Unit tests for the BASELINE quick_baseline_log plumbing.

Two pieces are under test:

1. :func:`turbo_optimize.logs.append_baseline` — must append the
   ``Quick baseline log:`` line iff a non-empty path is provided.
2. :func:`turbo_optimize.orchestrator.campaign._coerce_quick_baseline_log` —
   pulls the field out of the BASELINE phase_result JSON, strips whitespace,
   rejects non-strings and empty strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_optimize.logs import (
    append_baseline,
    init_optimize_log,
    optimize_log_path,
)
from turbo_optimize.orchestrator.campaign import _coerce_quick_baseline_log


def _prime_log(tmp_path: Path) -> Path:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    init_optimize_log(
        campaign_dir,
        {
            "target_op": "gemm",
            "target_backend": "triton",
            "target_lang": "triton",
            "target_gpu": "mi300x",
        },
    )
    return campaign_dir


def test_append_baseline_includes_quick_log_when_provided(tmp_path: Path) -> None:
    campaign_dir = _prime_log(tmp_path)
    append_baseline(
        campaign_dir,
        backend="triton",
        gpu="mi300x",
        commit="abc123",
        aggregate_score={"Forward TFLOPS": 100.0, "Backward TFLOPS": 50.0},
        all_check_pass=True,
        quick_baseline_log="rounds/round-1/artifacts/quick_baseline.log",
    )
    text = optimize_log_path(campaign_dir).read_text(encoding="utf-8")
    assert "## Baseline Entry" in text
    assert "- Detailed data: rounds/round-1/summary.md" in text
    assert (
        "- Quick baseline log: rounds/round-1/artifacts/quick_baseline.log" in text
    )


def test_append_baseline_omits_quick_log_line_when_missing(tmp_path: Path) -> None:
    campaign_dir = _prime_log(tmp_path)
    append_baseline(
        campaign_dir,
        backend="triton",
        gpu="mi300x",
        commit=None,
        aggregate_score=None,
        all_check_pass=False,
    )
    text = optimize_log_path(campaign_dir).read_text(encoding="utf-8")
    assert "Quick baseline log" not in text
    assert "- All Check: FAIL" in text


def test_append_baseline_omits_quick_log_line_when_empty_string(tmp_path: Path) -> None:
    campaign_dir = _prime_log(tmp_path)
    append_baseline(
        campaign_dir,
        backend="triton",
        gpu="mi300x",
        commit=None,
        aggregate_score={"Forward TFLOPS": 12.3},
        all_check_pass=True,
        quick_baseline_log="",
    )
    text = optimize_log_path(campaign_dir).read_text(encoding="utf-8")
    assert "Quick baseline log" not in text


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"quick_baseline_log": "rounds/round-1/artifacts/quick_baseline.log"},
         "rounds/round-1/artifacts/quick_baseline.log"),
        ({"quick_baseline_log": "  rounds/round-1/artifacts/quick_baseline.log  "},
         "rounds/round-1/artifacts/quick_baseline.log"),
        ({"quick_baseline_log": ""}, None),
        ({"quick_baseline_log": "   "}, None),
        ({"quick_baseline_log": None}, None),
        ({"quick_baseline_log": 42}, None),
        ({}, None),
    ],
)
def test_coerce_quick_baseline_log(payload: dict, expected: str | None) -> None:
    assert _coerce_quick_baseline_log(payload) == expected


def test_coerce_quick_baseline_log_rejects_non_dict() -> None:
    assert _coerce_quick_baseline_log("not a dict") is None  # type: ignore[arg-type]
    assert _coerce_quick_baseline_log(None) is None  # type: ignore[arg-type]
