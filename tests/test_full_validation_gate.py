"""Tests for the quick→full validation gate in ``_run_round``.

Question 3 from the campaign review: before this change the
orchestrator accepted any round that passed only the quick validation
level, even though ``prompts/validate.md`` claimed the orchestrator
would escalate to ``full`` for ACCEPT candidates. The fix runs a second
VALIDATE with ``validation_level="full"`` whenever quick proposes
ACCEPTED / ACCEPT_PENDING_NOISE, and the full-run decision is the one
written to ``logs/optimize.md`` / ``state.history``.

Two scenarios:

* quick ACCEPT + full ACCEPT → final ACCEPT, both levels recorded.
* quick ACCEPT + full correctness_ok=False → final ROLLBACK, best_round
  remains at the baseline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import init_optimize_log, init_performance_trend
from turbo_optimize.orchestrator import campaign as campaign_mod
from turbo_optimize.orchestrator.phases import (
    analyze as analyze_phase,
    optimize as optimize_phase,
    validate as validate_phase,
)
from turbo_optimize.orchestrator.run_phase import PhaseOutcome
from turbo_optimize.state import RunState, record_round_event


_BASE_MANIFEST = {
    "target_op": "gemm_fp8_blockwise",
    "target_backend": "triton",
    "target_gpu": "mi300x",
    "execution_mode": "repo",
    "primary_metric": "Forward TFLOPS",
    "kernel_source": "primus_turbo/triton/kernels/gemm_fp8_blockwise.py",
    "test_command": "pytest tests",
    "benchmark_command": "python bench.py",
    "quick_command": "python quick.py",
}


def _outcome(structured: dict) -> PhaseOutcome:
    return PhaseOutcome(
        phase="FAKE", messages_log=Path("/dev/null"), structured=structured
    )


def _bootstrap(tmp_path: Path) -> tuple[CampaignParams, RunState]:
    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    campaign_dir.mkdir()
    (campaign_dir / "logs").mkdir()
    (campaign_dir / "rounds").mkdir()
    state_dir.mkdir()
    (campaign_dir / "manifest.yaml").write_text(
        yaml.safe_dump(_BASE_MANIFEST, sort_keys=False), encoding="utf-8"
    )

    params = CampaignParams(
        prompt="unit-test",
        campaign_id="cmp",
        campaign_dir=campaign_dir,
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
        primary_metric="Forward TFLOPS",
        target_op="gemm_fp8_blockwise",
        target_backend="triton",
        target_gpu="mi300x",
        debug_retry=0,
    )
    init_optimize_log(campaign_dir, _BASE_MANIFEST)
    init_performance_trend(campaign_dir)

    state = RunState(
        campaign_id="cmp",
        current_phase="ANALYZE",
        current_round=2,
        best_round=1,
        best_score={"Forward TFLOPS": 100.0, "Backward TFLOPS": 50.0},
    )
    record_round_event(
        state,
        round_n=1,
        decision="BASELINE",
        score={"Forward TFLOPS": 100.0, "Backward TFLOPS": 50.0},
        description="Baseline",
    )
    return params, state


def _validate_ok(level: str, fwd: float, bwd: float) -> dict:
    return {
        "correctness_ok": True,
        "validation_level": level,
        "aggregate_score": {"Forward TFLOPS": fwd, "Backward TFLOPS": bwd},
        "score_vector": [
            {
                "shape": {"M": 8192, "N": 8192, "K": 8192},
                "check": "PASS",
                "metrics": {"Forward TFLOPS": fwd, "Backward TFLOPS": bwd},
            }
        ],
        "trend_row": {
            "fwd_avg": fwd,
            "fwd_peak": fwd,
            "bwd_avg": bwd,
            "bwd_peak": bwd,
            "step_geomean": (fwd * bwd) ** 0.5,
        },
        "notes": "mock",
    }


def _validate_fail(level: str) -> dict:
    return {
        "correctness_ok": False,
        "validation_level": level,
        "aggregate_score": {"Forward TFLOPS": 0.0, "Backward TFLOPS": 0.0},
        "score_vector": [
            {
                "shape": {"M": 8192, "N": 8192, "K": 8192},
                "check": "FAIL",
                "metrics": {"Forward TFLOPS": 0.0, "Backward TFLOPS": 0.0},
            }
        ],
        "notes": "full-run surfaces a bug quick did not",
    }


def test_quick_accept_plus_full_accept_finalizes_as_accepted(monkeypatch, tmp_path):
    params, state = _bootstrap(tmp_path)
    validate_levels: list[str] = []

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({"primary_hypothesis": "pipeline mbarriers"})

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        return _outcome({"build_ok": True, "diff_summary": "pipelining"})

    async def fake_validate(p, *, round_n, validation_level):
        validate_levels.append(validation_level)
        return _outcome(_validate_ok(validation_level, fwd=118.0, bwd=58.0))

    monkeypatch.setattr(analyze_phase, "run", fake_analyze)
    monkeypatch.setattr(optimize_phase, "run", fake_optimize)
    monkeypatch.setattr(validate_phase, "run", fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert validate_levels == ["quick", "full"]
    assert state.best_round == 2
    last_event = state.history[-1]
    assert last_event["decision"] == "ACCEPTED"

    trend = (params.campaign_dir / "logs" / "performance_trend.md").read_text(
        encoding="utf-8"
    )
    assert "ACCEPTED" in trend


def test_quick_accept_plus_full_fail_rolls_back(monkeypatch, tmp_path):
    """The quick pass was a false positive: full validation uncovers
    correctness failure on wider shapes → final ROLLBACK, baseline kept."""
    params, state = _bootstrap(tmp_path)
    validate_levels: list[str] = []

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({"primary_hypothesis": "loosen epilogue dtype"})

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        return _outcome({"build_ok": True, "diff_summary": "dtype relax"})

    async def fake_validate(p, *, round_n, validation_level):
        validate_levels.append(validation_level)
        if validation_level == "quick":
            return _outcome(_validate_ok(validation_level, fwd=120.0, bwd=60.0))
        return _outcome(_validate_fail(validation_level))

    monkeypatch.setattr(analyze_phase, "run", fake_analyze)
    monkeypatch.setattr(optimize_phase, "run", fake_optimize)
    monkeypatch.setattr(validate_phase, "run", fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert validate_levels == ["quick", "full"]
    assert state.best_round == 1
    last_event = state.history[-1]
    assert last_event["decision"] == "ROLLED BACK"


def test_rollback_skips_full_validation(monkeypatch, tmp_path):
    """Quick already says ROLLBACK → no full run, no additional cost."""
    params, state = _bootstrap(tmp_path)
    validate_levels: list[str] = []

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({"primary_hypothesis": "halve BLOCK_M"})

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        return _outcome({"build_ok": True, "diff_summary": "block size shrink"})

    async def fake_validate(p, *, round_n, validation_level):
        validate_levels.append(validation_level)
        return _outcome(_validate_ok(validation_level, fwd=95.0, bwd=45.0))

    monkeypatch.setattr(analyze_phase, "run", fake_analyze)
    monkeypatch.setattr(optimize_phase, "run", fake_optimize)
    monkeypatch.setattr(validate_phase, "run", fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert validate_levels == ["quick"]
    last_event = state.history[-1]
    assert last_event["decision"] == "ROLLED BACK"
