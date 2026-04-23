"""Tests for the OPTIMIZE+VALIDATE retry-with-fix micro-loop.

Exercises :func:`turbo_optimize.orchestrator.campaign._run_round` with
monkey-patched phase runners so no Claude subprocess is spawned. Each
test swaps in a tiny coroutine that simulates the phase's structured
output, then asserts on the sequence of OPTIMIZE / VALIDATE invocations
and on the final ``state.history`` record.

Three scenarios are pinned down:

* A build failure on attempt 1 is retried; attempt 2 succeeds and the
  round accepts through the full-validation gate.
* Persistent build failures exhaust ``debug_retry`` and end in ROLLBACK.
* A performance regression (non-retryable reason) causes an immediate
  ROLLBACK without consuming a retry slot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import pytest

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


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------


def _outcome(structured: dict) -> PhaseOutcome:
    return PhaseOutcome(
        phase="FAKE", messages_log=Path("/dev/null"), structured=structured
    )


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


def _make_params(tmp_path: Path, debug_retry: int) -> CampaignParams:
    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    campaign_dir.mkdir()
    (campaign_dir / "logs").mkdir()
    (campaign_dir / "rounds").mkdir()
    state_dir.mkdir()

    import yaml

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
        debug_retry=debug_retry,
    )
    init_optimize_log(campaign_dir, _BASE_MANIFEST)
    init_performance_trend(campaign_dir)
    return params


def _make_state(baseline_fwd: float = 100.0, baseline_bwd: float = 50.0) -> RunState:
    state = RunState(
        campaign_id="cmp",
        current_phase="ANALYZE",
        current_round=2,
        best_round=1,
        best_score={"Forward TFLOPS": baseline_fwd, "Backward TFLOPS": baseline_bwd},
    )
    record_round_event(
        state,
        round_n=1,
        decision="BASELINE",
        score={"Forward TFLOPS": baseline_fwd, "Backward TFLOPS": baseline_bwd},
        description="Baseline",
    )
    return state


def _patch_phases(
    monkeypatch,
    fake_analyze: Callable,
    fake_optimize: Callable,
    fake_validate: Callable,
) -> None:
    monkeypatch.setattr(analyze_phase, "run", fake_analyze)
    monkeypatch.setattr(optimize_phase, "run", fake_optimize)
    monkeypatch.setattr(validate_phase, "run", fake_validate)


def _validate_ok(round_n: int, level: str, fwd: float, bwd: float) -> dict:
    return {
        "correctness_ok": True,
        "validation_level": level,
        "aggregate_score": {"Forward TFLOPS": fwd, "Backward TFLOPS": bwd},
        "score_vector": [
            {
                "shape": {"M": 8192, "N": 8192, "K": 8192},
                "check": "PASS",
                "metrics": {
                    "Forward TFLOPS": fwd,
                    "Backward TFLOPS": bwd,
                },
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


def _validate_correctness_fail(round_n: int, level: str) -> dict:
    return {
        "correctness_ok": False,
        "validation_level": level,
        "aggregate_score": {"Forward TFLOPS": 0.0, "Backward TFLOPS": 0.0},
        "score_vector": [
            {
                "shape": {"M": 8192, "N": 8192, "K": 8192},
                "check": "FAIL",
                "metrics": {
                    "Forward TFLOPS": 0.0,
                    "Backward TFLOPS": 0.0,
                },
            }
        ],
        "notes": "wrong output on transpose path",
        "benchmark_csv": f"rounds/round-{round_n}/artifacts/benchmark.csv",
    }


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


def test_retry_recovers_after_build_fix(monkeypatch, tmp_path):
    """Attempt 1: build_ok=False → retry; attempt 2: build_ok=True and
    correct → quick ACCEPT → full ACCEPT."""
    params = _make_params(tmp_path, debug_retry=3)
    state = _make_state()

    attempts = {"optimize": 0}
    validate_levels: list[str] = []
    validate_force_flags: list[bool] = []
    retry_contexts_seen: list[str | None] = []

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({
            "primary_hypothesis": "use bf16 epilogue for 128b alignment",
            "evidence": "profile",
            "expected_gain_pct": 7.0,
        })

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        attempts["optimize"] += 1
        retry_contexts_seen.append(retry_context)
        build_ok = attempts["optimize"] >= 2
        return _outcome({
            "build_ok": build_ok,
            "diff_summary": "epilogue dtype cast",
            "build_log": f"rounds/round-{round_n}/artifacts/build.log",
            "notes": "missing import" if not build_ok else "built",
        })

    async def fake_validate(p, *, round_n, validation_level, force=False):
        validate_levels.append(validation_level)
        validate_force_flags.append(force)
        if attempts["optimize"] == 1:
            return _outcome(_validate_correctness_fail(round_n, validation_level))
        return _outcome(_validate_ok(round_n, validation_level, fwd=115.0, bwd=60.0))

    _patch_phases(monkeypatch, fake_analyze, fake_optimize, fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert attempts["optimize"] == 2
    assert retry_contexts_seen[0] is None
    assert retry_contexts_seen[1] is not None
    assert "Previous attempt #1 failed" in retry_contexts_seen[1]
    assert "build_log" in retry_contexts_seen[1]
    assert validate_levels == ["quick", "quick", "full"]
    # attempt 1 hits fresh JSON path (no cache bypass needed); attempt 2
    # is a debug-retry → must force-refresh the stale quick JSON; the
    # full-validation gate writes to a different suffix so it doesn't
    # need force.
    assert validate_force_flags == [False, True, False]

    assert state.best_round == 2
    last_event = state.history[-1]
    assert last_event["decision"] == "ACCEPTED"


def test_retry_exhausted_falls_through_to_rollback(monkeypatch, tmp_path):
    """``debug_retry=2`` means 3 total attempts; all build-fail → ROLLBACK."""
    params = _make_params(tmp_path, debug_retry=2)
    state = _make_state()

    attempts = {"optimize": 0}
    validate_levels: list[str] = []
    validate_force_flags: list[bool] = []

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({"primary_hypothesis": "vectorize loads"})

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        attempts["optimize"] += 1
        return _outcome({
            "build_ok": False,
            "diff_summary": "typo in kernel name",
            "build_log": f"rounds/round-{round_n}/artifacts/build.log",
            "notes": "NameError",
        })

    async def fake_validate(p, *, round_n, validation_level, force=False):
        validate_levels.append(validation_level)
        validate_force_flags.append(force)
        return _outcome(_validate_correctness_fail(round_n, validation_level))

    _patch_phases(monkeypatch, fake_analyze, fake_optimize, fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert attempts["optimize"] == 3  # 1 original + 2 retries
    assert validate_levels == ["quick", "quick", "quick"]  # no full gate reached
    # Every retry past attempt 1 must force-refresh the quick JSON so
    # the cached first-attempt FAIL is not silently replayed.
    assert validate_force_flags == [False, True, True]
    last_event = state.history[-1]
    assert last_event["decision"] == "ROLLED BACK"
    assert state.best_round == 1


def test_regression_does_not_trigger_retry(monkeypatch, tmp_path):
    """Soft regression (non-retryable reason) → rollback on first attempt."""
    params = _make_params(tmp_path, debug_retry=3)
    state = _make_state(baseline_fwd=200.0, baseline_bwd=100.0)

    attempts = {"optimize": 0}
    validate_levels: list[str] = []

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({"primary_hypothesis": "smaller blocks"})

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        attempts["optimize"] += 1
        return _outcome({"build_ok": True, "diff_summary": "block size tweak"})

    async def fake_validate(p, *, round_n, validation_level, force=False):
        validate_levels.append(validation_level)
        # correct but slower than baseline -> ROLLBACK via regression / no improvement
        return _outcome(_validate_ok(round_n, validation_level, fwd=180.0, bwd=80.0))

    _patch_phases(monkeypatch, fake_analyze, fake_optimize, fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert attempts["optimize"] == 1
    assert validate_levels == ["quick"]  # full gate not reached
    last_event = state.history[-1]
    assert last_event["decision"] == "ROLLED BACK"


def test_debug_retry_zero_disables_retry(monkeypatch, tmp_path):
    """``--debug-retry 0`` means the first build failure goes straight to
    ROLLBACK (max_attempts=1)."""
    params = _make_params(tmp_path, debug_retry=0)
    state = _make_state()

    attempts = {"optimize": 0}

    async def fake_analyze(p, *, round_n, retry_hint=None):
        return _outcome({"primary_hypothesis": "unroll inner k"})

    async def fake_optimize(
        p, *, round_n, hypothesis, rebuild_required, retry_context=None
    ):
        attempts["optimize"] += 1
        return _outcome({
            "build_ok": False,
            "diff_summary": "bad unroll",
            "build_log": "rounds/round-2/artifacts/build.log",
            "notes": "ptx compile error",
        })

    async def fake_validate(p, *, round_n, validation_level, force=False):
        return _outcome(_validate_correctness_fail(round_n, validation_level))

    _patch_phases(monkeypatch, fake_analyze, fake_optimize, fake_validate)

    asyncio.run(campaign_mod._run_round(params, state))

    assert attempts["optimize"] == 1
    last_event = state.history[-1]
    assert last_event["decision"] == "ROLLED BACK"


def test_is_retryable_bug_classification():
    """Direct guardrail for the prefix list; if scoring.py's reason
    strings drift the orchestrator must re-examine the retryable set."""
    from turbo_optimize.scoring import DecisionResult

    def _dec(decision: str, reason: str) -> DecisionResult:
        return DecisionResult(
            decision=decision,
            reason=reason,
            improvement_pct={},
            regressions=[],
            noise_check_required=False,
        )

    assert campaign_mod._is_retryable_bug(_dec("ROLLBACK", "build failed"))
    assert campaign_mod._is_retryable_bug(
        _dec("ROLLBACK", "correctness failed (any Check != PASS counts)")
    )
    assert campaign_mod._is_retryable_bug(
        _dec("ROLLBACK", "benchmark Check=FAIL in at least one shape")
    )
    assert not campaign_mod._is_retryable_bug(
        _dec("ROLLBACK", "no metric improved over current best")
    )
    assert not campaign_mod._is_retryable_bug(
        _dec("ROLLBACK", "core shape regressed 8.12% (>= 5.0%)")
    )
    assert not campaign_mod._is_retryable_bug(_dec("ACCEPTED", "baseline round"))
