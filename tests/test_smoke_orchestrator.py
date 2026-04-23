"""End-to-end smoke test for the kernel-optimize orchestrator.

The test monkey-patches every `phase.run()` coroutine so no Claude
session is ever opened. Each fake phase writes exactly the files and
phase_result JSON that `campaign.py` expects, which lets us exercise
the full state machine (DEFINE_TARGET -> DONE) against `max_iterations=3`.

Asserts cover:
* `state/run.json` reaches `DONE` with best_round populated
* `manifest.yaml` / `logs/optimize.md` / `logs/performance_trend.md`
  exist and contain the Rule-8 mandated sections
* round-1 and round-2 summaries exist
* termination reason is recorded in the optimize log
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator import run_phase as run_phase_module
from turbo_optimize.orchestrator.phases import (
    analyze as analyze_phase,
    baseline as baseline_phase,
    define_target as define_target_phase,
    optimize as optimize_phase,
    prepare_environment as prepare_environment_phase,
    profile as profile_phase,
    read_historical_tips as read_historical_tips_phase,
    report as report_phase,
    stagnation_review as stagnation_review_phase,
    survey_related_work as survey_related_work_phase,
    validate as validate_phase,
)
from turbo_optimize.state import phase_result_path, write_phase_result


# ---------------------------------------------------------------------
# fake phase runners
# ---------------------------------------------------------------------


def _manifest_stub(campaign_dir: Path) -> dict[str, Any]:
    return {
        "target_op": "gemm_fp8_blockwise",
        "target_backend": "triton",
        "target_lang": "triton",
        "target_gpu": "mi300x",
        "execution_mode": "repo",
        "project_skill": "primus-turbo-develop",
        "primary_metric": "Forward TFLOPS",
        "performance_target": None,
        "target_shapes": [{"M": 8192, "N": 8192, "K": 8192}],
        "representative_shapes": None,
        "kernel_source": "primus_turbo/triton/kernels/gemm_fp8_blockwise.py",
        "test_command": "pytest -q tests/test_gemm_fp8_blockwise.py",
        "benchmark_command": "python benchmarks/bench_gemm_fp8_blockwise.py --csv",
        "quick_command": "python ${CAMPAIGN_DIR}/quick_test_bench.py",
        "profile_command": "python ${CAMPAIGN_DIR}/profile_op_shape.py",
        "base_branch": "main",
        "git_commit": "deadbeef",
        "git_branch": "main",
        "max_iterations": None,
        "max_duration": None,
    }


def _write_round_summary(campaign_dir: Path, round_n: int, body: str) -> None:
    round_dir = campaign_dir / "rounds" / f"round-{round_n}"
    (round_dir / "kernel_snapshot").mkdir(parents=True, exist_ok=True)
    (round_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (round_dir / "summary.md").write_text(body, encoding="utf-8")


async def _fake_define_target_run(params: CampaignParams):
    assert params.campaign_dir is not None
    manifest_path = params.campaign_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_manifest_stub(params.campaign_dir), sort_keys=False),
        encoding="utf-8",
    )
    result = _manifest_stub(params.campaign_dir)
    write_phase_result(params.state_dir, "DEFINE_TARGET", result)
    return run_phase_module.PhaseOutcome(
        phase="DEFINE_TARGET",
        messages_log=params.campaign_dir / "profiles" / "_transcript_define_target.jsonl",
        structured=result,
    )


async def _fake_prepare_environment_run(params: CampaignParams):
    assert params.campaign_dir is not None
    (params.campaign_dir / "rounds" / "round-1" / "kernel_snapshot").mkdir(
        parents=True, exist_ok=True
    )
    result = {
        "snapshot_ok": True,
        "base_branch_confirmed": True,
        "base_branch_expected": "main",
        "base_branch_observed": "main",
        "base_commit_observed": "deadbeef",
        "workspace_clean": True,
        "submodule_state_ignored": True,
        "submodule_state": " c26064254 3rdparty/composable_kernel (therock-7.10)",
    }
    write_phase_result(params.state_dir, "PREPARE_ENVIRONMENT", result)
    return run_phase_module.PhaseOutcome(
        phase="PREPARE_ENVIRONMENT",
        messages_log=params.campaign_dir / "profiles" / "_pe.jsonl",
        structured=result,
    )


async def _fake_survey_run(params: CampaignParams):
    assert params.campaign_dir is not None
    (params.campaign_dir / "related_work.md").write_text(
        "# related work\n\n(mock)\n", encoding="utf-8"
    )
    result = {"papers": []}
    write_phase_result(params.state_dir, "SURVEY_RELATED_WORK", result)
    return run_phase_module.PhaseOutcome(
        phase="SURVEY_RELATED_WORK",
        messages_log=params.campaign_dir / "profiles" / "_sw.jsonl",
        structured=result,
    )


async def _fake_tips_run(params: CampaignParams):
    assert params.campaign_dir is not None
    (params.campaign_dir / "tips_summary.md").write_text(
        "# tips summary\n\n(mock)\n", encoding="utf-8"
    )
    result = {"tips_loaded": 0}
    write_phase_result(params.state_dir, "READ_HISTORICAL_TIPS", result)
    return run_phase_module.PhaseOutcome(
        phase="READ_HISTORICAL_TIPS",
        messages_log=params.campaign_dir / "profiles" / "_tips.jsonl",
        structured=result,
    )


async def _fake_baseline_run(params: CampaignParams):
    assert params.campaign_dir is not None
    _write_round_summary(
        params.campaign_dir,
        1,
        (
            "# Round 1 (BASELINE)\n\n"
            "## Summary\n- Baseline round, no code change.\n\n"
            "## Single change\nNo code change. Baseline round.\n\n"
            "## Decision\nBASELINE\n"
        ),
    )
    (params.campaign_dir / "quick_test_bench.py").write_text(
        "SHAPES = [(8192, 8192, 8192)]\n", encoding="utf-8"
    )
    quick_baseline_log_rel = "rounds/round-1/artifacts/quick_baseline.log"
    (params.campaign_dir / quick_baseline_log_rel).write_text(
        "shape=(8192,8192,8192) Check=PASS Forward TFLOPS=100.0 "
        "Backward TFLOPS=50.0\n",
        encoding="utf-8",
    )
    manifest = _manifest_stub(params.campaign_dir)
    manifest["representative_shapes"] = [{"M": 8192, "N": 8192, "K": 8192}]
    (params.campaign_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    result = {
        "test_pass": True,
        "benchmark_csv": "rounds/round-1/artifacts/benchmark.csv",
        "quick_baseline_log": quick_baseline_log_rel,
        "primary_metric": "Forward TFLOPS",
        "aggregate_score": {
            "Forward TFLOPS": 100.0,
            "Backward TFLOPS": 50.0,
        },
        "score_vector": [
            {
                "shape": {"M": 8192, "N": 8192, "K": 8192},
                "check": "PASS",
                "metrics": {
                    "Forward TFLOPS": 100.0,
                    "Backward TFLOPS": 50.0,
                },
            }
        ],
        "trend_row": {
            "fwd_avg": 100.0,
            "fwd_peak": 100.0,
            "bwd_avg": 50.0,
            "bwd_peak": 50.0,
            "step_geomean": 70.710678,
        },
        "representative_shapes": [{"M": 8192, "N": 8192, "K": 8192}],
        "git_commit": "deadbeef",
        "notes": "mock baseline",
    }
    write_phase_result(params.state_dir, "BASELINE", result)
    return run_phase_module.PhaseOutcome(
        phase="BASELINE",
        messages_log=params.campaign_dir / "profiles" / "_bl.jsonl",
        structured=result,
    )


_ANALYZE_HYPOTHESES = [
    "vectorize epilogue store for 128b alignment",
    "increase BLOCK_K to 64 for better cache reuse",
    "unroll inner k loop by factor 2",
]


async def _fake_analyze_run(params: CampaignParams, *, round_n: int, retry_hint: str | None = None):
    assert params.campaign_dir is not None
    idx = (round_n - 2) % len(_ANALYZE_HYPOTHESES)
    hypothesis = _ANALYZE_HYPOTHESES[idx]
    result = {
        "primary_hypothesis": hypothesis,
        "evidence": "profile shows memory-bound on epilogue",
        "expected_gain_pct": 8.0,
        "risk": "correctness regression on transpose path",
    }
    write_phase_result(params.state_dir, "ANALYZE", result, round_n=round_n)
    return run_phase_module.PhaseOutcome(
        phase="ANALYZE",
        messages_log=params.campaign_dir / "profiles" / f"_an_{round_n}.jsonl",
        structured=result,
    )


async def _fake_optimize_run(
    params: CampaignParams,
    *,
    round_n: int,
    hypothesis: dict,
    rebuild_required: bool,
    retry_context: str | None = None,
):
    assert params.campaign_dir is not None
    result = {
        "build_ok": True,
        "changes_summary": f"mock change for round-{round_n}",
        "files_touched": ["primus_turbo/triton/kernels/gemm_fp8_blockwise.py"],
        "rebuild_required": bool(rebuild_required),
        "retry_context_seen": bool(retry_context),
    }
    write_phase_result(params.state_dir, "OPTIMIZE", result, round_n=round_n)
    return run_phase_module.PhaseOutcome(
        phase="OPTIMIZE",
        messages_log=params.campaign_dir / "profiles" / f"_op_{round_n}.jsonl",
        structured=result,
    )


async def _fake_validate_run(
    params: CampaignParams,
    *,
    round_n: int,
    validation_level: str,
    force: bool = False,
):
    assert params.campaign_dir is not None
    _write_round_summary(
        params.campaign_dir,
        round_n,
        (
            f"# Round {round_n}\n\n## Decision\nACCEPTED\n\n"
            f"## Aggregate score\nForward TFLOPS: {100.0 + round_n * 5:.1f}\n"
        ),
    )
    fwd = 100.0 + round_n * 5.0
    bwd = 50.0 + round_n * 2.0
    result = {
        "correctness_ok": True,
        "validation_level": validation_level,
        "aggregate_score": {
            "Forward TFLOPS": fwd,
            "Backward TFLOPS": bwd,
        },
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
            "fwd_peak": fwd + 2.0,
            "bwd_avg": bwd,
            "bwd_peak": bwd + 1.0,
            "step_geomean": (fwd * bwd) ** 0.5,
        },
        "changes_summary": f"mock validate round-{round_n}",
    }
    write_phase_result(params.state_dir, "VALIDATE", result, round_n=round_n)
    return run_phase_module.PhaseOutcome(
        phase="VALIDATE",
        messages_log=params.campaign_dir / "profiles" / f"_va_{round_n}.jsonl",
        structured=result,
    )


async def _fake_stagnation_run(
    params: CampaignParams,
    *,
    rollback_streak: int,
    current_round: int | None = None,
):
    result = {
        "decision": "PROCEED",
        "notes": "mock stagnation review",
        "current_round": current_round,
    }
    write_phase_result(params.state_dir, "STAGNATION_REVIEW", result)
    return run_phase_module.PhaseOutcome(
        phase="STAGNATION_REVIEW",
        messages_log=Path("/dev/null"),
        structured=result,
    )


async def _fake_profile_run(
    params: CampaignParams, *, round_n: int, trigger: str, force: bool = False
):
    assert params.campaign_dir is not None
    result = {
        "round": round_n,
        "trigger": trigger,
        "skipped": True,
        "skip_reason": "rocprof tools not installed in smoke test",
        "artifacts_dir": f"profiles/round-{round_n}_{trigger}",
        "tools": [],
    }
    from turbo_optimize.state import phase_result_path as _prp

    expected = _prp(params.state_dir, "PROFILE", round_n, suffix=trigger)
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text(json.dumps(result), encoding="utf-8")
    return run_phase_module.PhaseOutcome(
        phase="PROFILE",
        messages_log=params.campaign_dir / "profiles" / f"_pf_{round_n}_{trigger}.jsonl",
        structured=result,
    )


async def _fake_report_run(params: CampaignParams, *, termination: dict):
    result = {
        "termination": termination,
        "summary": "mock final report",
        "tips_appended": [
            {
                "category": "failure",
                "round": 2,
                "status": "ROLLED_BACK",
                "takeaway": (
                    "Triton does not fuse scale broadcast when BLOCK_K > 64 "
                    "on gfx942; manifests as ~20% Fwd drop."
                ),
                "applicability": (
                    "Reuse on gfx942/triton fp8 ops with block-scaled "
                    "tensors; not applicable to gfx950 or cuBLAS paths."
                ),
            }
        ],
    }
    write_phase_result(params.state_dir, "REPORT", result)
    return run_phase_module.PhaseOutcome(
        phase="REPORT",
        messages_log=Path("/dev/null"),
        structured=result,
    )


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def patched_phases(monkeypatch):
    """Rebind each phase module's `run` to a fake that never calls Claude.

    `campaign.py` imports the phase modules (not their `run` attribute
    directly), so patching `module.run` is enough — every
    `analyze_phase.run(...)` call inside campaign.py resolves at call
    time against the same module object.
    """
    monkeypatch.setattr(define_target_phase, "run", _fake_define_target_run)
    monkeypatch.setattr(prepare_environment_phase, "run", _fake_prepare_environment_run)
    monkeypatch.setattr(survey_related_work_phase, "run", _fake_survey_run)
    monkeypatch.setattr(read_historical_tips_phase, "run", _fake_tips_run)
    monkeypatch.setattr(baseline_phase, "run", _fake_baseline_run)
    monkeypatch.setattr(analyze_phase, "run", _fake_analyze_run)
    monkeypatch.setattr(optimize_phase, "run", _fake_optimize_run)
    monkeypatch.setattr(validate_phase, "run", _fake_validate_run)
    monkeypatch.setattr(stagnation_review_phase, "run", _fake_stagnation_run)
    monkeypatch.setattr(profile_phase, "run", _fake_profile_run)
    monkeypatch.setattr(report_phase, "run", _fake_report_run)
    return None


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


def test_smoke_campaign_reaches_done(tmp_path, patched_phases, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "agent").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)

    monkeypatch.delenv("CI", raising=False)

    params = CampaignParams(
        prompt="optimize gemm fp8 blockwise triton mi300x",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
        max_iterations=3,
    )

    campaign_dir_holder: dict[str, Path] = {}

    async def _runner():
        from turbo_optimize.manifest import mark_confirmed
        from turbo_optimize.orchestrator import campaign as campaign_module

        original_confirm = campaign_module._phase_confirm_manifest

        async def auto_confirm(p, s):
            assert p.campaign_dir is not None
            campaign_dir_holder["dir"] = p.campaign_dir
            mark_confirmed(p.campaign_dir)
            await original_confirm(p, s)

        monkeypatch.setattr(
            campaign_module, "_phase_confirm_manifest", auto_confirm
        )

        rc = await campaign_module.run_campaign(params)
        return rc

    rc = asyncio.run(_runner())
    assert rc == 0, f"campaign exited with {rc}"

    campaign_dir = campaign_dir_holder["dir"]

    assert (campaign_dir / "manifest.yaml").exists()
    log = (campaign_dir / "logs" / "optimize.md").read_text(encoding="utf-8")
    assert "Baseline" in log
    assert "Termination Check" in log
    assert (
        "Quick baseline log: rounds/round-1/artifacts/quick_baseline.log" in log
    ), "append_baseline should record the quick_baseline_log path"
    assert (campaign_dir / "rounds" / "round-1" / "artifacts" / "quick_baseline.log").exists()
    warm_sh = campaign_dir / "warm_restart.sh"
    assert warm_sh.exists() and os.access(warm_sh, os.X_OK)
    body = warm_sh.read_text(encoding="utf-8")
    assert campaign_dir.name in body
    assert "primus-turbo-optimize -s" in body

    trend = (campaign_dir / "logs" / "performance_trend.md").read_text(encoding="utf-8")
    assert "BASELINE" in trend
    assert "ACCEPTED" in trend
    assert "Fwd Avg TFLOPS" in trend
    assert "Bwd Avg TFLOPS" in trend
    assert "Step Geomean TFLOPS" in trend
    # baseline row uses 3-decimal precision: fwd_avg=100.000, bwd_avg=50.000
    assert "| 100.000 |" in trend
    assert "| 50.000 |" in trend
    # baseline row renders vs Baseline as an em dash
    baseline_row = next(
        line for line in trend.splitlines() if line.startswith("| 1 | BASELINE")
    )
    assert "| — |" in baseline_row
    # step geomean on baseline = sqrt(100 * 50) ≈ 70.711
    assert "70.711" in trend
    # round-2 vs Baseline row carries the three-part delta (step/fwd/bwd)
    r2_row = next(
        line for line in trend.splitlines() if line.startswith("| 2 |")
    )
    assert "step +" in r2_row
    assert "fwd +" in r2_row
    assert "bwd +" in r2_row
    assert "%" in r2_row
    # round-3 validate: fwd=115, bwd=56 → step=sqrt(115*56)≈80.250
    assert "80.250" in trend

    assert (campaign_dir / "rounds" / "round-1" / "summary.md").exists()
    assert (campaign_dir / "rounds" / "round-2" / "summary.md").exists()

    namespaced_state = state_dir / campaign_dir.name
    assert namespaced_state.is_dir(), (
        "state_dir should be nested under campaign_id after namespacing"
    )
    assert not (state_dir / "run.json").exists(), (
        "top-level state/run.json must not be created once namespacing is on"
    )
    run_json = json.loads(
        (namespaced_state / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["current_phase"] == "DONE"
    assert run_json["best_round"] in {1, 2, 3}
    assert (namespaced_state / "phase_result").is_dir()

    cost_md_path = campaign_dir / "logs" / "cost.md"
    assert cost_md_path.exists(), "logs/cost.md must be initialized at campaign start"
    cost_md = cost_md_path.read_text(encoding="utf-8")
    assert "Campaign Cost Log" in cost_md
    assert "| Cost USD | Cumulative USD |" in cost_md

    report_result = json.loads(
        (namespaced_state / "phase_result" / "report.json").read_text(encoding="utf-8")
    )
    tips_appended = report_result.get("tips_appended")
    assert isinstance(tips_appended, list) and tips_appended, (
        "REPORT structured result must carry a non-empty tips_appended "
        "list whenever Claude (or the fake) distilled cross-op lessons"
    )
    for entry in tips_appended:
        assert {"category", "round", "status", "takeaway", "applicability"} <= set(
            entry
        ), (
            "each tips_appended entry must carry category/round/status/"
            "takeaway/applicability; missing fields means the distillation "
            "quality bar was skipped"
        )
        assert entry["category"] in {"failure", "success"}


def test_dry_run_plan(tmp_path, capsys):
    workspace = tmp_path / "ws"
    (workspace / "agent").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)

    params = CampaignParams(
        prompt="dry run check",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
        dry_run=True,
    )

    from turbo_optimize.orchestrator.campaign import run_campaign

    rc = asyncio.run(run_campaign(params))
    out = capsys.readouterr().out
    assert rc == 0
    assert "primus-turbo-optimize dry-run plan" in out
    assert "DEFINE_TARGET" in out
    assert "BASELINE" in out
    assert "REPORT" in out
