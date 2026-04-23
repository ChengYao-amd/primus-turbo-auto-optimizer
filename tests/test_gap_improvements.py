"""Targeted unit tests for the 2026-04-22 campaign improvements.

Each test maps to one of the gaps identified in the campaign review:

* Gap D — VALIDATE phase downgrades empty / malformed phase_result to
  a ROLLBACK-looking payload with ``failure_category=schema_invalid``.
* Gap E — ``parse_bench_csv`` picks up stddev companion columns and
  the decision gate widens the noise threshold accordingly.
* Gap F — ``check_hypothesis_duplicate`` flags hypotheses that overlap
  on ``modified_files`` even when the prose looks different.
* Gap G — ``_build_retry_context`` forwards ``failure_category`` +
  ``failure_summary`` into the next OPTIMIZE prompt and persists the
  per-round failure ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import (
    IneffectiveDirection,
    append_verified_ineffective,
    extract_history,
)
from turbo_optimize.orchestrator import campaign as campaign_mod
from turbo_optimize.orchestrator.phases import validate as validate_phase
from turbo_optimize.scoring import (
    DecisionResult,
    ScoreVector,
    ShapeResult,
    check_hypothesis_duplicate,
    decide_accept_rollback,
    parse_bench_csv,
)


# ---------------------------------------------------------------------
# Gap D — VALIDATE schema coercion
# ---------------------------------------------------------------------


def test_validate_coerces_empty_result_to_schema_invalid():
    out = validate_phase._coerce_structured_result(
        {}, round_n=10, level="quick"
    )
    assert out["failure_category"] == "schema_invalid"
    assert out["correctness_ok"] is False
    assert out["score_vector"] == []
    assert out["aggregate_score"] == {}
    assert "empty or non-dict" in out["notes"]


def test_validate_coerces_missing_score_vector_to_schema_invalid():
    raw = {
        "round": 5,
        "correctness_ok": True,
        "build_ok": True,
        "aggregate_score": {"Forward TFLOPS": 100.0},
    }
    out = validate_phase._coerce_structured_result(raw, round_n=5, level="quick")
    assert out["failure_category"] == "schema_invalid"
    assert out["correctness_ok"] is False
    assert "score_vector empty" in out["notes"]


def test_validate_preserves_valid_result():
    raw = {
        "round": 3,
        "validation_level": "quick",
        "correctness_ok": True,
        "build_ok": True,
        "aggregate_score": {"Forward TFLOPS": 105.0},
        "score_vector": [
            {
                "shape": {"M": 8192},
                "check": "PASS",
                "metrics": {"Forward TFLOPS": 105.0},
            }
        ],
        "failure_category": None,
        "failure_summary": None,
    }
    out = validate_phase._coerce_structured_result(raw, round_n=3, level="quick")
    assert out["failure_category"] is None
    assert out["correctness_ok"] is True
    assert out["score_vector"][0]["metrics"]["Forward TFLOPS"] == 105.0


def test_validate_normalises_unknown_failure_category():
    raw = {
        "round": 4,
        "validation_level": "quick",
        "correctness_ok": False,
        "build_ok": True,
        "aggregate_score": {"Forward TFLOPS": 0.0},
        "score_vector": [
            {"shape": {}, "check": "FAIL", "metrics": {"Forward TFLOPS": 0.0}}
        ],
        "failure_category": "totally_made_up",
        "failure_summary": "kernel crashed",
    }
    out = validate_phase._coerce_structured_result(raw, round_n=4, level="quick")
    assert out["failure_category"] == "other"


# ---------------------------------------------------------------------
# Gap E — stddev column parsing + noise gate widening
# ---------------------------------------------------------------------


def test_parse_bench_csv_reads_stddev_pct_column(tmp_path: Path):
    csv = tmp_path / "bench.csv"
    csv.write_text(
        "M,N,K,Check,Forward TFLOPS,Forward TFLOPS_stddev_pct,repeats\n"
        "8192,8192,8192,PASS,120.0,3.5,5\n"
        "4096,4096,4096,PASS,100.0,1.0,5\n",
        encoding="utf-8",
    )
    parse = parse_bench_csv(csv, "Forward TFLOPS")
    assert len(parse.rows) == 2
    first = parse.rows[0]
    assert first.metrics["Forward TFLOPS"] == pytest.approx(120.0)
    assert first.metrics_stddev_pct["Forward TFLOPS"] == pytest.approx(3.5)
    assert first.repeats == 5


def test_parse_bench_csv_converts_raw_stddev_to_percent(tmp_path: Path):
    csv = tmp_path / "bench.csv"
    csv.write_text(
        "M,N,K,Check,Forward TFLOPS,Forward TFLOPS_stddev\n"
        "8192,8192,8192,PASS,200.0,6.0\n",
        encoding="utf-8",
    )
    parse = parse_bench_csv(csv, "Forward TFLOPS")
    row = parse.rows[0]
    assert row.metrics_stddev_pct["Forward TFLOPS"] == pytest.approx(3.0)


def _score_vector(fwd: float, stddev_pct: float | None) -> ScoreVector:
    metrics_stddev = {"Forward TFLOPS": stddev_pct} if stddev_pct is not None else {}
    row = ShapeResult(
        shape={"M": 8192},
        check="PASS",
        metrics={"Forward TFLOPS": fwd},
        metrics_stddev_pct=metrics_stddev,
    )
    return ScoreVector(per_shape=[row], aggregate={"Forward TFLOPS": fwd})


def test_decision_widens_noise_threshold_with_observed_stddev():
    baseline = _score_vector(100.0, stddev_pct=None)
    candidate = _score_vector(101.2, stddev_pct=4.0)
    decision = decide_accept_rollback(
        candidate, baseline, "Forward TFLOPS", correctness_ok=True
    )
    assert decision.decision == "ACCEPT_PENDING_NOISE"
    assert "observed stddev 4.00%" in decision.reason


def test_decision_uses_static_threshold_when_stddev_absent():
    baseline = _score_vector(100.0, stddev_pct=None)
    candidate = _score_vector(101.0, stddev_pct=None)
    decision = decide_accept_rollback(
        candidate, baseline, "Forward TFLOPS", correctness_ok=True
    )
    assert decision.decision == "ACCEPT_PENDING_NOISE"
    assert "observed stddev" not in decision.reason


# ---------------------------------------------------------------------
# Gap F — modified_files dedup signal
# ---------------------------------------------------------------------


def test_duplicate_flagged_on_file_overlap_even_without_text_match():
    verified = [
        IneffectiveDirection(
            round=3,
            direction="switch LDS layout for B matrix",
            reason="no metric improved",
            modified_files=[
                "primus_turbo/triton/kernels/gemm_fp8_blockwise.py",
                "primus_turbo/triton/kernels/gemm_helpers.py",
            ],
        )
    ]
    match = check_hypothesis_duplicate(
        "rework the epilogue dtype path",
        verified,
        planned_modified_files=[
            "primus_turbo/triton/kernels/gemm_fp8_blockwise.py",
            "primus_turbo/triton/kernels/gemm_helpers.py",
        ],
    )
    assert match is not None
    assert match.signal == "files"
    assert match.file_overlap == pytest.approx(1.0)


def test_duplicate_flagged_on_text_overlap_when_files_unknown():
    verified = [
        IneffectiveDirection(
            round=2,
            direction="vectorize epilogue store for 128b alignment",
            reason="no metric improved",
        )
    ]
    match = check_hypothesis_duplicate(
        "vectorize epilogue store for better 128b alignment",
        verified,
        planned_modified_files=None,
    )
    assert match is not None
    assert match.signal == "text"
    assert match.similarity >= 0.6


def test_duplicate_miss_when_neither_signal_fires():
    verified = [
        IneffectiveDirection(
            round=2,
            direction="vectorize epilogue store",
            reason="no metric improved",
            modified_files=["primus_turbo/triton/kernels/gemm_fp8_blockwise.py"],
        )
    ]
    match = check_hypothesis_duplicate(
        "rework BLOCK_K tiling to reduce HBM pressure",
        verified,
        planned_modified_files=[
            "primus_turbo/triton/kernels/gemm_fp8_new_path.py"
        ],
    )
    assert match is None


def test_verified_ineffective_sidecar_roundtrips_modified_files(tmp_path: Path):
    campaign_dir = tmp_path / "campaign"
    (campaign_dir / "logs").mkdir(parents=True)
    (campaign_dir / "logs" / "optimize.md").write_text(
        "# Optimization History\n\n"
        "## Verified Ineffective Directions\n\n"
        "| Direction | Round | Reason |\n"
        "|-----------|-------|--------|\n"
        "| vectorize epilogue store | round-2 | no metric improved |\n",
        encoding="utf-8",
    )
    (campaign_dir / "logs" / "performance_trend.md").write_text(
        "# Performance Trend\n", encoding="utf-8"
    )

    append_verified_ineffective(
        campaign_dir,
        round_n=2,
        direction="vectorize epilogue store",
        reason="no metric improved",
        modified_files=["primus_turbo/triton/kernels/gemm_fp8_blockwise.py"],
    )

    history = extract_history(campaign_dir)
    assert len(history.verified_ineffective) >= 1
    entry = [e for e in history.verified_ineffective if e.round == 2][0]
    assert entry.modified_files == [
        "primus_turbo/triton/kernels/gemm_fp8_blockwise.py"
    ]


# ---------------------------------------------------------------------
# Gap G — failure_category / failure_summary forwarded into retry_context
# ---------------------------------------------------------------------


def _make_params_gapG(tmp_path: Path) -> CampaignParams:
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace = tmp_path / "ws"
    for d in (campaign_dir, state_dir, workspace):
        d.mkdir(parents=True)
    (campaign_dir / "logs").mkdir()
    return CampaignParams(
        prompt="unit",
        campaign_id="cmp",
        campaign_dir=campaign_dir,
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
    )


def _rollback_decision() -> DecisionResult:
    return DecisionResult(
        decision="ROLLBACK",
        reason="correctness failed (any Check != PASS counts)",
        improvement_pct={},
        regressions=[],
        noise_check_required=False,
    )


def test_retry_context_includes_failure_category_and_summary():
    ctx = campaign_mod._build_retry_context(
        attempt=1,
        decision=_rollback_decision(),
        opt_result={"build_ok": True, "notes": "rebuilt ok"},
        val_result={
            "correctness_ok": False,
            "build_ok": True,
            "failure_category": "runtime_oom",
            "failure_summary": "kernel hit OOM at M=8192 with BLOCK_M=256; "
            "see artifacts/run.log. Try shrinking BLOCK_M to 128.",
            "failure_log_path": "rounds/round-2/artifacts/run.log",
            "score_vector": [],
            "benchmark_csv": None,
        },
    )
    assert "failure_category: `runtime_oom`" in ctx
    assert "failure_summary: kernel hit OOM" in ctx
    assert "failure_log_path: rounds/round-2/artifacts/run.log" in ctx


def test_retry_context_warns_when_category_repeats():
    ctx = campaign_mod._build_retry_context(
        attempt=2,
        decision=_rollback_decision(),
        opt_result={"build_ok": True},
        val_result={
            "correctness_ok": False,
            "build_ok": True,
            "failure_category": "runtime_oom",
            "failure_summary": "still OOM at the same shape",
            "failure_log_path": None,
        },
        previous_failures=[
            {
                "attempt": 1,
                "category": "runtime_oom",
                "summary": "OOM on first attempt",
            }
        ],
    )
    assert "Failure history in this round" in ctx
    assert "category `runtime_oom` repeated across attempts" in ctx


def test_failure_ledger_accumulates_across_attempts(tmp_path: Path):
    params = _make_params_gapG(tmp_path)
    decision = _rollback_decision()
    val_result = {
        "correctness_ok": False,
        "build_ok": True,
        "failure_category": "snr_fail",
        "failure_summary": "SNR 18 dB < 30 dB",
        "failure_log_path": "rounds/round-2/artifacts/run.log",
    }
    campaign_mod._append_failure_ledger(
        params, round_n=2, attempt=1, val_result=val_result, decision=decision
    )
    val_result2 = dict(val_result, failure_summary="SNR still 19 dB")
    campaign_mod._append_failure_ledger(
        params, round_n=2, attempt=2, val_result=val_result2, decision=decision
    )

    ledger_path = params.state_dir / "failures" / "round2.json"
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["attempt"] == 1
    assert data[1]["attempt"] == 2
    assert data[1]["summary"] == "SNR still 19 dB"


def test_prepare_env_prompt_ignores_submodules():
    from turbo_optimize.skills import load_prompt_template

    template = load_prompt_template("prepare_environment")
    assert "--ignore-submodules=all" in template, (
        "PREPARE_ENVIRONMENT gate must ignore submodule drift when "
        "computing workspace_clean; checking for the exact CLI flag"
    )
    assert "submodule_state_ignored" in template, (
        "schema must advertise the relaxation so reviewers can audit"
    )


def test_enforce_base_branch_gate_passes_when_only_submodules_changed():
    """The Python gate should NOT fail when the prompt reports
    workspace_clean=true (having ignored submodule drift)."""
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
        git_commit=True,
    )
    prepare_result = {
        "base_branch_confirmed": True,
        "base_branch_expected": "main",
        "base_branch_observed": "main",
        "workspace_clean": True,
        "submodule_state_ignored": True,
        "submodule_state": " M 3rdparty/composable_kernel",
    }
    campaign_mod._enforce_base_branch_gate(params, prepare_result)


def test_enforce_base_branch_gate_still_blocks_true_dirty_tree():
    """When the prompt reports workspace_clean=false (parent-repo
    untracked/modified files remain after the submodule-ignore pass),
    the gate must still raise."""
    from turbo_optimize.manifest import ManifestError

    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
        git_commit=True,
    )
    prepare_result = {
        "base_branch_confirmed": True,
        "base_branch_expected": "main",
        "base_branch_observed": "main",
        "workspace_clean": False,
        "submodule_state_ignored": True,
    }
    with pytest.raises(ManifestError, match="workspace_clean=false"):
        campaign_mod._enforce_base_branch_gate(params, prepare_result)
