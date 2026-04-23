"""Tests for the REVIEW phase integration into the campaign orchestrator.

These cover the tolerant-mode mapping between the REVIEW JSON / Python
fallback bundle and the final :class:`DecisionResult` produced by
``_run_round``. The heavy end-to-end path (``_run_review_phase`` calling
the live LLM) is not exercised here; it is covered by the smoke test
harness when the connector is mocked.

The scope is intentionally narrow:

1. :func:`_apply_review_verdict` must preserve the numeric decision on
   ``AGREE``, annotate it on ``DOWNGRADE_TO_NOISE_BOUND``, force
   ``ROLLBACK`` on ``DOWNGRADE_TO_ROLLBACK`` / ``ESCALATE_HUMAN``, and
   pass-through on unknown verdicts.
2. :func:`_build_review_bundle` and
   :func:`_parse_val_csv_to_score_vector` must tolerate missing CSVs
   without crashing — a REVIEW run that arrives with half the artefacts
   still has to produce a well-formed :class:`ReviewBundle`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator import campaign as campaign_mod
from turbo_optimize.scoring import (
    REVIEW_VERDICT_AGREE,
    REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND,
    REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK,
    REVIEW_VERDICT_ESCALATE_HUMAN,
    DecisionResult,
    ReviewBundle,
    ReviewSignal,
)


def _accept_decision() -> DecisionResult:
    return DecisionResult(
        decision="ACCEPTED",
        reason="improvement over noise band",
        improvement_pct={"Forward TFLOPS": 4.2, "Backward TFLOPS": 0.1},
        regressions=[],
        noise_check_required=False,
    )


def _signal(name: str, severity: str, passed: bool = True) -> ReviewSignal:
    return ReviewSignal(
        name=name,
        passed=passed,
        severity=severity,
        details={},
        note=f"{name}:{severity}",
    )


def _bundle(verdict: str, signals: list[ReviewSignal] | None = None) -> ReviewBundle:
    return ReviewBundle(
        signals=signals
        or [
            _signal("hypothesis_metric_alignment", "info"),
            _signal("off_target_gain", "info"),
            _signal("quick_vs_full_agreement", "info"),
            _signal("noise_band", "info"),
            _signal("correctness_bit_identity", "info"),
        ],
        tolerant_verdict=verdict,
        tolerant_reason=f"python fallback -> {verdict}",
    )


# ---------------------------------------------------------------------
# _apply_review_verdict — primary tolerant-mode behaviour
# ---------------------------------------------------------------------


def test_apply_review_verdict_agree_preserves_decision():
    decision = _accept_decision()
    merged = campaign_mod._apply_review_verdict(
        decision,
        {"review_verdict": REVIEW_VERDICT_AGREE, "review_reason": "all clean"},
        _bundle(REVIEW_VERDICT_AGREE),
    )
    assert merged == decision


def test_apply_review_verdict_noise_bound_keeps_accept_and_annotates_reason():
    decision = _accept_decision()
    merged = campaign_mod._apply_review_verdict(
        decision,
        {
            "review_verdict": REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND,
            "review_reason": "hypothesis predicted +3% forward, observed +0.4%",
        },
        _bundle(REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND),
    )
    assert merged.decision == "ACCEPTED"
    assert "DOWNGRADE_TO_NOISE_BOUND" in merged.reason
    assert "hypothesis predicted +3%" in merged.reason
    assert merged.noise_check_required is True
    assert merged.improvement_pct == decision.improvement_pct


def test_apply_review_verdict_rollback_forces_rollback():
    decision = _accept_decision()
    merged = campaign_mod._apply_review_verdict(
        decision,
        {
            "review_verdict": REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK,
            "review_reason": "off-target gain and alignment both block",
        },
        _bundle(REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK),
    )
    assert merged.decision == "ROLLBACK"
    assert "DOWNGRADE_TO_ROLLBACK" in merged.reason
    assert "off-target gain" in merged.reason
    # Numeric context is preserved so the round log still shows what
    # the measurement saw before the downgrade.
    assert merged.improvement_pct == decision.improvement_pct


def test_apply_review_verdict_escalate_human_forces_rollback_with_correctness_note(
    caplog: pytest.LogCaptureFixture,
):
    decision = _accept_decision()
    merged = campaign_mod._apply_review_verdict(
        decision,
        {
            "review_verdict": REVIEW_VERDICT_ESCALATE_HUMAN,
            "review_reason": "out_snr dropped to 42 dB but hypothesis claims bit-identity",
        },
        _bundle(REVIEW_VERDICT_ESCALATE_HUMAN),
    )
    assert merged.decision == "ROLLBACK"
    assert "ESCALATE_HUMAN" in merged.reason
    assert "correctness claim" in merged.reason


def test_apply_review_verdict_uses_fallback_when_llm_omits_verdict():
    decision = _accept_decision()
    merged = campaign_mod._apply_review_verdict(
        decision,
        {},
        _bundle(REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK),
    )
    # Empty dict -> bundle's tolerant_verdict is used as the verdict.
    assert merged.decision == "ROLLBACK"
    assert "DOWNGRADE_TO_ROLLBACK" in merged.reason


def test_apply_review_verdict_unknown_verdict_is_passthrough(
    caplog: pytest.LogCaptureFixture,
):
    decision = _accept_decision()
    merged = campaign_mod._apply_review_verdict(
        decision,
        {"review_verdict": "MAYBE", "review_reason": "unclear"},
        _bundle(REVIEW_VERDICT_AGREE),
    )
    assert merged == decision


# ---------------------------------------------------------------------
# _parse_val_csv_to_score_vector / _build_review_bundle — graceful
# degradation on missing files
# ---------------------------------------------------------------------


def _campaign_params(tmp_path: Path) -> CampaignParams:
    return CampaignParams(
        target_op="grouped_gemm_fp8",
        target_backend="triton",
        campaign_dir=tmp_path,
        state_dir=tmp_path / "state",
        primary_metric="Forward TFLOPS,Backward TFLOPS",
    )


def test_parse_val_csv_returns_none_for_missing_file(tmp_path: Path):
    params = _campaign_params(tmp_path)
    result = campaign_mod._parse_val_csv_to_score_vector(
        params, "rounds/round-2/artifacts/validate_quick.csv", "Forward TFLOPS"
    )
    assert result is None


def test_parse_val_csv_returns_none_for_empty_hint(tmp_path: Path):
    params = _campaign_params(tmp_path)
    assert (
        campaign_mod._parse_val_csv_to_score_vector(params, None, "Forward TFLOPS")
        is None
    )
    assert (
        campaign_mod._parse_val_csv_to_score_vector(params, "", "Forward TFLOPS")
        is None
    )


def test_baseline_score_vector_returns_none_when_round1_missing(tmp_path: Path):
    params = _campaign_params(tmp_path)
    # No rounds/round-1 directory at all.
    assert campaign_mod._baseline_score_vector(params, "Forward TFLOPS") is None


def test_build_review_bundle_tolerates_completely_missing_artefacts(tmp_path: Path):
    params = _campaign_params(tmp_path)
    bundle = campaign_mod._build_review_bundle(
        params,
        hypothesis={"primary_hypothesis": "remove O(G) scan"},
        opt_result={"modified_files": []},
        quick_val_result={},
        full_val_result={},
    )
    # The bundle must still be well-formed even with no candidates.
    assert isinstance(bundle, ReviewBundle)
    assert {s.name for s in bundle.signals} == {
        "hypothesis_metric_alignment",
        "off_target_gain",
        "quick_vs_full_agreement",
        "noise_band",
        "correctness_bit_identity",
    }
    # With no data the tolerant verdict falls through to AGREE (no
    # blocking evidence). The point of this assertion is to pin the
    # behaviour so future scoring changes make a conscious choice.
    assert bundle.tolerant_verdict == REVIEW_VERDICT_AGREE


# ---------------------------------------------------------------------
# Sanity: PHASE_ORDER and PHASE_TIMEOUT_DEFAULTS expose REVIEW
# ---------------------------------------------------------------------


def test_phase_order_contains_review_between_validate_and_decide():
    from turbo_optimize.state import PHASE_ORDER

    idx_validate = PHASE_ORDER.index("VALIDATE")
    idx_review = PHASE_ORDER.index("REVIEW")
    idx_decide = PHASE_ORDER.index("DECIDE")
    assert idx_validate < idx_review < idx_decide


def test_phase_timeout_defaults_contain_review():
    from turbo_optimize.config import PHASE_TIMEOUT_DEFAULTS

    assert "REVIEW" in PHASE_TIMEOUT_DEFAULTS
    cfg = PHASE_TIMEOUT_DEFAULTS["REVIEW"]
    for key in ("idle", "wall", "retries", "retriable"):
        assert key in cfg
