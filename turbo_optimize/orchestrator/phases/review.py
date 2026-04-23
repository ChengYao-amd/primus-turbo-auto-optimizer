"""REVIEW phase runner (tolerant mode).

REVIEW sits between ``VALIDATE (full)`` and ``DECIDE`` and only runs
when the numeric decision computed from ``VALIDATE (full)`` is
ACCEPT-flavoured. Its job is to sanity-check that the measured gain
is caused by the hypothesis the agent proposed, not by measurement
noise or an unrelated codepath. In tolerant mode only three hard
rules (hypothesis-metric alignment, off-target gain,
correctness-bit-identity) can force a downgrade; the two soft rules
are surfaced for the human reader but do not on their own reject a
round.

The phase is deliberately thin: Python pre-computes all five signals
in :func:`turbo_optimize.scoring.compute_review_signals`, and this
runner hands them to Claude plus the structured inputs. Claude
re-derives the per-rule verdicts from the same inputs and returns a
structured JSON.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from turbo_optimize.config import CampaignParams
from turbo_optimize.manifest import read_manifest
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.scoring import (
    REVIEW_VERDICTS,
    ReviewBundle,
    compute_review_signals,
)
from turbo_optimize.skills import render_prompt
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "REVIEW"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Glob",
    "Grep",
]


def _coerce_result(
    result: dict[str, Any] | None,
    *,
    round_n: int,
    fallback: ReviewBundle,
) -> dict[str, Any]:
    """Normalise the LLM's REVIEW JSON against the fallback Python bundle.

    Missing / malformed output is coerced into the Python tolerant
    verdict with a ``notes`` trail so the orchestrator always has a
    well-formed ``review_verdict`` to act on — the ``DECIDE`` step
    never observes a ``None``.
    """
    if not isinstance(result, dict) or not result:
        log.error(
            "REVIEW round-%d: phase_result empty or not a dict; falling "
            "back to Python tolerant verdict %s",
            round_n,
            fallback.tolerant_verdict,
        )
        return _fallback_payload(round_n, fallback, reason="empty or non-dict JSON")

    verdict = result.get("review_verdict")
    if verdict not in REVIEW_VERDICTS:
        log.error(
            "REVIEW round-%d: verdict %r not in %s; falling back to %s",
            round_n,
            verdict,
            REVIEW_VERDICTS,
            fallback.tolerant_verdict,
        )
        return _fallback_payload(
            round_n,
            fallback,
            reason=f"verdict {verdict!r} not recognised",
        )

    result.setdefault("round", round_n)
    result.setdefault("review_mode", "tolerant")
    result.setdefault("review_reason", fallback.tolerant_reason)
    result.setdefault("python_signals", fallback.to_dict())
    return result


def _fallback_payload(
    round_n: int, fallback: ReviewBundle, *, reason: str
) -> dict[str, Any]:
    return {
        "round": round_n,
        "review_mode": "tolerant",
        "review_verdict": fallback.tolerant_verdict,
        "review_reason": (
            f"orchestrator fallback: {reason}; using Python signals "
            f"({fallback.tolerant_reason})"
        ),
        "rule_verdicts": {
            s.name: {
                "verdict": "block" if s.severity == "block" else (
                    "warn" if s.severity == "warn" else "pass"
                ),
                "note": s.note,
            }
            for s in fallback.signals
        },
        "python_signals": fallback.to_dict(),
        "notes": (
            "REVIEW phase produced no valid structured output; the "
            "orchestrator substituted the Python-computed tolerant "
            "verdict verbatim. Audit the phase transcript before "
            "trusting this round."
        ),
    }


async def run(
    params: CampaignParams,
    *,
    round_n: int,
    hypothesis: dict[str, Any],
    opt_result: dict[str, Any],
    quick_val_result: dict[str, Any],
    full_val_result: dict[str, Any],
    decision: dict[str, Any],
    review_bundle: ReviewBundle,
) -> PhaseOutcome:
    """Run REVIEW for ``round_n``.

    ``review_bundle`` is the pre-computed Python signal pack; callers
    build it via :func:`turbo_optimize.scoring.compute_review_signals`
    and forward it here so the same structured data appears in both
    the prompt and the phase_result on disk. The caller is responsible
    for routing the returned ``review_verdict`` back into the
    orchestrator's final ACCEPT / ROLLBACK action.
    """
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    expected = phase_result_path(params.state_dir, PHASE, round_n)
    expected.parent.mkdir(parents=True, exist_ok=True)

    prompt_vars = {
        "round_n": round_n,
        "target_op": params.target_op or manifest.get("target_op", ""),
        "target_backend": params.target_backend or manifest.get("target_backend", ""),
        "primary_metric": params.primary_metric or manifest.get("primary_metric", ""),
        "campaign_dir": str(params.campaign_dir),
        "hypothesis_json": json.dumps(hypothesis, ensure_ascii=False, indent=2),
        "optimize_result_json": json.dumps(opt_result, ensure_ascii=False, indent=2),
        "validate_quick_json": json.dumps(
            quick_val_result, ensure_ascii=False, indent=2
        ),
        "validate_full_json": json.dumps(
            full_val_result, ensure_ascii=False, indent=2
        ),
        "decision_json": json.dumps(decision, ensure_ascii=False, indent=2),
        "review_signals_json": json.dumps(
            review_bundle.to_dict(), ensure_ascii=False, indent=2
        ),
        "quick_csv_path": str(quick_val_result.get("benchmark_csv") or ""),
        "full_csv_path": str(full_val_result.get("benchmark_csv") or ""),
        "phase_result_path": str(expected),
    }
    prompt = render_prompt("review", prompt_vars)

    server = build_in_process_server(params)
    outcome = await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the REVIEW operator (tolerant mode). Re-derive the "
            "five rule verdicts independently from the structured "
            "inputs, then map them to a single review_verdict. Emit "
            "the JSON file only — no chat, no benchmark reruns, no "
            "code edits."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        max_turns=20,
        round_n=round_n,
    )
    outcome.structured = _coerce_result(
        outcome.structured,
        round_n=round_n,
        fallback=review_bundle,
    )
    return outcome
