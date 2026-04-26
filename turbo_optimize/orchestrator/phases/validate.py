"""VALIDATE phase runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import extract_history
from turbo_optimize.manifest import read_manifest
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import (
    load_skill_section,
    render_prompt,
    render_workspace_hygiene,
)
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "VALIDATE"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Bash",
    "Glob",
]


_KNOWN_FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "build_compile",
        "build_link",
        "runtime_assert",
        "runtime_oom",
        "runtime_hang",
        "snr_fail",
        "bench_regression",
        "schema_invalid",
        "other",
    }
)


def _coerce_structured_result(
    result: dict[str, Any] | None, *, round_n: int, level: str
) -> dict[str, Any]:
    """Ensure VALIDATE's structured JSON has the fields the scorer needs.

    The orchestrator used to crash when a round wrote an empty or
    half-populated phase_result (round-10 in the reference campaign).
    We now normalise bad output into a well-formed ROLLBACK-looking
    payload with ``failure_category="schema_invalid"`` so the standard
    decision path can classify it instead of ``_apply_decision`` blowing
    up on ``None`` scores.

    The checks here are deliberately minimal — any deeper contract
    (primary-metric presence, per-row shape/Check fields) still lives
    in :mod:`turbo_optimize.scoring`.
    """
    if not isinstance(result, dict) or not result:
        log.error(
            "VALIDATE round-%d %s: phase_result empty or not a dict; "
            "downgrading to schema_invalid",
            round_n,
            level,
        )
        return _schema_invalid_payload(round_n, level, "empty or non-dict JSON")

    complaints: list[str] = []
    score_vector = result.get("score_vector")
    if not isinstance(score_vector, list) or len(score_vector) == 0:
        complaints.append("score_vector empty or not a list")
    aggregate = result.get("aggregate_score")
    if not isinstance(aggregate, dict) or not aggregate:
        complaints.append("aggregate_score empty or not a dict")
    if not isinstance(result.get("correctness_ok"), bool):
        complaints.append("correctness_ok missing or non-bool")
    if not isinstance(result.get("build_ok"), bool):
        result["build_ok"] = True

    failure_category = result.get("failure_category")
    if failure_category is not None and failure_category not in _KNOWN_FAILURE_CATEGORIES:
        log.warning(
            "VALIDATE round-%d: unknown failure_category=%r; coercing to 'other'",
            round_n,
            failure_category,
        )
        result["failure_category"] = "other"

    if complaints:
        log.error(
            "VALIDATE round-%d %s phase_result fails schema: %s",
            round_n,
            level,
            "; ".join(complaints),
        )
        base = _schema_invalid_payload(round_n, level, "; ".join(complaints))
        base.update({k: v for k, v in result.items() if k not in base})
        base["correctness_ok"] = False
        base["failure_category"] = "schema_invalid"
        base["failure_summary"] = (
            "VALIDATE emitted an incomplete phase_result; the orchestrator "
            "downgraded this round to ROLLBACK. Fix the VALIDATE step so it "
            "always writes score_vector + aggregate_score before the JSON "
            "is marked ready."
        )
        return base
    return result


def _schema_invalid_payload(round_n: int, level: str, reason: str) -> dict[str, Any]:
    return {
        "round": round_n,
        "validation_level": level,
        "correctness_ok": False,
        "build_ok": True,
        "failure_category": "schema_invalid",
        "failure_summary": (
            f"phase_result malformed ({reason}); treat this as a failed "
            "VALIDATE with no usable score."
        ),
        "failure_log_path": None,
        "benchmark_csv": None,
        "score_vector": [],
        "aggregate_score": {},
        "trend_row": {
            "fwd_avg": 0.0,
            "fwd_peak": 0.0,
            "bwd_avg": None,
            "bwd_peak": None,
            "step_geomean": None,
        },
        "notes": f"auto-downgraded by orchestrator: {reason}",
    }


async def run(
    params: CampaignParams,
    *,
    round_n: int,
    validation_level: str = "quick",
    force: bool = False,
) -> PhaseOutcome:
    """Run one VALIDATE phase.

    ``validation_level`` is reflected in ``expected_output``
    (``validate_round<N>_quick.json`` vs ``..._full.json``). The
    orchestrator's full-validation gate depends on that separation: after
    a quick ACCEPT it invokes this phase again with ``level=full`` and
    must *not* hit run_phase's cache-reuse shortcut on the quick JSON
    that was just produced.

    ``force=True`` skips run_phase's cache-reuse shortcut even when the
    expected JSON already exists on disk. Used by the OPTIMIZE+VALIDATE
    debug-retry loop so that each retry's rewritten kernel is actually
    re-validated instead of reading back the first attempt's failing
    result.
    """
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    skill = load_skill_section(params.skills_root, PHASE)
    scoring_excerpt = load_skill_section(params.skills_root, "SCORING")
    rules_excerpt = load_skill_section(params.skills_root, "ITERATION_RULES")
    expected = phase_result_path(
        params.state_dir, PHASE, round_n, suffix=validation_level
    )
    expected.parent.mkdir(parents=True, exist_ok=True)

    history = extract_history(params.campaign_dir)
    current_best = {
        "round": history.current_best_round,
        "score": history.current_best_score,
    }

    prompt = render_prompt(
        "validate",
        {
            "skill_excerpt": skill,
            "scoring_excerpt": scoring_excerpt,
            "rules_excerpt": rules_excerpt,
            "test_command": params.test_command or manifest.get("test_command", ""),
            "benchmark_command": params.benchmark_command
            or manifest.get("benchmark_command", ""),
            "quick_command": params.quick_command
            or manifest.get("quick_command", ""),
            "primary_metric": params.primary_metric
            or manifest.get("primary_metric", ""),
            "campaign_dir": str(params.campaign_dir),
            "round_n": round_n,
            "validation_level": validation_level,
            "current_best_json": json.dumps(current_best, ensure_ascii=False, indent=2),
            "phase_result_path": str(expected),
            "workspace_hygiene_block": render_workspace_hygiene(
                params.workspace_root, params.campaign_dir
            ),
        },
    )
    server = build_in_process_server(params)
    outcome = await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the VALIDATE operator. Run correctness + benchmark, "
            "parse the CSV via MCP, write rounds/round-N/summary.md. "
            "Leave the final ACCEPT/ROLLBACK decision to Python."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        force=force,
        round_n=round_n,
        phase_variant=validation_level,
    )
    outcome.structured = _coerce_structured_result(
        outcome.structured, round_n=round_n, level=validation_level
    )
    return outcome
