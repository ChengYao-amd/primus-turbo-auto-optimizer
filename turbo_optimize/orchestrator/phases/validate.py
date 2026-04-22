"""VALIDATE phase runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


async def run(
    params: CampaignParams,
    *,
    round_n: int,
    validation_level: str = "quick",
) -> PhaseOutcome:
    """Run one VALIDATE phase.

    ``validation_level`` is reflected in ``expected_output``
    (``validate_round<N>_quick.json`` vs ``..._full.json``). The
    orchestrator's full-validation gate depends on that separation: after
    a quick ACCEPT it invokes this phase again with ``level=full`` and
    must *not* hit run_phase's cache-reuse shortcut on the quick JSON
    that was just produced.
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
    return await run_phase(
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
        max_turns=70,
        round_n=round_n,
        phase_variant=validation_level,
    )
