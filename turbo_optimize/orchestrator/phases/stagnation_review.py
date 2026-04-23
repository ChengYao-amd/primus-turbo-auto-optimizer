"""STAGNATION_REVIEW phase runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import extract_history
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import load_skill_section, render_prompt
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "STAGNATION_REVIEW"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Glob",
    "Grep",
    "Bash(rocprof:*)",
    "Bash(rocprofv3:*)",
    "Bash(python:*)",
]


async def run(
    params: CampaignParams,
    *,
    rollback_streak: int,
    current_round: int | None = None,
) -> PhaseOutcome:
    """Run STAGNATION_REVIEW.

    ``expected_output`` includes both ``rollback_streak`` and
    ``current_round`` in its suffix so each distinct stagnation episode
    produces its own JSON. Without the suffix, run_phase's cache-reuse
    shortcut would replay the first episode's directions on every later
    rollback_streak>=2 trigger, wasting the re-prompt entirely.
    """
    assert params.campaign_dir is not None
    skill = load_skill_section(params.skills_root, PHASE)
    suffix_parts = [f"streak{rollback_streak}"]
    if current_round is not None:
        suffix_parts.append(f"atround{current_round}")
    expected = phase_result_path(
        params.state_dir, PHASE, suffix="_".join(suffix_parts)
    )
    expected.parent.mkdir(parents=True, exist_ok=True)
    history = extract_history(params.campaign_dir)

    prompt = render_prompt(
        "stagnation_review",
        {
            "skill_excerpt": skill,
            "rollback_streak": rollback_streak,
            "campaign_dir": str(params.campaign_dir),
            "history_json": json.dumps(history.to_prompt_dict(), ensure_ascii=False, indent=2),
            "verified_ineffective_json": json.dumps(
                [d.__dict__ for d in history.verified_ineffective],
                ensure_ascii=False,
                indent=2,
            ),
            "phase_result_path": str(expected),
        },
    )
    server = build_in_process_server(params)
    return await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the STAGNATION_REVIEW operator. Propose 3+ "
            "fundamentally new directions. No code edits. No chat."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        max_turns=60,
    )
