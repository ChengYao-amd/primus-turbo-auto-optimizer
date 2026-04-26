"""OPTIMIZE phase runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.manifest import read_manifest
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import (
    load_skill_section,
    render_prompt,
    render_workspace_hygiene,
)
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "OPTIMIZE"
ALLOWED_TOOLS = [
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Bash(cp:*)",
    "Bash(mkdir:*)",
    "Bash(python:*)",
    "Bash(make:*)",
    "Bash(cmake:*)",
    "Bash(pip:*)",
]


async def run(
    params: CampaignParams,
    *,
    round_n: int,
    hypothesis: dict,
    rebuild_required: bool,
    retry_context: str | None = None,
) -> PhaseOutcome:
    """Run one OPTIMIZE phase.

    ``retry_context`` is injected by the orchestrator when the previous
    attempt's code change either failed to build or failed correctness
    checks. It contains a short markdown block (attempt index, failure
    reason, pointers to build/benchmark logs) and is rendered into the
    prompt so Claude can keep the hypothesis but patch the implementation.
    When set, the run_phase cache is bypassed so Claude is always re-
    invoked even though a stale phase_result JSON exists from the
    previous attempt.
    """
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    skill = load_skill_section(params.skills_root, PHASE)
    rules = load_skill_section(params.skills_root, "ITERATION_RULES")
    expected = phase_result_path(params.state_dir, PHASE, round_n)
    expected.parent.mkdir(parents=True, exist_ok=True)

    retry_block = ""
    if retry_context:
        retry_block = (
            "\n<retry_context>\n"
            f"{retry_context.strip()}\n"
            "</retry_context>\n\n"
            "The hypothesis above is UNCHANGED. Do not propose a new "
            "direction; fix the concrete implementation bug described "
            "in `<retry_context>` and re-emit the structured OPTIMIZE "
            "result.\n"
        )

    prompt = render_prompt(
        "optimize",
        {
            "skill_excerpt": skill,
            "rules_excerpt": rules,
            "hypothesis_json": json.dumps(hypothesis, ensure_ascii=False, indent=2),
            "kernel_source": params.kernel_source or manifest.get("kernel_source", ""),
            "campaign_dir": str(params.campaign_dir),
            "round_n": round_n,
            "rebuild_required": "true" if rebuild_required else "false",
            "phase_result_path": str(expected),
            "retry_context_block": retry_block,
            "workspace_hygiene_block": render_workspace_hygiene(
                params.workspace_root, params.campaign_dir
            ),
        },
    )
    return await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the OPTIMIZE operator. Implement the single primary "
            "hypothesis. No tests/benchmarks here. No git commits. No chat."
        ),
        allowed_tools=ALLOWED_TOOLS,
        expected_output=expected,
        force=bool(retry_context),
        round_n=round_n,
    )
