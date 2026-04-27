"""READ_HISTORICAL_TIPS phase runner."""

from __future__ import annotations

from pathlib import Path

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.manifest import read_manifest
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import load_skill_section, render_prompt
from turbo_optimize.state import phase_result_path


PHASE = "READ_HISTORICAL_TIPS"
ALLOWED_TOOLS = [
    "Read",
    "Glob",
]


def tips_path(params: CampaignParams) -> Path:
    """Compute the absolute ``tips.md`` path for the campaign.

    Reuses :func:`turbo_optimize.mcp.tips._tips_path` so the prompt-side
    string and the MCP-side write/read paths can never drift. The root
    is resolved via :meth:`CampaignParams.resolved_tips_root`, which
    honours the ``TURBO_TIPS_ROOT`` environment variable and falls back
    to ``<tool_repo>/agent_data/historical_experience`` (see
    :func:`turbo_optimize.config.default_tips_root` for why this lives
    outside the optimized project's working tree).
    """
    from turbo_optimize.mcp.tips import _tips_path

    op = params.target_op or ""
    backend = params.target_backend or ""
    gpu = params.target_gpu or ""
    return _tips_path(params.resolved_tips_root(), gpu, op, backend)


async def run(params: CampaignParams) -> PhaseOutcome:
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    manifest_yaml = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(params.state_dir, PHASE)
    expected.parent.mkdir(parents=True, exist_ok=True)

    prompt = render_prompt(
        "read_historical_tips",
        {
            "skill_excerpt": skill,
            "manifest_yaml": manifest_yaml,
            "tips_path": str(tips_path(params)),
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
            "You are the READ_HISTORICAL_TIPS operator. Read the tips "
            "file via MCP query_tips, summarise applicable entries, "
            "write the JSON result. No chat."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
    )
