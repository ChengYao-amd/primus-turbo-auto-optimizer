"""PREPARE_ENVIRONMENT phase runner."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import init_optimize_log, init_performance_trend
from turbo_optimize.manifest import read_manifest
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import (
    load_skill_section,
    render_prompt,
    render_workspace_hygiene,
)
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "PREPARE_ENVIRONMENT"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash(git:*)",
    "Bash(mkdir:*)",
    "Bash(cp:*)",
    "Bash(ls:*)",
    "Glob",
    "Grep",
]


def bootstrap_logs(params: CampaignParams) -> None:
    """Create append-only log files before the phase runs."""
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    init_optimize_log(params.campaign_dir, manifest)
    init_performance_trend(params.campaign_dir)


async def run(params: CampaignParams) -> PhaseOutcome:
    assert params.campaign_dir is not None
    bootstrap_logs(params)
    manifest = read_manifest(params.campaign_dir)
    manifest_yaml = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)

    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(params.state_dir, PHASE)
    expected.parent.mkdir(parents=True, exist_ok=True)

    prompt = render_prompt(
        "prepare_environment",
        {
            "skill_excerpt": skill,
            "manifest_yaml": manifest_yaml,
            "campaign_dir": str(params.campaign_dir),
            "campaign_id": params.campaign_id or "",
            "phase_result_path": str(expected),
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
            "You are the PREPARE_ENVIRONMENT operator. Scaffold the "
            "campaign directory and snapshot the starting kernel. "
            "No code modifications; no chat."
        ),
        allowed_tools=ALLOWED_TOOLS,
        expected_output=expected,
        max_turns=40,
    )
