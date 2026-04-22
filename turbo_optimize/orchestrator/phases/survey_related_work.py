"""SURVEY_RELATED_WORK phase runner."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.manifest import read_manifest
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import load_skill_section, render_prompt
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "SURVEY_RELATED_WORK"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Bash(git:clone)",
    "Bash(git:log)",
]


async def run(params: CampaignParams) -> PhaseOutcome:
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    manifest_yaml = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)

    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(params.state_dir, PHASE)
    expected.parent.mkdir(parents=True, exist_ok=True)
    related_work_path = params.campaign_dir / "related_work.md"
    template_path = (
        params.skills_root
        / "skills"
        / "kernel-optimize"
        / "related-work-template.md"
    )

    prompt = render_prompt(
        "survey_related_work",
        {
            "skill_excerpt": skill,
            "manifest_yaml": manifest_yaml,
            "campaign_dir": str(params.campaign_dir),
            "related_work_path": str(related_work_path),
            "template_path": str(template_path),
            "phase_result_path": str(expected),
            "campaign_id": params.campaign_id or "",
            "target_op": params.target_op or manifest.get("target_op", ""),
            "target_backend": params.target_backend or manifest.get("target_backend", ""),
            "target_gpu": params.target_gpu or manifest.get("target_gpu", ""),
        },
    )

    server = build_in_process_server(params)
    return await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the SURVEY_RELATED_WORK operator. Gather external "
            "context and produce a related_work.md. No code edits; no chat."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        max_turns=60,
    )
