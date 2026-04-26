"""BASELINE phase (round-1). Runs focused test + benchmark, picks
representative shapes, seeds the round-1 summary and kernel snapshot."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.manifest import read_manifest, write_manifest
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import (
    load_skill_section,
    render_prompt,
    render_workspace_hygiene,
)
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "BASELINE"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
]


async def run(params: CampaignParams) -> PhaseOutcome:
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    manifest_yaml = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(params.state_dir, PHASE)
    expected.parent.mkdir(parents=True, exist_ok=True)

    prompt = render_prompt(
        "baseline",
        {
            "skill_excerpt": skill,
            "manifest_yaml": manifest_yaml,
            "campaign_dir": str(params.campaign_dir),
            "phase_result_path": str(expected),
            "primary_metric": params.primary_metric or manifest.get("primary_metric", ""),
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
            "You are the BASELINE operator. Run tests + benchmarks, "
            "choose representative shapes, write round-1 artifacts. "
            "No code edits to the kernel itself. No chat."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        round_n=1,
    )
    _sync_representative_shapes(params, outcome.structured)
    return outcome


def _sync_representative_shapes(
    params: CampaignParams, phase_result: dict | None
) -> None:
    if not phase_result:
        return
    shapes = phase_result.get("representative_shapes")
    if not shapes:
        return
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    if manifest.get("representative_shapes") != shapes:
        manifest["representative_shapes"] = shapes
        write_manifest(params.campaign_dir, manifest)
        log.info(
            "manifest.representative_shapes updated with %d entries", len(shapes)
        )
    params.representative_shapes = ", ".join(
        str(s) for s in shapes
    ) if isinstance(shapes, list) else str(shapes)
