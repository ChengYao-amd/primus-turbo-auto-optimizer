"""DEFINE_TARGET phase runner."""

from __future__ import annotations

import logging
from pathlib import Path

from turbo_optimize.config import CampaignParams, make_campaign_id
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import load_skill_section, render_prompt
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "DEFINE_TARGET"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Glob",
    "Grep",
]


def _yaml_value(value: object) -> str:
    """Render a CLI override so the prompt can embed it as-is.

    ``None`` becomes the YAML / JSON literal ``null`` (what the caller
    ultimately writes into ``manifest.yaml``); everything else stays as its
    string form.
    """
    if value is None:
        return "null"
    return str(value)


def ensure_campaign_dir(params: CampaignParams) -> Path:
    if params.campaign_id is None:
        params.campaign_id = make_campaign_id(params.prompt or "")
    campaign_root = params.workspace_root / "agent" / "workspace"
    campaign_dir = campaign_root / params.campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "logs").mkdir(exist_ok=True)
    (campaign_dir / "profiles").mkdir(exist_ok=True)
    (campaign_dir / "rounds").mkdir(exist_ok=True)
    params.campaign_dir = campaign_dir
    return campaign_dir


async def run(params: CampaignParams) -> PhaseOutcome:
    assert params.campaign_dir is not None, "campaign_dir must be set before DEFINE_TARGET"
    assert params.campaign_id is not None
    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(params.state_dir, PHASE)
    expected.parent.mkdir(parents=True, exist_ok=True)

    project_skill_path = params.skills_root / "skills" / params.project_skill
    manifest_path = params.campaign_dir / "manifest.yaml"

    prompt = render_prompt(
        "define_target",
        {
            "skill_excerpt": skill,
            "user_prompt": params.prompt or "<no prompt provided>",
            "project_skill_path": str(project_skill_path),
            "project_skill": params.project_skill,
            "campaign_dir": str(params.campaign_dir),
            "phase_result_path": str(expected),
            "manifest_path": str(manifest_path),
            "cli_max_iterations": _yaml_value(params.max_iterations),
            "cli_max_duration": _yaml_value(params.max_duration),
            "cli_base_branch": _yaml_value(params.base_branch),
        },
    )
    return await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the DEFINE_TARGET operator in the Primus-Turbo "
            "kernel-optimize loop. Emit structured files only; no chat."
        ),
        allowed_tools=ALLOWED_TOOLS,
        expected_output=expected,
    )
