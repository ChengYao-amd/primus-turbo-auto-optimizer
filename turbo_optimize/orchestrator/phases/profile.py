"""PROFILE phase runner.

Profiles the current kernel with ``rocprofv3`` and / or
``rocprof-compute`` at three trigger points:

* ``post_baseline`` — right after BASELINE completes, so the very first
  ANALYZE has a counter-level view to reason about.
* ``post_accept``   — after every ACCEPTED round, providing a refreshed
  profile against which the next ANALYZE can diff.
* ``pre_stagnation`` — right before STAGNATION_REVIEW so the rollback
  streak review has up-to-date counters for its new directions.

The phase is deliberately tolerant: when ``profile_command`` is empty
or both ``rocprof*`` tools are missing, the prompt emits
``skipped=true`` and the orchestrator only logs a warning instead of
aborting the campaign.
"""

from __future__ import annotations

import logging
from typing import Literal

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


PHASE = "PROFILE"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Bash",
    "Glob",
    "Grep",
]


Trigger = Literal["post_baseline", "post_accept", "pre_stagnation"]


async def run(
    params: CampaignParams,
    *,
    round_n: int,
    trigger: Trigger,
    force: bool = False,
) -> PhaseOutcome:
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    profile_command = params.profile_command or manifest.get("profile_command", "")

    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(
        params.state_dir, PHASE, round_n, suffix=trigger
    )
    expected.parent.mkdir(parents=True, exist_ok=True)

    representative = _representative_shape_hint(params, manifest)
    prompt = render_prompt(
        "profile",
        {
            "skill_excerpt": skill,
            "campaign_dir": str(params.campaign_dir),
            "round_n": round_n,
            "trigger": trigger,
            "target_op": params.target_op or manifest.get("target_op", ""),
            "target_backend": params.target_backend or manifest.get("target_backend", ""),
            "target_gpu": params.target_gpu or manifest.get("target_gpu", ""),
            "profile_command": profile_command,
            "representative_shape_hint": representative,
            "phase_result_path": str(expected),
            "workspace_hygiene_block": render_workspace_hygiene(
                params.workspace_root, params.campaign_dir
            ),
        },
    )

    outcome = await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the PROFILE operator. Capture rocprof counters "
            "for one representative shape. No kernel edits. No chat."
        ),
        allowed_tools=ALLOWED_TOOLS,
        expected_output=expected,
        max_turns=45,
        force=force,
        round_n=round_n,
        phase_variant=trigger,
    )
    return outcome


def _representative_shape_hint(params: CampaignParams, manifest: dict) -> str:
    """Best-effort one-line hint for the prompt.

    Whatever the manifest exposes (``representative_shapes`` first,
    falling back to ``target_shapes``) is rendered verbatim; the
    prompt itself is in charge of picking ONE entry, so the hint only
    has to be unambiguous enough for Claude to not hallucinate shapes.
    """
    rep = (
        params.representative_shapes
        or manifest.get("representative_shapes")
        or manifest.get("target_shapes")
        or ""
    )
    if rep:
        return str(rep)
    return (
        "(manifest has no shape; use the first shape from BASELINE's "
        "selection in the rounds/round-1 summary)"
    )
