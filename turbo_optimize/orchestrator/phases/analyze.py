"""ANALYZE phase runner."""

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
from turbo_optimize.skills import load_skill_section, render_prompt
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "ANALYZE"
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
    round_n: int,
    retry_hint: str | None = None,
) -> PhaseOutcome:
    assert params.campaign_dir is not None
    manifest = read_manifest(params.campaign_dir)
    skill = load_skill_section(params.skills_root, PHASE)
    rules = load_skill_section(params.skills_root, "ITERATION_RULES")
    expected = phase_result_path(params.state_dir, PHASE, round_n)
    expected.parent.mkdir(parents=True, exist_ok=True)

    history = extract_history(params.campaign_dir)
    history_dict = history.to_prompt_dict()

    prompt_vars = {
        "skill_excerpt": skill,
        "rules_excerpt": rules,
        "target_op": params.target_op or manifest.get("target_op", ""),
        "target_backend": params.target_backend or manifest.get("target_backend", ""),
        "target_gpu": params.target_gpu or manifest.get("target_gpu", ""),
        "primary_metric": params.primary_metric or manifest.get("primary_metric", ""),
        "campaign_dir": str(params.campaign_dir),
        "round_n": round_n,
        "history_json": json.dumps(history_dict, ensure_ascii=False, indent=2),
        "verified_ineffective_json": json.dumps(
            [d.__dict__ for d in history.verified_ineffective],
            ensure_ascii=False,
            indent=2,
        ),
        "directions_to_try_json": json.dumps(
            history.directions_to_try, ensure_ascii=False, indent=2
        ),
        "phase_result_path": str(expected),
    }
    prompt = render_prompt("analyze", prompt_vars)
    if retry_hint:
        prompt += f"\n\n[retry hint from orchestrator]: {retry_hint}\n"

    server = build_in_process_server(params)
    return await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the ANALYZE operator. One hypothesis per round, "
            "informed by history. Emit JSON to the phase_result_path. "
            "No code edits, no chat."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        max_turns=40,
        force=bool(retry_hint),
        round_n=round_n,
    )
