"""REPORT phase runner.

Besides the final-report write-up, this phase is the single point in the
campaign where historical tips get distilled and appended to the
workspace-level knowledge base at
``<workspace_root>/agent/historical_experience/<gpu>/<op>/<backend>/tips.md``.

Centralising tip writes here (rather than in VALIDATE per-round) lets
Claude compare the full accepted/rolled-back set before choosing which
lessons are worth persisting, so tips stay short, cross-op reusable, and
avoid streak-of-the-moment noise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import extract_history
from turbo_optimize.manifest import read_manifest
from turbo_optimize.mcp import build_in_process_server, mcp_allowed_tools
from turbo_optimize.orchestrator.phases.read_historical_tips import tips_path
from turbo_optimize.orchestrator.run_phase import PhaseOutcome, run_phase
from turbo_optimize.skills import load_skill_section, render_prompt
from turbo_optimize.state import phase_result_path


log = logging.getLogger(__name__)


PHASE = "REPORT"
ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Glob",
]


async def run(
    params: CampaignParams,
    *,
    termination: dict,
) -> PhaseOutcome:
    assert params.campaign_dir is not None
    skill = load_skill_section(params.skills_root, PHASE)
    expected = phase_result_path(params.state_dir, PHASE)
    expected.parent.mkdir(parents=True, exist_ok=True)
    history = extract_history(params.campaign_dir)

    manifest: dict = {}
    try:
        manifest = read_manifest(params.campaign_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("REPORT could not read manifest.yaml: %s", exc)

    prompt = render_prompt(
        "report",
        {
            "skill_excerpt": skill,
            "history_json": json.dumps(history.to_prompt_dict(), ensure_ascii=False, indent=2),
            "termination_json": json.dumps(termination, ensure_ascii=False, indent=2),
            "campaign_dir": str(params.campaign_dir),
            "phase_result_path": str(expected),
            "target_op": params.target_op or manifest.get("target_op", "") or "",
            "target_backend": params.target_backend
            or manifest.get("target_backend", "")
            or "",
            "target_gpu": params.target_gpu or manifest.get("target_gpu", "") or "",
            "tips_path": str(tips_path(params)),
        },
    )
    server = build_in_process_server(params)
    return await run_phase(
        PHASE,
        campaign_dir=params.campaign_dir,
        params=params,
        prompt=prompt,
        system_prompt=(
            "You are the REPORT operator. Write the Final Report section "
            "into logs/optimize.md, distil at most a handful of high-quality "
            "tips via `mcp__turbo__append_tip` (cross-op reusable, no shape-"
            "specific magic numbers), and emit the structured JSON summary. "
            "No other edits. No chat."
        ),
        allowed_tools=ALLOWED_TOOLS + mcp_allowed_tools(),
        mcp_servers={"turbo": server},
        expected_output=expected,
        max_turns=50,
    )
