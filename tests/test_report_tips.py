"""Tests for REPORT-phase tip distillation wiring.

The intent captured here:
* REPORT is the single write-point for the cross-campaign
  ``agent/historical_experience/.../tips.md`` knowledge base.
* The phase must expose ``mcp__turbo__*`` tools (for ``append_tip`` +
  history queries) AND the prompt must explicitly instruct Claude to
  use ``append_tip`` with the required quality bar.
* No other phase's prompt may still claim VALIDATE writes tips — that
  wording was a stale design note.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from turbo_optimize.config import CampaignParams
from turbo_optimize.mcp import mcp_allowed_tools
from turbo_optimize.orchestrator.phases import report as report_phase


PROMPT_DIR = (
    Path(__file__).resolve().parents[1] / "turbo_optimize" / "prompts"
)


def test_report_phase_mounts_mcp_turbo_server():
    """REPORT.run must call build_in_process_server and pass
    mcp_allowed_tools so Claude can actually invoke append_tip."""
    src = inspect.getsource(report_phase.run)
    assert "build_in_process_server" in src, (
        "REPORT must build the in-process MCP server; otherwise "
        "append_tip is unreachable"
    )
    assert "mcp_allowed_tools()" in src, (
        "REPORT must include mcp_allowed_tools() so the append_tip "
        "tool id is on the allow-list"
    )
    assert "mcp_servers" in src, "REPORT must pass mcp_servers to run_phase"


def test_report_phase_does_not_set_max_turns():
    """Turn caps are intentionally removed from every phase runner.

    The original motivation for this test was the opposite — REPORT
    needed a generous ``max_turns`` so tip distillation would not get
    truncated at turn 30. That problem is now handled more bluntly by
    not passing ``max_turns`` at all, which lets the Claude CLI fall
    back to its ``unlimited`` default. Cost control lives in
    ``PHASE_TIMEOUT_DEFAULTS`` (``wall_timeout_s``) instead of a
    per-phase turn counter; see ``turbo_optimize/config.py``.
    """
    src = inspect.getsource(report_phase.run)
    import re

    assert re.search(r"max_turns\s*=", src) is None, (
        "REPORT.run must NOT pass max_turns to run_phase; the campaign "
        "runner relies on the SDK's unlimited default and bounds cost "
        "via wall_timeout_s instead. If you are intentionally "
        "reintroducing a turn cap, add a justification here."
    )


def test_report_allowed_tools_include_mcp_append_tip():
    """Sanity check that the MCP allow-list produced by mcp_allowed_tools
    actually contains the append_tip identifier REPORT depends on."""
    tools = mcp_allowed_tools()
    assert "mcp__turbo__append_tip" in tools
    assert "mcp__turbo__query_tips" in tools


def test_report_prompt_requires_append_tip_call():
    """The REPORT prompt must explicitly instruct Claude to call
    mcp__turbo__append_tip (not Write) for persistence."""
    body = (PROMPT_DIR / "report.md").read_text(encoding="utf-8")
    assert "mcp__turbo__append_tip" in body, (
        "report.md must name the MCP tool so Claude picks it up"
    )
    assert "Write" in body and "do NOT use `Write` directly" in body, (
        "report.md must steer Claude away from raw Write for tips to "
        "preserve the fcntl lock semantics"
    )


def test_report_prompt_enforces_quality_bar():
    """The tip-distillation block must enumerate the four-field quality
    bar (context / signal / takeaway / applicability). Missing any of
    these would allow Claude to regress into narrative diary entries."""
    body = (PROMPT_DIR / "report.md").read_text(encoding="utf-8")
    for field in ("context", "signal", "takeaway", "applicability"):
        assert f"`{field}`" in body, (
            f"report.md quality bar must cite the `{field}` field verbatim"
        )


def test_report_prompt_separates_failure_and_success_tips():
    """The two-category design is load-bearing: failure tips cut future
    test cost; success tips seed future campaigns on related ops. The
    prompt must label both."""
    body = (PROMPT_DIR / "report.md").read_text(encoding="utf-8")
    assert "Failure tips" in body
    assert "Success tips" in body
    assert "cross-op reusable" in body or "cross-op-reusable" in body


def test_report_prompt_caps_tip_count():
    """Without an upper bound Claude will pad tips to look thorough,
    diluting the knowledge base. Prompt must cap explicitly."""
    body = (PROMPT_DIR / "report.md").read_text(encoding="utf-8")
    assert "at most 5 tips" in body.lower() or "at most 5" in body


def test_report_prompt_expected_json_schema_has_tips_appended():
    """The structured result REPORT writes must expose the
    `tips_appended` array so the orchestrator (and downstream analysis)
    can audit what actually got persisted vs. what was only drafted."""
    body = (PROMPT_DIR / "report.md").read_text(encoding="utf-8")
    assert '"tips_appended"' in body


def test_read_historical_tips_prompt_points_to_report_not_validate():
    """Regression guard: the stale design note that claimed VALIDATE
    appends tips must be gone; otherwise Claude gets conflicting
    instructions."""
    body = (PROMPT_DIR / "read_historical_tips.md").read_text(encoding="utf-8")
    assert "appended by VALIDATE" not in body, (
        "stale wording: tips are appended by REPORT, not VALIDATE"
    )
    assert "REPORT" in body, (
        "read_historical_tips.md must point at the REPORT phase as the "
        "tip write-point so Claude knows when the file gets updated"
    )


def test_report_prompt_template_placeholders_cover_campaign_scope():
    """op / backend / gpu must be injected so Claude picks the right
    tips_path; tips_path itself must be rendered in the prompt."""
    body = (PROMPT_DIR / "report.md").read_text(encoding="utf-8")
    for placeholder in (
        "{target_op}",
        "{target_backend}",
        "{target_gpu}",
        "{tips_path}",
    ):
        assert placeholder in body, (
            f"report.md must still declare the {placeholder} "
            f"substitution; render_prompt depends on it"
        )
