"""History tools exposed via the in-process MCP server.

These helpers read the same markdown files that `logs.extract_history`
parses. The tools intentionally mirror that function so Claude and the
Python orchestrator see a consistent view of the campaign state.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from turbo_optimize.logs import (
    extract_history,
    optimize_log_path,
    performance_trend_path,
)


if TYPE_CHECKING:
    from turbo_optimize.mcp import CampaignContext


def list_ineffective_directions_impl(ctx: "CampaignContext") -> dict[str, Any]:
    history = extract_history(ctx.campaign_dir)
    return {
        "campaign_dir": str(ctx.campaign_dir),
        "source": str(optimize_log_path(ctx.campaign_dir)),
        "items": [d.__dict__ for d in history.verified_ineffective],
    }


def query_trend_impl(ctx: "CampaignContext", limit: int) -> dict[str, Any]:
    history = extract_history(ctx.campaign_dir)
    rows = history.history_rows[-max(1, limit):] if history.history_rows else []
    return {
        "campaign_dir": str(ctx.campaign_dir),
        "source": str(performance_trend_path(ctx.campaign_dir)),
        "rows": [row.__dict__ for row in rows],
        "rollback_streak": history.rollback_streak,
        "current_best_round": history.current_best_round,
        "current_best_score": history.current_best_score,
    }


def read_best_summary_impl(ctx: "CampaignContext") -> dict[str, Any]:
    history = extract_history(ctx.campaign_dir)
    round_n = history.current_best_round
    if round_n is None:
        return {
            "campaign_dir": str(ctx.campaign_dir),
            "round": None,
            "body": "",
            "note": "no accepted round yet; nothing to return",
        }
    summary = ctx.campaign_dir / "rounds" / f"round-{round_n}" / "summary.md"
    body = ""
    if summary.exists():
        body = summary.read_text(encoding="utf-8")
    return {
        "campaign_dir": str(ctx.campaign_dir),
        "round": round_n,
        "source": str(summary),
        "body": body,
    }
