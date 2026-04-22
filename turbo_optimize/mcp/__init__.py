"""In-process SDK MCP server that exposes campaign + tips + verification tools.

All tools return the standard MCP content payload:
    {"content": [{"type": "text", "text": <json>}]}

On exception they return:
    {"content": [{"type": "text", "text": <message>}], "is_error": True}

The server is built once per phase via `build_in_process_server(params)` and
mounted on `ClaudeAgentOptions.mcp_servers={"turbo": server}`. Tools bind a
`CampaignContext` closure, so they always resolve paths relative to the
current campaign even when the Claude side calls them with a bare relative
path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from turbo_optimize.config import CampaignParams
from turbo_optimize.mcp.history import (
    read_best_summary_impl,
    list_ineffective_directions_impl,
    query_trend_impl,
)
from turbo_optimize.mcp.tips import (
    append_tip_impl,
    query_tips_impl,
)
from turbo_optimize.mcp.verification import (
    parse_bench_csv_impl,
    run_quick_validation_impl,
)


log = logging.getLogger(__name__)


MCP_SERVER_NAME = "turbo"


@dataclass
class CampaignContext:
    campaign_dir: Path
    workspace_root: Path
    skills_root: Path
    target_op: str | None
    target_backend: str | None
    target_gpu: str | None
    primary_metric: str | None
    quick_command: str | None


def _context_from_params(params: CampaignParams) -> CampaignContext:
    if params.campaign_dir is None:
        raise ValueError("CampaignParams.campaign_dir must be set before building MCP")
    return CampaignContext(
        campaign_dir=params.campaign_dir,
        workspace_root=params.workspace_root,
        skills_root=params.skills_root,
        target_op=params.target_op,
        target_backend=params.target_backend,
        target_gpu=params.target_gpu,
        primary_metric=params.primary_metric,
        quick_command=params.quick_command,
    )


def build_in_process_server(params: CampaignParams):
    """Return a configured SDK MCP server for the current campaign."""
    ctx = _context_from_params(params)

    @tool(
        "list_ineffective_directions",
        "List verified ineffective optimization directions recorded in the "
        "campaign's logs/optimize.md 'Verified Ineffective Directions' table.",
        {},
    )
    async def _list_ineffective(args: dict) -> dict:
        return _safe(list_ineffective_directions_impl, ctx)

    @tool(
        "query_trend",
        "Return the most recent rows of logs/performance_trend.md as structured "
        "data. Optional 'limit' argument caps the number of rows (default 10).",
        {"limit": int},
    )
    async def _query_trend(args: dict) -> dict:
        limit = int(args.get("limit") or 10)
        return _safe(query_trend_impl, ctx, limit)

    @tool(
        "read_best_summary",
        "Return the summary.md body of the current best accepted round, "
        "identified from logs/optimize.md.",
        {},
    )
    async def _read_best(args: dict) -> dict:
        return _safe(read_best_summary_impl, ctx)

    @tool(
        "query_tips",
        "Search agent/historical_experience/<gpu>/<op>/<backend>/tips.md for "
        "reusable lessons. Optional keyword filters entries by substring match. "
        "Any of op/backend/gpu left empty falls back to the current campaign's values.",
        {"op": str, "backend": str, "gpu": str, "keyword": str},
    )
    async def _query_tips(args: dict) -> dict:
        return _safe(
            query_tips_impl,
            ctx,
            (args.get("op") or ctx.target_op),
            (args.get("backend") or ctx.target_backend),
            (args.get("gpu") or ctx.target_gpu),
            args.get("keyword"),
        )

    @tool(
        "append_tip",
        "Append a reusable tip to agent/historical_experience/<gpu>/<op>/"
        "<backend>/tips.md. Fields: op/backend/gpu (default to campaign's), "
        "round (required), status (ACCEPTED|ROLLED_BACK), context, signal, "
        "takeaway, applicability.",
        {
            "op": str,
            "backend": str,
            "gpu": str,
            "round": int,
            "status": str,
            "context": str,
            "signal": str,
            "takeaway": str,
            "applicability": str,
        },
    )
    async def _append_tip(args: dict) -> dict:
        entry = {
            "round": args.get("round"),
            "status": args.get("status", "ACCEPTED"),
            "context": args.get("context", ""),
            "signal": args.get("signal", ""),
            "takeaway": args.get("takeaway", ""),
            "applicability": args.get("applicability", ""),
        }
        return _safe(
            append_tip_impl,
            ctx,
            args.get("op") or ctx.target_op,
            args.get("backend") or ctx.target_backend,
            args.get("gpu") or ctx.target_gpu,
            entry,
        )

    @tool(
        "run_quick_validation",
        "Execute the manifest.quick_command for the current campaign. "
        "Returns a structured {returncode, stdout_tail, stderr_tail, duration_s}.",
        {"timeout_s": int},
    )
    async def _quick_validate(args: dict) -> dict:
        timeout = int(args.get("timeout_s") or 1800)
        return _safe(run_quick_validation_impl, ctx, timeout)

    @tool(
        "parse_bench_csv",
        "Parse a benchmark CSV into {rows: [{shape, check, metrics}], "
        "aggregate: {metric: geomean}}. Provide csv_path (relative to "
        "campaign_dir) and optional primary_metric override.",
        {"csv_path": str, "primary_metric": str},
    )
    async def _parse_csv(args: dict) -> dict:
        return _safe(
            parse_bench_csv_impl,
            ctx,
            args.get("csv_path", ""),
            args.get("primary_metric") or (ctx.primary_metric or ""),
        )

    server = create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="0.1.0",
        tools=[
            _list_ineffective,
            _query_trend,
            _read_best,
            _query_tips,
            _append_tip,
            _quick_validate,
            _parse_csv,
        ],
    )
    return server


def mcp_allowed_tools() -> list[str]:
    """Convenience list of `mcp__turbo__<name>` identifiers for allow-lists."""
    names = [
        "list_ineffective_directions",
        "query_trend",
        "read_best_summary",
        "query_tips",
        "append_tip",
        "run_quick_validation",
        "parse_bench_csv",
    ]
    return [f"mcp__{MCP_SERVER_NAME}__{n}" for n in names]


def _safe(fn, *args, **kwargs) -> dict:
    import json

    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.exception("MCP tool %s failed", getattr(fn, "__name__", "<fn>"))
        return {
            "content": [{"type": "text", "text": f"error: {exc!s}"}],
            "is_error": True,
        }
    text = json.dumps(result, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}]}
