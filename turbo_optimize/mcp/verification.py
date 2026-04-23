"""Quick validation + CSV parsing tools for the MCP server."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turbo_optimize.scoring import compute_score_vector, parse_bench_csv, split_primary_metric


if TYPE_CHECKING:
    from turbo_optimize.mcp import CampaignContext


def run_quick_validation_impl(ctx: "CampaignContext", timeout_s: int) -> dict[str, Any]:
    if not ctx.quick_command:
        return {
            "ok": False,
            "note": "manifest.quick_command is empty; VALIDATE must provide it first",
        }
    cmd = ctx.quick_command
    start = time.monotonic()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(ctx.workspace_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    duration = time.monotonic() - start
    return {
        "command": cmd,
        "returncode": result.returncode,
        "duration_s": round(duration, 2),
        "stdout_tail": _tail(result.stdout, 4000),
        "stderr_tail": _tail(result.stderr, 4000),
    }


def parse_bench_csv_impl(
    ctx: "CampaignContext", csv_path: str, primary_metric: str
) -> dict[str, Any]:
    if not csv_path:
        raise ValueError("csv_path is required")
    abs_path = ctx.campaign_dir / csv_path
    if not abs_path.exists():
        alt = Path(csv_path)
        if alt.is_absolute() or (ctx.workspace_root / csv_path).exists():
            abs_path = alt if alt.is_absolute() else ctx.workspace_root / csv_path
    parse = parse_bench_csv(abs_path, primary_metric)
    score = compute_score_vector(parse)
    return {
        "path": str(abs_path),
        "primary_metric": split_primary_metric(primary_metric),
        "all_pass": parse.all_pass,
        "rows": [
            {
                "shape": row.shape,
                "check": row.check,
                "metrics": row.metrics,
                "metrics_stddev_pct": row.metrics_stddev_pct,
                "repeats": row.repeats,
            }
            for row in parse.rows
        ],
        "aggregate": score.aggregate,
    }


def _tail(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]
