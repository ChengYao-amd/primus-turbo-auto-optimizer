"""Build the multi-campaign overview payload + HTML.

The overview's data shape is intentionally narrow: KPI bar + table
rows. Each campaign also gets a sub-page at ``c/<id>/index.html``
written by :func:`write_overview` so the table rows can link in to
the full single-campaign dashboard.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from turbo_view import __version__
from turbo_view.discover import CampaignHandle, discover_campaigns
from turbo_view.io.loader import load_campaign
from turbo_view.model import CampaignBundle
from turbo_view.render.build import (
    _copy_assets,
    _payload_json_for_inline,
    _template_env,
    write_detail,
)
from turbo_view.render.payload import bundle_to_payload

log = logging.getLogger(__name__)

OVERVIEW_SCHEMA_VERSION = "1"


def _baseline_pct(bundle: CampaignBundle) -> float | None:
    """Best ACCEPTED step-geomean Δ% relative to BASELINE."""
    state = bundle.state
    if state is None or not state.history:
        return None
    base = next((h for h in state.history if h.decision == "BASELINE"), None)
    if base is None:
        return None
    base_score = float(base.score.get("step_geomean", 0.0))
    if not base_score:
        return None
    best = base_score
    for h in state.history:
        if h.decision in ("ACCEPTED", "ACCEPTED (noise-bounded)"):
            score = float(h.score.get("step_geomean", best))
            if score > best:
                best = score
    return ((best - base_score) / base_score) * 100.0


def _campaign_row(handle: CampaignHandle, bundle: CampaignBundle) -> dict[str, Any]:
    state = bundle.state
    params = state.params if state is not None else {}
    last_cost = bundle.cost[-1].cumulative_usd if bundle.cost else 0.0
    return {
        "campaign_id": handle.campaign_id,
        "op": params.get("target_op") if isinstance(params, dict) else None,
        "backend": params.get("backend") if isinstance(params, dict) else None,
        "gpu": params.get("gpu") if isinstance(params, dict) else None,
        "status": (state.current_phase if state is not None else "UNKNOWN"),
        "current_round": state.current_round if state is not None else None,
        "best_round": state.best_round if state is not None else None,
        "best_delta_pct": _baseline_pct(bundle),
        "cost_usd": last_cost,
        "started_at": state.started_at if state is not None else None,
        "last_update": state.last_update if state is not None else None,
        "round_count": len(bundle.rounds),
        "href": f"c/{handle.campaign_id}/index.html",
    }


def _kpi(rows: list[dict[str, Any]], wall_total_s: float) -> dict[str, Any]:
    active = sum(1 for r in rows if r["status"] != "DONE")
    cost_total = sum(r["cost_usd"] or 0.0 for r in rows)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    decisions_24h = 0
    for r in rows:
        last = r.get("last_update")
        if not last:
            continue
        try:
            ts = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            decisions_24h += 1
    return {
        "active_campaigns": active,
        "total_cost_usd": cost_total,
        "total_wall_hours": wall_total_s / 3600.0,
        "campaigns_active_24h": decisions_24h,
    }


def build_overview_payload(handles: Iterable[CampaignHandle]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    wall_total = 0.0
    for h in handles:
        bundle = load_campaign(h.campaign_dir)
        rows.append(_campaign_row(h, bundle))
        wall_total += sum(r.wall_s for r in bundle.cost)
    rows.sort(key=lambda r: r["campaign_id"])
    return {
        "schema_version": OVERVIEW_SCHEMA_VERSION,
        "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaigns": rows,
        "kpi": _kpi(rows, wall_total),
    }


def render_overview(payload: dict[str, Any], watch_mode: bool = False) -> str:
    env = _template_env()
    template = env.get_template("overview.html")
    return template.render(
        view_version=__version__,
        rendered_at=payload["rendered_at"],
        schema_version=payload["schema_version"],
        campaign_count=len(payload["campaigns"]),
        payload_json=_payload_json_for_inline(payload),
        watch_mode=watch_mode,
    )


def write_overview(root: Path, out_dir: Path, watch_mode: bool = False) -> Path:
    handles = discover_campaigns(root)
    if not handles:
        raise RuntimeError(f"no campaigns discovered under {root}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_overview_payload(handles)
    html = render_overview(payload, watch_mode=watch_mode)
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    from turbo_view.render.build import _sanitize
    (out_dir / "data.json").write_text(
        json.dumps(_sanitize(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _copy_assets(out_dir / "assets")

    for h in handles:
        sub_dir = out_dir / "c" / h.campaign_id
        bundle = load_campaign(h.campaign_dir)
        write_detail(
            bundle, sub_dir,
            copy_assets=False,
            asset_prefix="../../assets/",
            watch_mode=watch_mode,
        )

    return index.resolve()
