"""Single entry point: campaign directory -> ``CampaignBundle``.

PR-1 wired state / cost / perf / rounds / ineffective / optimize.md.
PR-2 added bench_shapes inside rounds.
PR-3 fills profiles + transcripts.
Post-PR-3: synthesize a fallback ``RunState`` when ``run.json`` is
missing — older campaigns predate the state writer but still have
cost.md / performance_trend.md, and rendering them blank made the
sticky bar useless.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from turbo_view.io.logs import (
    parse_cost_md,
    parse_optimize_md_sections,
    parse_perf_trend_md,
)
from turbo_view.io.profiles import load_profiles
from turbo_view.io.rounds import load_rounds
from turbo_view.io.state import load_run_state
from turbo_view.io.transcripts import load_transcripts
from turbo_view.model import CampaignBundle, CostRow, HistoryEntry, PerfRow, RunState

log = logging.getLogger(__name__)


def _load_ineffective(campaign_dir: Path) -> list[dict[str, Any]]:
    path = campaign_dir / "verified_ineffective.jsonl"
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("bad jsonl row in %s: %s", path, exc)
    return out


_ACCEPT_DECISIONS = frozenset({
    "BASELINE",
    "ACCEPTED",
    "ACCEPTED (noise-bounded)",
})


def _synthesize_state(
    campaign_dir: Path, cost: list[CostRow], perf: list[PerfRow],
) -> RunState | None:
    """Build a best-effort RunState when ``run.json`` is absent.

    Older campaigns predate the state writer or were copied without the
    state subdir. cost.md + performance_trend.md still carry enough to
    populate the sticky bar (campaign id, current phase, current/best
    round, started/last-update, best score) — we synthesize the rest as
    empty so the front-end's null-check ("no run.json") stops firing.
    """
    if not cost and not perf:
        return None
    last_cost = cost[-1] if cost else None
    first_cost = cost[0] if cost else None

    rounds_with_score = [r for r in perf if r.step_geomean]
    best_row: PerfRow | None = None
    for r in rounds_with_score:
        if r.status not in _ACCEPT_DECISIONS:
            continue
        if best_row is None or r.step_geomean > best_row.step_geomean:
            best_row = r
    best_score: dict[str, float] = (
        {"step_geomean": best_row.step_geomean} if best_row else {}
    )

    current_round = (
        last_cost.round if last_cost and last_cost.round is not None
        else (perf[-1].round if perf else 0)
    )
    current_phase = last_cost.phase if last_cost else "UNKNOWN"

    history = [
        HistoryEntry(
            round=p.round,
            decision=p.status,
            score={"step_geomean": p.step_geomean},
            description=p.description,
            at="",
        )
        for p in perf
    ]

    return RunState(
        campaign_id=campaign_dir.name,
        campaign_dir=campaign_dir,
        current_phase=current_phase,
        current_round=current_round,
        best_round=best_row.round if best_row else None,
        best_score=best_score,
        rollback_streak=0,
        started_at=first_cost.ts.isoformat() if first_cost else "",
        last_update=last_cost.ts.isoformat() if last_cost else "",
        params={},
        history=history,
    )


def load_campaign(campaign_dir: Path) -> CampaignBundle:
    campaign_dir = campaign_dir.resolve()
    state = load_run_state(campaign_dir)
    cost = parse_cost_md(campaign_dir / "logs" / "cost.md")
    perf = parse_perf_trend_md(campaign_dir / "logs" / "performance_trend.md")
    if state is None:
        state = _synthesize_state(campaign_dir, cost, perf)
    return CampaignBundle(
        state=state,
        cost=cost,
        perf=perf,
        rounds=load_rounds(campaign_dir),
        profiles=load_profiles(campaign_dir),
        ineffective=_load_ineffective(campaign_dir),
        transcripts=load_transcripts(campaign_dir),
        optimize_md_sections=parse_optimize_md_sections(
            campaign_dir / "logs" / "optimize.md"
        ),
    )
