"""Cost analytics (spec §6.1 panel 2 + panel 8).

Three derived aggregates:

* ``cumulative``        — running ``cumulative_usd`` over time, the
  authoritative line on the cost panel
* ``per_phase``         — sum of ``cost_usd`` / ``wall_s`` / ``turns``
  grouped by phase (post-arrow normalised), driving the stacked bar
* ``per_round``         — same dimensions grouped by round number,
  feeding the token/turn/wall mini line charts (panel 8)
* ``cost_per_improvement`` — best step-geomean Δ vs cumulative cost
  at each ACCEPTED point; ``None`` for ROLLBACK / BASELINE
"""

from __future__ import annotations

from typing import Any, Iterable

from turbo_view.model import CostRow, HistoryEntry, RunState

_ACCEPT_DECISIONS = frozenset({"BASELINE", "ACCEPTED", "ACCEPTED (noise-bounded)"})


def _phase_name(phase: str) -> str:
    """Return the canonical phase name (strip variant suffix).

    ``"VALIDATE (quick)"`` → ``"VALIDATE"``.  Variants are tracked
    separately at the row level, but phase-level rollups merge them.
    """
    p = phase.strip()
    paren = p.find("(")
    return p[:paren].strip() if paren > 0 else p


def cumulative_series(rows: list[CostRow]) -> list[dict[str, Any]]:
    return [
        {
            "x": row.ts.isoformat(),
            "y": row.cumulative_usd,
            "phase": row.phase,
            "round": row.round,
            "status": row.status,
        }
        for row in rows
    ]


def per_phase_breakdown(rows: list[CostRow]) -> list[dict[str, Any]]:
    """Aggregate cost / wall / turn / count grouped by canonical phase."""
    buckets: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for row in rows:
        ph = _phase_name(row.phase)
        if ph not in buckets:
            buckets[ph] = {"cost_usd": 0.0, "wall_s": 0.0, "turns": 0.0, "count": 0.0}
            order.append(ph)
        b = buckets[ph]
        b["cost_usd"] += row.cost_usd
        b["wall_s"] += row.wall_s
        b["turns"] += row.turns
        b["count"] += 1
    return [
        {
            "phase": ph,
            "cost_usd": round(buckets[ph]["cost_usd"], 6),
            "wall_s": round(buckets[ph]["wall_s"], 3),
            "turns": int(buckets[ph]["turns"]),
            "count": int(buckets[ph]["count"]),
        }
        for ph in order
    ]


def per_round_series(rows: list[CostRow]) -> list[dict[str, Any]]:
    """Aggregate ``wall_s`` / ``sdk_s`` / ``turns`` / ``cost_usd`` per round.

    Rows with ``round is None`` (e.g. ``DEFINE_TARGET``,
    ``PREPARE_ENVIRONMENT``) are bucketed under round 0 to keep the
    chart's x-axis fully populated.
    """
    buckets: dict[int, dict[str, float]] = {}
    for row in rows:
        rn = row.round if row.round is not None else 0
        b = buckets.setdefault(rn, {"wall_s": 0.0, "sdk_s": 0.0, "turns": 0.0, "cost_usd": 0.0})
        b["wall_s"] += row.wall_s
        b["sdk_s"] += row.sdk_s
        b["turns"] += row.turns
        b["cost_usd"] += row.cost_usd
    return [
        {
            "round": rn,
            "wall_s": round(buckets[rn]["wall_s"], 3),
            "sdk_s": round(buckets[rn]["sdk_s"], 3),
            "turns": int(buckets[rn]["turns"]),
            "cost_usd": round(buckets[rn]["cost_usd"], 6),
        }
        for rn in sorted(buckets)
    ]


def cost_per_improvement(
    rows: list[CostRow],
    history: Iterable[HistoryEntry],
) -> list[dict[str, Any]]:
    """For each ACCEPTED point: cumulative cost vs best step-geomean Δ.

    Δ is measured against the BASELINE entry; BASELINE itself is
    plotted at Δ=0. Useful as a marginal-utility curve.
    """
    history_list = list(history)
    baseline = next((h for h in history_list if h.decision == "BASELINE"), None)
    if baseline is None:
        return []
    base_score = float(baseline.score.get("step_geomean", 0.0))
    if not base_score:
        return []

    cum_by_round: dict[int, float] = {}
    for row in rows:
        if row.round is not None:
            cum_by_round[row.round] = max(cum_by_round.get(row.round, 0.0), row.cumulative_usd)

    out: list[dict[str, Any]] = []
    best = base_score
    for h in history_list:
        if h.decision not in _ACCEPT_DECISIONS:
            continue
        score = float(h.score.get("step_geomean", best))
        if score > best:
            best = score
        cum = cum_by_round.get(h.round)
        if cum is None:
            continue
        out.append({
            "round": h.round,
            "decision": h.decision,
            "cumulative_usd": cum,
            "step_geomean": score,
            "delta_pct": ((best - base_score) / base_score) * 100.0,
        })
    return out


def cost_panel(rows: list[CostRow], state: RunState | None) -> dict[str, Any]:
    history = state.history if state is not None else []
    return {
        "cumulative": cumulative_series(rows),
        "per_phase": per_phase_breakdown(rows),
        "per_round": per_round_series(rows),
        "cost_per_improvement": cost_per_improvement(rows, history),
        "total_usd": rows[-1].cumulative_usd if rows else 0.0,
        "total_wall_s": sum(r.wall_s for r in rows),
        "total_turns": sum(r.turns for r in rows),
    }


def token_turn_wall_panel(rows: list[CostRow]) -> dict[str, Any]:
    """Panel 8: 3 mini line charts (wall_s / sdk_s / turns) over rounds."""
    series = per_round_series(rows)
    return {
        "rounds": [s["round"] for s in series],
        "wall_s": [s["wall_s"] for s in series],
        "sdk_s": [s["sdk_s"] for s in series],
        "turns": [s["turns"] for s in series],
    }
