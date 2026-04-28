"""Convert ``CampaignBundle`` into a JSON-serialisable, chart-ready dict.

The payload is the single source of truth for the front end: it is
both inlined into ``index.html`` (via ``<script id="data">``) and
written to ``data.json`` so future watch-mode rebuilds can re-fetch
just the data without reloading the page.

Schema is versioned (``schema_version``); bump on breaking changes.

Schema timeline:

* ``"1"`` (PR-1): state, perf, perf_panel, rounds, ineffective,
  optimize_md_sections.
* ``"2"`` (PR-2): adds cost_panel, gantt_panel, heatmap_panel,
  token_turn_wall_panel; rounds rows expose ``bench_shapes``.
* ``"3"`` (PR-3): adds ``profile_panels`` (per round; P1/P9/P10/P11)
  + ``profile_global`` (P2/P3/P4 cross-round)
  + ``profile_diffs`` (P5, consecutive only); ``gantt_panel.events``
  populated from transcripts.
* ``"4"`` (post-review): drops ``optimize_md_sections`` (never
  consumed by the front-end); hoists global profile aggregates out of
  per-round panels; collapses ``profile_diffs`` to consecutive pairs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from turbo_view.analytics.cost import cost_panel, token_turn_wall_panel
from turbo_view.analytics.diff import round_diff
from turbo_view.analytics.gantt import gantt_panel
from turbo_view.analytics.heatmap import heatmap_panel
from turbo_view.analytics.profile import (
    family_rollup,
    gpu_resource_trends,
    profile_panel_for_round,
    round_over_round_topn,
)
from turbo_view.model import (
    CampaignBundle,
    HistoryEntry,
    PerfRow,
    RoundBundle,
    RunState,
    ShapeRow,
)

SCHEMA_VERSION = "4"

_ACCEPT_DECISIONS = frozenset({
    "BASELINE",
    "ACCEPTED",
    "ACCEPTED (noise-bounded)",
})


def _isoformat(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _state_payload(state: RunState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "campaign_id": state.campaign_id,
        "campaign_dir": str(state.campaign_dir),
        "current_phase": state.current_phase,
        "current_round": state.current_round,
        "best_round": state.best_round,
        "best_score": dict(state.best_score),
        "rollback_streak": state.rollback_streak,
        "started_at": state.started_at,
        "last_update": state.last_update,
        "params": dict(state.params),
        "history": [_history_entry_payload(h) for h in state.history],
    }


def _history_entry_payload(h: HistoryEntry) -> dict[str, Any]:
    return {
        "round": h.round,
        "decision": h.decision,
        "score": dict(h.score),
        "description": h.description,
        "at": h.at,
    }


def _perf_row_payload(row: PerfRow) -> dict[str, Any]:
    return {
        "round": row.round,
        "status": row.status,
        "description": row.description,
        "fwd_avg": row.fwd_avg,
        "fwd_peak": row.fwd_peak,
        "bwd_avg": row.bwd_avg,
        "bwd_peak": row.bwd_peak,
        "step_geomean": row.step_geomean,
        "vs_baseline": row.vs_baseline,
        "key_finding": row.key_finding,
    }


def _baseline_row(perf: list[PerfRow]) -> PerfRow | None:
    """Return the row tagged ``BASELINE`` (or ``None`` if absent)."""
    for row in perf:
        if row.status == "BASELINE":
            return row
    return None


def _best_row(perf: list[PerfRow], state: RunState | None) -> PerfRow | None:
    """Resolve the row that defines the campaign's "best" performance.

    Priority order:
    1. ``state.best_round`` if it points at a known perf row.
    2. ACCEPTED row with the highest ``step_geomean``.
    3. ``None`` when no accepted run has been recorded yet.
    """
    if state is not None and state.best_round is not None:
        for row in perf:
            if row.round == state.best_round:
                return row
    accepted = [r for r in perf if r.status in _ACCEPT_DECISIONS and r.status != "BASELINE"]
    if accepted:
        return max(accepted, key=lambda r: r.step_geomean)
    return None


def _baseline_geomean(perf: list[PerfRow]) -> float | None:
    """Step-geomean of the BASELINE row, if present."""
    base = _baseline_row(perf)
    return base.step_geomean if base else None


def _best_geomean(perf: list[PerfRow], state: RunState | None) -> float | None:
    """Resolve the best step-geomean.

    Prefers ``state.best_score['step_geomean']`` (authoritative);
    falls back to the row resolved by :func:`_best_row`.
    """
    if state and state.best_score.get("step_geomean") is not None:
        try:
            return float(state.best_score["step_geomean"])
        except (TypeError, ValueError):
            pass
    best = _best_row(perf, state)
    return best.step_geomean if best else None


def _kpi_summary(
    perf: list[PerfRow],
    state: RunState | None,
) -> dict[str, dict[str, float | None]]:
    """Sticky-bar KPI summary: baseline / best across step / fwd / bwd.

    Front end renders each entry as ``{baseline} -> {best} ({delta_pct} %)``.
    Percent is left to the front end so a missing baseline (or zero
    baseline) can render an em-dash without recomputing.
    """
    base = _baseline_row(perf)
    best = _best_row(perf, state)

    def _pair(field: str) -> dict[str, float | None]:
        b = getattr(base, field) if base else None
        x = getattr(best, field) if best else None
        return {"baseline": b, "best": x}

    return {
        "step": _pair("step_geomean"),
        "fwd":  _pair("fwd_avg"),
        "bwd":  _pair("bwd_avg"),
    }


def _perf_panel(perf: list[PerfRow], state: RunState | None) -> dict[str, Any]:
    """Build perf-trend chart payload.

    Front end consumes:
    * ``points``    — every round, scatter
    * ``accepted``  — step-line connecting BASELINE + ACCEPTED rounds
    * ``baseline``  — horizontal reference (or null)
    * ``best``      — horizontal reference (or null)
    * ``annotations.accept`` — vertical lines at ACCEPTED rounds
    """
    points = [
        {
            "round": r.round,
            "y": r.step_geomean,
            "fwd": r.fwd_avg,
            "bwd": r.bwd_avg,
            "status": r.status,
            "description": r.description,
        }
        for r in perf
    ]
    accepted = [
        {"round": r.round, "y": r.step_geomean, "status": r.status}
        for r in perf
        if r.status in _ACCEPT_DECISIONS
    ]
    accept_marks = [
        {"round": r.round, "label": r.description}
        for r in perf
        if r.status == "ACCEPTED" or r.status == "ACCEPTED (noise-bounded)"
    ]
    return {
        "points": points,
        "accepted": accepted,
        "baseline": _baseline_geomean(perf),
        "best": _best_geomean(perf, state),
        "annotations": {"accept": accept_marks},
    }


def _shape_row_payload(s: ShapeRow) -> dict[str, Any]:
    return {
        "label": s.label,
        "B": s.B, "M": s.M, "N": s.N, "K": s.K,
        "fwd_tflops": s.fwd_tflops,
        "bwd_tflops": s.bwd_tflops,
        "fwd_std": s.fwd_std,
        "bwd_std": s.bwd_std,
        "check": s.check,
    }


def _build_round_rows(
    rounds: dict[int, RoundBundle],
    state: RunState | None,
    perf: list[PerfRow],
) -> list[dict[str, Any]]:
    """Merge per-round data sources into a single flat list.

    Sources of truth, in priority order:
    1. ``state.history[i]`` for ``decision`` / ``description`` / ``score`` / ``at``
    2. ``perf[i]`` for performance numbers (fallback for ``description`` if history lacks it)
    3. ``rounds[i].summary_md_html`` for inline expansion
    4. ``rounds[i].bench_shapes`` (PR-2) and ``rounds[i].artifacts`` filenames
    The union of round numbers across all three sources is rendered;
    missing fields graceful-degrade to ``None`` / ``""``.
    """
    history_by_round: dict[int, HistoryEntry] = {}
    if state is not None:
        for h in state.history:
            history_by_round[h.round] = h

    perf_by_round: dict[int, PerfRow] = {p.round: p for p in perf}

    all_rounds: set[int] = set(history_by_round) | set(perf_by_round) | set(rounds)
    if state is not None and state.current_round:
        all_rounds.add(state.current_round)

    out: list[dict[str, Any]] = []
    for n in sorted(all_rounds):
        h = history_by_round.get(n)
        p = perf_by_round.get(n)
        rb = rounds.get(n)

        if h is not None:
            decision = h.decision
            description = h.description
            score = dict(h.score)
            at = h.at
        elif p is not None:
            decision = p.status
            description = p.description
            score = {"step_geomean": p.step_geomean}
            at = ""
        else:
            decision = "PENDING"
            description = ""
            score = {}
            at = ""

        out.append({
            "n": n,
            "decision": decision,
            "description": description,
            "score": score,
            "at": at,
            "perf": _perf_row_payload(p) if p is not None else None,
            "summary_md_html": rb.summary_md_html if rb is not None else None,
            "bench_shapes": ([_shape_row_payload(s) for s in rb.bench_shapes]
                             if rb is not None else []),
            "artifacts": ([a.name for a in rb.artifacts]
                          if rb is not None else []),
        })
    return out


def _ineffective_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "round": item.get("round"),
            "direction": item.get("direction", ""),
            "reason": item.get("reason", ""),
            "modified_files": list(item.get("modified_files") or []),
        })
    return out


def _profile_panels(bundle: CampaignBundle) -> dict[int, dict[str, Any]]:
    """Per-round profile panels containing only round-local data.

    Global aggregates (round-over-round, family rollup, GPU resource
    trends) are computed once in :func:`_profile_global` and exposed at
    the top level — embedding them per-round duplicated multi-MB of
    payload across 67-90 rounds (campaign at 9.6 MB → 4 MB drop).
    """
    return {
        n: profile_panel_for_round(p)
        for n, p in bundle.profiles.items()
    }


def _profile_global(bundle: CampaignBundle) -> dict[str, Any]:
    return {
        "round_over_round": round_over_round_topn(bundle.profiles),  # P2
        "family_rollup": family_rollup(bundle.profiles),             # P3
        "resources": gpu_resource_trends(bundle.profiles),           # P4
    }


def _consecutive_profile_diffs(
    profiles: dict[int, "ProfileBundle"],
) -> dict[str, Any]:
    """Diff each profile against its predecessor (PR-3 P5).

    The previous all-pairs strategy was O(N²) and wrote ~800 KB for
    a 90-round campaign. Consecutive pairs cover the only diff that
    appears by default (the "did this round move the needle" view);
    arbitrary R-vs-R remains available client-side from the per-round
    ``top_n`` arrays. Shape matches the old ``all_round_pairs`` so the
    front-end ``buildDiffTable`` consumer stays unchanged.
    """
    rounds = sorted(profiles)
    pairs: list[dict[str, Any]] = []
    for prev, curr in zip(rounds, rounds[1:]):
        pairs.append(round_diff(profiles[prev], profiles[curr]))
    return {"rounds": rounds, "pairs": pairs}


def bundle_to_payload(bundle: CampaignBundle) -> dict[str, Any]:
    state = bundle.state
    perf_rows = bundle.perf
    cost_rows = bundle.cost
    return {
        "schema_version": SCHEMA_VERSION,
        "state": _state_payload(state),
        "kpi_summary": _kpi_summary(perf_rows, state),
        "perf": [_perf_row_payload(r) for r in perf_rows],
        "perf_panel": _perf_panel(perf_rows, state),
        "cost_panel": cost_panel(cost_rows, state),
        "gantt_panel": gantt_panel(cost_rows, bundle.transcripts),
        "heatmap_panel": heatmap_panel(bundle.rounds),
        "token_turn_wall_panel": token_turn_wall_panel(cost_rows),
        "rounds": _build_round_rows(bundle.rounds, state, perf_rows),
        "ineffective": _ineffective_payload(bundle.ineffective),
        "profile_panels": {str(k): v for k, v in _profile_panels(bundle).items()},
        "profile_global": _profile_global(bundle),
        "profile_diffs": _consecutive_profile_diffs(bundle.profiles),
    }
