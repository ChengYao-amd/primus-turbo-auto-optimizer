"""R-vs-R diff (spec §6.2 P5).

Given two ``ProfileBundle`` instances, produce a side-by-side row
list of every kernel's total / avg / count plus Δ%. Kernels missing
in one side appear with ``None`` in the absent column.
"""

from __future__ import annotations

from typing import Any

from turbo_view.analytics.profile import _grouped, clean_name
from turbo_view.model import ProfileBundle


def _delta_pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return ((b - a) / a) * 100.0


def round_diff(left: ProfileBundle, right: ProfileBundle) -> dict[str, Any]:
    g_left = _grouped(left.dispatches)
    g_right = _grouped(right.dispatches)
    names = sorted(set(g_left) | set(g_right))

    rows: list[dict[str, Any]] = []
    for name in names:
        left_b = g_left.get(name)
        right_b = g_right.get(name)
        rows.append({
            "name_raw": name,
            "name_clean": clean_name(name),
            "left_total_us":  round(left_b["total_us"], 1) if left_b else None,
            "right_total_us": round(right_b["total_us"], 1) if right_b else None,
            "left_count":  int(left_b["count"]) if left_b else None,
            "right_count": int(right_b["count"]) if right_b else None,
            "delta_pct": _delta_pct(
                left_b["total_us"] if left_b else None,
                right_b["total_us"] if right_b else None,
            ),
        })
    rows.sort(key=lambda r: -(r["right_total_us"] or r["left_total_us"] or 0))
    return {
        "left_round": left.round,
        "right_round": right.round,
        "rows": rows,
    }


def all_round_pairs(profiles: dict[int, ProfileBundle]) -> dict[str, Any]:
    """Pre-compute (left, right) diffs for adjacent rounds.

    The front-end can use this for the default "show me consecutive
    rounds" view; user-driven dropdowns swap in arbitrary pairs at
    render time.
    """
    rounds = sorted(profiles)
    pairs: list[dict[str, Any]] = []
    for i in range(len(rounds) - 1):
        pairs.append(round_diff(profiles[rounds[i]], profiles[rounds[i + 1]]))
    return {
        "rounds": rounds,
        "pairs": pairs,
    }
