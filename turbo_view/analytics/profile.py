"""Profile sub-panel analytics (spec §6.2 P1 / P2 / P3 / P4 / P9).

Inputs are ``ProfileBundle.dispatches`` lists indexed by round.
Outputs are tiny chart-ready dicts the front-end can render with
vanilla Chart.js + a hand-rolled treemap layout.
"""

from __future__ import annotations

import re
from typing import Any

from turbo_view.model import KernelDispatch, ProfileBundle

# --- Family rules (spec §6.3) -------------------------------------

_FAMILY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("fwd_dgrad", re.compile(r"(^_grouped_fp8_persistent_gemm_kernel$)|(^_fp8_persistent_gemm_kernel$)")),
    ("wgrad",     re.compile(r"(^_grouped_variable_k_gemm_kernel$)|(_wgrad)")),
    ("quant",     re.compile(r"^void primus_turbo::unary_kernel.*bf16.*fp8")),
    ("amax",      re.compile(r"(^void primus_turbo::reduce_row_kernel.*AbsMax)|(^void primus_turbo::compute_scale_from_amax)")),
    ("elementwise", re.compile(r"^void at::native::vectorized_elementwise_kernel")),
]


def family_for(name: str) -> str:
    for fam, rx in _FAMILY_RULES:
        if rx.search(name):
            return fam
    return "other"


# --- Name cleanup (spec §6.2) -------------------------------------

_CLEAN_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^void primus_turbo::(\w+)<.*$"),  r"primus_turbo::\1<…>"),
    (re.compile(r"^void at::native::(?:\(anonymous namespace\)::)?(\w+)<.*$"),
                                                    r"at::native::\1<…>"),
]


def clean_name(raw: str) -> str:
    for rx, repl in _CLEAN_RULES:
        if rx.match(raw):
            return rx.sub(repl, raw)
    return raw


# --- Aggregation primitives ---------------------------------------


def _grouped(dispatches: list[KernelDispatch]) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {}
    for d in dispatches:
        bucket = groups.setdefault(d.name, {"total_us": 0.0, "count": 0,
                                            "vgpr_max": 0, "sgpr_max": 0,
                                            "lds_max": 0})
        bucket["total_us"] += d.dur_us
        bucket["count"] += 1
        bucket["vgpr_max"] = max(bucket["vgpr_max"], d.vgpr)
        bucket["sgpr_max"] = max(bucket["sgpr_max"], d.sgpr)
        bucket["lds_max"]  = max(bucket["lds_max"],  d.lds_bytes)
    return groups


# --- P1 / P9: top-N -----------------------------------------------


def top_n_kernels(dispatches: list[KernelDispatch], n: int = 12) -> list[dict[str, Any]]:
    groups = _grouped(dispatches)
    rows = [
        {
            "name_clean": clean_name(name),
            "name_raw": name,
            "family": family_for(name),
            "total_us": round(b["total_us"], 1),
            "avg_us": round(b["total_us"] / max(b["count"], 1), 2),
            "count": int(b["count"]),
        }
        for name, b in groups.items()
    ]
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows[:n]


# --- P2: round-over-round Δ ---------------------------------------


def round_over_round_topn(
    profiles: dict[int, ProfileBundle], n: int = 6,
) -> dict[str, Any]:
    """Track top-N kernels across rounds.

    The "top-N" set is the intersection of each round's top-N to avoid
    sparse-line whiplash; if intersection is empty, fall back to the
    union of the lowest round's top-N.
    """
    if not profiles:
        return {"rounds": [], "kernels": [], "series": []}

    by_round = {n_: {r["name_raw"]: r for r in top_n_kernels(p.dispatches, n=n*2)}
                for n_, p in profiles.items()}
    rounds = sorted(by_round)

    sets = [set(by_round[r].keys()) for r in rounds]
    common = set.intersection(*sets) if sets else set()
    if not common:
        common = set(list(by_round[rounds[0]].keys())[:n])

    kernels = sorted(common, key=lambda k: -by_round[rounds[0]][k]["total_us"])[:n]
    series = []
    for k in kernels:
        clean = clean_name(k)
        series.append({
            "name_clean": clean,
            "name_raw": k,
            "totals_us": [by_round[r].get(k, {"total_us": None})["total_us"] for r in rounds],
        })
    return {"rounds": rounds, "kernels": kernels, "series": series}


# --- P3: family rollup -------------------------------------------


def family_rollup(profiles: dict[int, ProfileBundle]) -> dict[str, Any]:
    if not profiles:
        return {"rounds": [], "families": [], "series": {}}
    rounds = sorted(profiles)
    families = ["fwd_dgrad", "wgrad", "quant", "amax", "elementwise", "other"]
    series: dict[str, list[float]] = {f: [] for f in families}
    for r in rounds:
        sums = {f: 0.0 for f in families}
        for d in profiles[r].dispatches:
            sums[family_for(d.name)] += d.dur_us
        for f in families:
            series[f].append(round(sums[f], 1))
    return {"rounds": rounds, "families": families, "series": series}


# --- P4: VGPR/SGPR/LDS trends -------------------------------------


def gpu_resource_trends(
    profiles: dict[int, ProfileBundle], target: str | None = None,
) -> dict[str, Any]:
    """Pick a target kernel and trace VGPR / SGPR / LDS across rounds.

    If ``target`` is omitted, pick the kernel with the largest
    cumulative time in the lowest round (spec §6.2 fallback).
    """
    if not profiles:
        return {"rounds": [], "target": None, "vgpr": [], "sgpr": [], "lds": []}
    rounds = sorted(profiles)
    if target is None:
        first_top = top_n_kernels(profiles[rounds[0]].dispatches, n=1)
        if not first_top:
            return {"rounds": rounds, "target": None,
                    "vgpr": [None] * len(rounds), "sgpr": [None] * len(rounds),
                    "lds": [None] * len(rounds)}
        target = first_top[0]["name_raw"]

    vgpr: list[int | None] = []
    sgpr: list[int | None] = []
    lds:  list[int | None] = []
    for r in rounds:
        groups = _grouped(profiles[r].dispatches)
        b = groups.get(target)
        vgpr.append(int(b["vgpr_max"]) if b else None)
        sgpr.append(int(b["sgpr_max"]) if b else None)
        lds.append(int(b["lds_max"]) if b else None)

    return {
        "rounds": rounds,
        "target": target,
        "target_clean": clean_name(target),
        "vgpr": vgpr,
        "sgpr": sgpr,
        "lds": lds,
    }


# --- P9: treemap layout (squarified) ------------------------------


def treemap_layout(top_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute rectangle layout for top kernels (squarified treemap).

    Layout fills a unit-square (0..1, 0..1) so the front-end can scale
    to any pixel area. Returned rectangles carry ``x`` / ``y`` /
    ``w`` / ``h`` plus the original row's keys.
    """
    items = [r for r in top_rows if r["total_us"] > 0]
    if not items:
        return []
    total = sum(r["total_us"] for r in items)
    items.sort(key=lambda r: -r["total_us"])

    cells: list[dict[str, Any]] = []
    x, y, w, h = 0.0, 0.0, 1.0, 1.0
    i = 0
    while i < len(items):
        row = []
        side = min(w, h)
        while i + len(row) < len(items):
            row_candidate = row + [items[i + len(row)]]
            if not _is_better(row, row_candidate, side, total):
                break
            row = row_candidate
        if not row:
            row = [items[i]]
        cells.extend(_layout_row(row, x, y, w, h, total))
        consumed = sum(r["total_us"] for r in row)
        frac = consumed / total
        if w >= h:
            x += w * frac
            w = w * (1 - frac)
        else:
            y += h * frac
            h = h * (1 - frac)
        i += len(row)
        total -= consumed
    return cells


def _is_better(row, row_candidate, side, total):
    if not row or total <= 0:
        return True
    return _worst_aspect(row_candidate, side, total) <= _worst_aspect(row, side, total)


def _worst_aspect(row, side, total):
    s = sum(r["total_us"] for r in row)
    if s <= 0:
        return float("inf")
    long_side = side
    short_side = (s / total) * (1.0 / max(long_side, 1e-9))
    worst = 0.0
    for r in row:
        a = r["total_us"]
        ratio = max(long_side ** 2 * a / (s ** 2),
                    s ** 2 / (long_side ** 2 * a))
        worst = max(worst, ratio)
    return worst


def _layout_row(row, x, y, w, h, total):
    s = sum(r["total_us"] for r in row)
    cells = []
    if w >= h:
        rect_w = (s / total) * w
        offset = 0.0
        for r in row:
            rect_h = h * (r["total_us"] / s)
            cells.append({**r, "x": x, "y": y + offset, "w": rect_w, "h": rect_h})
            offset += rect_h
    else:
        rect_h = (s / total) * h
        offset = 0.0
        for r in row:
            rect_w = w * (r["total_us"] / s)
            cells.append({**r, "x": x + offset, "y": y, "w": rect_w, "h": rect_h})
            offset += rect_w
    return cells


# --- Profile panel bundle (per round) -----------------------------


def profile_panel_for_round(bundle: ProfileBundle) -> dict[str, Any]:
    """Per-round panel containing only round-local data.

    Cross-round aggregates (P2/P3/P4) live at the payload top level
    (see ``payload._profile_global``); embedding them per-round
    duplicated multi-MB across 67–90-round campaigns.
    """
    top = top_n_kernels(bundle.dispatches, n=12)
    return {
        "round": bundle.round,
        "flavor": bundle.flavor,
        "summary_md_html": bundle.summary_md_html,
        "perfetto_results_path": (
            str(bundle.perfetto_json_path) if bundle.perfetto_json_path else None
        ),
        "top_n": top,                       # P1
        "treemap": treemap_layout(top),     # P9
    }
