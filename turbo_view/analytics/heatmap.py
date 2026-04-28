"""Heatmap analytics (spec §6.1 panel 4).

Builds a (rounds × shapes) matrix of forward / backward TFLOPS plus
Δ% vs baseline (the lowest-numbered round that has bench data).

Shapes that don't appear in every round are still listed; missing
cells render as ``null``.
"""

from __future__ import annotations

from typing import Any

from turbo_view.model import RoundBundle


def _baseline_round_n(rounds: dict[int, RoundBundle]) -> int | None:
    for n in sorted(rounds):
        if rounds[n].bench_shapes:
            return n
    return None


def _rows_with_bench(rounds: dict[int, RoundBundle]) -> list[int]:
    return sorted(n for n, rb in rounds.items() if rb.bench_shapes)


def _all_shape_labels(rounds: dict[int, RoundBundle]) -> list[str]:
    """Stable ordering: first occurrence in lowest round wins."""
    seen: dict[str, None] = {}
    for n in sorted(rounds):
        for s in rounds[n].bench_shapes:
            seen.setdefault(s.label)
    return list(seen.keys())


def _delta_pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return ((value - base) / base) * 100.0


def heatmap_panel(rounds: dict[int, RoundBundle]) -> dict[str, Any]:
    bench_rounds = _rows_with_bench(rounds)
    if not bench_rounds:
        return {
            "rounds": [], "shape_labels": [], "rows": [],
            "baseline_round": None,
        }

    labels = _all_shape_labels(rounds)
    baseline_n = _baseline_round_n(rounds)
    baseline_by_label: dict[str, dict[str, float]] = {}
    if baseline_n is not None:
        for s in rounds[baseline_n].bench_shapes:
            baseline_by_label[s.label] = {"fwd": s.fwd_tflops, "bwd": s.bwd_tflops}

    out_rows: list[dict[str, Any]] = []
    for n in bench_rounds:
        by_label = {s.label: s for s in rounds[n].bench_shapes}
        fwd: list[float | None] = []
        bwd: list[float | None] = []
        delta_fwd: list[float | None] = []
        delta_bwd: list[float | None] = []
        check: list[str | None] = []
        for label in labels:
            s = by_label.get(label)
            base = baseline_by_label.get(label)
            if s is None:
                fwd.append(None); bwd.append(None)
                delta_fwd.append(None); delta_bwd.append(None)
                check.append(None)
                continue
            fwd.append(s.fwd_tflops)
            bwd.append(s.bwd_tflops)
            delta_fwd.append(_delta_pct(s.fwd_tflops, base["fwd"] if base else None))
            delta_bwd.append(_delta_pct(s.bwd_tflops, base["bwd"] if base else None))
            check.append(s.check)
        out_rows.append({
            "round": n,
            "fwd": fwd,
            "bwd": bwd,
            "delta_fwd": delta_fwd,
            "delta_bwd": delta_bwd,
            "check": check,
        })

    return {
        "rounds": bench_rounds,
        "shape_labels": labels,
        "rows": out_rows,
        "baseline_round": baseline_n,
    }
