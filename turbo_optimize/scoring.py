"""Benchmark CSV parsing + accept / rollback decision logic.

The rules encoded here come from two sources:

* `workflow/optimize-loop.md` "Scoring Operations Specification" (lines
  121-171) — geometric mean aggregate score, correctness gate,
  multi-metric handling, noise assessment.
* `rules/iteration_rules.mdc` Rule 3 — mechanical accept / rollback
  decision with the additional "any shape regressed >= 5%" gate.

The two specifications combine as:
    - any `Check=FAIL` → reject
    - aggregate regression (worse than current best) → reject
    - any core shape regressed > 3% → reject (SKILL workflow)
    - any core shape regressed >= 5% → reject (iteration rules, redundant
      with the 3% rule but kept explicit for audit)
    - improvement < 2% → candidate, requires noise re-measurement
    - otherwise accept
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


log = logging.getLogger(__name__)


CHECK_COL = "Check"
DEFAULT_PRIMARY = "Forward TFLOPS"

REGRESS_PCT_SOFT = 3.0
REGRESS_PCT_HARD = 5.0
NOISE_THRESHOLD_PCT = 2.0


class ScoringError(Exception):
    pass


@dataclass
class ShapeResult:
    shape: dict[str, Any]
    check: str
    metrics: dict[str, float]

    def metric(self, name: str) -> float | None:
        return self.metrics.get(name)


@dataclass
class BenchmarkParse:
    primary_metric: list[str]
    rows: list[ShapeResult]
    all_pass: bool
    raw_headers: list[str] = field(default_factory=list)


@dataclass
class ScoreVector:
    per_shape: list[ShapeResult]
    aggregate: dict[str, float]


@dataclass
class DecisionResult:
    decision: str
    reason: str
    improvement_pct: dict[str, float]
    regressions: list[dict[str, Any]]
    noise_check_required: bool


def split_primary_metric(primary_metric: str) -> list[str]:
    if not primary_metric:
        return [DEFAULT_PRIMARY]
    parts = [p.strip() for p in primary_metric.split(",") if p.strip()]
    return parts or [DEFAULT_PRIMARY]


def parse_bench_csv(path: Path, primary_metric: str) -> BenchmarkParse:
    """Parse a benchmark CSV. Assumes headers plus one row per shape."""
    metrics = split_primary_metric(primary_metric)
    if not path.exists():
        raise ScoringError(f"benchmark CSV not found at {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ScoringError(f"CSV has no header: {path}")
        headers = list(reader.fieldnames)
        shape_cols = _detect_shape_columns(headers, metrics)
        rows: list[ShapeResult] = []
        all_pass = True
        for raw in reader:
            metrics_in_row: dict[str, float] = {}
            for metric in metrics:
                val = raw.get(metric)
                if val is None:
                    metrics_in_row[metric] = float("nan")
                    continue
                try:
                    metrics_in_row[metric] = float(val)
                except ValueError:
                    metrics_in_row[metric] = float("nan")
            check = str(raw.get(CHECK_COL, "")).strip().upper() or "UNKNOWN"
            if check != "PASS":
                all_pass = False
            rows.append(
                ShapeResult(
                    shape={col: raw.get(col) for col in shape_cols},
                    check=check,
                    metrics=metrics_in_row,
                )
            )
    return BenchmarkParse(
        primary_metric=metrics, rows=rows, all_pass=all_pass, raw_headers=headers
    )


def _detect_shape_columns(headers: Iterable[str], metrics: Iterable[str]) -> list[str]:
    metric_set = set(metrics)
    exclude = metric_set | {CHECK_COL}
    exclude |= {h for h in headers if _looks_like_metric(h) and h not in metric_set}
    return [h for h in headers if h not in exclude]


def _looks_like_metric(header: str) -> bool:
    keywords = ("TFLOPS", "TOPS", "GB/s", "ms", "latency", "Bandwidth")
    return any(k.lower() in header.lower() for k in keywords)


def compute_score_vector(parse: BenchmarkParse) -> ScoreVector:
    if not parse.rows:
        raise ScoringError("no benchmark rows to score")
    aggregate: dict[str, float] = {}
    for metric in parse.primary_metric:
        values = [r.metrics.get(metric) for r in parse.rows if r.check == "PASS"]
        values = [v for v in values if v is not None and not math.isnan(v) and v > 0]
        if not values:
            aggregate[metric] = 0.0
            continue
        log_sum = sum(math.log(v) for v in values)
        aggregate[metric] = math.exp(log_sum / len(values))
    return ScoreVector(per_shape=parse.rows, aggregate=aggregate)


def geomean(values: Iterable[float]) -> float:
    filtered = [v for v in values if v > 0 and not math.isnan(v)]
    if not filtered:
        return 0.0
    return math.exp(sum(math.log(v) for v in filtered) / len(filtered))


def compare_score(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
) -> dict[str, float]:
    """Percentage improvement of candidate over baseline per metric."""
    if not baseline:
        return {k: 0.0 for k in candidate}
    out: dict[str, float] = {}
    for metric, cand in candidate.items():
        base = baseline.get(metric)
        if base is None or base == 0:
            out[metric] = 0.0
            continue
        out[metric] = (cand - base) / base * 100.0
    return out


def find_per_shape_regressions(
    candidate: ScoreVector,
    baseline: ScoreVector | None,
    metric: str,
    *,
    threshold_pct: float = REGRESS_PCT_SOFT,
) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    regressions: list[dict[str, Any]] = []
    base_lookup = {_shape_key(r.shape): r for r in baseline.per_shape}
    for row in candidate.per_shape:
        base = base_lookup.get(_shape_key(row.shape))
        if base is None:
            continue
        base_val = base.metrics.get(metric)
        cand_val = row.metrics.get(metric)
        if not base_val or not cand_val or base_val <= 0:
            continue
        delta = (cand_val - base_val) / base_val * 100.0
        if delta < -threshold_pct:
            regressions.append(
                {
                    "shape": row.shape,
                    "metric": metric,
                    "baseline": base_val,
                    "candidate": cand_val,
                    "delta_pct": delta,
                }
            )
    return regressions


def _shape_key(shape: dict[str, Any]) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in shape.items()))


def decide_accept_rollback(
    candidate: ScoreVector,
    best: ScoreVector | None,
    primary_metric: str,
    *,
    correctness_ok: bool,
    build_ok: bool = True,
) -> DecisionResult:
    """Apply the hard gates and noise assessment from the workflow spec."""
    metrics = split_primary_metric(primary_metric)
    if not build_ok:
        return DecisionResult(
            decision="ROLLBACK",
            reason="build failed",
            improvement_pct={},
            regressions=[],
            noise_check_required=False,
        )
    if not correctness_ok:
        return DecisionResult(
            decision="ROLLBACK",
            reason="correctness failed (any Check != PASS counts)",
            improvement_pct={},
            regressions=[],
            noise_check_required=False,
        )
    if any(r.check != "PASS" for r in candidate.per_shape):
        return DecisionResult(
            decision="ROLLBACK",
            reason="benchmark Check=FAIL in at least one shape",
            improvement_pct={},
            regressions=[],
            noise_check_required=False,
        )

    baseline_aggregate = best.aggregate if best else {}
    improvement = compare_score(candidate.aggregate, baseline_aggregate)

    regressions: list[dict[str, Any]] = []
    for metric in metrics:
        regressions.extend(
            find_per_shape_regressions(
                candidate, best, metric, threshold_pct=REGRESS_PCT_SOFT
            )
        )

    if regressions:
        worst = min(r["delta_pct"] for r in regressions)
        if worst <= -REGRESS_PCT_HARD:
            return DecisionResult(
                decision="ROLLBACK",
                reason=f"core shape regressed {worst:.2f}% (>= {REGRESS_PCT_HARD}%)",
                improvement_pct=improvement,
                regressions=regressions,
                noise_check_required=False,
            )
        return DecisionResult(
            decision="ROLLBACK",
            reason=(
                f"core shape regressed {worst:.2f}% (> {REGRESS_PCT_SOFT}%); "
                "SKILL workflow rejects by default"
            ),
            improvement_pct=improvement,
            regressions=regressions,
            noise_check_required=False,
        )

    if best is None:
        return DecisionResult(
            decision="ACCEPTED",
            reason="baseline round; no prior best to compare",
            improvement_pct=improvement,
            regressions=[],
            noise_check_required=False,
        )

    for metric in metrics:
        if candidate.aggregate.get(metric, 0) < baseline_aggregate.get(metric, 0):
            return DecisionResult(
                decision="ROLLBACK",
                reason=f"aggregate {metric} regressed vs current best",
                improvement_pct=improvement,
                regressions=regressions,
                noise_check_required=False,
            )

    best_delta = max(improvement.values()) if improvement else 0.0
    any_improvement = best_delta > 0
    if not any_improvement:
        return DecisionResult(
            decision="ROLLBACK",
            reason="no metric improved over current best",
            improvement_pct=improvement,
            regressions=regressions,
            noise_check_required=False,
        )

    if 0 < best_delta < NOISE_THRESHOLD_PCT:
        return DecisionResult(
            decision="ACCEPT_PENDING_NOISE",
            reason=(
                f"improvement {best_delta:.2f}% < {NOISE_THRESHOLD_PCT}% noise "
                "threshold; require 3 re-measurements"
            ),
            improvement_pct=improvement,
            regressions=regressions,
            noise_check_required=True,
        )

    return DecisionResult(
        decision="ACCEPTED",
        reason=f"aggregate improved by {best_delta:.2f}%",
        improvement_pct=improvement,
        regressions=regressions,
        noise_check_required=False,
    )


# --- noise re-measurement ----------------------------------------------

def noise_summary(samples: list[dict[str, float]], metric: str) -> dict[str, float]:
    vals = [s.get(metric, float("nan")) for s in samples]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"mean": mean, "std": math.sqrt(var), "n": float(n)}


def accept_after_noise(
    original_improvement_pct: float,
    repeated: list[ScoreVector],
    baseline: ScoreVector,
    primary_metric: str,
) -> bool:
    """Post-re-measurement acceptance gate from scoring spec."""
    if original_improvement_pct <= 0:
        return False
    metrics = split_primary_metric(primary_metric)
    for metric in metrics:
        samples = [{metric: sv.aggregate.get(metric, 0.0)} for sv in repeated]
        stats = noise_summary(samples, metric)
        base = baseline.aggregate.get(metric, 0.0)
        if base <= 0 or stats["n"] == 0:
            return False
        mean_improvement = (stats["mean"] - base) / base * 100.0
        if mean_improvement <= 1.0:
            return False
        if stats["std"] >= original_improvement_pct / 2.0 * base / 100.0:
            return False
    return True


# --- hypothesis dedup ---------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in _WORD_RE.findall(text) if len(t) > 2}


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


@dataclass
class DuplicateMatch:
    direction: str
    reason: str
    round: int | None
    similarity: float


def check_hypothesis_duplicate(
    hypothesis: str,
    verified_ineffective: list[dict[str, Any]] | list,
    *,
    threshold: float = 0.6,
) -> DuplicateMatch | None:
    """Fuzzy match a new hypothesis against the verified ineffective list."""
    best: DuplicateMatch | None = None
    for entry in verified_ineffective:
        if hasattr(entry, "direction"):
            direction = entry.direction
            reason = entry.reason
            round_n = entry.round
        else:
            direction = entry.get("direction", "")
            reason = entry.get("reason", "")
            round_n = entry.get("round")
        score = similarity(hypothesis, direction)
        if score >= threshold and (best is None or score > best.similarity):
            best = DuplicateMatch(
                direction=direction,
                reason=reason,
                round=round_n,
                similarity=score,
            )
    return best
