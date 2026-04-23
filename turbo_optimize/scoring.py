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
    metrics_stddev_pct: dict[str, float] = field(default_factory=dict)
    repeats: int = 1

    def metric(self, name: str) -> float | None:
        return self.metrics.get(name)

    def stddev_pct(self, name: str) -> float | None:
        return self.metrics_stddev_pct.get(name)


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


_STDDEV_SUFFIXES: tuple[str, ...] = (
    "_stddev_pct",
    "_stddev",
    "_std_pct",
    "_std",
)


def _stddev_header_for(metric: str, headers: Iterable[str]) -> str | None:
    header_set = set(headers)
    for suffix in _STDDEV_SUFFIXES:
        candidate = f"{metric}{suffix}"
        if candidate in header_set:
            return candidate
    return None


def _parse_float(raw: Any) -> float:
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def parse_bench_csv(path: Path, primary_metric: str) -> BenchmarkParse:
    """Parse a benchmark CSV. Assumes headers plus one row per shape.

    The parser recognises optional companion columns ``<Metric>_stddev``,
    ``<Metric>_stddev_pct``, ``<Metric>_std``, ``<Metric>_std_pct``. When
    any of those is present, it is attached to
    ``ShapeResult.metrics_stddev_pct`` as a percentage — callers fall
    back to the hard-coded noise threshold only when no column matches.
    ``_stddev`` / ``_std`` columns are normalised into percentages using
    the metric's mean in the same row.
    """
    metrics = split_primary_metric(primary_metric)
    if not path.exists():
        raise ScoringError(f"benchmark CSV not found at {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ScoringError(f"CSV has no header: {path}")
        headers = list(reader.fieldnames)
        shape_cols = _detect_shape_columns(headers, metrics)
        stddev_columns = {
            metric: _stddev_header_for(metric, headers) for metric in metrics
        }
        has_repeats_col = "repeats" in headers
        rows: list[ShapeResult] = []
        all_pass = True
        for raw in reader:
            metrics_in_row: dict[str, float] = {}
            stddev_in_row: dict[str, float] = {}
            for metric in metrics:
                mean_val = _parse_float(raw.get(metric))
                metrics_in_row[metric] = mean_val
                std_col = stddev_columns.get(metric)
                if std_col is None:
                    continue
                std_val = _parse_float(raw.get(std_col))
                if math.isnan(std_val):
                    continue
                if std_col.endswith(("_stddev", "_std")):
                    if mean_val and not math.isnan(mean_val) and mean_val > 0:
                        stddev_in_row[metric] = std_val / mean_val * 100.0
                else:
                    stddev_in_row[metric] = std_val
            check = str(raw.get(CHECK_COL, "")).strip().upper() or "UNKNOWN"
            if check != "PASS":
                all_pass = False
            repeats = 1
            if has_repeats_col:
                try:
                    repeats = max(1, int(float(raw.get("repeats", 1) or 1)))
                except (TypeError, ValueError):
                    repeats = 1
            rows.append(
                ShapeResult(
                    shape={col: raw.get(col) for col in shape_cols},
                    check=check,
                    metrics=metrics_in_row,
                    metrics_stddev_pct=stddev_in_row,
                    repeats=repeats,
                )
            )
    return BenchmarkParse(
        primary_metric=metrics, rows=rows, all_pass=all_pass, raw_headers=headers
    )


def _detect_shape_columns(headers: Iterable[str], metrics: Iterable[str]) -> list[str]:
    metric_set = set(metrics)
    companion_cols: set[str] = set()
    for metric in metric_set:
        for suffix in _STDDEV_SUFFIXES:
            companion_cols.add(f"{metric}{suffix}")
    companion_cols.add("repeats")
    exclude = metric_set | {CHECK_COL} | companion_cols
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


def observed_noise_pct(score: ScoreVector, metrics: Iterable[str]) -> float:
    """Largest per-shape stddev% observed across ``metrics``.

    Returns 0.0 when no shape carried a stddev companion column, which
    keeps the decision gate on the original static noise threshold.
    """
    worst: float = 0.0
    for row in score.per_shape:
        for metric in metrics:
            std = row.metrics_stddev_pct.get(metric)
            if std is None or math.isnan(std):
                continue
            worst = max(worst, float(std))
    return worst


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

    observed_noise = observed_noise_pct(candidate, metrics)
    effective_noise = max(NOISE_THRESHOLD_PCT, observed_noise * 2.0)
    if 0 < best_delta < effective_noise:
        if observed_noise > 0:
            reason = (
                f"improvement {best_delta:.2f}% < {effective_noise:.2f}% noise "
                f"threshold (observed stddev {observed_noise:.2f}% x2); "
                "require 3 re-measurements"
            )
        else:
            reason = (
                f"improvement {best_delta:.2f}% < {NOISE_THRESHOLD_PCT}% noise "
                "threshold; require 3 re-measurements"
            )
        return DecisionResult(
            decision="ACCEPT_PENDING_NOISE",
            reason=reason,
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
    file_overlap: float = 0.0
    signal: str = "text"


def _file_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = {str(p).strip() for p in a if p}
    set_b = {str(p).strip() for p in b if p}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


def check_hypothesis_duplicate(
    hypothesis: str,
    verified_ineffective: list[dict[str, Any]] | list,
    *,
    threshold: float = 0.6,
    planned_modified_files: list[str] | None = None,
    file_overlap_threshold: float = 0.6,
) -> DuplicateMatch | None:
    """Fuzzy match a new hypothesis against the verified ineffective list.

    Two signals are considered, either sufficient on its own to flag a
    duplicate:

    * **Textual similarity** of the hypothesis prose against the
      historical ``direction`` string (the original v1 behaviour).
    * **Modified-file overlap** — when ``planned_modified_files`` is
      provided, a Jaccard >= ``file_overlap_threshold`` against the
      historical entry's ``modified_files`` counts as a duplicate even
      if the prose reads different. This catches the "reword the same
      edit" failure mode where ANALYZE keeps re-proposing tweaks to
      the same file block.

    The match reports both scores plus ``signal`` = ``text`` /
    ``files`` / ``both`` so the orchestrator can log which signal
    fired.
    """
    planned_files = list(planned_modified_files or [])
    best: DuplicateMatch | None = None
    for entry in verified_ineffective:
        if hasattr(entry, "direction"):
            direction = entry.direction
            reason = entry.reason
            round_n = entry.round
            files = getattr(entry, "modified_files", []) or []
        else:
            direction = entry.get("direction", "")
            reason = entry.get("reason", "")
            round_n = entry.get("round")
            files = entry.get("modified_files") or []
        text_score = similarity(hypothesis, direction)
        file_score = _file_jaccard(planned_files, files) if planned_files else 0.0
        text_hit = text_score >= threshold
        file_hit = file_score >= file_overlap_threshold
        if not (text_hit or file_hit):
            continue
        if text_hit and file_hit:
            signal = "both"
        elif text_hit:
            signal = "text"
        else:
            signal = "files"
        candidate = DuplicateMatch(
            direction=direction,
            reason=reason,
            round=round_n,
            similarity=text_score,
            file_overlap=file_score,
            signal=signal,
        )
        better = best is None or max(text_score, file_score) > max(
            best.similarity, best.file_overlap
        )
        if better:
            best = candidate
    return best
