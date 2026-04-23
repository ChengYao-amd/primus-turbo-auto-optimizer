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


# --- REVIEW signals (tolerant mode) -----------------------------------


REVIEW_VERDICT_AGREE = "AGREE"
REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND = "DOWNGRADE_TO_NOISE_BOUND"
REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK = "DOWNGRADE_TO_ROLLBACK"
REVIEW_VERDICT_ESCALATE_HUMAN = "ESCALATE_HUMAN"

REVIEW_VERDICTS: tuple[str, ...] = (
    REVIEW_VERDICT_AGREE,
    REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND,
    REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK,
    REVIEW_VERDICT_ESCALATE_HUMAN,
)


# Metric keyword → canonical Primus-Turbo metric name. Used by both the
# hypothesis-text parser and the off-target-gain attribution heuristic,
# so a single keyword table keeps them consistent.
_FWD_KEYWORDS: tuple[str, ...] = (
    "forward", "fwd", "_fw_", "_fwkernel_",
    "grouped_fp8_persistent", "persistent_gemm", "tile_cumsum",
)
_BWD_KEYWORDS: tuple[str, ...] = (
    "backward", "bwd", "_bw_", "_bwkernel_",
    "variable_k", "grouped_variable_k",
)


@dataclass
class ReviewSignal:
    """One structured finding inside a REVIEW phase output.

    ``severity`` is either ``"info"`` / ``"warn"`` / ``"block"``.
    Only ``"block"`` fires a verdict transition in tolerant mode, and
    only for the three hard-rule signals (hypothesis-metric alignment,
    off-target gain, correctness bit-identity). ``"warn"`` is used for
    the quick-vs-full agreement and noise-band signals which exist to
    inform operators but do not, on their own, overrule a numeric
    ACCEPT in tolerant mode.
    """

    name: str
    passed: bool
    severity: str
    details: dict[str, Any]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "details": dict(self.details),
            "note": self.note,
        }


@dataclass
class ReviewBundle:
    """Structured output of :func:`compute_review_signals`.

    * ``signals`` — the five named findings in fixed order so the
      prompt can refer to them by index.
    * ``tolerant_verdict`` — preliminary verdict the orchestrator would
      apply if the REVIEW phase's LLM returned no override; the LLM's
      ``review_verdict`` takes precedence when present.
    * ``tolerant_reason`` — human-readable explanation behind
      ``tolerant_verdict``; preserved in the phase result and in the
      decision reason when a downgrade actually fires.
    """

    signals: list[ReviewSignal]
    tolerant_verdict: str
    tolerant_reason: str

    def signal(self, name: str) -> ReviewSignal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "tolerant_verdict": self.tolerant_verdict,
            "tolerant_reason": self.tolerant_reason,
        }


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


# Schema normalization -------------------------------------------------
#
# Primus-Turbo campaigns traditionally emit two very different CSVs:
#
# 1. Full benchmark (``benchmark_command``) — canonical columns
#    ``Check``, ``Forward TFLOPS``, ``Backward TFLOPS``, optional
#    ``Forward Time (ms)`` / ``Backward Time (ms)``, plus shape
#    columns ``B, M, N, K, Case, Dtype, Granularity``. One row per
#    target shape (hundreds of rows for a full sweep).
# 2. Quick bench (``quick_command`` → ``quick_test_bench.py``) —
#    ``correct`` (True/False), ``fwd_tflops_mean``, ``fwd_tflops_std``,
#    ``bwd_tflops_mean``, ``bwd_tflops_std``, ``label, B, M, N, K``.
#    One row per representative shape.
#
# Both layouts describe the same underlying measurement. The scorer
# requires a single canonical schema so BASELINE (run via either
# harness) and the per-round VALIDATE (always run via the quick
# harness) compute the **same** aggregate over the **same** shapes.
# Without normalization the column lookup silently returns NaN for the
# quick schema, the per-shape regression gate never finds matching
# shapes, and the trend rows end up using different numbers across
# rounds — exactly the measurement-consistency failure described in
# ``docs/performance-measurement-confidence.md``.
#
# The rules below are intentionally narrow: they only rename columns
# for which the canonical name is absent, so a genuinely canonical CSV
# passes through unchanged and cannot accidentally be double-renamed.


_CANONICAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    # canonical_name: (alias1, alias2, ...)
    "Check": ("Check", "correct", "check", "check_pass", "correctness"),
    "Forward TFLOPS": (
        "Forward TFLOPS",
        "fwd_tflops_mean",
        "forward_tflops_mean",
        "fwd_tflops",
    ),
    "Backward TFLOPS": (
        "Backward TFLOPS",
        "bwd_tflops_mean",
        "backward_tflops_mean",
        "bwd_tflops",
    ),
    "Forward Time (ms)": (
        "Forward Time (ms)",
        "fwd_ms_mean",
        "forward_ms_mean",
        "fwd_ms",
    ),
    "Backward Time (ms)": (
        "Backward Time (ms)",
        "bwd_ms_mean",
        "backward_ms_mean",
        "bwd_ms",
    ),
}

_CANONICAL_STDDEV_ALIASES: dict[str, tuple[str, ...]] = {
    # canonical_metric -> alias columns that describe its stddev
    # (absolute units; the parser converts to percentage using the
    # row's mean). All aliases are appended with the canonical
    # ``_stddev`` suffix in the normalised row so the downstream
    # ``_stddev_header_for`` lookup finds them.
    "Forward TFLOPS": ("Forward TFLOPS_stddev", "fwd_tflops_std", "forward_tflops_std"),
    "Backward TFLOPS": ("Backward TFLOPS_stddev", "bwd_tflops_std", "backward_tflops_std"),
    "Forward Time (ms)": ("Forward Time (ms)_stddev", "fwd_ms_std", "forward_ms_std"),
    "Backward Time (ms)": ("Backward Time (ms)_stddev", "bwd_ms_std", "backward_ms_std"),
}


_BOOLEAN_PASS_VALUES = {"true", "1", "yes", "y", "pass", "passed", "ok"}
_BOOLEAN_FAIL_VALUES = {"false", "0", "no", "n", "fail", "failed"}


def _build_alias_rewrite(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return ``(old_to_new, canonical_headers)`` for column normalisation.

    * ``old_to_new`` maps every non-canonical alias to its canonical
      target, including stddev companions. Canonical columns map to
      themselves so the rewritten row dict preserves them as-is.
    * ``canonical_headers`` is the fully-rewritten header list used by
      the downstream shape-column / stddev-column detectors.

    Columns that are neither metric aliases nor canonical names (shape
    columns, ``repeats``, ``label``, ``TestID`` etc.) pass through
    unchanged.
    """
    present = set(headers)
    old_to_new: dict[str, str] = {}

    def _first_alias_in(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            if alias in present:
                return alias
        return None

    for canonical, aliases in _CANONICAL_COLUMN_ALIASES.items():
        if canonical in present:
            old_to_new[canonical] = canonical
            continue
        match = _first_alias_in(aliases)
        if match is not None:
            old_to_new[match] = canonical

    for canonical_metric, aliases in _CANONICAL_STDDEV_ALIASES.items():
        canonical_std = f"{canonical_metric}_stddev"
        if canonical_std in present:
            old_to_new[canonical_std] = canonical_std
            continue
        match = _first_alias_in(aliases)
        if match is not None:
            old_to_new[match] = canonical_std

    canonical_headers: list[str] = []
    for header in headers:
        canonical_headers.append(old_to_new.get(header, header))
    return old_to_new, canonical_headers


def _rewrite_row(raw: dict[str, Any], rewrite: dict[str, str]) -> dict[str, Any]:
    """Return a row dict whose keys are the canonical names."""
    if not rewrite:
        return dict(raw)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = rewrite.get(key, key)
        if canonical == "Check":
            out[canonical] = _normalise_check_value(value)
        else:
            out[canonical] = value
    return out


def _normalise_check_value(value: Any) -> str:
    """Map quick-bench ``correct=True/False`` values to the canonical
    ``PASS`` / ``FAIL`` strings the scorer expects.

    Non-string canonical values (``"PASS"`` / ``"FAIL"`` etc.) pass
    through untouched; unknown strings are upper-cased so downstream
    ``str(raw.get(CHECK_COL, "")).strip().upper()`` works identically
    to the unpatched path.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(value).strip()
    low = text.lower()
    if low in _BOOLEAN_PASS_VALUES:
        return "PASS"
    if low in _BOOLEAN_FAIL_VALUES:
        return "FAIL"
    return text


def parse_bench_csv(path: Path, primary_metric: str) -> BenchmarkParse:
    """Parse a benchmark CSV. Assumes headers plus one row per shape.

    The parser recognises optional companion columns ``<Metric>_stddev``,
    ``<Metric>_stddev_pct``, ``<Metric>_std``, ``<Metric>_std_pct``. When
    any of those is present, it is attached to
    ``ShapeResult.metrics_stddev_pct`` as a percentage — callers fall
    back to the hard-coded noise threshold only when no column matches.
    ``_stddev`` / ``_std`` columns are normalised into percentages using
    the metric's mean in the same row.

    To guarantee BASELINE and VALIDATE measure the *same* thing, the
    parser transparently normalises alternative column names emitted
    by ``quick_test_bench.py`` (``fwd_tflops_mean``, ``correct``,
    ``fwd_tflops_std`` ...) into the canonical Primus-Turbo schema
    (``Forward TFLOPS``, ``Check``, ``Forward TFLOPS_stddev`` ...).
    ``BenchmarkParse.raw_headers`` always reports the post-rewrite
    header list so downstream consumers can discover which columns the
    scorer actually saw. See :data:`_CANONICAL_COLUMN_ALIASES` for the
    full alias table.
    """
    metrics = split_primary_metric(primary_metric)
    if not path.exists():
        raise ScoringError(f"benchmark CSV not found at {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ScoringError(f"CSV has no header: {path}")
        raw_headers = list(reader.fieldnames)
        rewrite, canonical_headers = _build_alias_rewrite(raw_headers)
        shape_cols = _detect_shape_columns(canonical_headers, metrics)
        stddev_columns = {
            metric: _stddev_header_for(metric, canonical_headers)
            for metric in metrics
        }
        has_repeats_col = "repeats" in canonical_headers
        rows: list[ShapeResult] = []
        all_pass = True
        for raw in reader:
            row = _rewrite_row(raw, rewrite)
            metrics_in_row: dict[str, float] = {}
            stddev_in_row: dict[str, float] = {}
            for metric in metrics:
                mean_val = _parse_float(row.get(metric))
                metrics_in_row[metric] = mean_val
                std_col = stddev_columns.get(metric)
                if std_col is None:
                    continue
                std_val = _parse_float(row.get(std_col))
                if math.isnan(std_val):
                    continue
                if std_col.endswith(("_stddev", "_std")):
                    if mean_val and not math.isnan(mean_val) and mean_val > 0:
                        stddev_in_row[metric] = std_val / mean_val * 100.0
                else:
                    stddev_in_row[metric] = std_val
            check = str(row.get(CHECK_COL, "")).strip().upper() or "UNKNOWN"
            if check != "PASS":
                all_pass = False
            repeats = 1
            if has_repeats_col:
                try:
                    repeats = max(1, int(float(row.get("repeats", 1) or 1)))
                except (TypeError, ValueError):
                    repeats = 1
            rows.append(
                ShapeResult(
                    shape={col: row.get(col) for col in shape_cols},
                    check=check,
                    metrics=metrics_in_row,
                    metrics_stddev_pct=stddev_in_row,
                    repeats=repeats,
                )
            )
    return BenchmarkParse(
        primary_metric=metrics,
        rows=rows,
        all_pass=all_pass,
        raw_headers=canonical_headers,
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
    exclude |= {h for h in headers if _looks_like_noise_column(h)}
    return [h for h in headers if h not in exclude]


def _looks_like_metric(header: str) -> bool:
    keywords = ("TFLOPS", "TOPS", "GB/s", "ms", "latency", "Bandwidth")
    return any(k.lower() in header.lower() for k in keywords)


def _looks_like_noise_column(header: str) -> bool:
    """True for columns that vary run-to-run even for a fixed shape.

    The scorer builds per-row shape keys from the columns returned by
    :func:`_detect_shape_columns`. Correctness-signal columns (SNR
    values, random-seed diagnostics) must be excluded from that key
    set or round-N's shape keys never match round-1's — even though
    the underlying shape is identical. Matching by substrings here
    keeps the check permissive enough to absorb future column names
    (``out_snr_fwd`` / ``grad_snr_bwd`` ...).
    """
    low = header.lower()
    noise_markers = ("snr",)
    return any(marker in low for marker in noise_markers)


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


# Shape-identity helpers ------------------------------------------------
#
# BASELINE and VALIDATE are only comparable when they run the same
# shapes with the same schema. Each shape dict mixes semantic shape
# axes (``B, M, N, K``) with bookkeeping columns (``label``, ``Case``,
# ``TestID``, ``Platform``, ...). Matching by the full shape dict makes
# round-N reject shapes that BASELINE labelled with a different ``Case``
# name, even though the underlying ``B/M/N/K`` are identical. The
# geometry-only key below fixes that by keeping only the canonical
# numeric shape axes.


_SHAPE_AXIS_CANDIDATES: tuple[str, ...] = (
    "B",
    "M",
    "N",
    "K",
    "D",
    "H",
    "W",
    "seq_len",
    "num_heads",
    "num_kv_heads",
    "head_dim",
)


def _geometry_key(shape: dict[str, Any]) -> tuple:
    """Return a shape key based on numeric axes only.

    Uses :data:`_SHAPE_AXIS_CANDIDATES` as a priority list; missing
    axes are skipped so partial matches (``{M, N, K}`` when BASELINE
    only records those three) still work. Falls back to the full shape
    dict when none of the canonical axes are present — this preserves
    behaviour for exotic ops whose shape columns we don't yet know.
    """
    hits: list[tuple[str, str]] = []
    for axis in _SHAPE_AXIS_CANDIDATES:
        if axis in shape and shape[axis] is not None and shape[axis] != "":
            hits.append((axis, str(shape[axis])))
    if hits:
        return tuple(hits)
    return _shape_key(shape)


@dataclass
class ShapeConsistencyReport:
    """Outcome of comparing a candidate vector's shape set against the
    baseline.

    * ``consistent`` — True when every candidate shape has a matching
      geometry key in the baseline (extra baseline shapes are
      permitted; the baseline may run a broader sweep).
    * ``candidate_only`` / ``baseline_only`` — the geometry keys that
      appear on only one side. Rendered into the orchestrator warning
      so operators can see exactly which shapes drifted.
    * ``mismatch_reason`` — human-readable message when
      ``consistent=False``; empty when everything matches.
    """

    consistent: bool
    candidate_only: list[tuple]
    baseline_only: list[tuple]
    mismatch_reason: str = ""


def verify_shape_consistency(
    candidate: ScoreVector | None,
    baseline: ScoreVector | None,
) -> ShapeConsistencyReport:
    """Check whether ``candidate`` shapes are a subset of ``baseline``.

    Returns :class:`ShapeConsistencyReport`. A ``None`` vector on
    either side is treated as "unknown" and yields an inconclusive but
    consistent report — callers should only surface warnings when
    ``consistent=False``. Comparison uses :func:`_geometry_key` so a
    rename from ``DSV3-GateUP-sm`` (quick bench ``label``) to
    ``DeepSeek-V3-GateUP`` (full bench ``Case``) does not produce a
    false mismatch.
    """
    if candidate is None or baseline is None:
        return ShapeConsistencyReport(
            consistent=True,
            candidate_only=[],
            baseline_only=[],
            mismatch_reason="",
        )
    cand_keys = {_geometry_key(row.shape) for row in candidate.per_shape}
    base_keys = {_geometry_key(row.shape) for row in baseline.per_shape}
    cand_only = sorted(cand_keys - base_keys)
    base_only = sorted(base_keys - cand_keys)
    consistent = not cand_only
    reason = ""
    if not consistent:
        reason = (
            f"candidate contains {len(cand_only)} shape(s) absent from the "
            f"baseline — measurements are not directly comparable. "
            f"Unmatched candidate keys: {cand_only[:3]}"
            + (" ..." if len(cand_only) > 3 else "")
        )
    return ShapeConsistencyReport(
        consistent=consistent,
        candidate_only=cand_only,
        baseline_only=base_only,
        mismatch_reason=reason,
    )


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


# --- REVIEW signal extraction -----------------------------------------


def _classify_metric_axis(text: str) -> set[str]:
    """Return the set of canonical metric axes a free-text blob touches.

    Returns a subset of ``{"Forward", "Backward"}``. An empty set means
    the text carries no metric-direction keyword and callers should
    treat the axis as undetermined (never as "both").
    """
    low = text.lower()
    axes: set[str] = set()
    if any(k in low for k in _FWD_KEYWORDS):
        axes.add("Forward")
    if any(k in low for k in _BWD_KEYWORDS):
        axes.add("Backward")
    return axes


_PREDICTED_PCT_RE = re.compile(
    r"""
    (?:
        (?P<metric>forward|backward|fwd|bwd|step[_\s]geomean|geomean)
        [^+\-\d%]{0,20}
    )?
    (?P<sign>[+\-]?)
    (?P<value>\d+(?:\.\d+)?)
    \s*%
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_predicted_directions(hypothesis: dict[str, Any]) -> dict[str, float]:
    """Return ``{metric_axis: min_predicted_pct}`` parsed from hypothesis text.

    Scans ``expected_benefit`` first, then ``verification_signal``, then
    ``primary_hypothesis`` until a ``"+N%"`` pattern is found. If the
    surrounding context names a direction (``forward`` / ``backward``
    / ``step_geomean``), the percentage is attributed to that axis;
    otherwise it is attributed to every axis mentioned elsewhere in
    the hypothesis text (or dropped when no axis keyword is present).
    """
    out: dict[str, float] = {}
    sources = [
        str(hypothesis.get("expected_benefit") or ""),
        str(hypothesis.get("verification_signal") or ""),
        str(hypothesis.get("primary_hypothesis") or ""),
    ]
    text_axes = _classify_metric_axis(" ".join(sources))
    for text in sources:
        for m in _PREDICTED_PCT_RE.finditer(text):
            raw_metric = (m.group("metric") or "").lower()
            sign = m.group("sign") or ""
            value_str = m.group("value") or "0"
            try:
                value = float(value_str)
            except ValueError:
                continue
            if sign == "-":
                continue
            if raw_metric in ("forward", "fwd"):
                axes = {"Forward"}
            elif raw_metric in ("backward", "bwd"):
                axes = {"Backward"}
            elif raw_metric in ("step_geomean", "step geomean", "geomean"):
                axes = {"Forward", "Backward"}
            else:
                axes = set(text_axes) or {"Forward", "Backward"}
            for axis in axes:
                if value > 0 and (axis not in out or value < out[axis]):
                    out[axis] = value
    return out


def _classify_modified_paths(modified_files: Iterable[str]) -> set[str]:
    """Infer which kernel axis the OPTIMIZE diff touched.

    Returns a subset of ``{"Forward", "Backward"}``. Purely lexical —
    the heuristic cannot tell a fwd-kernel edit apart from a bwd-kernel
    edit in the same file, so a path that names neither axis is
    classified as ``{"Forward", "Backward"}`` (i.e. undetermined,
    treated by :func:`compute_review_signals` as "could have touched
    either").
    """
    axes: set[str] = set()
    for raw in modified_files:
        if not raw:
            continue
        hits = _classify_metric_axis(str(raw))
        axes |= hits
    if not axes:
        return {"Forward", "Backward"}
    return axes


def _as_canonical_metric(axis: str) -> str:
    if axis == "Forward":
        return "Forward TFLOPS"
    if axis == "Backward":
        return "Backward TFLOPS"
    return axis


def _metric_stddev_pct(vector: ScoreVector | None, metric: str) -> float:
    """Return the geometric-mean stddev% for ``metric`` across PASS shapes.

    Geomean (instead of max) is used because REVIEW's noise-band check
    is interested in the typical measurement noise rather than the
    worst outlier; :func:`observed_noise_pct` already reports the max
    for the accept/rollback decision.
    """
    if vector is None:
        return 0.0
    values: list[float] = []
    for row in vector.per_shape:
        if row.check != "PASS":
            continue
        std = row.metrics_stddev_pct.get(metric)
        if std is None or math.isnan(std) or std <= 0:
            continue
        values.append(float(std))
    if not values:
        return 0.0
    return geomean(values)


def _claims_numerical_equivalence(hypothesis: dict[str, Any]) -> bool:
    """Heuristic: True if the hypothesis asserts bit-identical / numerically
    equivalent output.

    Matches common wordings used in the skill prompt ("bit-identical",
    "numerically equivalent", "same output", "no numerical change").
    False when the hypothesis explicitly allows numerical drift
    ("within SNR threshold", "rtol", "absolute tolerance").
    """
    blob = " ".join(
        str(hypothesis.get(k) or "")
        for k in ("primary_hypothesis", "verification_signal", "expected_benefit")
    ).lower()
    positive = (
        "bit-identical",
        "bit identical",
        "numerically equivalent",
        "same output",
        "no numerical change",
        "preserves numerics",
        "algebraically equivalent",
    )
    negative = (
        "within snr",
        "rtol",
        "atol",
        "absolute tolerance",
        "numerical drift",
    )
    if any(tok in blob for tok in negative):
        return False
    return any(tok in blob for tok in positive)


def _min_snr_from_score_vector(score_vector_rows: list[dict[str, Any]]) -> float | None:
    """Return the worst SNR (dB) across ``out_snr`` / ``da_snr`` / ``db_snr``
    columns in the VALIDATE score-vector payload.

    VALIDATE serialises the CSV rows as the ``score_vector`` list on its
    phase_result; the per-shape dicts carry the raw SNR floats so the
    review gate can assert bit-identity without re-reading the CSV.
    Rows without any SNR column (rowwise / blockwise kernels) return
    ``None``.
    """
    if not score_vector_rows:
        return None
    snr_keys = ("out_snr", "da_snr", "db_snr")
    values: list[float] = []
    for row in score_vector_rows:
        if not isinstance(row, dict):
            continue
        for key in snr_keys:
            raw = row.get(key)
            if raw is None or raw == "":
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
    if not values:
        return None
    return min(values)


def compute_review_signals(
    *,
    hypothesis: dict[str, Any],
    opt_result: dict[str, Any],
    quick_val_result: dict[str, Any] | None,
    full_val_result: dict[str, Any] | None,
    quick_candidate: ScoreVector | None,
    full_candidate: ScoreVector | None,
    best: ScoreVector | None,
    primary_metric: str,
    mode: str = "tolerant",
) -> ReviewBundle:
    """Run the five hard rules of the REVIEW phase on structured inputs.

    The caller passes the ScoreVector objects already parsed from the
    quick and (optionally) full CSVs, plus the raw VALIDATE phase
    results for SNR / correctness extraction. The function returns a
    :class:`ReviewBundle` with one :class:`ReviewSignal` per rule and a
    preliminary ``tolerant_verdict`` computed entirely from the
    ``"block"`` / ``"warn"`` severity pattern:

    * ``ESCALATE_HUMAN``  — correctness-bit-identity block (hypothesis
      claimed numerical equivalence but measured SNR dropped).
    * ``DOWNGRADE_TO_ROLLBACK`` — hypothesis-metric-alignment AND
      off-target-gain both block (strong evidence the measured gain is
      not the one the hypothesis predicts).
    * ``DOWNGRADE_TO_NOISE_BOUND`` — exactly one of the two
      (alignment / off-target) blocks, or the decision's improvement
      sits inside the noise band even without a blocking signal.
    * ``AGREE`` — no blocking signals; any warn-severity findings are
      forwarded for information only.

    ``mode`` currently accepts ``"tolerant"`` only; strict mode would
    promote the two ``"warn"`` signals to ``"block"``.
    """
    if mode != "tolerant":
        raise ScoringError(f"only review mode 'tolerant' is implemented (got {mode!r})")

    metrics = split_primary_metric(primary_metric)
    improvement = compare_score(
        quick_candidate.aggregate if quick_candidate else {},
        best.aggregate if best else {},
    )

    align = _signal_hypothesis_metric_alignment(
        hypothesis=hypothesis,
        improvement=improvement,
        quick_candidate=quick_candidate,
    )
    off_target = _signal_off_target_gain(
        opt_result=opt_result,
        improvement=improvement,
    )
    quick_full = _signal_quick_vs_full_agreement(
        quick_candidate=quick_candidate,
        full_candidate=full_candidate,
        best=best,
        metrics=metrics,
    )
    noise = _signal_noise_band(
        improvement=improvement,
        quick_candidate=quick_candidate,
        metrics=metrics,
    )
    correctness = _signal_correctness_bit_identity(
        hypothesis=hypothesis,
        quick_val_result=quick_val_result,
        full_val_result=full_val_result,
    )

    signals = [align, off_target, quick_full, noise, correctness]
    verdict, reason = _tolerant_verdict(signals)
    return ReviewBundle(
        signals=signals,
        tolerant_verdict=verdict,
        tolerant_reason=reason,
    )


def _signal_hypothesis_metric_alignment(
    *,
    hypothesis: dict[str, Any],
    improvement: dict[str, float],
    quick_candidate: ScoreVector | None,
) -> ReviewSignal:
    predicted = _extract_predicted_directions(hypothesis)
    if not predicted:
        return ReviewSignal(
            name="hypothesis_metric_alignment",
            passed=True,
            severity="info",
            details={
                "predicted": {},
                "observed_pct": dict(improvement),
            },
            note=(
                "hypothesis did not name a target metric axis with a "
                "quantified delta; skipping alignment check"
            ),
        )

    misaligned: list[dict[str, Any]] = []
    for axis, predicted_pct in predicted.items():
        metric = _as_canonical_metric(axis)
        observed = improvement.get(metric, 0.0)
        stddev_pct = _metric_stddev_pct(quick_candidate, metric)
        sigma_ratio = observed / stddev_pct if stddev_pct > 0 else float("inf")
        ok = observed >= min(predicted_pct * 0.25, predicted_pct - 1.0)
        ok = ok and observed > 0
        ok = ok and sigma_ratio >= 2.0
        if not ok:
            misaligned.append(
                {
                    "metric": metric,
                    "predicted_pct": predicted_pct,
                    "observed_pct": observed,
                    "stddev_pct": stddev_pct,
                    "sigma_ratio": sigma_ratio,
                }
            )
    passed = not misaligned
    severity = "info" if passed else "block"
    note = (
        "predicted-axis gains are within 25% of the predicted magnitude "
        "and above 2σ of measurement noise"
        if passed
        else "predicted-axis gain is much smaller than the hypothesis claimed, "
        "or lives inside the measurement noise band"
    )
    return ReviewSignal(
        name="hypothesis_metric_alignment",
        passed=passed,
        severity=severity,
        details={
            "predicted_pct": predicted,
            "observed_pct": dict(improvement),
            "misaligned": misaligned,
        },
        note=note,
    )


def _signal_off_target_gain(
    *,
    opt_result: dict[str, Any],
    improvement: dict[str, float],
) -> ReviewSignal:
    modified_files = opt_result.get("modified_files") or []
    modified_axes = _classify_modified_paths(str(p) for p in modified_files)
    fwd_gain = improvement.get("Forward TFLOPS", 0.0)
    bwd_gain = improvement.get("Backward TFLOPS", 0.0)

    if modified_axes == {"Forward", "Backward"}:
        return ReviewSignal(
            name="off_target_gain",
            passed=True,
            severity="info",
            details={
                "modified_files": list(modified_files),
                "modified_axes": sorted(modified_axes),
                "fwd_gain_pct": fwd_gain,
                "bwd_gain_pct": bwd_gain,
            },
            note=(
                "modified files name neither fwd nor bwd exclusively; "
                "cannot attribute gain to a specific axis"
            ),
        )

    dominant_axis = "Forward" if fwd_gain >= bwd_gain else "Backward"
    passed = dominant_axis in modified_axes
    severity = "info" if passed else "block"
    note = (
        f"dominant gain axis ({dominant_axis}) is contained in the "
        "modified-path set"
        if passed
        else (
            f"dominant gain axis ({dominant_axis}) is NOT in the modified-path "
            f"set {sorted(modified_axes)} — the observed improvement is "
            "likely from a codepath the hypothesis did not touch"
        )
    )
    return ReviewSignal(
        name="off_target_gain",
        passed=passed,
        severity=severity,
        details={
            "modified_files": list(modified_files),
            "modified_axes": sorted(modified_axes),
            "dominant_axis": dominant_axis,
            "fwd_gain_pct": fwd_gain,
            "bwd_gain_pct": bwd_gain,
        },
        note=note,
    )


def _signal_quick_vs_full_agreement(
    *,
    quick_candidate: ScoreVector | None,
    full_candidate: ScoreVector | None,
    best: ScoreVector | None,
    metrics: list[str],
) -> ReviewSignal:
    if full_candidate is None or quick_candidate is None or best is None:
        return ReviewSignal(
            name="quick_vs_full_agreement",
            passed=True,
            severity="info",
            details={"full_available": full_candidate is not None},
            note="full-sweep CSV not provided; skipping quick-vs-full comparison",
        )
    quick_delta = compare_score(quick_candidate.aggregate, best.aggregate)
    full_delta = compare_score(full_candidate.aggregate, best.aggregate)
    sign_flips: list[str] = []
    for metric in metrics:
        q = quick_delta.get(metric, 0.0)
        f = full_delta.get(metric, 0.0)
        if q > 0 and f < 0:
            sign_flips.append(metric)
        elif q < 0 and f > 0:
            sign_flips.append(metric)
    passed = not sign_flips
    severity = "info" if passed else "warn"
    note = (
        "quick and full sweeps agree on the sign of every primary metric"
        if passed
        else (
            f"quick and full sweeps disagree on the sign of: "
            f"{', '.join(sign_flips)}; one of the two harnesses is sampling "
            "a non-representative shape set"
        )
    )
    return ReviewSignal(
        name="quick_vs_full_agreement",
        passed=passed,
        severity=severity,
        details={
            "full_available": True,
            "quick_delta_pct": quick_delta,
            "full_delta_pct": full_delta,
            "sign_flips": sign_flips,
        },
        note=note,
    )


def _signal_noise_band(
    *,
    improvement: dict[str, float],
    quick_candidate: ScoreVector | None,
    metrics: list[str],
) -> ReviewSignal:
    worst_sigma = float("inf")
    worst_metric = ""
    worst_stddev = 0.0
    worst_gain = 0.0
    for metric in metrics:
        gain = improvement.get(metric, 0.0)
        if gain <= 0:
            continue
        stddev = _metric_stddev_pct(quick_candidate, metric)
        sigma_ratio = gain / stddev if stddev > 0 else float("inf")
        if sigma_ratio < worst_sigma:
            worst_sigma = sigma_ratio
            worst_metric = metric
            worst_stddev = stddev
            worst_gain = gain
    if worst_metric == "":
        return ReviewSignal(
            name="noise_band",
            passed=True,
            severity="info",
            details={"reason": "no positive improvement to evaluate"},
            note="no metric improved; noise-band check skipped",
        )
    passed = worst_sigma >= 2.0 or worst_gain >= NOISE_THRESHOLD_PCT
    severity = "info" if passed else "warn"
    note = (
        f"{worst_metric} gain ({worst_gain:.2f}%) is {worst_sigma:.1f}σ above "
        f"stddev {worst_stddev:.2f}% and clears the {NOISE_THRESHOLD_PCT}% "
        "absolute noise band"
        if passed
        else (
            f"{worst_metric} gain ({worst_gain:.2f}%) is only {worst_sigma:.1f}σ "
            f"above stddev {worst_stddev:.2f}% and below the "
            f"{NOISE_THRESHOLD_PCT}% absolute noise band"
        )
    )
    return ReviewSignal(
        name="noise_band",
        passed=passed,
        severity=severity,
        details={
            "worst_metric": worst_metric,
            "worst_gain_pct": worst_gain,
            "worst_stddev_pct": worst_stddev,
            "sigma_ratio": worst_sigma,
            "absolute_threshold_pct": NOISE_THRESHOLD_PCT,
        },
        note=note,
    )


def _signal_correctness_bit_identity(
    *,
    hypothesis: dict[str, Any],
    quick_val_result: dict[str, Any] | None,
    full_val_result: dict[str, Any] | None,
) -> ReviewSignal:
    claims_equivalent = _claims_numerical_equivalence(hypothesis)
    min_snr = None
    score_vectors: list[list[dict[str, Any]]] = []
    for val in (quick_val_result, full_val_result):
        if not isinstance(val, dict):
            continue
        rows = val.get("score_vector")
        if isinstance(rows, list):
            score_vectors.append(rows)
    for rows in score_vectors:
        candidate_min = _min_snr_from_score_vector(rows)
        if candidate_min is None:
            continue
        min_snr = candidate_min if min_snr is None else min(min_snr, candidate_min)

    if min_snr is None:
        return ReviewSignal(
            name="correctness_bit_identity",
            passed=True,
            severity="info",
            details={
                "claims_equivalent": claims_equivalent,
                "min_snr_db": None,
            },
            note=(
                "no SNR columns in VALIDATE output (kernel has no per-output "
                "correctness signal); cannot verify bit-identity claim"
            ),
        )

    bit_identity_threshold_db = 80.0
    passed = (not claims_equivalent) or min_snr >= bit_identity_threshold_db
    severity = "info" if passed else "block"
    note = (
        f"observed worst SNR {min_snr:.1f} dB — "
        + (
            "hypothesis does not claim numerical equivalence, so any finite "
            "SNR is acceptable"
            if not claims_equivalent
            else f">= {bit_identity_threshold_db} dB threshold; claim holds"
        )
        if passed
        else (
            f"hypothesis claims numerical equivalence but worst SNR "
            f"{min_snr:.1f} dB is below the {bit_identity_threshold_db} dB "
            "bit-identity threshold"
        )
    )
    return ReviewSignal(
        name="correctness_bit_identity",
        passed=passed,
        severity=severity,
        details={
            "claims_equivalent": claims_equivalent,
            "min_snr_db": min_snr,
            "threshold_db": bit_identity_threshold_db,
        },
        note=note,
    )


def _tolerant_verdict(signals: list[ReviewSignal]) -> tuple[str, str]:
    def _find(name: str) -> ReviewSignal | None:
        for s in signals:
            if s.name == name:
                return s
        return None

    corr = _find("correctness_bit_identity")
    align = _find("hypothesis_metric_alignment")
    off_target = _find("off_target_gain")
    noise = _find("noise_band")

    if corr is not None and corr.severity == "block":
        return (
            REVIEW_VERDICT_ESCALATE_HUMAN,
            f"correctness_bit_identity blocked: {corr.note}",
        )

    align_block = align is not None and align.severity == "block"
    off_block = off_target is not None and off_target.severity == "block"

    if align_block and off_block:
        return (
            REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK,
            (
                "hypothesis_metric_alignment AND off_target_gain both blocked — "
                "the measured gain does not line up with the hypothesis and "
                "does not come from the modified codepath"
            ),
        )
    if align_block:
        return (
            REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND,
            f"hypothesis_metric_alignment blocked: {align.note if align else ''}",
        )
    if off_block:
        return (
            REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND,
            f"off_target_gain blocked: {off_target.note if off_target else ''}",
        )
    if noise is not None and not noise.passed:
        return (
            REVIEW_VERDICT_AGREE,
            f"noise_band warning tolerated in tolerant mode: {noise.note}",
        )
    return (REVIEW_VERDICT_AGREE, "no blocking signals")


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
