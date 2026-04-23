"""Append-only writers for `logs/optimize.md` and `logs/performance_trend.md`.

Hard constraint from `iteration_rules.mdc` Rule 8:
    "Do not delete, truncate, rewrite, or replace existing content in
    either file. Only append new sections, rows, or correction notes."

This module only exposes `append_*` helpers. It never overwrites. The
structured history extractor `extract_history` parses the current file
contents into a Python dict so that ANALYZE prompts can embed verified
ineffective directions, directions-to-try, the history table, and the
current best — no Claude markdown parsing required.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


OPTIMIZE_LOG = "logs/optimize.md"
PERF_TREND = "logs/performance_trend.md"
COST_LOG = "logs/cost.md"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _now_seconds() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_file(path: Path, header: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")


def _append(path: Path, text: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} must be initialized with a header before appending"
        )
    with path.open("a", encoding="utf-8") as f:
        if not text.endswith("\n"):
            text += "\n"
        f.write(text)


# --- section-scoped editors --------------------------------------------
#
# ``optimize.md`` is seeded from :data:`OPTIMIZE_HEADER_TEMPLATE` with a
# fixed set of ``## <section>`` headings (Baseline / Optimization History
# / Current Best / Directions to Try / Verified Ineffective Directions /
# Final Report). Early versions of the helpers below used plain
# :func:`_append`, which dropped every new entry at EOF and left the
# template sections empty. Readers and dedup parsers that look inside
# specific headings therefore saw nothing. The editors below locate the
# owning heading and insert / upsert into its body, which both preserves
# Rule 8's append-only intent inside each section and keeps the
# rendered file structured.
#
# ``_split_section`` is the only place that has to know how a section
# ends: we stop at the next ``## `` (two hashes + space) heading, which
# skips ``### round-N`` and ``### Baseline Entry`` sub-headings.


def _split_section(text: str, header: str) -> tuple[str, str, str]:
    """Locate ``header`` and return ``(prefix, body, suffix)``.

    * ``prefix`` ends at (and includes) the header line.
    * ``body`` is everything between the header and the next ``## ``
      heading — or EOF if this is the last top-level section.
    * ``suffix`` starts at the next ``## `` heading (possibly empty).

    Raises :class:`KeyError` when the exact header line is missing so
    callers can fall back to EOF appends for old logs that predate the
    structured template.
    """
    lines = text.splitlines(keepends=True)
    start: int | None = None
    for i, line in enumerate(lines):
        if line.rstrip("\r\n") == header:
            start = i
            break
    if start is None:
        raise KeyError(header)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line.startswith("## ") and not line.startswith("### "):
            end = j
            break
    prefix = "".join(lines[: start + 1])
    body = "".join(lines[start + 1 : end])
    suffix = "".join(lines[end:])
    return prefix, body, suffix


def _normalize_body(body: str, *, has_suffix: bool) -> str:
    """Ensure the body ends with exactly one trailing blank line when it
    is followed by another ``## `` section, or with a single newline at
    EOF.
    """
    if not body:
        return "\n" if has_suffix else ""
    body = body.rstrip("\n") + "\n"
    if has_suffix:
        body += "\n"
    return body


def _strip_placeholder_lines(body: str, placeholders: set[str] | None) -> str:
    if not placeholders:
        return body
    kept: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.strip() in placeholders:
            continue
        kept.append(line)
    return "".join(kept)


def _upsert_section_body(
    path: Path,
    header: str,
    new_body: str,
    *,
    fallback_append: str | None = None,
) -> None:
    """Replace the body of ``header`` with ``new_body``.

    ``new_body`` should already contain trailing newlines; this helper
    only normalizes the padding before the following ``## `` heading so
    sections stay visually separated.
    """
    text = path.read_text(encoding="utf-8")
    try:
        prefix, _old, suffix = _split_section(text, header)
    except KeyError:
        if fallback_append is not None:
            _append(path, fallback_append)
        return
    body = _normalize_body(new_body, has_suffix=bool(suffix))
    path.write_text(prefix + body + suffix, encoding="utf-8")


def _append_in_section(
    path: Path,
    header: str,
    addition: str,
    *,
    placeholders: set[str] | None = None,
    fallback_append: str | None = None,
) -> None:
    """Append ``addition`` inside the body of ``header``.

    Lines in the existing body whose stripped form appears in
    ``placeholders`` are dropped first, so the template's
    ``_to be filled in…`` lines vanish the moment real content lands.
    """
    text = path.read_text(encoding="utf-8")
    try:
        prefix, body, suffix = _split_section(text, header)
    except KeyError:
        if fallback_append is not None:
            _append(path, fallback_append)
        else:
            _append(path, addition)
        return
    body = _strip_placeholder_lines(body, placeholders)
    if body and not body.endswith("\n"):
        body += "\n"
    combined = body + addition
    combined = _normalize_body(combined, has_suffix=bool(suffix))
    path.write_text(prefix + combined + suffix, encoding="utf-8")


# --- optimize.md --------------------------------------------------------

OPTIMIZE_HEADER_TEMPLATE = """# {target_op} {target_backend} Optimization Log

## Basic Information
- Target operator: {target_op}
- Implementation language: {target_lang}
- Backend: {target_backend}
- Target GPU: {target_gpu}
- Campaign: {campaign_dir}
- Start time: {start_time}
- Current status: Optimizing (round-0)

## Baseline
_to be filled in after round-1_

## Optimization History

## Current Best
_to be updated per accepted round_

## Directions to Try

## Verified Ineffective Directions
| Direction | Version | Failure Reason |
|-----------|---------|---------------|

## Final Report
_filled in when campaign terminates_
"""


def optimize_log_path(campaign_dir: Path) -> Path:
    return campaign_dir / OPTIMIZE_LOG


def init_optimize_log(campaign_dir: Path, params: dict[str, Any]) -> Path:
    path = optimize_log_path(campaign_dir)
    if path.exists():
        return path
    header = OPTIMIZE_HEADER_TEMPLATE.format(
        target_op=params.get("target_op", "<target_op>"),
        target_backend=params.get("target_backend", "<target_backend>"),
        target_lang=params.get("target_lang", "<target_lang>"),
        target_gpu=params.get("target_gpu", "<target_gpu>"),
        campaign_dir=str(campaign_dir),
        start_time=_now(),
    )
    _ensure_file(path, header)
    return path


BASELINE_PLACEHOLDER = "_to be filled in after round-1_"
CURRENT_BEST_PLACEHOLDER = "_to be updated per accepted round_"
FINAL_REPORT_PLACEHOLDER = "_filled in when campaign terminates_"


def append_baseline(
    campaign_dir: Path,
    *,
    backend: str,
    gpu: str,
    commit: str | None,
    aggregate_score: dict[str, float] | None,
    all_check_pass: bool,
    rounds_link: str = "rounds/round-1/summary.md",
    quick_baseline_log: str | None = None,
) -> None:
    """Upsert the ``## Baseline`` section body.

    BASELINE runs once per campaign but `_phase_baseline` can be
    re-entered (e.g. after resume) so we overwrite the body rather than
    append — this removes the ``_to be filled in after round-1_``
    placeholder on first write and stays idempotent on re-runs.

    ``quick_baseline_log`` is the campaign-relative path to the log
    produced by BASELINE step 6. Passing ``None`` or an empty string
    drops the ``Quick baseline log:`` line entirely so phases that
    bail before emitting the log still produce a well-formed block.
    """
    score_str = _fmt_scores(aggregate_score)
    lines = [
        f"### Baseline Entry ({_now()})",
        f"- Backend: {backend}",
        f"- GPU: {gpu}",
        f"- Commit: {commit or 'n/a'}",
        "- Validation level: full",
        f"- Aggregate score: {score_str}",
        f"- All Check: {'PASS' if all_check_pass else 'FAIL'}",
        f"- Detailed data: {rounds_link}",
    ]
    if quick_baseline_log:
        lines.append(f"- Quick baseline log: {quick_baseline_log}")
    new_body = "\n".join(lines) + "\n"
    fallback = "\n\n## Baseline\n" + new_body
    _upsert_section_body(
        optimize_log_path(campaign_dir),
        "## Baseline",
        new_body,
        fallback_append=fallback,
    )


def append_round_entry(
    campaign_dir: Path,
    *,
    round_n: int,
    description: str,
    validation_level: str,
    hypothesis: str,
    changes: str,
    aggregate_score_delta: str,
    test_result: str,
    decision: str,
    notes: str | None = None,
) -> None:
    """Insert a ``### round-N`` entry at the end of ``## Optimization
    History`` (not at EOF). Each round is an append because every round
    is unique; the helper falls back to EOF append when the heading is
    absent so pre-template logs keep working.
    """
    block = (
        f"### round-{round_n} — {description}\n"
        f"- Time: {_now()}\n"
        f"- Validation level: {validation_level}\n"
        f"- Hypothesis: {hypothesis}\n"
        f"- Changes: {changes}\n"
        f"- Result: {aggregate_score_delta}\n"
        f"- Test: {test_result}\n"
        f"- Decision: {decision}\n"
        f"- Detailed data: rounds/round-{round_n}/summary.md\n"
    )
    if notes:
        block += f"- Notes: {notes}\n"
    _append_in_section(
        optimize_log_path(campaign_dir),
        "## Optimization History",
        block,
        fallback_append="\n" + block,
    )


def upsert_directions_to_try(
    campaign_dir: Path,
    *,
    round_n: int,
    directions: list[dict[str, Any]],
) -> None:
    """Replace ``## Directions to Try`` with the STAGNATION_REVIEW pick.

    ``directions`` is the raw ``new_directions`` array from the phase
    result. Each entry needs at least a ``title`` field; ``category``
    and ``hypothesis`` are rendered when present. The checkbox style
    (``- [ ] ...``) matches the reference campaign so ``parse_directions
    _to_try`` continues to work unchanged.

    Skipping this call (e.g. when ``directions`` is empty) is a no-op
    so phases that decide "no new directions" don't wipe the section.
    """
    cleaned: list[str] = []
    for entry in directions or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        category = str(entry.get("category") or "").strip()
        hypothesis = str(entry.get("hypothesis") or "").strip()
        tail_parts: list[str] = []
        if category:
            tail_parts.append(f"[{category}]")
        if hypothesis:
            tail_parts.append(hypothesis)
        tail = " — " + " ".join(tail_parts) if tail_parts else ""
        cleaned.append(f"- [ ] {title}{tail}")
    if not cleaned:
        return
    caption = f"_Updated after round-{round_n} stagnation review ({_now()})_"
    new_body = caption + "\n\n" + "\n".join(cleaned) + "\n"
    fallback = "\n\n## Directions to Try\n" + new_body
    _upsert_section_body(
        optimize_log_path(campaign_dir),
        "## Directions to Try",
        new_body,
        fallback_append=fallback,
    )


def upsert_current_best(
    campaign_dir: Path,
    *,
    best_round: int | None,
    best_score: dict[str, float] | None,
    baseline_score: dict[str, float] | None,
) -> None:
    """Replace ``## Current Best`` with a three-column improvement table.

    Only metrics present in ``best_score`` are rendered. A metric with a
    ``None``/missing baseline keeps the baseline cell as ``-`` so the row
    still renders during resume-from-cache cases where the baseline has
    not yet been persisted into ``state.history``.
    """
    metrics = _current_best_rows(best_score, baseline_score)
    if not metrics:
        return
    header_lines = [
        "| Metric | Baseline | Current Best | Improvement |",
        "|---|---:|---:|---:|",
    ]
    round_caption = (
        f"_Updated after round-{best_round} ({_now()})_"
        if best_round is not None
        else f"_Updated ({_now()})_"
    )
    lines = [round_caption, ""] + header_lines + metrics
    new_body = "\n".join(lines) + "\n"
    fallback = "\n\n## Current Best\n" + new_body
    _upsert_section_body(
        optimize_log_path(campaign_dir),
        "## Current Best",
        new_body,
        fallback_append=fallback,
    )


def _current_best_rows(
    best_score: dict[str, float] | None,
    baseline_score: dict[str, float] | None,
) -> list[str]:
    if not best_score:
        return []
    baseline = baseline_score or {}
    out: list[str] = []
    for metric, cur in best_score.items():
        if cur is None:
            continue
        base = baseline.get(metric)
        base_cell = "-" if base is None else f"{float(base):.3f}"
        cur_cell = f"{float(cur):.3f}"
        improvement = "-"
        if base not in (None, 0):
            try:
                delta = (float(cur) - float(base)) / float(base) * 100.0
            except (TypeError, ValueError, ZeroDivisionError):
                delta = None
            if delta is not None:
                improvement = f"{delta:+.2f}%"
        out.append(
            f"| {metric} | {base_cell} | {cur_cell} | {improvement} |"
        )
    return out


VERIFIED_INEFFECTIVE_SIDECAR = "verified_ineffective.jsonl"


def append_verified_ineffective(
    campaign_dir: Path,
    *,
    round_n: int,
    direction: str,
    reason: str,
    modified_files: list[str] | None = None,
) -> None:
    """Insert a row under the ``## Verified Ineffective Directions`` table
    and mirror the structured payload into the JSONL sidecar.

    The markdown row stays schema-stable so existing parsers keep
    working; the sidecar is the source of truth for dedup because
    Jaccard overlap on file paths is a stronger signal than textual
    similarity of the hypothesis prose. When the section exists but the
    table header is missing we seed a fresh header so the inserted row
    does not land as orphan text.
    """
    row = f"| {direction} | round-{round_n} | {reason} |"
    path = optimize_log_path(campaign_dir)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        prefix, body, suffix = _split_section(
            text, "## Verified Ineffective Directions"
        )
    except KeyError:
        _append(path, row + "\n")
    else:
        new_body = _insert_into_ineffective_table(body, row)
        new_body = _normalize_body(new_body, has_suffix=bool(suffix))
        path.write_text(prefix + new_body + suffix, encoding="utf-8")

    sidecar = campaign_dir / VERIFIED_INEFFECTIVE_SIDECAR
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_n,
        "direction": direction,
        "reason": reason,
        "modified_files": list(modified_files or []),
    }
    with sidecar.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


_INEFFECTIVE_HEADER = "| Direction | Version | Failure Reason |"
_INEFFECTIVE_SEPARATOR = "|-----------|---------|----------------|"


def _insert_into_ineffective_table(body: str, row: str) -> str:
    """Insert ``row`` at the end of the existing table inside ``body``.

    If the body has no table at all we re-create the canonical header +
    separator from the template and place the row underneath, so the
    section stays readable even after manual edits.
    """
    lines = body.splitlines()
    has_header = any(l.strip().startswith("| Direction |") for l in lines)
    has_sep = any(l.strip().startswith("|--") for l in lines)
    if not has_header or not has_sep:
        return (
            _INEFFECTIVE_HEADER
            + "\n"
            + _INEFFECTIVE_SEPARATOR
            + "\n"
            + row
            + "\n"
        )
    insert_at = len(lines)
    while insert_at > 0 and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, row)
    return "\n".join(lines) + "\n"


def append_final_report(campaign_dir: Path, body: str) -> None:
    """Append the final-report block into the ``## Final Report`` section.

    The termination-check block (written separately) and the full report
    payload both live inside the same ``## Final Report`` section body.
    The ``_filled in when campaign terminates_`` placeholder is scrubbed
    on first write.
    """
    addition = body.rstrip() + "\n"
    fallback = "\n\n## Final Report\n" + addition
    _append_in_section(
        optimize_log_path(campaign_dir),
        "## Final Report",
        addition,
        placeholders={FINAL_REPORT_PLACEHOLDER},
        fallback_append=fallback,
    )


def append_termination_block(
    campaign_dir: Path,
    *,
    checks: dict[str, bool],
    passed: list[str],
) -> None:
    """Append the ``### Termination Check`` block into ``## Final Report``.

    Kept here (rather than inline in ``campaign.py``) so all writes to
    ``optimize.md`` go through the section-aware helpers and share
    placeholder-scrubbing semantics.
    """
    mapping = {
        "T1": "performance_target",
        "T2": "hardware efficiency",
        "T3": "max_iterations reached",
        "T4": "max_duration reached",
        "T5": "user requested stop",
    }
    lines = ["### Termination Check"]
    for key in ("T1", "T2", "T3", "T4", "T5"):
        marker = "PASS" if checks.get(key) else "no"
        lines.append(f"- {key} {mapping[key]}: {marker}")
    lines.append(f"-> Satisfied condition(s): {', '.join(passed)}")
    addition = "\n".join(lines) + "\n"
    fallback = "\n\n## Final Report\n" + addition
    _append_in_section(
        optimize_log_path(campaign_dir),
        "## Final Report",
        addition,
        placeholders={FINAL_REPORT_PLACEHOLDER},
        fallback_append=fallback,
    )


# --- performance_trend.md ----------------------------------------------

PERF_HEADER = (
    "# Performance Trend\n\n"
    "| Round | Status | Description "
    "| Fwd Avg TFLOPS | Fwd Peak TFLOPS | Bwd Avg TFLOPS | Bwd Peak TFLOPS "
    "| Step Geomean TFLOPS | vs Baseline | Key Finding |\n"
    "|-------|--------|-------------"
    "|----------------|-----------------|----------------|-----------------"
    "|---------------------|-------------|-------------|\n"
)


def performance_trend_path(campaign_dir: Path) -> Path:
    return campaign_dir / PERF_TREND


def init_performance_trend(campaign_dir: Path) -> Path:
    path = performance_trend_path(campaign_dir)
    if not path.exists():
        _ensure_file(path, PERF_HEADER)
    return path


def append_trend_row(
    campaign_dir: Path,
    *,
    round_n: int,
    status: str,
    description: str,
    fwd_avg: float | None,
    fwd_peak: float | None,
    bwd_avg: float | None,
    bwd_peak: float | None,
    step_geomean: float | None,
    vs_baseline: dict[str, float | None] | None,
    key_finding: str,
) -> None:
    """Append one row to `logs/performance_trend.md` per Rule 8.

    Columns:
      * `fwd_avg / fwd_peak / bwd_avg / bwd_peak` — per-direction aggregates
        and per-shape max (None → `-`).
      * `step_geomean` — `sqrt(fwd_avg * bwd_avg)`, or `fwd_avg` if the kernel
        has no backward path. None → `-`.
      * `vs_baseline` — a dict with three float|None keys, rendered as
        `step ±X%, fwd ±Y%, bwd ±Z%`. Pass `None` for the baseline row
        itself; it renders as `—`.
    """

    def _fmt_value(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    vs_base = _format_vs_baseline(vs_baseline)
    row = (
        f"| {round_n} | {status} | {description} "
        f"| {_fmt_value(fwd_avg)} | {_fmt_value(fwd_peak)} "
        f"| {_fmt_value(bwd_avg)} | {_fmt_value(bwd_peak)} "
        f"| {_fmt_value(step_geomean)} | {vs_base} | {key_finding} |\n"
    )
    _append(performance_trend_path(campaign_dir), row)


def _format_vs_baseline(
    vs_baseline: dict[str, float | None] | None,
) -> str:
    """Render the three-part vs Baseline cell.

    None or empty → `—`. Missing key within the dict → `<name> —`.
    """
    if not vs_baseline:
        return "—"
    parts: list[str] = []
    for key in ("step", "fwd", "bwd"):
        if key not in vs_baseline:
            continue
        value = vs_baseline[key]
        if value is None:
            parts.append(f"{key} —")
        else:
            parts.append(f"{key} {value:+.3f}%")
    return ", ".join(parts) if parts else "—"


# --- cost.md ------------------------------------------------------------

COST_HEADER = (
    "# Campaign Cost Log\n\n"
    "Append-only record of every ``run_phase`` invocation. One row per\n"
    "phase execution (including cached reuses). ``Cumulative USD`` runs\n"
    "across the whole campaign, across resumes.\n\n"
    "| Time | Phase | Round | Status | Wall s | SDK s | Turns "
    "| Cost USD | Cumulative USD |\n"
    "|------|-------|-------|--------|--------|-------|-------"
    "|----------|----------------|\n"
)


def cost_log_path(campaign_dir: Path) -> Path:
    return campaign_dir / COST_LOG


def init_cost_log(campaign_dir: Path) -> Path:
    path = cost_log_path(campaign_dir)
    if not path.exists():
        _ensure_file(path, COST_HEADER)
    return path


def append_cost_row(
    campaign_dir: Path,
    *,
    phase: str,
    round_n: int | None = None,
    status: str,
    wall_s: float,
    sdk_s: float | None = None,
    turns: int = 0,
    cost_usd: float = 0.0,
    phase_variant: str | None = None,
) -> float:
    """Append one row to ``logs/cost.md`` and return the new cumulative USD.

    ``status`` is passed through verbatim so callers can distinguish
    ``ok`` (Claude ran to completion), ``cached`` (run_phase reused a
    prior phase_result), ``interrupted`` (SIGINT), or
    ``error:<ExcName>``.

    ``phase_variant`` renders as ``PHASE (variant)``; used by VALIDATE
    to separate quick / full rows visually without inventing a new
    phase name.
    """
    path = cost_log_path(campaign_dir)
    if not path.exists():
        init_cost_log(campaign_dir)
    previous = _last_cumulative_cost(path)
    cumulative = previous + max(0.0, float(cost_usd))
    phase_label = f"{phase} ({phase_variant})" if phase_variant else phase
    round_cell = "-" if round_n is None else str(round_n)
    sdk_cell = "-" if sdk_s is None else f"{sdk_s:.1f}"
    row = (
        f"| {_now_seconds()} | {phase_label} | {round_cell} | {status} "
        f"| {wall_s:.1f} | {sdk_cell} | {turns} "
        f"| ${cost_usd:.4f} | ${cumulative:.4f} |\n"
    )
    _append(path, row)
    return cumulative


_COST_ROW_RE = re.compile(
    r"^\|[^|]*"                          # Time
    r"\|[^|]*"                          # Phase
    r"\|[^|]*"                          # Round
    r"\|[^|]*"                          # Status
    r"\|[^|]*"                          # Wall s
    r"\|[^|]*"                          # SDK s
    r"\|[^|]*"                          # Turns
    r"\|[^|]*"                          # Cost USD
    r"\|\s*\$(?P<cum>[\d.]+)\s*\|"      # Cumulative USD
)


def _last_cumulative_cost(path: Path) -> float:
    """Return the ``Cumulative USD`` value from the last data row.

    Zero when the file only contains the header, is empty, or is
    malformed. Parsing is deliberately permissive: missing columns,
    empty strings, and non-numeric cumulative values all fall back to
    ``0.0`` so a single bad row can't poison the rest of the log.
    """
    if not path.exists():
        return 0.0
    last_value = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").replace(":", "").strip()) <= {"-", " "}:
            continue
        m = _COST_ROW_RE.match(line)
        if not m:
            continue
        try:
            last_value = float(m.group("cum"))
        except ValueError:
            continue
    return last_value


# --- history extraction -------------------------------------------------

@dataclass
class IneffectiveDirection:
    round: int
    direction: str
    reason: str
    modified_files: list[str] = field(default_factory=list)


@dataclass
class TrendRow:
    round: int
    status: str
    description: str
    fwd_avg: float | None
    fwd_peak: float | None
    bwd_avg: float | None
    bwd_peak: float | None
    step_geomean: float | None
    vs_baseline_step_pct: float | None
    vs_baseline_fwd_pct: float | None
    vs_baseline_bwd_pct: float | None
    key_finding: str

    @property
    def vs_baseline_pct(self) -> float | None:
        """Backward-compat alias: callers that only care about the overall
        step improvement can still read ``row.vs_baseline_pct``."""
        return self.vs_baseline_step_pct


@dataclass
class History:
    current_best_round: int | None = None
    current_best_score: dict[str, float] = field(default_factory=dict)
    history_rows: list[TrendRow] = field(default_factory=list)
    verified_ineffective: list[IneffectiveDirection] = field(default_factory=list)
    directions_to_try: list[str] = field(default_factory=list)
    rollback_streak: int = 0
    total_rounds: int = 0
    accepted_rounds: int = 0

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "current_best_round": self.current_best_round,
            "current_best_score": self.current_best_score,
            "history_rows": [row.__dict__ for row in self.history_rows],
            "verified_ineffective": [d.__dict__ for d in self.verified_ineffective],
            "directions_to_try": list(self.directions_to_try),
            "rollback_streak": self.rollback_streak,
            "total_rounds": self.total_rounds,
            "accepted_rounds": self.accepted_rounds,
        }


_NUM_CELL = r"[\d.\-]+|-|—"
_TREND_ROW_RE = re.compile(
    r"^\|\s*(?P<round>\d+)\s*"
    r"\|\s*(?P<status>[A-Z_/ ]+)\s*"
    r"\|\s*(?P<desc>[^|]*)"
    rf"\|\s*(?P<fwd_avg>{_NUM_CELL})\s*"
    rf"\|\s*(?P<fwd_peak>{_NUM_CELL})\s*"
    rf"\|\s*(?P<bwd_avg>{_NUM_CELL})\s*"
    rf"\|\s*(?P<bwd_peak>{_NUM_CELL})\s*"
    rf"\|\s*(?P<step>{_NUM_CELL})\s*"
    r"\|\s*(?P<vs>[^|]*)"
    r"\|\s*(?P<finding>[^|]*)\|"
)

_VS_BASELINE_COMPONENT_RE = re.compile(
    r"(?P<key>step|fwd|bwd)\s*(?P<sign>[+-])?(?P<val>[\d.]+)%"
)


def parse_trend_rows(text: str) -> list[TrendRow]:
    rows: list[TrendRow] = []
    for line in text.splitlines():
        m = _TREND_ROW_RE.match(line.strip())
        if not m:
            continue
        try:
            round_n = int(m.group("round"))
        except ValueError:
            continue
        step_pct, fwd_pct, bwd_pct = _parse_vs_baseline(m.group("vs"))
        rows.append(
            TrendRow(
                round=round_n,
                status=m.group("status").strip(),
                description=m.group("desc").strip(),
                fwd_avg=_parse_float(m.group("fwd_avg")),
                fwd_peak=_parse_float(m.group("fwd_peak")),
                bwd_avg=_parse_float(m.group("bwd_avg")),
                bwd_peak=_parse_float(m.group("bwd_peak")),
                step_geomean=_parse_float(m.group("step")),
                vs_baseline_step_pct=step_pct,
                vs_baseline_fwd_pct=fwd_pct,
                vs_baseline_bwd_pct=bwd_pct,
                key_finding=m.group("finding").strip(),
            )
        )
    return rows


def _parse_vs_baseline(
    cell: str,
) -> tuple[float | None, float | None, float | None]:
    out: dict[str, float | None] = {"step": None, "fwd": None, "bwd": None}
    for m in _VS_BASELINE_COMPONENT_RE.finditer(cell):
        sign = -1.0 if m.group("sign") == "-" else 1.0
        try:
            out[m.group("key")] = sign * float(m.group("val"))
        except ValueError:
            continue
    return out["step"], out["fwd"], out["bwd"]


def _parse_float(s: str) -> float | None:
    s = s.strip()
    if s in ("", "-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pct(s: str) -> float | None:
    s = s.strip().rstrip("%")
    if s in ("", "—", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


_INEFFECTIVE_ROW_RE = re.compile(
    r"^\|\s*(?P<dir>[^|]+)\|\s*round-(?P<round>\d+)\s*\|\s*(?P<reason>[^|]+)\|"
)


def parse_verified_ineffective(text: str) -> list[IneffectiveDirection]:
    out: list[IneffectiveDirection] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Verified Ineffective Directions"):
            in_block = True
            continue
        if in_block and stripped.startswith("## "):
            break
        if not in_block:
            continue
        m = _INEFFECTIVE_ROW_RE.match(stripped)
        if not m:
            continue
        header_markers = ("direction", "---")
        direction = m.group("dir").strip()
        if direction.lower() in header_markers:
            continue
        try:
            round_n = int(m.group("round"))
        except ValueError:
            continue
        out.append(
            IneffectiveDirection(
                round=round_n,
                direction=direction,
                reason=m.group("reason").strip(),
            )
        )
    return out


def parse_directions_to_try(text: str) -> list[str]:
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Directions to Try"):
            in_block = True
            continue
        if in_block and stripped.startswith("## "):
            break
        if not in_block:
            continue
        m = re.match(r"^- \[( |x)\]\s+(.+)$", stripped)
        if m:
            checked = m.group(1) == "x"
            text_part = m.group(2)
            if checked:
                continue
            out.append(text_part)
    return out


def parse_current_best(text: str) -> tuple[int | None, dict[str, float]]:
    """Return `(best_round, {metric: value})` from the `Optimization History` block."""
    best_round: int | None = None
    best_score: dict[str, float] = {}
    history_iter = re.finditer(
        r"^### round-(?P<round>\d+)\s+—.*?(?=^### round-|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    for match in history_iter:
        block = match.group(0)
        if "Decision: accept" not in block.lower() and "decision: accepted" not in block.lower():
            continue
        try:
            round_n = int(match.group("round"))
        except ValueError:
            continue
        scores = _extract_scores_from_entry(block)
        if scores:
            best_round = round_n
            best_score = scores
    return best_round, best_score


_SCORE_RE = re.compile(r"(?P<metric>[A-Za-z][A-Za-z0-9_ ]+):\s*(?P<value>[\d.]+)")


def _extract_scores_from_entry(block: str) -> dict[str, float]:
    m = re.search(r"Result:\s*(?P<body>.+?)(?:\n|$)", block)
    if not m:
        return {}
    body = m.group("body")
    scores: dict[str, float] = {}
    for match in _SCORE_RE.finditer(body):
        try:
            scores[match.group("metric").strip()] = float(match.group("value"))
        except ValueError:
            continue
    return scores


def _merge_modified_files_sidecar(
    campaign_dir: Path, entries: list[IneffectiveDirection]
) -> list[IneffectiveDirection]:
    """Overlay ``modified_files`` from the JSONL sidecar onto parsed entries.

    Matching is done on ``(round, direction)``; entries without a
    matching sidecar row keep an empty ``modified_files`` list so older
    campaigns still work.
    """
    sidecar = campaign_dir / VERIFIED_INEFFECTIVE_SIDECAR
    if not sidecar.exists():
        return entries
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError:
        return entries
    lookup: dict[tuple[int, str], list[str]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (int(payload.get("round", 0) or 0), str(payload.get("direction", "")))
        files = payload.get("modified_files") or []
        if isinstance(files, list):
            lookup[key] = [str(f) for f in files]
    for entry in entries:
        files = lookup.get((entry.round, entry.direction))
        if files:
            entry.modified_files = files
    return entries


def extract_history(campaign_dir: Path) -> History:
    optimize_text = ""
    trend_text = ""
    p1 = optimize_log_path(campaign_dir)
    p2 = performance_trend_path(campaign_dir)
    if p1.exists():
        optimize_text = p1.read_text(encoding="utf-8")
    if p2.exists():
        trend_text = p2.read_text(encoding="utf-8")

    rows = parse_trend_rows(trend_text)
    best_round, best_score = parse_current_best(optimize_text)
    verified = parse_verified_ineffective(optimize_text)
    verified = _merge_modified_files_sidecar(campaign_dir, verified)
    directions = parse_directions_to_try(optimize_text)

    rollback_streak = 0
    for row in reversed(rows):
        if row.status.upper().startswith("ROLL"):
            rollback_streak += 1
        else:
            break

    accepted_rounds = sum(1 for r in rows if r.status.upper().startswith("ACC"))
    return History(
        current_best_round=best_round,
        current_best_score=best_score,
        history_rows=rows,
        verified_ineffective=verified,
        directions_to_try=directions,
        rollback_streak=rollback_streak,
        total_rounds=len(rows),
        accepted_rounds=accepted_rounds,
    )


def _fmt_scores(scores: dict[str, float] | None) -> str:
    if not scores:
        return "n/a"
    return ", ".join(f"{k}: {v:.3f}" for k, v in scores.items())
