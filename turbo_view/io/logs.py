"""Pipe-table parsers for the three append-only markdown logs.

Owned by ``turbo_optimize.logs``; we re-implement read-side parsing
here to stay decoupled. Conventions:

* All parsers return ``[]`` / ``{}`` when the file is missing.
* Cells containing ``→`` (Unicode 2192) keep only the value after
  the arrow — matches the convention in
  ``workspace/.../logs/plot_trend.py``.
* Money cells (``$0.8695``) drop the ``$``.
* The ``Round`` column accepts ``-`` for no-round phases.
* ``Status`` is preserved verbatim — downstream colours / filters key
  off it.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from turbo_view.model import CostRow, PerfRow

log = logging.getLogger(__name__)

_ARROW = "→"


def _split_pipe_row(line: str) -> list[str] | None:
    """Return cells of a markdown pipe row or ``None`` if it isn't one."""
    s = line.strip()
    if not s.startswith("|") or not s.endswith("|"):
        return None
    parts = [c.strip() for c in s.strip("|").split("|")]
    return parts


def _is_separator(cells: list[str]) -> bool:
    return all(set(c) <= set("-:") and c for c in cells)


def _post_arrow(cell: str) -> str:
    if _ARROW in cell:
        return cell.split(_ARROW)[-1].strip()
    return cell.strip()


def _to_float(cell: str, default: float = 0.0) -> float:
    cell = _post_arrow(cell).lstrip("$").replace(",", "").strip()
    if not cell or cell in {"-", "—"}:
        return default
    try:
        return float(cell)
    except ValueError:
        return default


def _to_int(cell: str, default: int = 0) -> int:
    cell = _post_arrow(cell).strip()
    if not cell or cell in {"-", "—"}:
        return default
    try:
        return int(cell)
    except ValueError:
        return default


def _to_round(cell: str) -> int | None:
    cell = cell.strip()
    if not cell or cell in {"-", "—"}:
        return None
    try:
        return int(cell)
    except ValueError:
        return None


_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def _to_ts(cell: str) -> datetime | None:
    cell = cell.strip()
    if not _TS_RE.match(cell):
        return None
    try:
        return datetime.strptime(cell, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


_COST_HEADER = ("Time", "Phase", "Round", "Status", "Wall s", "SDK s",
                "Turns", "Cost USD", "Cumulative USD")


def parse_cost_md(path: Path) -> list[CostRow]:
    if not path.is_file():
        return []
    rows: list[CostRow] = []
    seen_header = False
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = _split_pipe_row(line)
        if cells is None or len(cells) < 9:
            continue
        if _is_separator(cells):
            continue
        if not seen_header:
            if tuple(cells[:9]) == _COST_HEADER:
                seen_header = True
            continue
        ts = _to_ts(cells[0])
        if ts is None:
            log.debug("skipping cost row with bad timestamp: %r", cells[0])
            continue
        rows.append(
            CostRow(
                ts=ts,
                phase=cells[1],
                round=_to_round(cells[2]),
                status=cells[3],
                wall_s=_to_float(cells[4]),
                sdk_s=_to_float(cells[5]),
                turns=_to_int(cells[6]),
                cost_usd=_to_float(cells[7]),
                cumulative_usd=_to_float(cells[8]),
            )
        )
    return rows


_PERF_HEADER = ("Round", "Status", "Description",
                "Fwd Avg TFLOPS", "Fwd Peak TFLOPS",
                "Bwd Avg TFLOPS", "Bwd Peak TFLOPS",
                "Step Geomean TFLOPS", "vs Baseline", "Key Finding")


def parse_perf_trend_md(path: Path) -> list[PerfRow]:
    if not path.is_file():
        return []
    rows: list[PerfRow] = []
    seen_header = False
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = _split_pipe_row(line)
        if cells is None or len(cells) < 10:
            continue
        if _is_separator(cells):
            continue
        if not seen_header:
            if tuple(cells[:10]) == _PERF_HEADER:
                seen_header = True
            continue
        try:
            round_n = int(cells[0].strip())
        except ValueError:
            continue
        rows.append(
            PerfRow(
                round=round_n,
                status=cells[1],
                description=cells[2],
                fwd_avg=_to_float(cells[3]),
                fwd_peak=_to_float(cells[4]),
                bwd_avg=_to_float(cells[5]),
                bwd_peak=_to_float(cells[6]),
                step_geomean=_to_float(cells[7]),
                vs_baseline=_post_arrow(cells[8]) or cells[8],
                key_finding=cells[9],
            )
        )
    return rows


_H2_RE = re.compile(r"^## (?!#)(.+?)\s*$")


def parse_optimize_md_sections(path: Path) -> dict[str, str]:
    """Split ``optimize.md`` into ``{heading_text: body}``.

    Splits on ``## ``-prefixed lines (level-2 headings only); ``### ``
    lines stay inside their parent body. Returns ``{}`` if the file is
    missing or empty.
    """
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _H2_RE.match(line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip("\n")
            current = m.group(1).strip()
            buf = []
        else:
            if current is not None:
                buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip("\n")
    return sections
