"""Index ``<campaign>/rounds/round-N/``.

PR-1: summary.md only.
PR-2: bench_shapes + artifacts directory listing.
PR-3: ``kernel_snapshot_dir`` detection.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from turbo_view.io.bench import parse_benchmark_csv
from turbo_view.io.markdown import render_markdown
from turbo_view.model import RoundBundle, ShapeRow

log = logging.getLogger(__name__)

_ROUND_RE = re.compile(r"^round-(\d+)$")


def _read_summary_html(round_dir: Path) -> str | None:
    summary = round_dir / "summary.md"
    if not summary.is_file():
        return None
    try:
        text = summary.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("failed to read %s: %s", summary, exc)
        return None
    return render_markdown(text)


def _list_artifacts(round_dir: Path) -> list[Path]:
    """List ``round_dir/artifacts/*`` files (one level, no recursion).

    Used to expose round-level files (``benchmark.csv``,
    ``benchmark_compact.md`` …) in the table-row expansion.
    """
    art = round_dir / "artifacts"
    if not art.is_dir():
        return []
    return sorted(p for p in art.iterdir() if p.is_file())


def _load_bench_shapes(round_dir: Path) -> list[ShapeRow]:
    return parse_benchmark_csv(round_dir / "artifacts" / "benchmark.csv")


def _detect_kernel_snapshot_dir(round_dir: Path) -> Path | None:
    """``rounds/round-N/kernel_snapshot/`` if present (PR-3 territory)."""
    cand = round_dir / "kernel_snapshot"
    return cand if cand.is_dir() else None


def load_rounds(campaign_dir: Path) -> dict[int, RoundBundle]:
    root = campaign_dir / "rounds"
    if not root.is_dir():
        return {}
    out: dict[int, RoundBundle] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        m = _ROUND_RE.match(entry.name)
        if not m:
            continue
        n = int(m.group(1))
        out[n] = RoundBundle(
            n=n,
            summary_md_html=_read_summary_html(entry),
            bench_shapes=_load_bench_shapes(entry),
            artifacts=_list_artifacts(entry),
            kernel_snapshot_dir=_detect_kernel_snapshot_dir(entry),
        )
    return out
