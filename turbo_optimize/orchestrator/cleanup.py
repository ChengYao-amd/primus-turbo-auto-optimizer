"""Manual cleanup of stray files that leaked into ``workspace_root``.

Background: Claude sometimes writes auxiliary files (CSV dumps, kernel
copies with a missing trailing slash, ad-hoc logs) to the current
working directory during BASELINE / OPTIMIZE / VALIDATE. The prompt-side
``<workspace_hygiene>`` block is the primary defense, but real-world
runs have already produced stray artifacts (e.g.
``agent_workspace/Primus-Turbo/gemm_fp8_kernel.py``) and future runs can
still regress. This module is the paved path to reclaim them.

Strategy:

1. ``git status --porcelain=v1 -z`` inside ``workspace_root`` tells us
   exactly which files are untracked (those starting with ``??``).
2. Filter to *top-level* files (no ``/`` in the path). Stuff under
   ``primus_turbo/`` / ``benchmarks/`` / ``tests/`` is real source that
   the repo might legitimately evolve, even if untracked locally — we
   never touch it.
3. Move each stray file to
   ``<campaign_dir>/_stray/<timestamp>/<basename>`` so the campaign
   folder owns every artifact attributable to the run. Preserving the
   originals under ``_stray/`` (instead of deleting) keeps the cleanup
   reversible.

The function is pure Python; the CLI wrapper lives in ``cli.py`` as
``primus-turbo-optimize --cleanup-stray <campaign_id>``.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


log = logging.getLogger(__name__)


@dataclass
class StrayReport:
    workspace_root: Path
    campaign_dir: Path
    stray_files: list[Path]
    dest_dir: Path | None = None
    moved: list[Path] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.moved is None:
            self.moved = []


def discover_stray_files(workspace_root: Path) -> list[Path]:
    """List *top-level untracked* files in ``workspace_root``.

    Directories are ignored (we never recurse into untracked dirs —
    e.g. ``build/`` might legitimately exist for out-of-tree compilers).
    The result is stable and purely read-only: safe to call repeatedly
    for a dry-run preview.
    """
    if not (workspace_root / ".git").exists():
        raise RuntimeError(
            f"{workspace_root} is not a git worktree; cleanup needs git "
            "to classify tracked vs untracked files"
        )
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=str(workspace_root),
        check=True,
        capture_output=True,
        text=True,
    )
    stray: list[Path] = []
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        if not entry.startswith("?? "):
            continue
        relpath = entry[3:]
        if "/" in relpath or "\\" in relpath:
            continue
        candidate = workspace_root / relpath
        if candidate.is_file():
            stray.append(candidate)
    return sorted(stray)


def cleanup_stray_files(
    campaign_dir: Path,
    workspace_root: Path,
    *,
    apply: bool = False,
    timestamp: str | None = None,
) -> StrayReport:
    """Move top-level untracked files into the campaign's ``_stray/`` bucket.

    ``apply=False`` (the default) returns a report with the discovered
    files but does not touch anything — this is the "dry-run preview"
    mode surfaced to the CLI so users can review before committing.

    When ``apply=True``, files are moved (``Path.rename`` first, falling
    back to read+write+unlink if the two paths sit on different
    filesystems). The destination directory name is derived from
    ``timestamp`` (defaults to ``YYYYMMDD_HHMMSS`` at call time) so
    successive cleanups don't collide.
    """
    campaign_dir = Path(campaign_dir).resolve()
    workspace_root = Path(workspace_root).resolve()
    stray = discover_stray_files(workspace_root)
    report = StrayReport(
        workspace_root=workspace_root,
        campaign_dir=campaign_dir,
        stray_files=stray,
    )
    if not stray or not apply:
        return report

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = campaign_dir / "_stray" / ts
    dest_dir.mkdir(parents=True, exist_ok=True)
    report.dest_dir = dest_dir

    for src in stray:
        dest = dest_dir / src.name
        suffix = 1
        while dest.exists():
            dest = dest_dir / f"{src.stem}.{suffix}{src.suffix}"
            suffix += 1
        try:
            src.rename(dest)
        except OSError:
            dest.write_bytes(src.read_bytes())
            src.unlink()
        report.moved.append(dest)
        log.info("moved stray %s -> %s", src, dest)
    return report


def format_report(report: StrayReport) -> str:
    lines: list[str] = []
    if not report.stray_files:
        lines.append(
            f"workspace_root={report.workspace_root} is clean "
            "(no top-level untracked files)."
        )
        return "\n".join(lines)
    lines.append(
        f"Stray top-level untracked files in {report.workspace_root}:"
    )
    for f in report.stray_files:
        lines.append(f"  - {f.relative_to(report.workspace_root)}")
    if report.dest_dir is not None:
        lines.append(f"Moved {len(report.moved)} file(s) to {report.dest_dir}")
    else:
        lines.append(
            "Re-run with --apply to move these files under "
            f"{report.campaign_dir}/_stray/<timestamp>/"
        )
    return "\n".join(lines)
