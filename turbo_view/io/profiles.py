"""Index ``<campaign>/profiles/round-N_<flavor>/`` (spec §3.1 + §3.2).

Per directory we collect:

* ``profile_summary.md`` rendered to XSS-safe HTML
* the freshest ``rocprofv3/<host>/<pid>_kernel_trace.csv`` (one host
  policy per spec §3.2)
* an optional ``rocprofv3/<host>/<pid>_results.json`` path (passthrough
  for the Perfetto deep-link in panel P10)

A single round can have multiple flavours (``baseline`` /
``post_accept`` / ``pre_stagnation``). The loader returns a dict keyed
by round number; if multiple flavours exist for the same round, the
lexicographically smallest one wins (spec doesn't specify, this is
the predictable default).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from turbo_view.io.markdown import render_markdown
from turbo_view.model import KernelDispatch, ProfileBundle

log = logging.getLogger(__name__)

_DIR_RE = re.compile(r"^round-(\d+)_(.+)$")


def _kernel_dispatches(csv_path: Path) -> list[KernelDispatch]:
    if not csv_path.is_file():
        return []
    dispatches: list[KernelDispatch] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row.get("Kernel_Name"):
                    continue
                try:
                    start = int(row.get("Start_Timestamp", 0) or 0)
                    end = int(row.get("End_Timestamp", 0) or 0)
                except ValueError:
                    continue
                if end <= start:
                    continue
                dispatches.append(KernelDispatch(
                    name=row["Kernel_Name"],
                    start_ns=start,
                    end_ns=end,
                    vgpr=_iget(row, "VGPR_Count"),
                    sgpr=_iget(row, "SGPR_Count"),
                    lds_bytes=_iget(row, "LDS_Block_Size_Bytes", "LDS_Block_Size"),
                    scratch_bytes=_iget(row, "Scratch_Size"),
                    wg_x=_iget(row, "Workgroup_Size_X"),
                    grid_x=_iget(row, "Grid_Size_X"),
                ))
    except (OSError, csv.Error) as exc:
        log.warning("failed to read %s: %s", csv_path, exc)
        return []
    return dispatches


def _iget(row: dict, *keys: str) -> int:
    for k in keys:
        v = row.get(k)
        if v in (None, ""):
            continue
        try:
            return int(float(v))
        except (TypeError, ValueError):
            continue
    return 0


def _newest_kernel_trace(rocprof_dir: Path) -> Path | None:
    """``rocprofv3/<host>/<pid>_kernel_trace.csv`` newest by mtime."""
    if not rocprof_dir.is_dir():
        return None
    candidates = sorted(rocprof_dir.glob("*/*_kernel_trace.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _newest_results_json(rocprof_dir: Path) -> Path | None:
    if not rocprof_dir.is_dir():
        return None
    candidates = sorted(rocprof_dir.glob("*/*_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_summary_html(profile_dir: Path) -> str | None:
    summary = profile_dir / "profile_summary.md"
    if not summary.is_file():
        return None
    try:
        return render_markdown(summary.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("failed to read %s: %s", summary, exc)
        return None


def load_profiles(campaign_dir: Path) -> dict[int, ProfileBundle]:
    root = campaign_dir / "profiles"
    if not root.is_dir():
        return {}
    out: dict[int, ProfileBundle] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        m = _DIR_RE.match(entry.name)
        if not m:
            continue
        n, flavor = int(m.group(1)), m.group(2)
        rocprof = entry / "rocprofv3"
        bundle = ProfileBundle(
            round=n,
            flavor=flavor,
            summary_md_html=_read_summary_html(entry),
            dispatches=_kernel_dispatches(_newest_kernel_trace(rocprof) or Path("/dev/null")),
            perfetto_json_path=_newest_results_json(rocprof),
        )
        existing = out.get(n)
        if existing is None or bundle.flavor < existing.flavor:
            out[n] = bundle
    return out
