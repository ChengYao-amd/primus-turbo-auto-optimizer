"""Long-lived historical experience tips (`agent/historical_experience/.../tips.md`).

The path convention is:
    agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md

Appends use a simple lock file so multiple campaigns running in parallel
on the same host do not interleave partial writes.
"""

from __future__ import annotations

import fcntl
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from turbo_optimize.mcp import CampaignContext


TIPS_ROOT_REL = Path("agent") / "historical_experience"


def _tips_path(workspace_root: Path, gpu: str, op: str, backend: str) -> Path:
    return (
        workspace_root
        / TIPS_ROOT_REL
        / gpu
        / op
        / backend.lower()
        / "tips.md"
    )


def query_tips_impl(
    ctx: "CampaignContext",
    op: str | None,
    backend: str | None,
    gpu: str | None,
    keyword: str | None,
) -> dict[str, Any]:
    if not op or not backend or not gpu:
        return {
            "items": [],
            "note": "op/backend/gpu missing; campaign context did not supply them",
        }
    path = _tips_path(ctx.workspace_root, gpu, op, backend)
    if not path.exists():
        return {
            "path": str(path),
            "items": [],
            "note": "tips file does not exist yet",
        }
    text = path.read_text(encoding="utf-8")
    entries = _split_entries(text)
    if keyword:
        k = keyword.lower()
        entries = [e for e in entries if k in e["body"].lower()]
    return {
        "path": str(path),
        "items": entries,
        "count": len(entries),
    }


def append_tip_impl(
    ctx: "CampaignContext",
    op: str | None,
    backend: str | None,
    gpu: str | None,
    entry: dict[str, Any],
) -> dict[str, Any]:
    if not op or not backend or not gpu:
        raise ValueError("op / backend / gpu must all be provided (or in params)")
    path = _tips_path(ctx.workspace_root, gpu, op, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Historical Experience Tips\n\n"
            f"Operator: {op} — Backend: {backend} — GPU: {gpu}\n\n",
            encoding="utf-8",
        )
    block = _format_tip(entry)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(block)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return {"path": str(path), "ok": True}


def _format_tip(entry: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    round_n = entry.get("round")
    status = entry.get("status", "ACCEPTED")
    head = f"\n### {ts} round-{round_n} — {status}\n"
    fields = [
        ("Context", entry.get("context", "")),
        ("Signal", entry.get("signal", "")),
        ("Takeaway", entry.get("takeaway", "")),
        ("Applicability", entry.get("applicability", "")),
    ]
    body = "".join(f"- {label}: {value}\n" for label, value in fields if value)
    return head + body


_ENTRY_HEADER_RE = re.compile(r"^### .*$", re.MULTILINE)


def _split_entries(text: str) -> list[dict[str, Any]]:
    starts = [m.start() for m in _ENTRY_HEADER_RE.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    entries: list[dict[str, Any]] = []
    for s, e in zip(starts, starts[1:]):
        block = text[s:e].strip()
        head, _, body_rest = block.partition("\n")
        entries.append({"heading": head.lstrip("# ").strip(), "body": block})
    return entries
