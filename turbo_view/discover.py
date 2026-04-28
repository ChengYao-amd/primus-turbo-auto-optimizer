"""Locate ``primus-turbo-optimize`` campaigns under a root directory.

Search policy (spec §3.2):

1. Look at ``<root>/state/*/run.json`` first; the matching campaign
   directory is ``<root>``. This is the canonical layout when the
   user points at a single workspace.
2. Otherwise walk ``<root>`` up to depth 4 looking for any
   ``state/<id>/run.json``. Each unique campaign root is returned
   exactly once.
3. A directory containing both ``state/<id>/run.json`` and
   ``manifest.yaml`` is treated as a valid campaign root even when
   only one of the two layouts matches.

The returned objects describe the campaign root directory and the
canonical campaign id (basename of the ``state/<id>`` folder).
``mtime`` (seconds since epoch) is included so the overview page can
render "started" / "last update" without re-reading ``run.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MAX_DEPTH = 4


@dataclass(slots=True, frozen=True)
class CampaignHandle:
    campaign_id: str
    campaign_dir: Path           # the directory containing ``state/<id>/``
    state_run_json: Path         # ``campaign_dir/state/<id>/run.json``
    mtime: float


def _make_handle(run_json: Path) -> CampaignHandle | None:
    """``run_json`` is ``<root>/state/<id>/run.json``; derive id + root."""
    state_dir = run_json.parent              # state/<id>
    if state_dir.parent.name != "state":
        return None
    campaign_dir = state_dir.parent.parent   # <root>
    try:
        st = run_json.stat()
    except OSError:
        return None
    return CampaignHandle(
        campaign_id=state_dir.name,
        campaign_dir=campaign_dir.resolve(),
        state_run_json=run_json.resolve(),
        mtime=st.st_mtime,
    )


def _run_json_parses(run_json: Path) -> bool:
    try:
        json.loads(run_json.read_text(encoding="utf-8"))
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _is_valid_campaign(root: Path) -> bool:
    """Filter pass: must look like a campaign workspace.

    A campaign is valid if it has either a ``manifest.yaml`` or a
    parseable ``run.json`` inside ``state/<id>/``.
    """
    if (root / "manifest.yaml").is_file():
        return True
    state = root / "state"
    if not state.is_dir():
        return False
    for child in state.iterdir():
        if not child.is_dir():
            continue
        rj = child / "run.json"
        if rj.is_file():
            try:
                json.loads(rj.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            return True
    return False


def _add_handle(
    handles: dict[Path, CampaignHandle], h: CampaignHandle, parses: bool,
) -> None:
    """Insert ``h`` into the dedup dict.

    When two state ids live under the same ``campaign_dir``, prefer
    the one whose ``run.json`` parses successfully; otherwise keep
    the earliest by id (deterministic tie-breaker).
    """
    existing = handles.get(h.campaign_dir)
    if existing is None:
        handles[h.campaign_dir] = h
        return
    if parses and not _run_json_parses(existing.state_run_json):
        handles[h.campaign_dir] = h
        return
    if existing.campaign_id > h.campaign_id and parses:
        handles[h.campaign_dir] = h


def discover_campaigns(root: Path) -> list[CampaignHandle]:
    root = root.resolve()
    if not root.is_dir():
        return []

    handles: dict[Path, CampaignHandle] = {}

    state_top = root / "state"
    if state_top.is_dir():
        for run_json in sorted(state_top.glob("*/run.json")):
            if not run_json.is_file():
                continue
            h = _make_handle(run_json)
            if h is None:
                continue
            if _is_valid_campaign(h.campaign_dir):
                _add_handle(handles, h, _run_json_parses(run_json))

    if not handles:
        for run_json in _walk_run_jsons(root, MAX_DEPTH):
            h = _make_handle(run_json)
            if h is None:
                continue
            if _is_valid_campaign(h.campaign_dir):
                _add_handle(handles, h, _run_json_parses(run_json))

    return sorted(handles.values(), key=lambda h: h.campaign_id)


def _walk_run_jsons(root: Path, max_depth: int):
    """Bounded BFS for ``state/<id>/run.json`` under ``root``."""
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        cur, depth = pending.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_file():
                if (entry.name == "run.json"
                        and entry.parent.parent.name == "state"):
                    yield entry
            elif entry.is_dir() and depth < max_depth:
                pending.append((entry, depth + 1))
