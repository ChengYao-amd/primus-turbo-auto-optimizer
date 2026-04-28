"""Read ``state/<id>/run.json`` produced by ``turbo_optimize``.

We cannot import ``turbo_optimize.state`` (strict isolation per spec
§10), so we re-parse the JSON. The schema is owned by
``turbo_optimize`` — graceful-degrade on any deviation: return ``None``
and let the caller display "N/A".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from turbo_view.model import HistoryEntry, RunState

log = logging.getLogger(__name__)


def _find_run_json(campaign_dir: Path) -> Path | None:
    """Locate ``run.json`` under ``state/*/run.json``.

    Picks the lexicographically first match — campaigns produced by
    ``turbo_optimize`` always have exactly one state subdir whose name
    matches ``campaign_id``.
    """
    state_root = campaign_dir / "state"
    if not state_root.is_dir():
        return None
    for sub in sorted(state_root.iterdir()):
        candidate = sub / "run.json"
        if candidate.is_file():
            return candidate
    return None


def load_run_state(campaign_dir: Path) -> RunState | None:
    path = _find_run_json(campaign_dir)
    if path is None:
        log.debug("no run.json under %s/state/*/", campaign_dir)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to parse %s: %s", path, exc)
        return None

    history = [
        HistoryEntry(
            round=int(h["round"]),
            decision=str(h["decision"]),
            score=dict(h.get("score") or {}),
            description=str(h.get("description", "")),
            at=str(h.get("at", "")),
        )
        for h in data.get("history", [])
    ]
    return RunState(
        campaign_id=str(data.get("campaign_id", campaign_dir.name)),
        campaign_dir=campaign_dir,
        current_phase=str(data.get("current_phase", "UNKNOWN")),
        current_round=int(data.get("current_round", 0)),
        best_round=(int(data["best_round"]) if data.get("best_round") is not None else None),
        best_score=dict(data.get("best_score") or {}),
        rollback_streak=int(data.get("rollback_streak", 0)),
        started_at=str(data.get("started_at", "")),
        last_update=str(data.get("last_update", "")),
        params=dict(data.get("params") or {}),
        history=history,
    )
