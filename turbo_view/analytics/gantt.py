"""Gantt analytics (spec §6.1 panel 3).

Each ``CostRow`` becomes a horizontal block keyed off ``phase`` for
colouring. Status ``error:*`` / ``idle_timeout_*`` / ``wall_timeout``
flips a separate flag the front-end uses to render hatched fill.

Debug-event overlays (``transcripts`` ⇒ idle_timeout / retry /
wall_timeout / interrupted) land in PR-3; field reserved here as
``"events": []``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from turbo_view.model import CostRow, TranscriptEvent

_ABNORMAL_STATUS_PREFIXES = ("error", "idle_timeout", "wall_timeout", "interrupted")


def _is_abnormal(status: str) -> bool:
    return any(status.startswith(p) for p in _ABNORMAL_STATUS_PREFIXES)


def gantt_blocks(rows: list[CostRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        end = row.ts + timedelta(seconds=max(row.wall_s, 0.0))
        out.append({
            "phase": row.phase,
            "round": row.round,
            "status": row.status,
            "abnormal": _is_abnormal(row.status),
            "start_ts": row.ts.isoformat(),
            "end_ts": end.isoformat(),
            "dur_s": row.wall_s,
            "cost_usd": row.cost_usd,
            "turns": row.turns,
        })
    return out


def transcript_event_overlay(
    transcripts: dict[str, list[TranscriptEvent]],
) -> list[dict[str, Any]]:
    """Flatten debug events from every phase transcript.

    Empty until PR-3 wires transcript loading. Kept here so panel 3
    consumers can rely on ``gantt_panel.events`` always existing.
    """
    overlay: list[dict[str, Any]] = []
    for phase, events in transcripts.items():
        for ev in events:
            if ev.kind not in {"idle_timeout", "retry_attempt", "wall_timeout", "interrupted"}:
                continue
            overlay.append({
                "phase": phase,
                "kind": ev.kind,
                "ts": ev.ts.isoformat() if ev.ts is not None else None,
                "fields": dict(ev.fields),
            })
    return overlay


def gantt_panel(
    rows: list[CostRow],
    transcripts: dict[str, list[TranscriptEvent]] | None = None,
) -> dict[str, Any]:
    return {
        "blocks": gantt_blocks(rows),
        "events": transcript_event_overlay(transcripts or {}),
    }
