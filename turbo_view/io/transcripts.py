"""Stream-read ``<campaign>/profiles/_transcript_<phase>.jsonl``.

Each line is a JSON object; we tolerate occasional corrupt lines
(skip with a warning) so a single bad write doesn't take the whole
phase's overlay down.

PR-3 returns the parsed events grouped by phase. PR-5 piggybacks on
the same parser for live-tail.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from turbo_view.model import TranscriptEvent

log = logging.getLogger(__name__)

_FILE_RE = re.compile(r"^_transcript_(.+)\.jsonl$")


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Python 3.11+: fromisoformat handles trailing 'Z'.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_transcript_file(path: Path, phase: str) -> list[TranscriptEvent]:
    if not path.is_file():
        return []
    events: list[TranscriptEvent] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.warning("bad json at %s:%d: %s", path, lineno, exc)
                    continue
                if not isinstance(obj, dict):
                    continue
                events.append(TranscriptEvent(
                    phase=phase,
                    ts=_parse_ts(obj.get("ts")),
                    kind=str(obj.get("kind", "")),
                    fields={k: v for k, v in obj.items() if k not in ("ts", "kind")},
                ))
    except OSError as exc:
        log.warning("failed to read %s: %s", path, exc)
        return []
    return events


def load_transcripts(campaign_dir: Path) -> dict[str, list[TranscriptEvent]]:
    root = campaign_dir / "profiles"
    if not root.is_dir():
        return {}
    out: dict[str, list[TranscriptEvent]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        m = _FILE_RE.match(entry.name)
        if not m:
            continue
        phase = m.group(1)
        out[phase] = parse_transcript_file(entry, phase)
    return out
