"""Persistent run state for the campaign orchestrator.

Two groups of files live under `params.state_dir`:

* `run.json` — the single mutable snapshot of `(campaign_id, current_phase,
  current_round, best_round, best_score, rollback_streak, params)`.
  Written atomically after every phase transition so that `-s <campaign>`
  can resume from the right state-machine node.
* `phase_result/<phase>[_<round>].json` — structured per-phase outputs
  produced by Claude via the `Write` tool and validated by Python after
  the phase returns.

All writes use a temp-file + `os.replace` dance; readers are guaranteed
to see a fully written payload or the previous one, never a half file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from turbo_optimize.config import CampaignParams


log = logging.getLogger(__name__)


PHASE_ORDER: tuple[str, ...] = (
    "DEFINE_TARGET",
    "USER_CONFIRM_MANIFEST",
    "PREPARE_ENVIRONMENT",
    "SURVEY_RELATED_WORK",
    "READ_HISTORICAL_TIPS",
    "BASELINE",
    "ANALYZE",
    "OPTIMIZE",
    "VALIDATE",
    "DECIDE",
    "STAGNATION_REVIEW",
    "TERMINATION_CHECK",
    "REPORT",
    "DONE",
)


@dataclass
class RunState:
    campaign_id: str | None = None
    campaign_dir: str | None = None
    current_phase: str = "DEFINE_TARGET"
    current_round: int = 0
    best_round: int | None = None
    best_score: dict[str, float] = field(default_factory=dict)
    rollback_streak: int = 0
    started_at: str | None = None
    last_update: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(**data)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def run_json_path(state_dir: Path) -> Path:
    return state_dir / "run.json"


def phase_result_path(
    state_dir: Path,
    phase: str,
    round_n: int | None = None,
    *,
    suffix: str | None = None,
) -> Path:
    """Return the on-disk path a phase writes its structured JSON to.

    ``suffix`` is an optional tag appended after the round number. Used by
    VALIDATE to keep ``quick`` and ``full`` results on separate paths so
    the full-validation gate can re-run the phase without colliding with
    (or reusing via the run_phase cache) the quick result it just
    produced.
    """
    parts = [phase.lower()]
    if round_n is not None:
        parts.append(f"round{round_n}")
    if suffix:
        parts.append(suffix)
    name = "_".join(parts) + ".json"
    return state_dir / "phase_result" / name


def save_run_state(state_dir: Path, state: RunState) -> None:
    state.last_update = _now()
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    _atomic_write(run_json_path(state_dir), payload)
    log.debug("run state saved: phase=%s round=%s", state.current_phase, state.current_round)


def load_run_state(state_dir: Path) -> RunState | None:
    path = run_json_path(state_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to load %s: %s", path, exc)
        return None
    return RunState.from_dict(data)


def init_run_state(params: CampaignParams) -> RunState:
    state = RunState(
        campaign_id=params.campaign_id,
        campaign_dir=str(params.campaign_dir) if params.campaign_dir else None,
        current_phase="DEFINE_TARGET",
        current_round=0,
        started_at=_now(),
        params=params.to_dict(),
    )
    return state


def load_or_init_run(params: CampaignParams) -> tuple[RunState, bool]:
    """Return `(state, resumed)`.

    If `params.campaign_id` is already set and a matching `run.json`
    exists, load it and keep going. Otherwise create a fresh state that
    begins at DEFINE_TARGET.
    """
    existing = load_run_state(params.state_dir)
    if (
        existing is not None
        and params.campaign_id
        and existing.campaign_id == params.campaign_id
    ):
        log.info(
            "resuming campaign %s at phase=%s round=%s",
            existing.campaign_id,
            existing.current_phase,
            existing.current_round,
        )
        return existing, True
    return init_run_state(params), False


def write_phase_result(
    state_dir: Path,
    phase: str,
    data: dict[str, Any],
    round_n: int | None = None,
) -> Path:
    path = phase_result_path(state_dir, phase, round_n)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    _atomic_write(path, payload)
    return path


def load_phase_result(
    state_dir: Path,
    phase: str,
    round_n: int | None = None,
) -> dict[str, Any] | None:
    path = phase_result_path(state_dir, phase, round_n)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to parse phase result %s: %s", path, exc)
        return None


def advance_phase(state: RunState, next_phase: str) -> None:
    if next_phase not in PHASE_ORDER:
        raise ValueError(f"unknown phase '{next_phase}'")
    state.current_phase = next_phase


def record_round_event(
    state: RunState,
    *,
    round_n: int,
    decision: str,
    score: dict[str, float] | None,
    description: str,
) -> None:
    state.history.append(
        {
            "round": round_n,
            "decision": decision,
            "score": score or {},
            "description": description,
            "at": _now(),
        }
    )
