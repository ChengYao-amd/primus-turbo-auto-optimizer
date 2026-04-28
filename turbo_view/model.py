"""Typed data layer for primus-turbo-view.

Every analytics + render call site consumes dataclasses defined here.
``slots=True`` everywhere so typos in field names raise instead of
silently creating new attributes — important because the IO layer maps
heterogeneous text artifacts (markdown tables, JSON, YAML, CSV) onto
these classes and we want loud failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HistoryEntry:
    round: int
    decision: str
    score: dict[str, float]
    description: str
    at: str


@dataclass(slots=True)
class RunState:
    campaign_id: str
    campaign_dir: Path
    current_phase: str
    current_round: int
    best_round: int | None
    best_score: dict[str, float]
    rollback_streak: int
    started_at: str
    last_update: str
    params: dict[str, Any]
    history: list[HistoryEntry]


@dataclass(slots=True)
class CostRow:
    ts: datetime
    phase: str
    round: int | None
    status: str
    wall_s: float
    sdk_s: float
    turns: int
    cost_usd: float
    cumulative_usd: float


@dataclass(slots=True)
class PerfRow:
    round: int
    status: str
    description: str
    fwd_avg: float
    fwd_peak: float
    bwd_avg: float
    bwd_peak: float
    step_geomean: float
    vs_baseline: str
    key_finding: str


@dataclass(slots=True)
class ShapeRow:
    label: str
    B: int
    M: int
    N: int
    K: int
    fwd_tflops: float
    bwd_tflops: float
    fwd_std: float | None
    bwd_std: float | None
    check: str | None


@dataclass(slots=True)
class KernelDispatch:
    name: str
    start_ns: int
    end_ns: int
    vgpr: int
    sgpr: int
    lds_bytes: int
    scratch_bytes: int
    wg_x: int
    grid_x: int

    @property
    def dur_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000.0


@dataclass(slots=True)
class ProfileBundle:
    round: int
    flavor: str
    summary_md_html: str | None
    dispatches: list[KernelDispatch]
    perfetto_json_path: Path | None


@dataclass(slots=True)
class TranscriptEvent:
    phase: str
    ts: datetime | None
    kind: str
    fields: dict[str, Any]


@dataclass(slots=True)
class RoundBundle:
    n: int
    summary_md_html: str | None
    bench_shapes: list[ShapeRow]
    artifacts: list[Path]
    kernel_snapshot_dir: Path | None


@dataclass(slots=True)
class CampaignBundle:
    state: RunState | None
    cost: list[CostRow]
    perf: list[PerfRow]
    rounds: dict[int, RoundBundle]
    profiles: dict[int, ProfileBundle]
    ineffective: list[dict[str, Any]]
    transcripts: dict[str, list[TranscriptEvent]]
    optimize_md_sections: dict[str, str]
