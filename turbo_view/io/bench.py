"""Parse ``rounds/round-N/artifacts/benchmark.csv`` (per spec §3.2).

Two schemas are supported:

A. ``primus-turbo`` 288-shape full bench
   ``label,B,M,N,K,Forward TFLOPS,Backward TFLOPS``

B. ``quick_test_bench`` 5-shape sampled bench
   ``label,B,M,N,K,fwd_tflops_mean,fwd_tflops_std,bwd_tflops_mean,bwd_tflops_std,correct``

Both normalize to ``ShapeRow``. Missing columns degrade to ``None``;
malformed numeric cells are skipped (per file-level graceful-degrade).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from turbo_view.model import ShapeRow

log = logging.getLogger(__name__)

_FWD_KEYS = ("Forward TFLOPS", "fwd_tflops_mean", "fwd_tflops")
_BWD_KEYS = ("Backward TFLOPS", "bwd_tflops_mean", "bwd_tflops")
_FWD_STD_KEYS = ("fwd_tflops_std",)
_BWD_STD_KEYS = ("bwd_tflops_std",)
_CHECK_KEYS = ("correct", "check")


def _first_present(row: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return None


def _to_float_opt(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_int_opt(text: str | None, default: int = 0) -> int:
    if text is None:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_benchmark_csv(path: Path) -> list[ShapeRow]:
    if not path.is_file():
        return []
    rows: list[ShapeRow] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                if not raw:
                    continue
                fwd = _to_float_opt(_first_present(raw, _FWD_KEYS))
                bwd = _to_float_opt(_first_present(raw, _BWD_KEYS))
                if fwd is None and bwd is None:
                    continue
                rows.append(
                    ShapeRow(
                        label=str(raw.get("label", "")),
                        B=_to_int_opt(raw.get("B")),
                        M=_to_int_opt(raw.get("M")),
                        N=_to_int_opt(raw.get("N")),
                        K=_to_int_opt(raw.get("K")),
                        fwd_tflops=fwd if fwd is not None else 0.0,
                        bwd_tflops=bwd if bwd is not None else 0.0,
                        fwd_std=_to_float_opt(_first_present(raw, _FWD_STD_KEYS)),
                        bwd_std=_to_float_opt(_first_present(raw, _BWD_STD_KEYS)),
                        check=(_first_present(raw, _CHECK_KEYS) or None),
                    )
                )
    except (OSError, csv.Error) as exc:
        log.warning("failed to read %s: %s", path, exc)
        return []
    return rows
