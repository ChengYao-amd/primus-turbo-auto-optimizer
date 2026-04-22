"""Unit tests for ``turbo_optimize.logs.append_cost_row`` and helpers.

The cost log is the single source of truth for "how much has this
campaign spent so far". We pin:

1. Header initialization is idempotent and not destructive.
2. Each ``append_cost_row`` call produces exactly one table row whose
   columns line up with the header.
3. The cumulative column increases monotonically across rows and is
   recovered correctly after parsing an existing file (the resume
   path).
4. ``phase_variant`` renders as ``PHASE (variant)`` so VALIDATE quick /
   full rows stay distinguishable at a glance.
5. ``round_n=None`` renders ``-`` (no-round phases such as
   ``DEFINE_TARGET`` / ``PREPARE_ENVIRONMENT``).
6. Malformed existing rows do not poison future appends (the parser
   skips them and falls back to zero).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_optimize.logs import (
    COST_HEADER,
    _last_cumulative_cost,
    append_cost_row,
    cost_log_path,
    init_cost_log,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_init_cost_log_writes_header_once(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()

    path = init_cost_log(campaign)

    assert path == cost_log_path(campaign)
    assert path.exists()
    first = _read(path)
    assert first == COST_HEADER
    assert "Campaign Cost Log" in first

    init_cost_log(campaign)
    second = _read(path)
    assert first == second


def test_append_cost_row_appends_single_row_and_returns_cumulative(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    init_cost_log(campaign)

    cumulative = append_cost_row(
        campaign,
        phase="OPTIMIZE",
        round_n=1,
        status="ok",
        wall_s=12.3,
        sdk_s=11.8,
        turns=4,
        cost_usd=0.12,
    )

    assert cumulative == pytest.approx(0.12)
    body = _read(cost_log_path(campaign))
    data_rows = [
        line for line in body.splitlines()
        if line.startswith("|") and "OPTIMIZE" in line
    ]
    assert len(data_rows) == 1
    row = data_rows[0]
    assert "| OPTIMIZE |" in row
    assert "| 1 |" in row
    assert "| ok |" in row
    assert "| 12.3 |" in row
    assert "| 11.8 |" in row
    assert "| 4 |" in row
    assert "| $0.1200 | $0.1200 |" in row


def test_append_cost_row_accumulates_across_calls(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()

    append_cost_row(
        campaign, phase="A", round_n=1, status="ok",
        wall_s=1.0, sdk_s=0.9, turns=1, cost_usd=0.10,
    )
    append_cost_row(
        campaign, phase="B", round_n=1, status="ok",
        wall_s=2.0, sdk_s=1.9, turns=2, cost_usd=0.25,
    )
    third = append_cost_row(
        campaign, phase="C", round_n=2, status="ok",
        wall_s=3.0, sdk_s=2.9, turns=3, cost_usd=0.05,
    )

    assert third == pytest.approx(0.40)
    body = _read(cost_log_path(campaign))
    assert body.count("\n|") >= 5
    assert "$0.1000 |" in body
    assert "$0.3500 |" in body
    assert "$0.4000 |" in body


def test_cached_rows_contribute_zero_but_still_logged(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    append_cost_row(
        campaign, phase="A", round_n=1, status="ok",
        wall_s=1.0, sdk_s=0.9, turns=1, cost_usd=0.50,
    )
    cumulative = append_cost_row(
        campaign, phase="A", round_n=1, status="cached",
        wall_s=0.01, sdk_s=None, turns=0, cost_usd=0.0,
    )
    assert cumulative == pytest.approx(0.50)

    body = _read(cost_log_path(campaign))
    cached_rows = [line for line in body.splitlines() if "| cached |" in line]
    assert len(cached_rows) == 1
    assert "| - |" in cached_rows[0]  # sdk_s None rendered as "-"
    assert "| $0.0000 | $0.5000 |" in cached_rows[0]


def test_phase_variant_renders_as_parenthesized_suffix(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    append_cost_row(
        campaign, phase="VALIDATE", round_n=2, status="ok",
        wall_s=10.0, sdk_s=9.5, turns=5, cost_usd=0.20,
        phase_variant="quick",
    )
    append_cost_row(
        campaign, phase="VALIDATE", round_n=2, status="ok",
        wall_s=40.0, sdk_s=38.1, turns=8, cost_usd=0.60,
        phase_variant="full",
    )
    body = _read(cost_log_path(campaign))
    assert "| VALIDATE (quick) |" in body
    assert "| VALIDATE (full) |" in body


def test_none_round_renders_as_dash(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    append_cost_row(
        campaign, phase="DEFINE_TARGET", round_n=None, status="ok",
        wall_s=5.0, sdk_s=4.7, turns=2, cost_usd=0.08,
    )
    body = _read(cost_log_path(campaign))
    rows = [line for line in body.splitlines() if "DEFINE_TARGET" in line]
    assert len(rows) == 1
    assert "| DEFINE_TARGET | - |" in rows[0]


def test_last_cumulative_cost_recovers_from_existing_file(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    append_cost_row(
        campaign, phase="A", round_n=1, status="ok",
        wall_s=1.0, sdk_s=0.9, turns=1, cost_usd=0.10,
    )
    append_cost_row(
        campaign, phase="B", round_n=1, status="ok",
        wall_s=2.0, sdk_s=1.9, turns=2, cost_usd=0.20,
    )

    assert _last_cumulative_cost(cost_log_path(campaign)) == pytest.approx(0.30)

    resumed = append_cost_row(
        campaign, phase="C", round_n=2, status="ok",
        wall_s=0.5, sdk_s=0.4, turns=1, cost_usd=0.05,
    )
    assert resumed == pytest.approx(0.35)


def test_last_cumulative_cost_is_zero_for_missing_or_header_only(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    assert _last_cumulative_cost(cost_log_path(campaign)) == 0.0

    init_cost_log(campaign)
    assert _last_cumulative_cost(cost_log_path(campaign)) == 0.0


def test_malformed_row_falls_back_to_zero(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    init_cost_log(campaign)
    path = cost_log_path(campaign)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("| 2026-01-01 | X | 1 | ok | 1.0 | 0.9 | 1 | $0.10 | not-a-number |\n")

    cumulative = append_cost_row(
        campaign, phase="NEXT", round_n=1, status="ok",
        wall_s=1.0, sdk_s=0.9, turns=1, cost_usd=0.25,
    )
    assert cumulative == pytest.approx(0.25)


def test_append_cost_row_autoinits_when_file_missing(tmp_path):
    campaign = tmp_path / "camp"
    campaign.mkdir()
    path = cost_log_path(campaign)
    assert not path.exists()

    cumulative = append_cost_row(
        campaign, phase="FIRST", round_n=1, status="ok",
        wall_s=1.0, sdk_s=0.9, turns=1, cost_usd=0.07,
    )
    assert cumulative == pytest.approx(0.07)
    body = _read(path)
    assert "Campaign Cost Log" in body
    assert "| FIRST |" in body
