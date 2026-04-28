"""Payload contract tests for ``turbo_view.render.payload``.

The payload is the single contract between Python and the front end.
We assert top-level keys, value shapes, and a few invariants:

* Schema version present and stable for PR-1.
* ``state`` is None when run.json is missing.
* ``perf_panel.baseline`` and ``perf_panel.best`` are derived from
  the right source (BASELINE row / state.best_score).
* ``rounds`` includes every round from history + perf + rounds dir
  + state.current_round, sorted ascending; the in-flight round shows
  as ``decision == "PENDING"`` when no history entry exists.
* The full payload survives a JSON round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbo_view.io.loader import load_campaign
from turbo_view.render.payload import SCHEMA_VERSION, bundle_to_payload

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_payload_top_level_keys():
    payload = bundle_to_payload(load_campaign(FIXTURE))
    assert payload["schema_version"] == SCHEMA_VERSION
    expected = {
        "schema_version", "state", "kpi_summary",
        "perf", "perf_panel",
        "cost_panel", "gantt_panel", "heatmap_panel",
        "token_turn_wall_panel",
        "rounds", "ineffective",
        "profile_panels", "profile_global", "profile_diffs",
    }
    assert set(payload) == expected


def test_payload_state_block():
    payload = bundle_to_payload(load_campaign(FIXTURE))
    state = payload["state"]
    assert state is not None
    assert state["campaign_id"] == "campaign_mini"
    assert state["current_phase"] == "ANALYZE"
    assert state["current_round"] == 4
    assert state["best_round"] == 2
    assert state["history"][0]["decision"] == "BASELINE"
    assert state["history"][2]["decision"] == "ROLLBACK"


def test_payload_kpi_summary_baseline_and_best_per_axis():
    """KPI summary feeds the sticky bar baseline -> best (delta %) cells.

    The fixture's BASELINE is round 1 and the best ACCEPTED row is
    round 2; values come straight from ``performance_trend.md``.
    """
    kpi = bundle_to_payload(load_campaign(FIXTURE))["kpi_summary"]
    assert kpi["step"]["baseline"] == pytest.approx(1309.062)
    assert kpi["step"]["best"]     == pytest.approx(1314.927)
    assert kpi["fwd"]["baseline"]  == pytest.approx(1308.688)
    assert kpi["fwd"]["best"]      == pytest.approx(1318.307)
    assert kpi["bwd"]["baseline"]  == pytest.approx(1309.436)
    assert kpi["bwd"]["best"]      == pytest.approx(1311.556)


def test_payload_kpi_summary_empty_when_no_data(tmp_path: Path):
    """An empty campaign has neither baseline nor best -- all None."""
    kpi = bundle_to_payload(load_campaign(tmp_path))["kpi_summary"]
    for axis in ("step", "fwd", "bwd"):
        assert kpi[axis] == {"baseline": None, "best": None}


def test_payload_perf_panel_baseline_and_best():
    payload = bundle_to_payload(load_campaign(FIXTURE))
    panel = payload["perf_panel"]
    assert panel["baseline"] == pytest.approx(1309.062)
    assert panel["best"] == pytest.approx(1314.927)
    rounds = [p["round"] for p in panel["points"]]
    assert rounds == [1, 2, 3]
    accepted = [a["round"] for a in panel["accepted"]]
    assert accepted == [1, 2]
    accept_marks = [a["round"] for a in panel["annotations"]["accept"]]
    assert accept_marks == [2]


def test_payload_rounds_merges_all_sources_including_in_flight():
    payload = bundle_to_payload(load_campaign(FIXTURE))
    rows = payload["rounds"]
    assert [r["n"] for r in rows] == [1, 2, 3, 4]

    by_n = {r["n"]: r for r in rows}
    assert by_n[1]["decision"] == "BASELINE"
    assert by_n[2]["decision"] == "ACCEPTED"
    assert by_n[3]["decision"] == "ROLLBACK"
    assert by_n[4]["decision"] == "PENDING"
    assert by_n[2]["summary_md_html"] is not None
    assert "1314.927" in by_n[2]["summary_md_html"]
    assert by_n[2]["perf"]["fwd_avg"] == pytest.approx(1318.307)
    assert by_n[4]["perf"] is None


def test_payload_ineffective_normalised():
    payload = bundle_to_payload(load_campaign(FIXTURE))
    items = payload["ineffective"]
    assert len(items) == 1
    assert items[0]["round"] == 3
    assert "waves_per_eu" in items[0]["direction"]
    assert isinstance(items[0]["modified_files"], list)


def test_payload_state_none_when_run_json_absent(tmp_path: Path):
    payload = bundle_to_payload(load_campaign(tmp_path))
    assert payload["state"] is None
    assert payload["perf"] == []
    assert payload["rounds"] == []
    assert payload["perf_panel"]["baseline"] is None
    assert payload["perf_panel"]["best"] is None


def test_payload_is_json_serialisable():
    payload = bundle_to_payload(load_campaign(FIXTURE))
    text = json.dumps(payload)
    assert json.loads(text) == payload


def test_payload_cost_panel_present_with_totals():
    panel = bundle_to_payload(load_campaign(FIXTURE))["cost_panel"]
    assert panel["total_usd"] == pytest.approx(7.6112)
    assert panel["total_turns"] >= 1
    assert any(p["phase"] == "ANALYZE" for p in panel["per_phase"])


def test_payload_gantt_panel_blocks_present():
    panel = bundle_to_payload(load_campaign(FIXTURE))["gantt_panel"]
    assert len(panel["blocks"]) == 11


def test_payload_heatmap_panel_has_baseline_round():
    panel = bundle_to_payload(load_campaign(FIXTURE))["heatmap_panel"]
    assert panel["baseline_round"] == 1
    assert panel["rounds"] == [1, 2, 3]
    assert len(panel["shape_labels"]) == 5


def test_payload_token_turn_wall_arrays_match_lengths():
    panel = bundle_to_payload(load_campaign(FIXTURE))["token_turn_wall_panel"]
    n = len(panel["rounds"])
    assert len(panel["wall_s"]) == n
    assert len(panel["sdk_s"]) == n
    assert len(panel["turns"]) == n


def test_payload_rounds_carry_bench_shapes_and_artifacts():
    rows = bundle_to_payload(load_campaign(FIXTURE))["rounds"]
    by_n = {r["n"]: r for r in rows}
    assert len(by_n[1]["bench_shapes"]) == 5
    assert by_n[2]["bench_shapes"][0]["check"] == "PASS"
    assert "benchmark.csv" in by_n[1]["artifacts"]


def test_payload_profile_panels_keyed_by_round_string():
    panels = bundle_to_payload(load_campaign(FIXTURE))["profile_panels"]
    assert sorted(panels) == ["1", "2"]
    assert panels["1"]["flavor"] == "baseline"
    assert panels["1"]["top_n"][0]["family"] == "fwd_dgrad"


def test_payload_profile_global_carries_cross_round_aggregates():
    g = bundle_to_payload(load_campaign(FIXTURE))["profile_global"]
    assert g["family_rollup"]["rounds"] == [1, 2]
    assert g["round_over_round"]["rounds"] == [1, 2]
    assert g["resources"]["rounds"] == [1, 2]


def test_payload_profile_diffs_pair_consecutive_rounds():
    diffs = bundle_to_payload(load_campaign(FIXTURE))["profile_diffs"]
    assert diffs["rounds"] == [1, 2]
    assert len(diffs["pairs"]) == 1


def test_payload_gantt_panel_events_overlay_from_transcripts():
    panel = bundle_to_payload(load_campaign(FIXTURE))["gantt_panel"]
    kinds = [e["kind"] for e in panel["events"]]
    assert "idle_timeout" in kinds
    assert "retry_attempt" in kinds
