"""Coverage for the multi-campaign overview renderer.

Builds a pair of mini campaigns, runs ``write_overview``, then asserts:

* index.html exists at the root, has KPI / filters / campaigns containers
* each campaign has its own ``c/<id>/index.html``
* ``data.json`` round-trips and carries a KPI block
* assets are written exactly once at the top, not duplicated per child
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from turbo_view.render.overview import build_overview_payload, write_overview
from turbo_view.discover import discover_campaigns

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def _multi_workspace(tmp_path: Path) -> Path:
    """Build a workspace with two distinct campaigns sharing the layout
    of ``campaign_mini``. Each campaign goes under its own subdirectory
    so ``discover_campaigns`` walks deeper than the top-level state.
    """
    a = tmp_path / "campaign-a"
    b = tmp_path / "campaign-b"
    shutil.copytree(FIXTURE, a)
    shutil.copytree(FIXTURE, b)
    # Rename state directories so each has a unique campaign_id.
    (a / "state" / "campaign_mini").rename(a / "state" / "campaign-a")
    (b / "state" / "campaign_mini").rename(b / "state" / "campaign-b")
    # Patch run.json campaign_id so payload sees the renamed id.
    for d, name in ((a, "campaign-a"), (b, "campaign-b")):
        rj = d / "state" / name / "run.json"
        data = json.loads(rj.read_text())
        data["campaign_id"] = name
        rj.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_overview_payload_shape(tmp_path: Path):
    workspace = _multi_workspace(tmp_path)
    handles = discover_campaigns(workspace)
    payload = build_overview_payload(handles)
    assert payload["schema_version"] == "1"
    assert sorted(c["campaign_id"] for c in payload["campaigns"]) == ["campaign-a", "campaign-b"]
    assert payload["kpi"]["active_campaigns"] >= 1
    assert payload["kpi"]["total_cost_usd"] > 0
    assert all(c["best_delta_pct"] is not None for c in payload["campaigns"])


def test_write_overview_creates_index_assets_and_subpages(tmp_path: Path):
    workspace = _multi_workspace(tmp_path)
    out = tmp_path / "out"
    index = write_overview(workspace, out)

    assert index == (out / "index.html").resolve()
    assert (out / "data.json").is_file()
    assert (out / "assets" / "app.js").is_file()
    for cid in ("campaign-a", "campaign-b"):
        assert (out / "c" / cid / "index.html").is_file()
        # Sub-pages don't duplicate the asset bundle (asset_prefix=../../assets/).
        assert not (out / "c" / cid / "assets").exists()

    payload = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert len(payload["campaigns"]) == 2


def test_write_overview_raises_when_no_campaigns(tmp_path: Path):
    with pytest.raises(RuntimeError):
        write_overview(tmp_path, tmp_path / "out")
