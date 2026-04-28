"""``turbo_view.io.loader.load_campaign`` glues every PR-1 IO module.

The result is a fully-populated ``CampaignBundle`` for the mini
fixture; missing optional pieces graceful-degrade to empty / None.
"""

from __future__ import annotations

from pathlib import Path

from turbo_view.io.loader import load_campaign
from turbo_view.model import CampaignBundle

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_load_campaign_full():
    bundle = load_campaign(FIXTURE)
    assert isinstance(bundle, CampaignBundle)
    assert bundle.state is not None
    assert bundle.state.campaign_id == "campaign_mini"
    assert len(bundle.cost) == 11
    assert len(bundle.perf) == 3
    assert sorted(bundle.rounds.keys()) == [1, 2, 3, 4]
    assert len(bundle.ineffective) == 1
    assert bundle.ineffective[0]["round"] == 3
    assert "Baseline" in bundle.optimize_md_sections
    assert sorted(bundle.profiles) == [1, 2]
    assert sorted(bundle.transcripts) == ["ANALYZE", "OPTIMIZE"]


def test_load_campaign_handles_bare_directory(tmp_path: Path):
    bundle = load_campaign(tmp_path)
    assert bundle.state is None
    assert bundle.cost == []
    assert bundle.perf == []
    assert bundle.rounds == {}
    assert bundle.ineffective == []
    assert bundle.optimize_md_sections == {}


def test_load_campaign_state_none_uses_dirname_fallback(tmp_path: Path):
    bundle = load_campaign(tmp_path)
    assert bundle.state is None
