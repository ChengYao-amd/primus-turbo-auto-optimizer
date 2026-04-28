"""Render-layer integration tests.

We assert the contract the front-end relies on:

* ``render_detail`` returns ``(html, payload)``.
* ``index.html`` contains the inlined ``<script id="data">`` JSON.
* All three PR-1 panel containers (``#p6``, ``#p1``, ``#p5``) appear.
* Vendored Chart.js script tag is present.
* ``</script>`` substrings in payload values are escaped so they
  cannot break out of the host script tag.
* ``write_detail`` produces ``index.html`` + ``data.json`` + ``assets/``
  with the expected vendored asset filenames.
* ``data.json`` round-trips through ``json.loads``.
"""

from __future__ import annotations

import json
from pathlib import Path

from turbo_view.io.loader import load_campaign
from turbo_view.render.build import render_detail, write_detail

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_render_detail_returns_html_and_payload():
    bundle = load_campaign(FIXTURE)
    html, payload = render_detail(bundle)
    assert isinstance(html, str)
    assert isinstance(payload, dict)
    assert payload["state"]["campaign_id"] == "campaign_mini"


def test_render_detail_html_has_panels_and_data_block():
    bundle = load_campaign(FIXTURE)
    html, _ = render_detail(bundle)

    assert '<script id="data" type="application/json">' in html
    for pid in ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"):
        assert f'id="{pid}"' in html
    assert "chart.umd.min.js" in html
    assert "chartjs-plugin-annotation.min.js" in html
    assert "app.js" in html
    assert "app.css" in html
    assert "campaign_mini" in html


def test_render_detail_escapes_script_close_tag_in_payload():
    """Inlined JSON must escape ``</script>`` so embedded user
    content cannot terminate the host ``<script>`` element early."""
    bundle = load_campaign(FIXTURE)
    bundle.ineffective.append({"round": 99, "direction": "</script><img>",
                               "reason": "x", "modified_files": []})
    html, _ = render_detail(bundle)
    inline_block = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert "</script>" not in inline_block
    assert "<\\/script>" in inline_block


def test_write_detail_creates_index_data_and_assets(tmp_path: Path):
    bundle = load_campaign(FIXTURE)
    out = tmp_path / "view"
    index_path = write_detail(bundle, out)

    assert index_path == (out / "index.html").resolve()
    assert (out / "index.html").is_file()
    assert (out / "data.json").is_file()
    assert (out / "assets" / "app.js").is_file()
    assert (out / "assets" / "app.css").is_file()
    assert (out / "assets" / "markdown.css").is_file()
    assert (out / "assets" / "chart.umd.min.js").is_file()
    assert (out / "assets" / "chartjs-plugin-annotation.min.js").is_file()

    assert (out / "index.html").stat().st_size > 5_000

    payload = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "4"
    assert [r["n"] for r in payload["rounds"]] == [1, 2, 3, 4]
    assert payload["cost_panel"]["total_usd"] > 0
    assert payload["heatmap_panel"]["baseline_round"] == 1
    assert sorted(payload["profile_panels"]) == ["1", "2"]
    assert any(e["kind"] == "idle_timeout" for e in payload["gantt_panel"]["events"])


def test_write_detail_handles_bare_directory(tmp_path: Path):
    bundle = load_campaign(tmp_path / "empty_campaign")
    (tmp_path / "empty_campaign").mkdir()
    out = tmp_path / "view2"
    index_path = write_detail(bundle, out)
    assert index_path.is_file()
    payload = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert payload["state"] is None
    assert payload["rounds"] == []
