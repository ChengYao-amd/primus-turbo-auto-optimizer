"""``turbo_view.io.rounds`` builds ``RoundBundle``s from disk.

For PR-1 only ``summary.md`` and the round directory itself are
required. Bench parsing + kernel-snapshot detection land in PR-2/PR-3.
We pin:

* All four mini-fixture rounds are discovered.
* ``round-N`` numerics are extracted (``-`` separator, padded fine).
* ``summary_md_html`` is rendered HTML, not raw markdown.
* Missing ``summary.md`` ⇒ ``summary_md_html is None``, no crash.
"""

from __future__ import annotations

from pathlib import Path

from turbo_view.io.rounds import load_rounds

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_load_rounds_discovers_all_four():
    rounds = load_rounds(FIXTURE)
    assert sorted(rounds.keys()) == [1, 2, 3, 4]


def test_load_rounds_renders_summary_to_html():
    rounds = load_rounds(FIXTURE)
    r2 = rounds[2]
    assert r2.n == 2
    assert r2.summary_md_html is not None
    assert "<h1>" in r2.summary_md_html
    assert "<strong>1314.927 TFLOPS</strong>" in r2.summary_md_html
    assert "ACCEPTED" in r2.summary_md_html


def test_load_rounds_missing_summary_yields_none(tmp_path: Path):
    (tmp_path / "rounds" / "round-7").mkdir(parents=True)
    rounds = load_rounds(tmp_path)
    assert 7 in rounds
    assert rounds[7].summary_md_html is None


def test_load_rounds_returns_empty_when_no_rounds_dir(tmp_path: Path):
    assert load_rounds(tmp_path) == {}


def test_load_rounds_picks_up_bench_shapes_and_artifacts():
    rounds = load_rounds(FIXTURE)
    assert len(rounds[1].bench_shapes) == 5
    labels = [s.label for s in rounds[2].bench_shapes]
    assert labels[0] == "B32_M64_N5760_K2880"
    artifact_names = [p.name for p in rounds[1].artifacts]
    assert "benchmark.csv" in artifact_names
