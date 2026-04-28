"""Coverage for ``turbo_view.io.transcripts.load_transcripts``."""

from __future__ import annotations

from pathlib import Path

from turbo_view.io.transcripts import load_transcripts, parse_transcript_file

FIXTURE = Path(__file__).parent / "fixtures" / "campaign_mini"


def test_load_transcripts_finds_both_phases():
    out = load_transcripts(FIXTURE)
    assert sorted(out) == ["ANALYZE", "OPTIMIZE"]
    assert any(ev.kind == "idle_timeout" for ev in out["ANALYZE"])
    assert any(ev.kind == "retry_attempt" for ev in out["OPTIMIZE"])


def test_transcript_ts_parses_iso8601_with_z_suffix():
    events = parse_transcript_file(
        FIXTURE / "profiles" / "_transcript_ANALYZE.jsonl",
        phase="ANALYZE",
    )
    first = events[0]
    assert first.ts is not None
    assert first.ts.year == 2025


def test_transcript_skips_bad_lines(tmp_path: Path):
    src = tmp_path / "_transcript_X.jsonl"
    src.write_text(
        '{"ts":"2025-01-01T00:00:00Z","kind":"phase_begin"}\n'
        "garbage\n"
        '{"ts":"2025-01-01T00:00:01Z","kind":"phase_end"}\n',
        encoding="utf-8",
    )
    events = parse_transcript_file(src, phase="X")
    assert [e.kind for e in events] == ["phase_begin", "phase_end"]


def test_load_transcripts_empty_when_no_profiles_dir(tmp_path: Path):
    assert load_transcripts(tmp_path) == {}
