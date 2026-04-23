"""Section-scoped writers for ``logs/optimize.md``.

These tests fence in the two bugs fixed alongside the 2026-04-22 campaign
review:

1. ``baseline.md`` no longer tells Claude to append a BASELINE row to
   ``performance_trend.md``; Python owns that write, so we only assert
   the prompt contract here (no duplicate instruction) — the end-to-end
   single-row behaviour is exercised by ``test_smoke_orchestrator``.

2. ``append_baseline`` / ``append_round_entry`` /
   ``append_verified_ineffective`` / ``upsert_current_best`` /
   ``upsert_directions_to_try`` / ``append_termination_block`` /
   ``append_final_report`` all land their content inside the owning
   ``## <section>`` heading instead of EOF. The template sections stop
   being empty and re-runs stay idempotent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_optimize.logs import (
    append_baseline,
    append_final_report,
    append_round_entry,
    append_termination_block,
    append_verified_ineffective,
    extract_history,
    init_optimize_log,
    init_performance_trend,
    optimize_log_path,
    upsert_current_best,
    upsert_directions_to_try,
)


# ---------------------------------------------------------------------
# Fixture — a primed optimize.md identical in structure to a fresh
# campaign directory, plus an initialized performance_trend.md so any
# helper that touches it (none currently, but the smoke invariants rely
# on it existing) does not trip on a missing file.
# ---------------------------------------------------------------------


@pytest.fixture
def campaign_dir(tmp_path: Path) -> Path:
    d = tmp_path / "campaign"
    d.mkdir()
    init_optimize_log(
        d,
        {
            "target_op": "gemm_fp8_blockwise",
            "target_backend": "TRITON",
            "target_lang": "TRITON",
            "target_gpu": "gfx950",
        },
    )
    init_performance_trend(d)
    return d


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section_body(text: str, header: str) -> str:
    """Return the body of ``## <header>`` up to the next ``## `` heading.

    Thin test helper — if the production code regresses to EOF-append
    the returned body will be empty, so these tests directly encode the
    ``section-scoped insertion`` contract.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.rstrip() == header:
            start = i
            break
    else:
        pytest.fail(f"section {header!r} not found in text:\n{text}")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## ") and not lines[j].startswith("### "):
            end = j
            break
    return "".join(lines[start + 1 : end])


# ---------------------------------------------------------------------
# Bug 1 — baseline.md prompt must not instruct Claude to append a trend
# row (Python owns ``performance_trend.md`` end-to-end).
# ---------------------------------------------------------------------


def test_baseline_prompt_does_not_instruct_claude_to_write_trend() -> None:
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "turbo_optimize"
        / "prompts"
        / "baseline.md"
    )
    text = prompt_path.read_text(encoding="utf-8")
    # the old wording put Claude on the hook; this must stay removed so
    # only `append_trend_row` in the orchestrator writes the BASELINE row
    forbidden = [
        "Append the first row to",
        "using the Rule 8 table format (status `BASELINE`",
    ]
    for phrase in forbidden:
        assert phrase not in text, (
            f"baseline.md still tells Claude to write performance_trend.md "
            f"({phrase!r}); remove the duplicate write to keep the BASELINE "
            f"row unique."
        )
    # the prompt must instead make the single-writer contract explicit
    assert "orchestrator owns both files" in text


# ---------------------------------------------------------------------
# Bug 2 — section-scoped insertion (the four sections that were empty
# in the 202604221559 campaign).
# ---------------------------------------------------------------------


def test_append_baseline_lands_inside_baseline_section(campaign_dir: Path) -> None:
    append_baseline(
        campaign_dir,
        backend="TRITON",
        gpu="gfx950",
        commit="deadbeef",
        aggregate_score={"Forward TFLOPS": 577.00, "Backward TFLOPS": 458.96},
        all_check_pass=True,
        quick_baseline_log="rounds/round-1/artifacts/quick_baseline.log",
    )
    text = _read(optimize_log_path(campaign_dir))
    baseline_body = _section_body(text, "## Baseline")

    assert "Baseline Entry" in baseline_body
    assert "- Backend: TRITON" in baseline_body
    assert "- GPU: gfx950" in baseline_body
    assert "- Commit: deadbeef" in baseline_body
    assert "Forward TFLOPS: 577.000" in baseline_body
    assert (
        "- Quick baseline log: rounds/round-1/artifacts/quick_baseline.log"
        in baseline_body
    )
    # the placeholder from the template must be gone now
    assert "_to be filled in after round-1_" not in baseline_body
    # it must not leak into Optimization History or Current Best
    assert "Baseline Entry" not in _section_body(text, "## Optimization History")
    assert "Baseline Entry" not in _section_body(text, "## Current Best")


def test_append_baseline_is_idempotent(campaign_dir: Path) -> None:
    """Resume re-enters `_phase_baseline`; upsert semantics prevent the
    duplicated `## Baseline Entry` rows we saw in the 202604221559 log.
    """
    kwargs = dict(
        backend="TRITON",
        gpu="gfx950",
        commit="abc123",
        aggregate_score={"Forward TFLOPS": 100.0, "Backward TFLOPS": 50.0},
        all_check_pass=True,
    )
    append_baseline(campaign_dir, **kwargs)
    append_baseline(campaign_dir, **kwargs)

    baseline_body = _section_body(_read(optimize_log_path(campaign_dir)), "## Baseline")
    assert baseline_body.count("Baseline Entry") == 1


def test_append_round_entry_lands_in_optimization_history(campaign_dir: Path) -> None:
    append_round_entry(
        campaign_dir,
        round_n=2,
        description="swap to persistent launch",
        validation_level="quick",
        hypothesis="persistent launch improves low-occupancy shapes",
        changes="primus_turbo/triton/gemm/gemm_fp8_kernel.py",
        aggregate_score_delta="+3.14% fwd",
        test_result="PASS",
        decision="accept",
        notes="no backward regression",
    )
    text = _read(optimize_log_path(campaign_dir))
    history_body = _section_body(text, "## Optimization History")

    assert "### round-2 — swap to persistent launch" in history_body
    assert "- Hypothesis: persistent launch improves low-occupancy shapes" in history_body
    assert "- Decision: accept" in history_body
    # must not contaminate neighbour sections
    for other in ("## Baseline", "## Current Best", "## Final Report"):
        assert "### round-2" not in _section_body(text, other)


def test_append_round_entry_stacks_multiple_rounds(campaign_dir: Path) -> None:
    for n in (2, 3, 4):
        append_round_entry(
            campaign_dir,
            round_n=n,
            description=f"candidate-{n}",
            validation_level="quick",
            hypothesis=f"hyp-{n}",
            changes="some/file.py",
            aggregate_score_delta="+0.00%",
            test_result="PASS",
            decision="rollback" if n % 2 == 0 else "accept",
        )
    body = _section_body(_read(optimize_log_path(campaign_dir)), "## Optimization History")
    idx_2 = body.index("### round-2")
    idx_3 = body.index("### round-3")
    idx_4 = body.index("### round-4")
    assert idx_2 < idx_3 < idx_4, "round blocks must stay in append order"


def test_upsert_current_best_replaces_body_and_renders_baseline(
    campaign_dir: Path,
) -> None:
    upsert_current_best(
        campaign_dir,
        best_round=3,
        best_score={"Forward TFLOPS": 600.0, "Backward TFLOPS": 480.0},
        baseline_score={"Forward TFLOPS": 577.0, "Backward TFLOPS": 458.0},
    )
    text = _read(optimize_log_path(campaign_dir))
    body = _section_body(text, "## Current Best")

    assert "_to be updated per accepted round_" not in body
    assert "_Updated after round-3" in body
    assert "| Metric | Baseline | Current Best | Improvement |" in body
    assert "| Forward TFLOPS | 577.000 | 600.000 | +3.99% |" in body
    assert "| Backward TFLOPS | 458.000 | 480.000 | +4.80% |" in body


def test_upsert_current_best_overwrites_on_repeat(campaign_dir: Path) -> None:
    upsert_current_best(
        campaign_dir,
        best_round=3,
        best_score={"Forward TFLOPS": 600.0},
        baseline_score={"Forward TFLOPS": 577.0},
    )
    upsert_current_best(
        campaign_dir,
        best_round=5,
        best_score={"Forward TFLOPS": 620.0},
        baseline_score={"Forward TFLOPS": 577.0},
    )
    body = _section_body(_read(optimize_log_path(campaign_dir)), "## Current Best")
    # only the latest round's table stays
    assert "_Updated after round-5" in body
    assert "_Updated after round-3" not in body
    assert "620.000" in body
    assert "600.000" not in body


def test_upsert_current_best_handles_missing_baseline(campaign_dir: Path) -> None:
    """On the BASELINE round itself we don't have a baseline to diff
    against. The helper must still render the current score and show
    ``-`` for both the baseline column and the improvement column.
    """
    upsert_current_best(
        campaign_dir,
        best_round=1,
        best_score={"Forward TFLOPS": 577.0, "Backward TFLOPS": 458.0},
        baseline_score=None,
    )
    body = _section_body(_read(optimize_log_path(campaign_dir)), "## Current Best")
    assert "| Forward TFLOPS | - | 577.000 | - |" in body
    assert "| Backward TFLOPS | - | 458.000 | - |" in body


def test_append_verified_ineffective_inserts_row_under_table(
    campaign_dir: Path,
) -> None:
    append_verified_ineffective(
        campaign_dir,
        round_n=2,
        direction="BLOCK_N=256 autotune",
        reason="weak skinny shapes regressed",
        modified_files=["primus_turbo/triton/gemm/gemm_fp8_kernel.py"],
    )
    append_verified_ineffective(
        campaign_dir,
        round_n=4,
        direction="num_warps=4 for BLOCK_M=256",
        reason="small shape regressed",
    )
    text = _read(optimize_log_path(campaign_dir))
    body = _section_body(text, "## Verified Ineffective Directions")

    assert "| Direction | Version | Failure Reason |" in body
    assert "| BLOCK_N=256 autotune | round-2 | weak skinny shapes regressed |" in body
    assert (
        "| num_warps=4 for BLOCK_M=256 | round-4 | small shape regressed |"
        in body
    )
    # rows must sit under the separator (below it, not above)
    assert body.index("|---") < body.index("round-2")

    history = extract_history(campaign_dir)
    round2 = [e for e in history.verified_ineffective if e.round == 2][0]
    assert round2.modified_files == [
        "primus_turbo/triton/gemm/gemm_fp8_kernel.py"
    ]


def test_upsert_directions_to_try_renders_stagnation_output(
    campaign_dir: Path,
) -> None:
    upsert_directions_to_try(
        campaign_dir,
        round_n=5,
        directions=[
            {
                "title": "Expand BLOCK_K tiling",
                "category": "tile",
                "hypothesis": "reduces HBM pressure on small shapes",
            },
            {
                "title": "Overlap quantization with gemm",
                "category": "pipeline",
            },
            {"title": ""},
            "not-a-dict",
        ],
    )
    body = _section_body(_read(optimize_log_path(campaign_dir)), "## Directions to Try")
    assert "- [ ] Expand BLOCK_K tiling — [tile] reduces HBM pressure on small shapes" in body
    assert "- [ ] Overlap quantization with gemm — [pipeline]" in body
    # empty title / non-dict entries are discarded
    assert body.count("- [ ] ") == 2


def test_upsert_directions_to_try_is_noop_on_empty(campaign_dir: Path) -> None:
    """An empty ``new_directions`` result must leave the previous
    section untouched rather than erasing it — otherwise a weak
    STAGNATION_REVIEW would wipe prior guidance.
    """
    upsert_directions_to_try(
        campaign_dir,
        round_n=5,
        directions=[
            {"title": "keep this around"},
        ],
    )
    upsert_directions_to_try(campaign_dir, round_n=6, directions=[])
    body = _section_body(_read(optimize_log_path(campaign_dir)), "## Directions to Try")
    assert "keep this around" in body


def test_termination_and_final_report_share_final_report_section(
    campaign_dir: Path,
) -> None:
    append_termination_block(
        campaign_dir,
        checks={"T1": False, "T2": False, "T3": True, "T4": False, "T5": False},
        passed=["T3"],
    )
    append_final_report(campaign_dir, body='{"reason": "T3", "best_round": 5}')

    text = _read(optimize_log_path(campaign_dir))
    final_body = _section_body(text, "## Final Report")

    assert "### Termination Check" in final_body
    assert "- T3 max_iterations reached: PASS" in final_body
    assert '"reason": "T3"' in final_body
    assert "_filled in when campaign terminates_" not in final_body
    # a second `## Final Report` must not exist
    assert text.count("## Final Report") == 1
