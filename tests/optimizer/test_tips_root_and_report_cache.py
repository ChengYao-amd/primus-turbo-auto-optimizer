"""Tests for the historical-tips relocation + REPORT cache invalidation.

Two related fixes are exercised here:

1. **Tips root relocation.** The cross-campaign tips knowledge base lives
   at ``<tips_root>/<gpu>/<op>/<backend>/tips.md``, where
   ``tips_root`` defaults to ``<tool_repo>/agent_data/historical_experience``
   and is overridable via ``TURBO_TIPS_ROOT``. The previous default
   placed the file under ``workspace_root/agent/historical_experience``,
   which the rollback step (``git clean -fd`` inside ``workspace_root``)
   would erase along with every other untracked file. These tests pin
   the new resolution rule so a future refactor that re-couples the
   path back to ``workspace_root`` fails loudly.

2. **REPORT cache invalidation.** ``run_phase`` reuses
   ``state/.../phase_result/report.json`` when present, which on a
   warm-restart skips the LLM session entirely and therefore skips
   ``mcp__turbo__append_tip``. The orchestrator now invalidates that
   cache when the on-disk report no longer matches the current
   :class:`RunState` (best round changed, or more rounds happened),
   so newly accepted / rolled-back rounds get a chance to persist tips.
   These tests exercise the four cases — fresh / new best / new total
   / corrupt — that the staleness rule has to cover.
"""

from __future__ import annotations

from pathlib import Path

from turbo_optimize.config import (
    CampaignParams,
    default_tips_root,
)
from turbo_optimize.mcp import _context_from_params
from turbo_optimize.mcp.tips import _tips_path, append_tip_impl, query_tips_impl
from turbo_optimize.orchestrator.campaign import _invalidate_stale_report_cache
from turbo_optimize.orchestrator.phases.read_historical_tips import (
    tips_path as phase_tips_path,
)
from turbo_optimize.state import RunState, phase_result_path, write_phase_result


# --- tips_root resolution -------------------------------------------------


def test_default_tips_root_uses_tool_repo_when_env_unset(monkeypatch):
    """Without ``TURBO_TIPS_ROOT`` the default points at
    ``<tool_repo>/agent_data/historical_experience``. Locking this
    mapping prevents an accidental revert to the workspace-rooted
    layout that the 202604231519 campaign got bitten by."""
    monkeypatch.delenv("TURBO_TIPS_ROOT", raising=False)
    root = default_tips_root()
    assert root.name == "historical_experience"
    assert root.parent.name == "agent_data"
    # tool_repo must NOT be the optimized-project workspace; verify by
    # checking that ``turbo_optimize/`` lives next to ``agent_data/``.
    assert (root.parent.parent / "turbo_optimize").is_dir(), (
        f"default_tips_root resolved to {root} whose grandparent is "
        f"not the orchestrator repo (no turbo_optimize/ sibling)"
    )


def test_default_tips_root_honours_env_override(monkeypatch, tmp_path):
    """``TURBO_TIPS_ROOT`` lets the user share one knowledge base across
    multiple worktrees / virtualenvs. The override must be absolute
    (resolve user-home + symlinks) and bypass the tool-repo default."""
    custom = tmp_path / "shared_tips_root"
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(custom))
    root = default_tips_root()
    assert root == custom.resolve()


def test_resolved_tips_root_falls_back_to_default(monkeypatch, tmp_path):
    """``CampaignParams.tips_root=None`` (default + every old
    ``run.json`` written before this field existed) must resolve via
    :func:`default_tips_root`. Otherwise resumed campaigns would crash
    with ``None`` propagating into ``_tips_path``."""
    custom = tmp_path / "env_tips"
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(custom))
    params = CampaignParams(
        campaign_dir=tmp_path / "campaign",
        workspace_root=tmp_path / "ws",
    )
    assert params.tips_root is None
    assert params.resolved_tips_root() == custom.resolve()


def test_resolved_tips_root_uses_explicit_field_when_set(tmp_path, monkeypatch):
    """Explicit ``tips_root`` overrides both env and default. The
    orchestrator never sets this today, but the field exists so the
    ``run.json`` resume path can pin a specific root if needed."""
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(tmp_path / "should_be_ignored"))
    explicit = tmp_path / "explicit"
    params = CampaignParams(
        campaign_dir=tmp_path / "campaign",
        workspace_root=tmp_path / "ws",
        tips_root=explicit,
    )
    assert params.resolved_tips_root() == explicit.resolve()


def test_tips_path_is_decoupled_from_workspace_root(monkeypatch, tmp_path):
    """Regression guard: ``tips_path(params)`` must NOT include
    ``workspace_root`` anywhere in its components. The previous layout
    nested the file inside ``workspace_root``, which is exactly the
    coupling the rollback bug exploited."""
    tips_root = tmp_path / "tips"
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(tips_root))
    workspace = tmp_path / "isolated_ws"
    params = CampaignParams(
        campaign_dir=tmp_path / "campaign",
        workspace_root=workspace,
        target_op="grouped_gemm_fp8_tensorwise",
        target_backend="TRITON",
        target_gpu="gfx950",
    )
    path = phase_tips_path(params)
    parts = set(Path(*path.parts).parts)
    assert "isolated_ws" not in parts, (
        f"tips_path must not be rooted in workspace_root; got {path}"
    )
    assert path == tips_root.resolve() / "gfx950" / "grouped_gemm_fp8_tensorwise" / "triton" / "tips.md"


def test_tips_path_lowercases_backend_segment(monkeypatch, tmp_path):
    """Backend casing has historically varied between manifests
    (``TRITON`` vs ``triton`` vs ``Triton``); the on-disk segment must
    always be lowercase so query/append from the same campaign agree."""
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(tmp_path / "tips"))
    params = CampaignParams(
        campaign_dir=tmp_path / "c",
        workspace_root=tmp_path / "ws",
        target_op="op",
        target_backend="TRITON",
        target_gpu="gfx950",
    )
    assert phase_tips_path(params).parts[-2] == "triton"


# --- MCP append/query under the new root ---------------------------------


def test_append_tip_writes_under_tips_root_not_workspace(monkeypatch, tmp_path):
    """The MCP ``append_tip`` tool must materialise the file under
    ``ctx.tips_root``. This is the single test that would have caught
    the original bug — before the relocation, a green campaign + a
    rollback step would silently empty the tips file because it lived
    inside the swept ``workspace_root`` tree."""
    tips_root = tmp_path / "tips"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(tips_root))
    params = CampaignParams(
        campaign_dir=tmp_path / "campaign",
        workspace_root=workspace,
        target_op="op",
        target_backend="TRITON",
        target_gpu="gfx950",
    )
    (tmp_path / "campaign").mkdir()
    ctx = _context_from_params(params)

    out = append_tip_impl(
        ctx,
        op="op",
        backend="TRITON",
        gpu="gfx950",
        entry={
            "round": 1,
            "status": "ACCEPTED",
            "context": "ctx",
            "signal": "sig",
            "takeaway": "take",
            "applicability": "appl",
        },
    )
    written = Path(out["path"])
    assert written.exists()
    assert written.is_relative_to(tips_root.resolve())
    assert not written.is_relative_to(workspace.resolve()), (
        "append_tip must not write inside workspace_root; "
        "rollback would erase it"
    )


def test_query_tips_reads_under_tips_root(monkeypatch, tmp_path):
    """Round-trip: ``append_tip`` followed by ``query_tips`` returns the
    just-written entry. Pins the read-side to the same root used by
    the write-side; otherwise READ_HISTORICAL_TIPS would silently
    return zero entries on a campaign that just appended dozens."""
    tips_root = tmp_path / "tips"
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(tips_root))
    params = CampaignParams(
        campaign_dir=tmp_path / "campaign",
        workspace_root=tmp_path / "ws",
        target_op="op",
        target_backend="TRITON",
        target_gpu="gfx950",
    )
    (tmp_path / "campaign").mkdir()
    (tmp_path / "ws").mkdir()
    ctx = _context_from_params(params)

    append_tip_impl(
        ctx,
        op="op",
        backend="TRITON",
        gpu="gfx950",
        entry={
            "round": 7,
            "status": "ROLLED_BACK",
            "context": "ctxA",
            "signal": "sigA",
            "takeaway": "do not X",
            "applicability": "all gfx950 fp8 paths",
        },
    )
    out = query_tips_impl(
        ctx, op="op", backend="TRITON", gpu="gfx950", keyword=None
    )
    assert out["count"] == 1
    assert "do not X" in out["items"][0]["body"]


def test_tips_path_helper_matches_mcp_path(monkeypatch, tmp_path):
    """Regression guard: the prompt-rendered ``tips_path`` (computed by
    the orchestrator phase helper) must match the path the MCP server
    actually writes to. Drift here means READ_HISTORICAL_TIPS would
    look at a different file from REPORT, defeating the whole knowledge
    base."""
    tips_root = tmp_path / "tips"
    monkeypatch.setenv("TURBO_TIPS_ROOT", str(tips_root))
    params = CampaignParams(
        campaign_dir=tmp_path / "campaign",
        workspace_root=tmp_path / "ws",
        target_op="grouped_gemm_fp8_tensorwise",
        target_backend="TRITON",
        target_gpu="gfx950",
    )
    via_phase = phase_tips_path(params)
    via_mcp = _tips_path(
        params.resolved_tips_root(),
        gpu="gfx950",
        op="grouped_gemm_fp8_tensorwise",
        backend="TRITON",
    )
    assert via_phase == via_mcp


# --- REPORT cache invalidation -------------------------------------------


def _make_state(*, best: int, current: int) -> RunState:
    """Minimal RunState fixture; the helper only inspects ``best_round``
    and ``current_round`` so the rest of the fields stay default."""
    return RunState(
        campaign_id="cid",
        campaign_dir="/tmp/c",
        current_phase="TERMINATION_CHECK",
        current_round=current,
        best_round=best,
    )


def _seed_report_cache(
    state_dir: Path, *, cached_best: int | None, cached_total: int | None
) -> Path:
    """Drop a fake ``report.json`` with the requested ``final_best_aggregate.round``
    and ``total_rounds`` so the staleness rule has something to compare
    against."""
    payload: dict = {}
    if cached_best is not None:
        payload["final_best_aggregate"] = {"round": cached_best}
    if cached_total is not None:
        payload["total_rounds"] = cached_total
    return write_phase_result(state_dir, "REPORT", payload)


def test_report_cache_invalidation_no_op_when_cache_missing(tmp_path):
    """Cache miss is a no-op: ``run_phase`` will run REPORT for the
    first time anyway, and there is no file to unlink."""
    state = _make_state(best=34, current=50)
    _invalidate_stale_report_cache(tmp_path, state)
    # phase_result_path now does NOT exist; helper must not raise.
    assert not phase_result_path(tmp_path, "REPORT").exists()


def test_report_cache_invalidation_keeps_fresh_cache(tmp_path):
    """When the cached report's best/total agree with the current
    state, the cache is fresh and must be preserved — otherwise we'd
    pay for an LLM rerun on every termination."""
    cache = _seed_report_cache(tmp_path, cached_best=34, cached_total=50)
    state = _make_state(best=34, current=50)
    _invalidate_stale_report_cache(tmp_path, state)
    assert cache.exists(), "fresh REPORT cache must not be unlinked"


def test_report_cache_invalidation_unlinks_when_best_round_changed(tmp_path):
    """A new best round (warm-restart picked up a better ACCEPT than
    the previous REPORT saw) is the headline staleness signal: the
    "Final Best" table and the success-tip pool both shifted."""
    cache = _seed_report_cache(tmp_path, cached_best=34, cached_total=50)
    state = _make_state(best=77, current=100)
    _invalidate_stale_report_cache(tmp_path, state)
    assert not cache.exists(), (
        "REPORT cache must be unlinked when best_round changed; "
        "otherwise warm-restart's new ACCEPTs never get tip-distilled"
    )


def test_report_cache_invalidation_unlinks_when_more_rounds_happened(tmp_path):
    """Even when the best stayed the same, additional ROLLBACKs may
    carry reusable failure tips. ``total_rounds`` drift must therefore
    also force a rerun."""
    cache = _seed_report_cache(tmp_path, cached_best=34, cached_total=50)
    state = _make_state(best=34, current=80)
    _invalidate_stale_report_cache(tmp_path, state)
    assert not cache.exists(), (
        "REPORT cache must be unlinked when total_rounds grew; new "
        "ROLLBACKs may carry failure tips worth distilling"
    )


def test_report_cache_invalidation_no_op_on_unparseable_cache(tmp_path):
    """A corrupt/old-schema cache file is left in place: ``run_phase``
    already handles unloadable caches by re-running the phase, so this
    helper should not race with that recovery path."""
    cache_path = phase_result_path(tmp_path, "REPORT")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not-json-at-all", encoding="utf-8")
    state = _make_state(best=34, current=50)
    _invalidate_stale_report_cache(tmp_path, state)
    assert cache_path.exists(), (
        "corrupt REPORT cache should be left for run_phase to handle; "
        "double-handling could lose data on a partial write"
    )
