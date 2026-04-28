"""Tests for :func:`turbo_optimize.orchestrator.campaign._rollback_kernel`.

The function's failure mode pre-fix — copying only
``params.kernel_source`` to ``workspace_root/<basename>`` and leaving
every other modified file in place — caused a silent-rollback bug in
the 2026-04-23 grouped-GEMM campaign: rounds 4-6 each rolled back, each
rollback left its diff on disk, and round 7 finally crashed with
``FileNotFoundError`` on top of the accumulated mess. These tests
pin the fixed behaviour (git reset + git clean + submodule update)
and exercise the snapshot-copy fallback that remains in place for
non-git workspaces.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator import campaign as campaign_mod
from turbo_optimize.state import RunState


# --- helpers ---------------------------------------------------------


def _git(cwd: Path, *args: str) -> None:
    """Run a git subcommand silently; fail loud on non-zero exit."""
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(root: Path) -> None:
    """Create a minimal git repo with identity config + one commit.

    ``git commit`` requires ``user.email`` + ``user.name``; set them
    locally (not globally, to avoid the "never modify global git
    config" safety rule) so the test is self-contained.
    """
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "rollback-test")


def _make_workspace_with_baseline(tmp_path: Path) -> Path:
    """Return a git repo containing a two-level kernel file committed."""
    ws = tmp_path / "workspace"
    _init_repo(ws)

    kernel_dir = ws / "primus_turbo" / "triton" / "grouped_gemm"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "grouped_gemm_fp8_kernel.py").write_text(
        "# baseline kernel\n", encoding="utf-8"
    )
    cpp_dir = ws / "csrc" / "pytorch" / "quantization"
    cpp_dir.mkdir(parents=True)
    (cpp_dir / "quantization.cpp").write_text(
        "// baseline cpp\n", encoding="utf-8"
    )
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "baseline")
    return ws


def _make_params(workspace: Path, campaign_dir: Path, state_dir: Path) -> CampaignParams:
    return CampaignParams(
        prompt="test rollback",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
        campaign_dir=campaign_dir,
        kernel_source="primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py",
    )


def _make_state(best_round: int | None) -> RunState:
    """Lightweight RunState; only ``best_round`` is read by rollback."""
    state = RunState(campaign_id="unit-test-campaign")
    state.best_round = best_round
    return state


# --- tests -----------------------------------------------------------


def test_git_rollback_reverts_modified_tracked_files(tmp_path, caplog):
    """Canonical success path: tracked-file edits from the failed round
    are reverted to HEAD by ``git reset --hard``. Before the fix only
    the file listed in ``kernel_source`` was restored, and even that
    went to the wrong path."""
    caplog.set_level("INFO", logger="turbo_optimize.orchestrator.campaign")

    workspace = _make_workspace_with_baseline(tmp_path)
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    kernel = workspace / "primus_turbo" / "triton" / "grouped_gemm" / "grouped_gemm_fp8_kernel.py"
    cpp = workspace / "csrc" / "pytorch" / "quantization" / "quantization.cpp"
    kernel.write_text("# round-7 half-baked edit\n", encoding="utf-8")
    cpp.write_text("// round-7 fused quant draft\n", encoding="utf-8")

    params = _make_params(workspace, campaign_dir, state_dir)
    state = _make_state(best_round=3)

    campaign_mod._rollback_kernel(params, state, round_n=7)

    assert kernel.read_text(encoding="utf-8") == "# baseline kernel\n", (
        "git reset --hard must revert the Triton file that the failed "
        "round modified"
    )
    assert cpp.read_text(encoding="utf-8") == "// baseline cpp\n", (
        "git reset --hard must revert ALL modified tracked files, "
        "not just ``kernel_source``"
    )
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "git-reset workspace" in messages
    assert "round-7 changes discarded" in messages


def test_git_rollback_removes_untracked_files(tmp_path):
    """Round-7 OPTIMIZE wrote a stray ``grouped_gemm_fp8_kernel.py``
    at the workspace root (the old snapshot-copy path's off-target
    write). ``git clean -fd`` must remove every untracked file the
    failed round emitted, at any depth."""
    workspace = _make_workspace_with_baseline(tmp_path)
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    stray_root = workspace / "grouped_gemm_fp8_kernel.py"
    stray_root.write_text("# stray copy from broken rollback\n", encoding="utf-8")
    stray_nested = workspace / "csrc" / "kernels" / "quantization" / "quantization.hip"
    stray_nested.parent.mkdir(parents=True, exist_ok=True)
    stray_nested.write_text("// half-baked hip kernel\n", encoding="utf-8")

    params = _make_params(workspace, campaign_dir, state_dir)
    state = _make_state(best_round=3)

    campaign_mod._rollback_kernel(params, state, round_n=7)

    assert not stray_root.exists(), (
        "git clean -fd must delete untracked files at the repo root"
    )
    assert not stray_nested.exists(), (
        "git clean -fd must delete untracked files inside nested dirs"
    )


def test_git_rollback_runs_even_without_best_round(tmp_path):
    """The git path should work even before any round has been
    accepted (``best_round=None``). HEAD is still a real commit
    (PREPARE_ENVIRONMENT's base point) and ``git reset --hard HEAD``
    simply brings the workspace back to it. The old file-copy path
    early-returned in this case, which meant a rollback-before-first-
    accept silently did nothing."""
    workspace = _make_workspace_with_baseline(tmp_path)
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    kernel = workspace / "primus_turbo" / "triton" / "grouped_gemm" / "grouped_gemm_fp8_kernel.py"
    kernel.write_text("# bad round-2 edit\n", encoding="utf-8")

    params = _make_params(workspace, campaign_dir, state_dir)
    state = _make_state(best_round=None)

    campaign_mod._rollback_kernel(params, state, round_n=2)

    assert kernel.read_text(encoding="utf-8") == "# baseline kernel\n"


def test_rollback_falls_back_to_snapshot_when_not_a_git_repo(tmp_path, caplog):
    """A workspace without ``.git`` triggers the legacy snapshot-copy
    path. The log must clearly warn so an operator who hit this case
    can fix their setup rather than trust the limited fallback."""
    caplog.set_level("WARNING", logger="turbo_optimize.orchestrator.campaign")

    workspace = tmp_path / "plain_workspace"
    workspace.mkdir()
    (workspace / "primus_turbo" / "triton" / "grouped_gemm").mkdir(parents=True)
    stale = workspace / "primus_turbo" / "triton" / "grouped_gemm" / "grouped_gemm_fp8_kernel.py"
    stale.write_text("# stale round-4 edit\n", encoding="utf-8")

    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    snapshot_root = campaign_dir / "rounds" / "round-3" / "kernel_snapshot"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "grouped_gemm_fp8_kernel.py").write_text(
        "# baseline kernel\n", encoding="utf-8"
    )

    params = _make_params(workspace, campaign_dir, state_dir)
    state = _make_state(best_round=3)

    campaign_mod._rollback_kernel(params, state, round_n=4)

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "has no .git directory" in messages
    assert "falling back to snapshot-copy" in messages
    copied = workspace / "grouped_gemm_fp8_kernel.py"
    assert copied.exists(), (
        "snapshot-copy fallback still uses the pre-existing (buggy) "
        "flat-copy semantics; the test pins that behaviour rather "
        "than asserting it is correct, because the warning log is "
        "what operators should actually rely on to fix the setup"
    )


def test_rollback_falls_back_when_git_not_on_path(tmp_path, monkeypatch, caplog):
    """If ``git`` is absent from ``PATH``, rollback must not crash
    the campaign: log a warning and drop into snapshot-copy."""
    caplog.set_level("WARNING", logger="turbo_optimize.orchestrator.campaign")

    workspace = _make_workspace_with_baseline(tmp_path)
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    snapshot_root = campaign_dir / "rounds" / "round-3" / "kernel_snapshot"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "grouped_gemm_fp8_kernel.py").write_text(
        "# restored via snapshot\n", encoding="utf-8"
    )

    real_run = subprocess.run

    def _fake_run(argv, *args, **kwargs):
        if isinstance(argv, list) and argv and argv[0] == "git":
            raise FileNotFoundError("git: no such executable")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(
        "turbo_optimize.orchestrator.campaign.subprocess.run", _fake_run
    )

    params = _make_params(workspace, campaign_dir, state_dir)
    state = _make_state(best_round=3)

    campaign_mod._rollback_kernel(params, state, round_n=5)

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "`git` executable not on PATH" in messages
    assert "falling back to snapshot-copy" in messages


def test_git_rollback_cleans_submodule_untracked_files(tmp_path):
    """Round-7 left 40+ generated ``*_hip.hpp`` files inside the
    ``3rdparty/composable_kernel`` submodule on the real campaign.
    Plain ``git submodule update --recursive`` would leave them in
    place because they are untracked in the submodule; only
    ``git submodule foreach ... git clean -fd`` removes them. Pin
    this behaviour with a tiny fixture submodule so the fix is
    tested end-to-end."""
    parent = _make_workspace_with_baseline(tmp_path)
    sub_src = tmp_path / "sub_src"
    _init_repo(sub_src)
    (sub_src / "baseline.h").write_text("// sub baseline\n", encoding="utf-8")
    _git(sub_src, "add", "-A")
    _git(sub_src, "commit", "-q", "-m", "sub baseline")

    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(sub_src),
        "3rdparty/sub",
    )
    _git(parent, "commit", "-q", "-m", "add submodule")

    stray = parent / "3rdparty" / "sub" / "generated_hip.hpp"
    stray.write_text("// generated during round-7\n", encoding="utf-8")

    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    params = _make_params(parent, campaign_dir, state_dir)
    state = _make_state(best_round=3)

    campaign_mod._rollback_kernel(params, state, round_n=7)

    assert not stray.exists(), (
        "submodule foreach + git clean -fd must delete untracked files "
        "dumped into a submodule during the failed round"
    )


def test_git_rollback_surfaces_command_failures_as_fallback(
    tmp_path, monkeypatch, caplog
):
    """If ``git reset`` fails mid-rollback (hook rejection, corrupt
    index, …), we should not crash the campaign: log the stderr head
    and fall back to snapshot-copy."""
    caplog.set_level("WARNING", logger="turbo_optimize.orchestrator.campaign")

    workspace = _make_workspace_with_baseline(tmp_path)
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    campaign_dir.mkdir()
    state_dir.mkdir()

    snapshot_root = campaign_dir / "rounds" / "round-3" / "kernel_snapshot"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "grouped_gemm_fp8_kernel.py").write_text(
        "# restored via snapshot\n", encoding="utf-8"
    )

    real_run = subprocess.run

    def _fake_run(argv, *args, **kwargs):
        if isinstance(argv, list) and argv[:2] == ["git", "reset"]:
            raise subprocess.CalledProcessError(
                returncode=128,
                cmd=argv,
                stderr="fatal: simulated reset failure\n",
            )
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(
        "turbo_optimize.orchestrator.campaign.subprocess.run", _fake_run
    )

    params = _make_params(workspace, campaign_dir, state_dir)
    state = _make_state(best_round=3)

    campaign_mod._rollback_kernel(params, state, round_n=6)

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "rc=128" in messages
    assert "falling back to snapshot-copy" in messages
