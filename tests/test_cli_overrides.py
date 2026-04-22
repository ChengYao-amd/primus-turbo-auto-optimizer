"""Tests for CLI-authoritative overrides that reach DEFINE_TARGET.

Two behaviours are pinned down here because both were silently dropped in
earlier revisions:

* ``--max-iterations N`` and ``--max-duration`` must be rendered into the
  DEFINE_TARGET prompt verbatim, so Claude writes the exact same value into
  ``manifest.yaml``. Claude has no way to "guess" the right value otherwise.
* ``--git-commit`` / ``--no-git-commit`` is CLI-authoritative. The default is
  off, and :meth:`CampaignParams.merge_manifest` must not let a stale
  ``git_commit: true`` in a manifest re-enable commits behind the user's
  back.
"""

from __future__ import annotations

from pathlib import Path

from turbo_optimize.cli import _build_parser, _build_params
from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator.phases.define_target import _yaml_value
from turbo_optimize.skills import render_prompt


def _parse(argv: list[str]):
    parser = _build_parser()
    return parser.parse_args(argv)


def test_git_commit_default_is_off():
    args = _parse(["-p", "opt gemm"])
    params = _build_params(args)
    assert params.git_commit is False


def test_git_commit_flag_turns_on():
    args = _parse(["-p", "opt gemm", "--git-commit"])
    params = _build_params(args)
    assert params.git_commit is True


def test_no_git_commit_flag_stays_off():
    args = _parse(["-p", "opt gemm", "--no-git-commit"])
    params = _build_params(args)
    assert params.git_commit is False


def test_max_iterations_surfaces_on_params():
    args = _parse(["-p", "opt gemm", "--max-iterations", "3"])
    params = _build_params(args)
    assert params.max_iterations == 3


def test_merge_manifest_does_not_re_enable_git_commit():
    """A stale `git_commit: true` in manifest.yaml must not flip the CLI
    default back on."""
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
        git_commit=False,
    )
    params.merge_manifest(
        {
            "target_op": "gemm",
            "target_backend": "triton",
            "target_gpu": "mi300x",
            "execution_mode": "repo",
            "primary_metric": "tflops",
            "kernel_source": "a.py",
            "test_command": "pytest",
            "benchmark_command": "python bench.py",
            "git_commit": True,
        }
    )
    assert params.git_commit is False


def test_merge_manifest_respects_explicit_git_commit_on():
    """``--git-commit`` must also survive merge_manifest even if the manifest
    omits the field."""
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
        git_commit=True,
    )
    params.merge_manifest({"target_op": "gemm"})
    assert params.git_commit is True


def test_yaml_value_renders_none_as_literal_null():
    assert _yaml_value(None) == "null"
    assert _yaml_value(3) == "3"
    assert _yaml_value("4h") == "4h"


def test_define_target_prompt_embeds_cli_overrides(tmp_path):
    """render_prompt('define_target', ...) must inline the three CLI
    override lines so Claude sees them in its context."""
    skill = "<skill-excerpt>"
    text = render_prompt(
        "define_target",
        {
            "skill_excerpt": skill,
            "user_prompt": "optimize gemm fp8 blockwise",
            "project_skill_path": "/p/skills/primus-turbo",
            "project_skill": "primus-turbo",
            "campaign_dir": str(tmp_path),
            "phase_result_path": str(tmp_path / "state" / "phase" / "d.json"),
            "manifest_path": str(tmp_path / "manifest.yaml"),
            "cli_max_iterations": _yaml_value(3),
            "cli_max_duration": _yaml_value(None),
            "cli_git_commit": "false",
        },
    )
    assert "max_iterations: 3" in text
    assert "max_duration:   null" in text
    assert "git_commit:     false" in text
    assert "AUTHORITATIVE" in text
