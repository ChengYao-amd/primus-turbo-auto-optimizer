"""Tests for CLI-authoritative overrides that reach DEFINE_TARGET.

Three behaviours are pinned down here because all three were silently
dropped in earlier revisions:

* ``--max-iterations N`` and ``--max-duration`` must be rendered into the
  DEFINE_TARGET prompt verbatim, so Claude writes the exact same value into
  ``manifest.yaml``. Claude has no way to "guess" the right value otherwise.
* ``--base-branch`` is CLI-authoritative: it overrides whatever Claude
  writes into ``manifest.yaml``.
* ``git_commit`` / ``git_branch`` are NOT CLI knobs anymore. They are
  forced to ``true`` / ``auto`` by
  :data:`turbo_optimize.orchestrator.campaign.FORCED_GIT_COMMIT` /
  :data:`turbo_optimize.orchestrator.campaign.FORCED_GIT_BRANCH`, and the
  CLI deliberately exposes no flag to change them; any ``git_commit`` /
  ``git_branch`` keys in a legacy ``manifest.yaml`` are ignored.
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


def test_max_iterations_surfaces_on_params():
    args = _parse(["-p", "opt gemm", "--max-iterations", "3"])
    params = _build_params(args)
    assert params.max_iterations == 3


def test_cli_has_no_git_commit_flag():
    """The former ``--git-commit`` / ``--no-git-commit`` flags must not
    reappear: the orchestrator forces git_commit=true unconditionally
    (see :data:`FORCED_GIT_COMMIT`)."""
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-p", "opt gemm", "--git-commit"])
    with pytest.raises(SystemExit):
        parser.parse_args(["-p", "opt gemm", "--no-git-commit"])


def test_campaign_params_has_no_git_commit_attr():
    """CampaignParams must not expose a mutable ``git_commit`` field; the
    value is a module-level constant on purpose."""
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
    )
    assert not hasattr(params, "git_commit")
    assert not hasattr(params, "git_branch")


def test_merge_manifest_ignores_legacy_git_keys():
    """Legacy manifests that still carry git_commit / git_branch must
    not crash merge_manifest; the keys are silently dropped."""
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
    )
    params.merge_manifest(
        {
            "target_op": "gemm",
            "target_backend": "triton",
            "git_commit": True,
            "git_branch": "optimize/legacy",
        }
    )
    assert not hasattr(params, "git_commit")
    assert not hasattr(params, "git_branch")


def test_base_branch_default_is_none():
    args = _parse(["-p", "opt gemm"])
    params = _build_params(args)
    assert params.base_branch is None


def test_base_branch_flag_sets_value():
    args = _parse(["-p", "opt gemm", "--base-branch", "develop"])
    params = _build_params(args)
    assert params.base_branch == "develop"


def test_base_branch_cli_overrides_manifest():
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
        base_branch="develop",
    )
    params.merge_manifest({"target_op": "gemm", "base_branch": "main"})
    assert params.base_branch == "develop"


def test_base_branch_falls_back_to_manifest():
    params = CampaignParams(
        prompt="opt gemm",
        workspace_root=Path("/tmp/ws"),
        state_dir=Path("/tmp/ws/state"),
        skills_root=Path("/tmp/skills"),
    )
    params.merge_manifest({"target_op": "gemm", "base_branch": "main"})
    assert params.base_branch == "main"


def test_expand_campaign_vars_replaces_placeholder():
    params = CampaignParams(
        prompt="opt gemm",
        campaign_dir=Path("/workspace/campaign_001"),
        workspace_root=Path("/workspace"),
        state_dir=Path("/workspace/state"),
        skills_root=Path("/workspace/skills"),
    )
    params.merge_manifest({
        "target_op": "gemm",
        "quick_command": "python ${CAMPAIGN_DIR}/quick_test_bench.py",
        "profile_command": "python ${CAMPAIGN_DIR}/profile_op_shape.py",
        "related_work_file": "${CAMPAIGN_DIR}/related_work.md",
    })
    assert params.quick_command == "python /workspace/campaign_001/quick_test_bench.py"
    assert params.profile_command == "python /workspace/campaign_001/profile_op_shape.py"
    assert params.related_work_file == "/workspace/campaign_001/related_work.md"


def test_expand_campaign_vars_noop_without_placeholder():
    params = CampaignParams(
        prompt="opt gemm",
        campaign_dir=Path("/workspace/campaign_001"),
        workspace_root=Path("/workspace"),
        state_dir=Path("/workspace/state"),
        skills_root=Path("/workspace/skills"),
    )
    params.merge_manifest({
        "target_op": "gemm",
        "quick_command": "python quick_test_bench.py",
        "profile_command": "python profile_op_shape.py",
    })
    assert params.quick_command == "python quick_test_bench.py"
    assert params.profile_command == "python profile_op_shape.py"


def test_yaml_value_renders_none_as_literal_null():
    assert _yaml_value(None) == "null"
    assert _yaml_value(3) == "3"
    assert _yaml_value("4h") == "4h"


def test_define_target_prompt_embeds_cli_overrides(tmp_path):
    """render_prompt('define_target', ...) must inline the four CLI
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
            "cli_base_branch": _yaml_value("develop"),
        },
    )
    assert "max_iterations: 3" in text
    assert "max_duration:   null" in text
    assert "base_branch:    develop" in text
    assert "AUTHORITATIVE" in text
    assert "git_commit: true" in text, (
        "template must surface the forced git policy so reviewers know "
        "commits happen even though the flag is not in the CLI"
    )
    assert "git_branch: auto" in text
    assert "${CAMPAIGN_DIR}/quick_test_bench.py" in text, (
        "quick_command placeholder must use ${CAMPAIGN_DIR} template variable"
    )
    assert "${CAMPAIGN_DIR}/profile_op_shape.py" in text, (
        "profile_command placeholder must use ${CAMPAIGN_DIR} template variable"
    )
    assert "${CAMPAIGN_DIR}/related_work.md" in text, (
        "related_work_file placeholder must use ${CAMPAIGN_DIR} template variable"
    )


def test_define_target_prompt_base_branch_null_when_unset(tmp_path):
    """When CLI does not pass --base-branch, the prompt shows 'null'
    so Claude falls back to its default ('main')."""
    text = render_prompt(
        "define_target",
        {
            "skill_excerpt": "x",
            "user_prompt": "opt gemm",
            "project_skill_path": "/p",
            "project_skill": "p",
            "campaign_dir": str(tmp_path),
            "phase_result_path": str(tmp_path / "d.json"),
            "manifest_path": str(tmp_path / "m.yaml"),
            "cli_max_iterations": _yaml_value(None),
            "cli_max_duration": _yaml_value(None),
            "cli_base_branch": _yaml_value(None),
        },
    )
    assert "base_branch:    null" in text
