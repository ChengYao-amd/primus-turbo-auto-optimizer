"""Tests for ``turbo_optimize.orchestrator.warm_restart.write_script``.

The warm-restart helper writes a bash wrapper to ``<campaign_dir>/
warm_restart.sh`` so the operator can resume a campaign without having
to remember the absolute paths. These tests cover:

* the happy path (file written, executable bit set, paths burned in);
* the guarded no-ops (campaign_id / campaign_dir missing);
* the generated script's bash syntax (``bash -n``);
* the user-facing flags (``--help``, missing args -> exit 2, one-flag
  is sufficient to succeed once ``primus-turbo-optimize`` itself is
  shadowed out with a stub on ``$PATH``);
* shell-injection safety when the campaign_id contains metacharacters.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator import warm_restart


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="bash is required for warm_restart script tests",
)


def _params(tmp_path: Path, *, campaign_id: str = "demo_campaign_20260420") -> CampaignParams:
    campaign_dir = tmp_path / "agent" / "workspace" / campaign_id
    campaign_dir.mkdir(parents=True)
    return CampaignParams(
        prompt="stub",
        campaign_id=campaign_id,
        campaign_dir=campaign_dir,
        workspace_root=tmp_path,
        skills_root=tmp_path / "agent",
        state_dir=tmp_path / "state",
    )


def test_write_script_returns_none_without_campaign_dir(tmp_path: Path) -> None:
    params = CampaignParams(
        prompt="stub",
        workspace_root=tmp_path,
        skills_root=tmp_path / "agent",
        state_dir=tmp_path / "state",
    )
    assert warm_restart.write_script(params) is None


def test_write_script_returns_none_without_campaign_id(tmp_path: Path) -> None:
    params = _params(tmp_path)
    params.campaign_id = None
    assert warm_restart.write_script(params) is None


def test_write_script_creates_executable_file(tmp_path: Path) -> None:
    params = _params(tmp_path)
    path = warm_restart.write_script(params)
    assert path is not None
    assert path.name == "warm_restart.sh"
    mode = path.stat().st_mode
    assert mode & stat.S_IXUSR, "owner should have execute bit"
    assert mode & stat.S_IXGRP, "group should have execute bit"
    assert mode & stat.S_IXOTH, "other should have execute bit"


def test_write_script_burns_in_paths_and_campaign_id(tmp_path: Path) -> None:
    params = _params(tmp_path, campaign_id="gemm_fp8_blockwise_20260420")
    path = warm_restart.write_script(params)
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "CAMPAIGN_ID=gemm_fp8_blockwise_20260420" in text
    assert f"WORKSPACE_ROOT={params.workspace_root}" in text
    assert f"SKILLS_ROOT={params.skills_root}" in text
    assert f"STATE_DIR={params.state_dir}" in text
    assert "primus-turbo-optimize -s \"$CAMPAIGN_ID\"" in text


def test_write_script_bash_syntax_is_valid(tmp_path: Path) -> None:
    params = _params(tmp_path)
    path = warm_restart.write_script(params)
    assert path is not None
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_script_help_flag_exits_zero_and_mentions_flags(tmp_path: Path) -> None:
    params = _params(tmp_path)
    path = warm_restart.write_script(params)
    assert path is not None
    result = subprocess.run(
        ["bash", str(path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "-i, --iterations" in out
    assert "-d, --duration" in out
    assert "At least one of -i / -d is required" in out


def test_script_without_args_exits_2_and_errors_out(tmp_path: Path) -> None:
    params = _params(tmp_path)
    path = warm_restart.write_script(params)
    assert path is not None
    result = subprocess.run(
        ["bash", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "supply at least one of -i / -d" in result.stderr


_STUB_ARGV_MARKER = "__STUB_ARGV__"


def _stub_primus_on_path(tmp_path: Path) -> dict[str, str]:
    """Return an env dict whose PATH fronts a stub ``primus-turbo-optimize``.

    The stub prints a marker line, then one argv element per NUL-
    terminated record. The marker lets the assertions skip past the
    ``+ cmd...`` trace that ``warm_restart.sh`` emits just before
    ``exec``, without accidentally swallowing real args that happen to
    contain whitespace.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "primus-turbo-optimize"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' '{_STUB_ARGV_MARKER}'\n"
        "printf '%s\\0' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return env


def _decode_stub_argv(stdout: str) -> list[str]:
    _, _, tail = stdout.partition(_STUB_ARGV_MARKER + "\n")
    return [arg for arg in tail.split("\0") if arg]


def test_script_iteration_flag_forwards_expected_args(tmp_path: Path) -> None:
    params = _params(tmp_path)
    path = warm_restart.write_script(params)
    assert path is not None
    env = _stub_primus_on_path(tmp_path)
    result = subprocess.run(
        ["bash", str(path), "-i", "7"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    argv = _decode_stub_argv(result.stdout)
    assert "-s" in argv and params.campaign_id in argv
    assert "--max-iterations" in argv
    assert argv[argv.index("--max-iterations") + 1] == "7"
    assert "--workspace-root" in argv
    assert "--max-duration" not in argv


def test_script_accepts_both_iteration_and_duration_with_passthrough(
    tmp_path: Path,
) -> None:
    params = _params(tmp_path)
    path = warm_restart.write_script(params)
    assert path is not None
    env = _stub_primus_on_path(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(path),
            "-i",
            "5",
            "-d",
            "2h",
            "--",
            "--debug-retry",
            "4",
            "-v",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    argv = _decode_stub_argv(result.stdout)
    assert "--max-iterations" in argv
    assert argv[argv.index("--max-iterations") + 1] == "5"
    assert "--max-duration" in argv
    assert argv[argv.index("--max-duration") + 1] == "2h"
    assert "--debug-retry" in argv
    assert argv[argv.index("--debug-retry") + 1] == "4"
    assert "-v" in argv


def test_script_rewrite_is_idempotent(tmp_path: Path) -> None:
    params = _params(tmp_path)
    first = warm_restart.write_script(params)
    assert first is not None
    body1 = first.read_text(encoding="utf-8")
    second = warm_restart.write_script(params)
    assert second == first
    body2 = second.read_text(encoding="utf-8")
    nonts1 = "\n".join(
        line for line in body1.splitlines() if "Generated by primus-turbo-optimize" not in line
    )
    nonts2 = "\n".join(
        line for line in body2.splitlines() if "Generated by primus-turbo-optimize" not in line
    )
    assert nonts1 == nonts2


def test_script_quotes_campaign_id_with_special_chars(tmp_path: Path) -> None:
    params = _params(tmp_path, campaign_id="weird id; echo pwned")
    path = warm_restart.write_script(params)
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "CAMPAIGN_ID='weird id; echo pwned'" in text
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
