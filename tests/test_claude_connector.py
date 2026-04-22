"""Tests for ClaudeCodeConnector env plumbing, especially IS_SANDBOX.

Claude Code's native root check rejects ``--dangerously-skip-permissions``
(the CLI flag that SDK ``permission_mode="bypassPermissions"`` maps to)
when ``euid == 0``. Inside containers we're already sandboxed and exporting
``IS_SANDBOX=1`` is Anthropic's documented opt-in. The connector must
insert that flag automatically so callers don't have to remember.
"""

from __future__ import annotations

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from turbo_optimize.model_connnector.claude_code_connector import (
    ClaudeCodeConnector,
    _needs_sandbox_flag,
)


def _make_options(**kwargs) -> ClaudeAgentOptions:
    kwargs.setdefault("system_prompt", "sys")
    return ClaudeAgentOptions(**kwargs)


def _force_root(monkeypatch, *, is_root: bool) -> None:
    monkeypatch.setattr(
        "turbo_optimize.model_connnector.claude_code_connector._is_root",
        lambda: is_root,
    )


def test_sandbox_auto_inject_for_root_bypass(monkeypatch):
    _force_root(monkeypatch, is_root=True)
    options = _make_options(permission_mode="bypassPermissions")
    conn = ClaudeCodeConnector(options=options, load_auth=False)
    assert conn._options.env.get("IS_SANDBOX") == "1"


def test_sandbox_not_injected_for_non_root(monkeypatch):
    _force_root(monkeypatch, is_root=False)
    options = _make_options(permission_mode="bypassPermissions")
    conn = ClaudeCodeConnector(options=options, load_auth=False)
    assert "IS_SANDBOX" not in conn._options.env


def test_sandbox_not_injected_without_bypass(monkeypatch):
    _force_root(monkeypatch, is_root=True)
    options = _make_options(permission_mode=None)
    conn = ClaudeCodeConnector(options=options, load_auth=False)
    assert "IS_SANDBOX" not in conn._options.env


def test_sandbox_respects_explicit_opt_out(monkeypatch):
    """If the caller sets IS_SANDBOX=0 we must not rewrite it to 1."""
    _force_root(monkeypatch, is_root=True)
    options = _make_options(
        permission_mode="bypassPermissions",
        env={"IS_SANDBOX": "0"},
    )
    conn = ClaudeCodeConnector(options=options, load_auth=False)
    assert conn._options.env["IS_SANDBOX"] == "0"


def test_sandbox_fills_empty_value(monkeypatch):
    """Empty-string is treated as "not configured" and filled in."""
    _force_root(monkeypatch, is_root=True)
    options = _make_options(
        permission_mode="bypassPermissions",
        env={"IS_SANDBOX": ""},
    )
    conn = ClaudeCodeConnector(options=options, load_auth=False)
    assert conn._options.env["IS_SANDBOX"] == "1"


def test_needs_sandbox_flag_pure(monkeypatch):
    """Direct unit test of the helper, independent of __init__ wiring."""
    _force_root(monkeypatch, is_root=True)
    opts = _make_options(permission_mode="bypassPermissions")
    assert _needs_sandbox_flag(opts, {}) is True
    assert _needs_sandbox_flag(opts, {"IS_SANDBOX": "1"}) is False
    assert _needs_sandbox_flag(opts, {"IS_SANDBOX": "0"}) is False

    _force_root(monkeypatch, is_root=False)
    assert _needs_sandbox_flag(opts, {}) is False
