"""Tests for the CLI-exposed model / effort settings.

Covers:
* Hardcoded fallbacks match the advertised contract
  (``claude-opus-4-7`` + ``max``) so docs and code cannot drift silently.
* :meth:`CampaignParams.resolve_runtime_defaults` applies the fallback
  when both CLI and resumed state are silent, respects explicit values,
  and rejects unknown effort tags before any SDK call.
* :func:`turbo_optimize.orchestrator.run_phase._build_options` pipes
  ``model`` and ``effort`` into the final ``ClaudeAgentOptions``.

The older env-variable fallback (``ANTHROPIC_MODEL`` /
``CLAUDE_CODE_EFFORT_LEVEL``) was dropped: a single slug lives in one
place (fallback constants) and everything else flows through the CLI or
``run.json``. No test here touches ``os.environ`` for those names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turbo_optimize.config import (
    EFFORT_CHOICES,
    EFFORT_FALLBACK,
    MODEL_FALLBACK,
    CampaignParams,
)
from turbo_optimize.orchestrator.run_phase import (
    PhaseInvocation,
    _build_options,
)


def test_fallbacks_are_claude_opus_4_7_and_max():
    assert MODEL_FALLBACK == "claude-opus-4-7"
    assert EFFORT_FALLBACK == "max"
    assert "max" in EFFORT_CHOICES


def test_resolve_runtime_defaults_fills_from_fallback():
    params = CampaignParams(prompt="x")
    params.resolve_runtime_defaults()
    assert params.model == MODEL_FALLBACK
    assert params.effort == EFFORT_FALLBACK


def test_resolve_runtime_defaults_cli_beats_fallback():
    params = CampaignParams(
        prompt="x", model="claude-sonnet-4-5", effort="low"
    )
    params.resolve_runtime_defaults()
    assert params.model == "claude-sonnet-4-5"
    assert params.effort == "low"


def test_resolve_runtime_defaults_ignores_env(monkeypatch):
    """Setting the legacy env names must not leak into the params."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-garbage")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "nope")
    params = CampaignParams(prompt="x")
    params.resolve_runtime_defaults()
    assert params.model == MODEL_FALLBACK
    assert params.effort == EFFORT_FALLBACK


def test_resolve_runtime_defaults_rejects_bad_effort():
    params = CampaignParams(prompt="x", effort="turbo")
    with pytest.raises(ValueError, match="invalid effort"):
        params.resolve_runtime_defaults()


def test_build_options_pipes_model_and_effort():
    invocation = PhaseInvocation(
        phase="TEST",
        prompt="noop",
        allowed_tools=["Read"],
        system_prompt="sys",
        cwd=Path("/tmp"),
        model="claude-opus-4-7",
        effort="max",
    )
    options = _build_options(invocation)
    assert options.model == "claude-opus-4-7"
    assert options.effort == "max"


def test_build_options_omits_model_when_none():
    """When model/effort are unset the SDK default path is preserved."""
    invocation = PhaseInvocation(
        phase="TEST",
        prompt="noop",
        allowed_tools=["Read"],
        system_prompt="sys",
        cwd=Path("/tmp"),
        model=None,
        effort=None,
    )
    options = _build_options(invocation)
    assert options.model is None
    assert options.effort is None
