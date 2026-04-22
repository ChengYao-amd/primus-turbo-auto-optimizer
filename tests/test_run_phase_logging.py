"""Verify that ``run_phase`` emits INFO progress lines and aggregates cost.

We substitute a fake ``ClaudeCodeConnector`` so no Claude subprocess is
spawned; the fake yields a canned stream that includes a ``ResultMessage``
so the totals-accumulator branch is exercised.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable

import pytest
from claude_agent_sdk import ResultMessage

from turbo_optimize.config import CampaignParams
from turbo_optimize.orchestrator import run_phase as run_phase_module


class _FakeClient:
    async def interrupt(self):  # pragma: no cover - SIGINT path not exercised
        return None


class _FakeConnector:
    """Async context manager stand-in for ``ClaudeCodeConnector``.

    Yields a pre-scripted list of SDK messages from ``ask`` regardless of
    the prompt text.
    """

    def __init__(self, messages: Iterable[object]):
        self._messages = list(messages)
        self._client = _FakeClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def ask(self, prompt: str):  # noqa: D401 - matches real signature
        for msg in self._messages:
            yield msg


def _make_result(cost: float, turns: int, duration_ms: int) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=duration_ms,
        duration_api_ms=duration_ms,
        is_error=False,
        num_turns=turns,
        session_id="sess-fake",
        total_cost_usd=cost,
    )


def test_run_phase_emits_begin_end_and_aggregates_cost(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="turbo_optimize.orchestrator.run_phase")

    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    campaign_dir.mkdir()
    state_dir.mkdir()

    params = CampaignParams(
        prompt="unit test",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
    )

    messages = [
        _make_result(cost=0.25, turns=3, duration_ms=1500),
        _make_result(cost=0.10, turns=2, duration_ms=800),
    ]

    def _factory(*, options):
        return _FakeConnector(messages)

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _factory)

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_TEST",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read", "Write"],
        )
    )

    assert outcome.phase == "UNIT_TEST"
    assert outcome.structured is None
    assert outcome.stopped is False

    messages_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "[UNIT_TEST] phase begin" in messages_text
    assert "tools=2" in messages_text
    assert "[UNIT_TEST] phase end" in messages_text
    assert "status=ok" in messages_text
    assert "turns=5" in messages_text
    assert "cost=$0.3500" in messages_text
    assert "sdk=0.8s" in messages_text

    cost_md = (campaign_dir / "logs" / "cost.md").read_text(encoding="utf-8")
    assert "Campaign Cost Log" in cost_md
    assert "| UNIT_TEST |" in cost_md
    assert "| ok |" in cost_md
    assert "| $0.3500 | $0.3500 |" in cost_md


def test_run_phase_reuses_cached_expected_output(tmp_path, monkeypatch, caplog):
    """If ``expected_output`` already exists (e.g. crashed before
    ``advance_phase``), skip the Claude session and return the cached JSON.

    Without this short-circuit, ``-s <campaign>`` resumes always re-pay for a
    phase the previous run actually finished — the exact failure mode behind
    the PREPARE_ENVIRONMENT crash.
    """
    caplog.set_level(logging.INFO, logger="turbo_optimize.orchestrator.run_phase")

    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    campaign_dir.mkdir()
    state_dir.mkdir()

    params = CampaignParams(
        prompt="unit test",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
    )

    expected = state_dir / "phase_result" / "unit_cache.json"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text('{"cached": true, "round": 1}', encoding="utf-8")

    def _boom(*, options):  # pragma: no cover - must not be called
        raise AssertionError("ClaudeCodeConnector must NOT be invoked when "
                             "expected_output is already present")

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _boom)

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_CACHE",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
            expected_output=expected,
        )
    )
    assert outcome.structured == {"cached": True, "round": 1}
    assert outcome.stopped is False
    messages_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "[UNIT_CACHE] reusing cached output" in messages_text
    assert "phase begin" not in messages_text

    cost_md = (campaign_dir / "logs" / "cost.md").read_text(encoding="utf-8")
    assert "| UNIT_CACHE |" in cost_md
    assert "| cached |" in cost_md
    assert "| $0.0000 | $0.0000 |" in cost_md


def test_run_phase_ignores_invalid_cache(tmp_path, monkeypatch, caplog):
    """A half-written cache must not block re-execution; fall through to the
    real Claude path (here mocked to succeed)."""
    caplog.set_level(logging.INFO, logger="turbo_optimize.orchestrator.run_phase")

    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    campaign_dir.mkdir()
    state_dir.mkdir()

    params = CampaignParams(
        prompt="unit test",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
    )

    expected = state_dir / "phase_result" / "unit_broken.json"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("not-json", encoding="utf-8")

    class _RewritingConnector(_FakeConnector):
        """Simulates Claude writing a valid JSON before the session ends."""

        def __init__(self, messages, target: Path):
            super().__init__(messages)
            self._target = target

        async def ask(self, prompt):
            self._target.write_text('{"rewritten": true}', encoding="utf-8")
            async for msg in super().ask(prompt):
                yield msg

    def _factory(*, options):
        return _RewritingConnector(
            [_make_result(cost=0.05, turns=1, duration_ms=120)],
            target=expected,
        )

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _factory)

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_BROKEN",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
            expected_output=expected,
        )
    )
    assert outcome.structured == {"rewritten": True}
    messages_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "cached output at" in messages_text
    assert "re-running phase" in messages_text
    assert "phase begin" in messages_text


def test_run_phase_dry_run_still_logs(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="turbo_optimize.orchestrator.run_phase")

    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    campaign_dir.mkdir()
    state_dir.mkdir()

    params = CampaignParams(
        prompt="unit test",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
        dry_run=True,
    )

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_DRY",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
        )
    )
    assert outcome.structured == {"dry_run": True, "phase": "UNIT_DRY"}
    messages_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "[dry-run] UNIT_DRY" in messages_text
