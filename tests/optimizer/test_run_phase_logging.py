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

    async def ask(
        self, prompt: str, *, idle_timeout_s: float | None = None
    ):  # noqa: D401 - matches real signature
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

        async def ask(self, prompt, *, idle_timeout_s: float | None = None):
            self._target.write_text('{"rewritten": true}', encoding="utf-8")
            async for msg in super().ask(prompt, idle_timeout_s=idle_timeout_s):
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


# ---------------------------------------------------------------------
# Wrap-up recovery when the main session forgets to Write the JSON
# ---------------------------------------------------------------------


from turbo_optimize.errors import PhaseExpectedOutputMissing


class _SilentConnector(_FakeConnector):
    """Connector whose ``ask`` yields a clean ResultMessage but writes
    nothing. Simulates the round-7 OPTIMIZE failure mode where the
    session closed cleanly (``is_error=True`` or just ran out of
    turns) without the model ever calling ``Write``.
    """


class _WrapUpWritingConnector(_FakeConnector):
    """Second-session stand-in that *does* write the expected JSON.

    Used to verify that the orchestrator's wrap-up recovery layer
    successfully reinvokes Claude with a scoped prompt and recovers
    from the missing-output state without killing the phase.
    """

    def __init__(self, messages, target: Path, payload: dict):
        super().__init__(messages)
        self._target = target
        self._payload = payload
        self.seen_prompts: list[str] = []

    async def ask(self, prompt, *, idle_timeout_s: float | None = None):
        self.seen_prompts.append(prompt)
        import json as _json

        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._target.write_text(_json.dumps(self._payload), encoding="utf-8")
        async for msg in super().ask(prompt, idle_timeout_s=idle_timeout_s):
            yield msg


def _wrap_up_params(tmp_path: Path) -> tuple[CampaignParams, Path, Path]:
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
    return params, campaign_dir, state_dir


def test_run_phase_wrap_up_recovery_writes_missing_output(
    tmp_path, monkeypatch, caplog
):
    """When the main session closes without writing the JSON, the
    wrap-up recovery launches a second session whose prompt is the
    dedicated ``<wrap_up_recovery>`` template, and the payload it
    writes is returned as the phase outcome. cost.md must record both
    the silent main attempt and the recovered wrap-up row.
    """
    caplog.set_level(logging.INFO, logger="turbo_optimize.orchestrator.run_phase")
    params, campaign_dir, state_dir = _wrap_up_params(tmp_path)

    expected = state_dir / "phase_result" / "unit_missing.json"
    recovery_payload = {"recovered": True, "round": 7}
    silent_messages = [_make_result(cost=1.0, turns=20, duration_ms=2000)]
    wrap_up_messages = [_make_result(cost=0.05, turns=2, duration_ms=500)]

    wrap_up_connector = _WrapUpWritingConnector(
        wrap_up_messages, target=expected, payload=recovery_payload
    )

    call_counter = {"n": 0}

    def _factory(*, options):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return _SilentConnector(silent_messages)
        return wrap_up_connector

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _factory)

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_MISS",
            campaign_dir=campaign_dir,
            params=params,
            prompt="ORIGINAL_PHASE_PROMPT_BODY",
            system_prompt="sys",
            allowed_tools=["Read", "Write"],
            expected_output=expected,
            round_n=7,
        )
    )

    assert call_counter["n"] == 2, "wrap-up recovery must launch a 2nd session"
    assert outcome.structured == recovery_payload
    assert wrap_up_connector.seen_prompts, "wrap-up session received no prompt"
    wrap_up_prompt = wrap_up_connector.seen_prompts[0]
    assert "<wrap_up_recovery>" in wrap_up_prompt
    assert str(expected) in wrap_up_prompt
    assert "ORIGINAL_PHASE_PROMPT_BODY" in wrap_up_prompt, (
        "the original prompt must be carried verbatim so the wrap-up "
        "knows the JSON schema"
    )

    messages_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "expected_output missing" in messages_text
    assert "launching wrap-up recovery" in messages_text
    assert "wrap-up attempt 1 produced" in messages_text

    cost_md = (campaign_dir / "logs" / "cost.md").read_text(encoding="utf-8")
    assert "| UNIT_MISS |" in cost_md
    assert "| UNIT_MISS (wrap_up) |" in cost_md, (
        "wrap-up row must be tagged so operators can distinguish it "
        "from the main attempt"
    )


def test_run_phase_raises_when_wrap_up_also_fails(tmp_path, monkeypatch, caplog):
    """If every wrap-up attempt also exits without writing the JSON,
    ``run_phase`` raises :class:`PhaseExpectedOutputMissing` — clearer
    than the bare ``FileNotFoundError`` the pre-recovery code threw.
    """
    caplog.set_level(logging.WARNING, logger="turbo_optimize.orchestrator.run_phase")
    params, campaign_dir, state_dir = _wrap_up_params(tmp_path)

    expected = state_dir / "phase_result" / "unit_stuck.json"

    silent_messages = [_make_result(cost=0.4, turns=15, duration_ms=1500)]

    def _factory(*, options):
        return _SilentConnector(silent_messages)

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _factory)

    with pytest.raises(PhaseExpectedOutputMissing) as exc_info:
        asyncio.run(
            run_phase_module.run_phase(
                phase="UNIT_STUCK",
                campaign_dir=campaign_dir,
                params=params,
                prompt="noop",
                system_prompt="sys",
                allowed_tools=["Read", "Write"],
                expected_output=expected,
                round_n=9,
                missing_output_recovery=2,
            )
        )

    err = exc_info.value
    assert err.phase == "UNIT_STUCK"
    assert err.recovery_attempts == 2
    assert str(expected) in str(err)


def test_run_phase_disables_recovery_when_budget_zero(
    tmp_path, monkeypatch, caplog
):
    """``missing_output_recovery=0`` restores the pre-recovery
    behaviour so callers / tests that explicitly want a hard failure
    (e.g. to assert a schema regression) can opt out."""
    caplog.set_level(logging.WARNING, logger="turbo_optimize.orchestrator.run_phase")
    params, campaign_dir, state_dir = _wrap_up_params(tmp_path)

    expected = state_dir / "phase_result" / "unit_no_recovery.json"

    def _factory(*, options):
        return _SilentConnector([_make_result(cost=0.2, turns=5, duration_ms=300)])

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _factory)

    with pytest.raises(PhaseExpectedOutputMissing) as exc_info:
        asyncio.run(
            run_phase_module.run_phase(
                phase="UNIT_NORECOV",
                campaign_dir=campaign_dir,
                params=params,
                prompt="noop",
                system_prompt="sys",
                allowed_tools=["Read", "Write"],
                expected_output=expected,
                missing_output_recovery=0,
            )
        )

    assert exc_info.value.recovery_attempts == 0
