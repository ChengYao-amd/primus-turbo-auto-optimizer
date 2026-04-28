"""Idle- and wall-timeout regression tests for :func:`run_phase`.

Covers the scenarios from ``docs/issue.md`` §13.1:

* Idle stall with no retry → :class:`PhaseIdleTimeout`.
* Idle stall on attempt 0 followed by success on attempt 1 → the phase
  returns cleanly and ``cost.md`` records ``idle_timeout_retry_ok``.
* Idle stall on every attempt (retries exhausted) → the phase raises
  :class:`PhaseIdleTimeout` and ``cost.md`` records
  ``idle_timeout_exhausted``.
* Steady-stream phase that never finishes within ``wall_timeout_s`` →
  :class:`PhaseWallTimeout`.
* Synthetic ``idle_timeout`` / ``retry_attempt`` / ``wall_timeout``
  events show up in ``_transcript_<phase>.jsonl``.

Every fake yields real ``ResultMessage`` instances so the
accumulator branch that updates ``totals`` is exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import pytest
from claude_agent_sdk import ResultMessage

from turbo_optimize.config import CampaignParams
from turbo_optimize.errors import PhaseIdleTimeout, PhaseWallTimeout
from turbo_optimize.orchestrator import run_phase as run_phase_module


class _FakeClient:
    def __init__(self) -> None:
        self.interrupt_calls = 0

    async def interrupt(self) -> None:
        self.interrupt_calls += 1


class _FakeConnector:
    """Async context-manager stand-in for :class:`ClaudeCodeConnector`.

    Subclasses plug behaviour via ``_stream``. ``idle_timeout_s`` is
    honoured on each ``__anext__``: if ``_stream`` decides to stall,
    the iterator awaits ``asyncio.sleep(idle_timeout_s * 2)`` so
    :func:`asyncio.wait_for` fires.
    """

    def __init__(self) -> None:
        self._client = _FakeClient()

    async def __aenter__(self) -> "_FakeConnector":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def ask(
        self, prompt: str, *, idle_timeout_s: float | None = None
    ):  # pragma: no cover - implemented in subclass
        raise NotImplementedError


class _ScriptedConnector(_FakeConnector):
    """Yields a canned list of messages. Optionally hangs before yielding.

    ``stall_before_index`` is the index of the first message to stall on.
    When set, the iterator awaits forever (via :func:`asyncio.wait_for`'s
    timeout) before emitting that message, simulating the HTTPS silent
    hang described in ``docs/issue.md`` §3.
    """

    def __init__(
        self,
        messages: Iterable[object],
        *,
        stall_before_index: int | None = None,
        stall_factor: float = 20.0,
    ) -> None:
        super().__init__()
        self._messages = list(messages)
        self._stall_before_index = stall_before_index
        self._stall_factor = stall_factor

    async def ask(self, prompt: str, *, idle_timeout_s: float | None = None):
        for i, msg in enumerate(self._messages):
            if self._stall_before_index is not None and i == self._stall_before_index:
                stall_for = (idle_timeout_s or 10.0) * self._stall_factor
                try:
                    await asyncio.wait_for(
                        asyncio.Event().wait(), timeout=stall_for
                    )
                except asyncio.TimeoutError:
                    raise
            if idle_timeout_s is not None:
                # Simulate a brief, below-threshold gap so the outer
                # ``wait_for`` has something to measure.
                await asyncio.sleep(min(0.01, idle_timeout_s / 10))
            yield msg


def _make_result(
    cost: float = 0.01,
    turns: int = 1,
    duration_ms: int = 100,
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=duration_ms,
        duration_api_ms=duration_ms,
        is_error=False,
        num_turns=turns,
        session_id="sess-fake",
        total_cost_usd=cost,
    )


def _make_params(tmp_path: Path) -> tuple[CampaignParams, Path, Path]:
    workspace = tmp_path / "ws"
    campaign_dir = tmp_path / "campaign"
    state_dir = tmp_path / "state"
    for p in (workspace, campaign_dir, state_dir):
        p.mkdir()
    params = CampaignParams(
        prompt="unit test",
        workspace_root=workspace,
        skills_root=Path("agent_workspace/Primus-Turbo/agent"),
        state_dir=state_dir,
    )
    return params, campaign_dir, state_dir


def _patch_connector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factory: Callable[[], _FakeConnector],
) -> None:
    def _outer(*, options):  # noqa: ARG001 - signature matches real ctor
        return factory()

    monkeypatch.setattr(run_phase_module, "ClaudeCodeConnector", _outer)


def _read_transcript(campaign_dir: Path, phase: str) -> list[dict]:
    path = campaign_dir / "profiles" / f"_transcript_{phase.lower()}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_idle_timeout_raises_without_retry(tmp_path, monkeypatch, caplog):
    """``retriable=False`` or ``max_retries=0`` → idle stall aborts the phase."""
    caplog.set_level(logging.WARNING, logger="turbo_optimize.orchestrator.run_phase")
    params, campaign_dir, _ = _make_params(tmp_path)

    messages = [_make_result(cost=0.10)]
    _patch_connector(
        monkeypatch,
        factory=lambda: _ScriptedConnector(messages, stall_before_index=0),
    )

    with pytest.raises(PhaseIdleTimeout) as ei:
        asyncio.run(
            run_phase_module.run_phase(
                phase="UNIT_IDLE_NORETRY",
                campaign_dir=campaign_dir,
                params=params,
                prompt="noop",
                system_prompt="sys",
                allowed_tools=["Read"],
                idle_timeout_s=0.1,
                wall_timeout_s=5.0,
                max_retries=0,
                retriable=False,
            )
        )

    assert ei.value.phase == "UNIT_IDLE_NORETRY"
    assert ei.value.elapsed_s > 0

    events = _read_transcript(campaign_dir, "UNIT_IDLE_NORETRY")
    kinds = [e["kind"] for e in events]
    assert "phase_begin" in kinds
    assert "idle_timeout" in kinds
    assert "phase_end" in kinds
    phase_end = next(e for e in events if e["kind"] == "phase_end")
    assert phase_end["status"] == "idle_timeout_exhausted"

    cost_md = (campaign_dir / "logs" / "cost.md").read_text(encoding="utf-8")
    assert "| UNIT_IDLE_NORETRY |" in cost_md
    assert "| idle_timeout_exhausted |" in cost_md


def test_idle_timeout_retry_then_success(tmp_path, monkeypatch, caplog):
    """Attempt 0 stalls; attempt 1 (fresh connector) completes cleanly."""
    caplog.set_level(logging.WARNING, logger="turbo_optimize.orchestrator.run_phase")
    params, campaign_dir, _ = _make_params(tmp_path)

    call_log: list[str] = []

    def _factory() -> _FakeConnector:
        attempt = len(call_log)
        call_log.append(f"attempt-{attempt}")
        if attempt == 0:
            return _ScriptedConnector(
                [_make_result(cost=0.05)], stall_before_index=0
            )
        return _ScriptedConnector([_make_result(cost=0.07, turns=2)])

    _patch_connector(monkeypatch, factory=_factory)

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_IDLE_RETRY_OK",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
            idle_timeout_s=0.1,
            wall_timeout_s=10.0,
            max_retries=1,
            retriable=True,
        )
    )

    assert outcome.phase == "UNIT_IDLE_RETRY_OK"
    assert outcome.stopped is False
    assert call_log == ["attempt-0", "attempt-1"]

    events = _read_transcript(campaign_dir, "UNIT_IDLE_RETRY_OK")
    kinds = [e["kind"] for e in events]
    assert kinds.count("attempt_begin") == 2
    assert kinds.count("attempt_end") == 2
    assert "idle_timeout" in kinds
    assert "retry_attempt" in kinds
    phase_end = next(e for e in events if e["kind"] == "phase_end")
    assert phase_end["status"] == "idle_timeout_retry_ok"

    cost_md = (campaign_dir / "logs" / "cost.md").read_text(encoding="utf-8")
    assert "| idle_timeout_retry_ok |" in cost_md
    # Cost aggregates across both attempts (0.07 from the retry; attempt 0
    # produced no ResultMessage because it stalled).
    assert "| $0.0700 |" in cost_md


def test_idle_timeout_retry_exhausted(tmp_path, monkeypatch):
    """Every attempt stalls → ``max_retries`` reached → propagate."""
    params, campaign_dir, _ = _make_params(tmp_path)

    def _factory() -> _FakeConnector:
        return _ScriptedConnector(
            [_make_result(cost=0.02)], stall_before_index=0
        )

    _patch_connector(monkeypatch, factory=_factory)

    with pytest.raises(PhaseIdleTimeout):
        asyncio.run(
            run_phase_module.run_phase(
                phase="UNIT_IDLE_EXHAUSTED",
                campaign_dir=campaign_dir,
                params=params,
                prompt="noop",
                system_prompt="sys",
                allowed_tools=["Read"],
                idle_timeout_s=0.1,
                wall_timeout_s=10.0,
                max_retries=2,
                retriable=True,
            )
        )

    events = _read_transcript(campaign_dir, "UNIT_IDLE_EXHAUSTED")
    idle_events = [e for e in events if e["kind"] == "idle_timeout"]
    retry_events = [e for e in events if e["kind"] == "retry_attempt"]
    assert len(idle_events) == 3  # attempt 0, 1, 2
    assert len(retry_events) == 2  # retries logged before attempts 1 and 2
    phase_end = next(e for e in events if e["kind"] == "phase_end")
    assert phase_end["status"] == "idle_timeout_exhausted"


def test_retriable_false_never_retries(tmp_path, monkeypatch):
    """``retriable=False`` must not retry even when ``max_retries > 0``."""
    params, campaign_dir, _ = _make_params(tmp_path)

    def _factory() -> _FakeConnector:
        return _ScriptedConnector(
            [_make_result(cost=0.01)], stall_before_index=0
        )

    _patch_connector(monkeypatch, factory=_factory)

    with pytest.raises(PhaseIdleTimeout):
        asyncio.run(
            run_phase_module.run_phase(
                phase="UNIT_NOT_RETRIABLE",
                campaign_dir=campaign_dir,
                params=params,
                prompt="noop",
                system_prompt="sys",
                allowed_tools=["Read"],
                idle_timeout_s=0.1,
                wall_timeout_s=5.0,
                max_retries=3,
                retriable=False,
            )
        )

    events = _read_transcript(campaign_dir, "UNIT_NOT_RETRIABLE")
    retry_events = [e for e in events if e["kind"] == "retry_attempt"]
    assert retry_events == []  # no retries attempted


class _SteadyStreamConnector(_FakeConnector):
    """Emits assistant-style placeholder dicts forever until cancelled.

    The goal is to trigger the *wall* timeout, not the idle one: each
    message arrives within ``interval_s`` < ``idle_timeout_s`` but the
    stream never produces a ``ResultMessage``.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        super().__init__()
        self._interval_s = interval_s

    async def ask(self, prompt: str, *, idle_timeout_s: float | None = None):
        from claude_agent_sdk import AssistantMessage, TextBlock

        while True:
            await asyncio.sleep(self._interval_s)
            yield AssistantMessage(
                content=[TextBlock(text="still working")],
                model="claude-fake",
            )


def test_wall_timeout_aborts_steady_stream(tmp_path, monkeypatch):
    """A phase that keeps emitting but never finishes → wall timeout."""
    params, campaign_dir, _ = _make_params(tmp_path)

    _patch_connector(monkeypatch, factory=lambda: _SteadyStreamConnector(0.02))

    with pytest.raises(PhaseWallTimeout) as ei:
        asyncio.run(
            run_phase_module.run_phase(
                phase="UNIT_WALL",
                campaign_dir=campaign_dir,
                params=params,
                prompt="noop",
                system_prompt="sys",
                allowed_tools=["Read"],
                idle_timeout_s=5.0,
                wall_timeout_s=0.3,
                max_retries=0,
                retriable=False,
            )
        )

    assert ei.value.phase == "UNIT_WALL"
    assert ei.value.elapsed_s >= 0.2

    events = _read_transcript(campaign_dir, "UNIT_WALL")
    kinds = [e["kind"] for e in events]
    assert "wall_timeout" in kinds
    phase_end = next(e for e in events if e["kind"] == "phase_end")
    assert phase_end["status"] == "wall_timeout"

    cost_md = (campaign_dir / "logs" / "cost.md").read_text(encoding="utf-8")
    assert "| wall_timeout |" in cost_md


def test_defaults_pulled_from_phase_timeout_table(tmp_path, monkeypatch):
    """Unspecified kwargs should resolve from ``PHASE_TIMEOUT_DEFAULTS``."""
    params, campaign_dir, _ = _make_params(tmp_path)

    captured: dict = {}

    def _factory() -> _FakeConnector:
        return _ScriptedConnector([_make_result(cost=0.01)])

    class _Tap(_FakeConnector):
        def __init__(self) -> None:
            super().__init__()
            self._inner = _factory()

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return await self._inner.__aexit__(exc_type, exc, tb)

        async def ask(self, prompt, *, idle_timeout_s=None):
            captured["idle_timeout_s"] = idle_timeout_s
            async for msg in self._inner.ask(prompt, idle_timeout_s=idle_timeout_s):
                yield msg

    _patch_connector(monkeypatch, factory=lambda: _Tap())

    outcome = asyncio.run(
        run_phase_module.run_phase(
            phase="ANALYZE",  # registered in PHASE_TIMEOUT_DEFAULTS
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
        )
    )
    assert outcome.phase == "ANALYZE"
    # Pin to whatever the table currently declares so the test tracks
    # the source of truth without hardcoding a magic number.
    from turbo_optimize.config import PHASE_TIMEOUT_DEFAULTS
    assert captured["idle_timeout_s"] == float(
        PHASE_TIMEOUT_DEFAULTS["ANALYZE"]["idle"]
    )


def test_variant_aware_timeout_lookup(tmp_path, monkeypatch):
    """``VALIDATE (quick)`` and ``VALIDATE (full)`` resolve to different wall budgets.

    The 2026-04-22 campaign showed ``VALIDATE (full)`` wall times up to
    3186s vs ``VALIDATE (quick)`` max 583s. A single ``VALIDATE`` entry
    could not cover both; the fix is to let
    :func:`get_phase_timeouts` consult ``"PHASE (variant)"`` first.
    """
    from turbo_optimize.config import PHASE_TIMEOUT_DEFAULTS, get_phase_timeouts

    quick = get_phase_timeouts("VALIDATE", "quick")
    full = get_phase_timeouts("VALIDATE", "full")
    assert quick["wall"] == PHASE_TIMEOUT_DEFAULTS["VALIDATE (quick)"]["wall"]
    assert full["wall"] == PHASE_TIMEOUT_DEFAULTS["VALIDATE (full)"]["wall"]
    assert float(full["wall"]) > float(quick["wall"])

    unknown_variant = get_phase_timeouts("VALIDATE", "bogus")
    assert unknown_variant["wall"] == PHASE_TIMEOUT_DEFAULTS["VALIDATE"]["wall"]

    # End-to-end: run_phase must pass the variant through to the lookup.
    params, campaign_dir, _ = _make_params(tmp_path)
    captured: dict = {}

    class _Tap(_FakeConnector):
        async def ask(self, prompt, *, idle_timeout_s=None):
            captured["idle_timeout_s"] = idle_timeout_s
            yield _make_result(cost=0.01)

    _patch_connector(monkeypatch, factory=lambda: _Tap())

    asyncio.run(
        run_phase_module.run_phase(
            phase="VALIDATE",
            phase_variant="full",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
        )
    )
    assert captured["idle_timeout_s"] == float(
        PHASE_TIMEOUT_DEFAULTS["VALIDATE (full)"]["idle"]
    )


def test_idle_timeout_disabled_with_zero(tmp_path, monkeypatch):
    """``idle_timeout_s=0`` opts *out* of the new guard (legacy callers)."""
    params, campaign_dir, _ = _make_params(tmp_path)

    captured: dict = {}

    class _Tap(_FakeConnector):
        async def ask(self, prompt, *, idle_timeout_s=None):
            captured["idle_timeout_s"] = idle_timeout_s
            yield _make_result(cost=0.01)

    _patch_connector(monkeypatch, factory=lambda: _Tap())

    asyncio.run(
        run_phase_module.run_phase(
            phase="UNIT_NOGUARD",
            campaign_dir=campaign_dir,
            params=params,
            prompt="noop",
            system_prompt="sys",
            allowed_tools=["Read"],
            idle_timeout_s=0,
            wall_timeout_s=0,
            max_retries=0,
            retriable=False,
        )
    )
    assert captured["idle_timeout_s"] is None
