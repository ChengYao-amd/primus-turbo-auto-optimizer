"""Tests for ClaudeCodeConnector env plumbing, especially IS_SANDBOX.

Claude Code's native root check rejects ``--dangerously-skip-permissions``
(the CLI flag that SDK ``permission_mode="bypassPermissions"`` maps to)
when ``euid == 0``. Inside containers we're already sandboxed and exporting
``IS_SANDBOX=1`` is Anthropic's documented opt-in. The connector must
insert that flag automatically so callers don't have to remember.

Also covers the long-running-tool-aware idle extension introduced after
two false-positives:

* 2026-04-23 ``VALIDATE (full)`` — a 498s ``pytest`` dominated a 360s
  idle budget; the original Bash-only extension fixed this.
* 2026-04-25 ``VALIDATE (quick)`` — a ``TaskOutput timeout=600000``
  (CLI hard-cap) ran to its full 600s native limit and the bare 600s
  idle fired at elapsed=600.1s, because the Bash-only helper ignored
  the ``TaskOutput`` block.  The current implementation extends to
  ``Bash`` / ``BashOutput`` / ``TaskOutput`` and waits on
  ``max(idle_timeout_s, tool_timeout_s) + LONG_RUNNING_IDLE_SLACK_S``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Iterable

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from turbo_optimize.model_connnector import claude_code_connector as connector_mod
from turbo_optimize.model_connnector.claude_code_connector import (
    BASH_IDLE_SLACK_S,
    ClaudeCodeConnector,
    LONG_RUNNING_IDLE_SLACK_S,
    _extract_pending_bash_timeout_s,
    _extract_pending_long_running_timeout_s,
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


def _bash_assistant(bash_timeout_ms: int | None, *, name: str = "Bash") -> AssistantMessage:
    """Build a minimal AssistantMessage with one tool_use block.

    Despite the name, ``name`` is configurable so the same helper can
    construct ``BashOutput`` / ``TaskOutput`` fixtures for the
    long-running-tool tests below.
    """
    input_payload: dict[str, object] = {"command": "echo hi"}
    if bash_timeout_ms is not None:
        input_payload["timeout"] = bash_timeout_ms
    return AssistantMessage(
        content=[ToolUseBlock(id="tu1", name=name, input=input_payload)],
        model="claude-fake",
    )


def test_extract_pending_long_running_timeout_bash_happy_path():
    msg = _bash_assistant(600_000)
    assert _extract_pending_long_running_timeout_s(msg) == 600.0


def test_extract_pending_long_running_timeout_task_output_happy_path():
    """``TaskOutput`` is the 2026-04-25 regression case: agent uses
    ``Task`` to spawn a subagent, then ``TaskOutput block=true
    timeout=…`` to wait on it.  The block is genuinely silent on the
    SDK stream for the full ``timeout``, so the idle budget MUST be
    extended even though the tool is not ``Bash``."""
    msg = _bash_assistant(600_000, name="TaskOutput")
    assert _extract_pending_long_running_timeout_s(msg) == 600.0


def test_extract_pending_long_running_timeout_bash_output_happy_path():
    """``BashOutput`` blocks polling on a backgrounded ``Bash``; same
    silent-stream semantics as ``TaskOutput``."""
    msg = _bash_assistant(120_000, name="BashOutput")
    assert _extract_pending_long_running_timeout_s(msg) == 120.0


def test_extract_pending_long_running_timeout_unknown_tool_is_zero():
    """Tools outside :data:`_LONG_RUNNING_TOOLS` (Read/Write/MCP/…) keep
    the strict idle guard — they complete promptly enough that no
    extension is required, and silencing the guard for them would mask
    real stalls."""
    msg = _bash_assistant(600_000, name="Read")
    assert _extract_pending_long_running_timeout_s(msg) == 0.0


def test_extract_pending_long_running_timeout_missing_field_is_zero():
    msg = _bash_assistant(None)
    assert _extract_pending_long_running_timeout_s(msg) == 0.0


def test_extract_pending_long_running_timeout_non_assistant_is_zero():
    msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content="ok", is_error=False)]
    )
    assert _extract_pending_long_running_timeout_s(msg) == 0.0


def test_extract_pending_long_running_timeout_takes_max_within_bash():
    msg = AssistantMessage(
        content=[
            ToolUseBlock(id="a", name="Bash", input={"command": "ls", "timeout": 30_000}),
            ToolUseBlock(
                id="b", name="Bash", input={"command": "pytest", "timeout": 600_000}
            ),
        ],
        model="claude-fake",
    )
    assert _extract_pending_long_running_timeout_s(msg) == 600.0


def test_extract_pending_long_running_timeout_takes_max_across_tool_kinds():
    """Mix of ``Bash`` + ``TaskOutput`` + ``BashOutput``: the largest
    timeout across all three tool kinds wins, not the largest within a
    single kind."""
    msg = AssistantMessage(
        content=[
            ToolUseBlock(
                id="a", name="Bash", input={"command": "ls", "timeout": 60_000}
            ),
            ToolUseBlock(
                id="b",
                name="BashOutput",
                input={"bash_id": "x", "timeout": 120_000},
            ),
            ToolUseBlock(
                id="c",
                name="TaskOutput",
                input={"task_id": "t", "block": True, "timeout": 600_000},
            ),
        ],
        model="claude-fake",
    )
    assert _extract_pending_long_running_timeout_s(msg) == 600.0


def test_extract_pending_long_running_timeout_ignores_non_positive_and_non_numeric():
    msg = AssistantMessage(
        content=[
            ToolUseBlock(id="a", name="Bash", input={"command": "ls", "timeout": -5}),
            ToolUseBlock(id="b", name="Bash", input={"command": "ls", "timeout": "5000"}),
            ToolUseBlock(id="c", name="Bash", input={"command": "ls", "timeout": 0}),
            ToolUseBlock(
                id="d", name="TaskOutput", input={"task_id": "t", "timeout": -10}
            ),
        ],
        model="claude-fake",
    )
    assert _extract_pending_long_running_timeout_s(msg) == 0.0


def test_bash_alias_still_resolves():
    """``_extract_pending_bash_timeout_s`` is kept as a deprecated
    alias for one release.  Callers who still import the old name must
    get the new behaviour, including ``TaskOutput`` recognition."""
    msg = _bash_assistant(600_000, name="TaskOutput")
    assert _extract_pending_bash_timeout_s(msg) == 600.0
    assert BASH_IDLE_SLACK_S == LONG_RUNNING_IDLE_SLACK_S


class _FakeSDKClient:
    """Minimal stand-in for ``ClaudeSDKClient`` used by the idle tests.

    ``messages`` and ``delays`` are parallel lists: for each yielded
    message the generator first sleeps ``delays[i]`` seconds.  That lets
    tests place deterministic gaps between the ``Bash`` tool_use and its
    eventual ``tool_result`` to exercise (or bypass) the idle guard.
    """

    def __init__(
        self, messages: Iterable[object], delays: Iterable[float]
    ) -> None:
        self._messages = list(messages)
        self._delays = list(delays)
        self.queries: list[str] = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    def receive_response(self) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            for delay, msg in zip(self._delays, self._messages):
                if delay > 0:
                    await asyncio.sleep(delay)
                yield msg

        return _gen()


def _patch_sdk_client(monkeypatch, fake: _FakeSDKClient) -> None:
    monkeypatch.setattr(
        connector_mod, "ClaudeSDKClient", lambda options: fake
    )


def _result_message(session_id: str = "sess-fake", cost: float = 0.0) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=10,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        total_cost_usd=cost,
    )


def test_idle_extended_when_bash_timeout_exceeds_budget(monkeypatch):
    """A 0.4s gap under a 0.1s idle must be tolerated when Bash.timeout=5s.

    Without the extension the ``tool_result`` gap (0.4s) would trip
    :class:`asyncio.TimeoutError`; with extension the effective budget is
    ``max(0.1, 5.0) + LONG_RUNNING_IDLE_SLACK_S`` (well over the gap) and
    the stream completes normally.
    """
    assistant = _bash_assistant(5_000)  # 5s Bash timeout
    tool_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content="ok", is_error=False)]
    )
    result = _result_message()
    fake = _FakeSDKClient(
        messages=[assistant, tool_result, result],
        delays=[0.01, 0.4, 0.01],
    )
    _patch_sdk_client(monkeypatch, fake)

    async def _drive() -> list[object]:
        async with ClaudeCodeConnector(load_auth=False) as conn:
            return [msg async for msg in conn.ask("hi", idle_timeout_s=0.1)]

    got = asyncio.run(_drive())
    assert [type(m).__name__ for m in got] == [
        "AssistantMessage",
        "UserMessage",
        "ResultMessage",
    ]
    assert fake.queries == ["hi"]


def test_idle_extended_when_task_output_timeout_exceeds_budget(monkeypatch):
    """Reproduces the 2026-04-25 ``VALIDATE (quick)`` failure mode.

    The agent issues ``TaskOutput timeout=5000`` (subagent wait) under
    a 0.1s idle budget, the ``tool_result`` arrives 0.4s later, and
    the orchestrator must wait it out instead of killing the phase as
    ``idle_timeout_exhausted``.  Before the 2026-04-25 fix this case
    would have raised :class:`asyncio.TimeoutError` because the
    extension helper only matched ``Bash``.
    """
    assistant = AssistantMessage(
        content=[
            ToolUseBlock(
                id="tu1",
                name="TaskOutput",
                input={"task_id": "t", "block": True, "timeout": 5_000},
            )
        ],
        model="claude-fake",
    )
    tool_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content="ok", is_error=False)]
    )
    fake = _FakeSDKClient(
        messages=[assistant, tool_result, _result_message()],
        delays=[0.01, 0.4, 0.01],
    )
    _patch_sdk_client(monkeypatch, fake)

    async def _drive() -> list[object]:
        async with ClaudeCodeConnector(load_auth=False) as conn:
            return [msg async for msg in conn.ask("hi", idle_timeout_s=0.1)]

    got = asyncio.run(_drive())
    assert [type(m).__name__ for m in got] == [
        "AssistantMessage",
        "UserMessage",
        "ResultMessage",
    ]


def test_idle_still_fires_without_pending_long_running_tool(monkeypatch):
    """Non-long-running tool_use (or one without timeout) keeps the
    strict idle guard."""
    assistant = AssistantMessage(
        content=[ToolUseBlock(id="r1", name="Read", input={"path": "x"})],
        model="claude-fake",
    )
    tool_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="r1", content="ok", is_error=False)]
    )
    fake = _FakeSDKClient(
        messages=[assistant, tool_result, _result_message()],
        delays=[0.01, 0.4, 0.01],
    )
    _patch_sdk_client(monkeypatch, fake)

    async def _drive() -> list[object]:
        async with ClaudeCodeConnector(load_auth=False) as conn:
            return [msg async for msg in conn.ask("hi", idle_timeout_s=0.1)]

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_drive())


def test_idle_resets_when_next_assistant_turn_has_no_long_running_tool(monkeypatch):
    """After a long-running turn, the next assistant turn must reset the budget.

    Stream layout:
      1. AssistantMessage(Bash, timeout=5000)      <- enables extension
      2. UserMessage(tool_result)                  <- covered by extension
      3. AssistantMessage(Read, no long-running)   <- resets pending=0
      4. UserMessage(tool_result_for_read)         <- 0.4s gap, should FAIL
    """
    bash_turn = _bash_assistant(5_000)
    bash_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content="ok", is_error=False)]
    )
    read_turn = AssistantMessage(
        content=[ToolUseBlock(id="r1", name="Read", input={"path": "x"})],
        model="claude-fake",
    )
    read_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="r1", content="ok", is_error=False)]
    )
    fake = _FakeSDKClient(
        messages=[bash_turn, bash_result, read_turn, read_result],
        delays=[0.01, 0.05, 0.01, 0.4],
    )
    _patch_sdk_client(monkeypatch, fake)

    async def _drive() -> None:
        async with ClaudeCodeConnector(load_auth=False) as conn:
            async for _ in conn.ask("hi", idle_timeout_s=0.1):
                pass

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_drive())


def test_long_running_slack_constant_is_applied(monkeypatch):
    """The ``+ LONG_RUNNING_IDLE_SLACK_S`` margin must be honoured.

    The default 30s slack is sized for production worst-cases (large
    subagent ``TaskOutput`` payload serialization) and is too long to
    sleep through in a unit test, so we monkeypatch the slack down to
    0.4s and arrange a tool_result gap of ``slack - 0.1s``.  Without
    the slack, ``effective_idle = max(0.05, 0.1) = 0.1s`` would be
    smaller than the 0.3s gap and the call would time out.  With the
    slack added the budget is ``0.1 + 0.4s = 0.5s`` so the gap fits
    inside.
    """
    monkeypatch.setattr(connector_mod, "LONG_RUNNING_IDLE_SLACK_S", 0.4)
    assistant = _bash_assistant(100)  # 0.1s
    gap = 0.4 - 0.1
    tool_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content="ok", is_error=False)]
    )
    fake = _FakeSDKClient(
        messages=[assistant, tool_result, _result_message()],
        delays=[0.01, gap, 0.01],
    )
    _patch_sdk_client(monkeypatch, fake)

    async def _drive() -> list[object]:
        async with ClaudeCodeConnector(load_auth=False) as conn:
            return [msg async for msg in conn.ask("hi", idle_timeout_s=0.05)]

    got = asyncio.run(_drive())
    assert len(got) == 3
