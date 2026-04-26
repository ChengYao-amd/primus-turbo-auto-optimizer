"""Connect to Claude Code via ``ClaudeSDKClient`` with session persistence.

The Claude Code CLI stores full conversation transcripts on disk (typically
under ``~/.claude/projects/<hash>/<session_id>.jsonl``). This module only
needs to remember the ``session_id`` string; passing it via
``ClaudeAgentOptions(resume=...)`` reloads the context for a new connection.

Authentication via environment variables (supports API key and gateway):
    ANTHROPIC_BASE_URL    Gateway / proxy URL. Optional; omit to call the
                          official Anthropic API directly.
    ANTHROPIC_AUTH_TOKEN  Bearer token (gateway style). Required if
                          ANTHROPIC_API_KEY is not set.
    ANTHROPIC_API_KEY     X-Api-Key header value. Required if
                          ANTHROPIC_AUTH_TOKEN is not set.

Requirements:
    pip install claude-agent-sdk
    npm install -g @anthropic-ai/claude-code

Run the demo:
    export ANTHROPIC_BASE_URL=https://your-gateway.example/...
    export ANTHROPIC_AUTH_TOKEN=...      # or ANTHROPIC_API_KEY=...
    python -m turbo_optimize.model_connnector.claude_code_connector
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)


DEFAULT_SESSION_FILE = Path(".claude_code_session.json")

LONG_RUNNING_IDLE_SLACK_S: float = 30.0
"""Extra wall-clock slack added on top of a long-running tool's ``timeout``.

Several Claude Code tools accept a ``timeout`` (milliseconds) that lets
them block the SDK stream legitimately while waiting on a subprocess or
background subagent.  After that native timeout fires, the SDK still
needs to deliver the resulting ``ToolResult`` back to us, which costs:

* JSON serialization of the result payload (small for ``Bash``,
  potentially MB-scale for ``TaskOutput`` carrying a subagent's
  accumulated output);
* one round-trip across the CLI ↔ orchestrator pipe.

30s is sized for the worst-case ``TaskOutput`` case observed on the
``optimize_grouped_gemm_fp8_tensorwise_triton_back_202604231519``
campaign on 2026-04-25, where a ``TaskOutput timeout=600000`` ran to its
full 600s native limit and the orchestrator's idle guard fired at
elapsed=600.1s — i.e. the SDK had less than 100ms of real headroom.
The previous 5s margin was tuned for ``Bash`` only and was too tight
once the agent started using ``Task`` subagents.

The idle clock resets on every new message, so this slack only adds
real latency when the underlying tool actually runs to its full
configured ``timeout`` (rare).
"""

BASH_IDLE_SLACK_S: float = LONG_RUNNING_IDLE_SLACK_S
"""Deprecated alias kept for one release. New code should reference
:data:`LONG_RUNNING_IDLE_SLACK_S` directly."""


_LONG_RUNNING_TOOLS: frozenset[str] = frozenset({"Bash", "BashOutput", "TaskOutput"})
"""Tool names whose ``timeout`` extends the orchestrator's idle budget.

* ``Bash`` — synchronous shell; ``timeout`` caps subprocess lifetime.
* ``BashOutput`` — blocks waiting for new output from a background
  Bash spawned via ``Bash`` with ``run_in_background=true``;
  ``timeout`` caps the wait.
* ``TaskOutput`` — blocks waiting for new output from a subagent
  spawned via ``Task``; ``timeout`` caps the wait.

Other tools (``Read`` / ``Write`` / ``Glob`` / ``Grep`` / MCP / WebFetch)
either complete quickly enough that no extension is needed or do not
expose a controllable ``timeout``, so they keep the strict idle guard.
"""


def _extract_pending_long_running_timeout_s(msg: object) -> float:
    """Return the longest blocking-tool ``timeout`` declared in ``msg``.

    Scans :data:`_LONG_RUNNING_TOOLS` (``Bash`` / ``BashOutput`` /
    ``TaskOutput``).  Each of these tool_use blocks may carry a
    ``timeout`` field in milliseconds; this helper converts to seconds
    so callers can compare against a seconds-scale idle budget.
    Multiple parallel calls in one turn collapse to the max — the one
    that dominates wall time also dominates the idle budget we need to
    reserve.  Returns ``0.0`` when ``msg`` is not an
    :class:`AssistantMessage` or has no in-scope tool_use carrying an
    explicit positive ``timeout``.

    The 2026-04-25 incident on the
    ``optimize_grouped_gemm_fp8_tensorwise_triton_back_202604231519``
    campaign triggered the ``TaskOutput`` extension: the agent issued
    ``TaskOutput block=true timeout=600000`` (CLI hard-cap) to wait on
    a subagent, the older Bash-only helper returned ``0.0`` because the
    block was not ``Bash``, the idle guard fell back to its bare 600s
    base, and the timer fired at elapsed=600.1s exactly when the
    subagent's native timeout was about to deliver the tool_result.
    """
    if not isinstance(msg, AssistantMessage):
        return 0.0
    longest_ms: float = 0.0
    for block in msg.content:
        if not isinstance(block, ToolUseBlock):
            continue
        if block.name not in _LONG_RUNNING_TOOLS:
            continue
        raw = block.input.get("timeout") if isinstance(block.input, dict) else None
        if not isinstance(raw, (int, float)) or raw <= 0:
            continue
        longest_ms = max(longest_ms, float(raw))
    return longest_ms / 1000.0


_extract_pending_bash_timeout_s = _extract_pending_long_running_timeout_s
"""Deprecated alias kept for one release.  New code should reference
:func:`_extract_pending_long_running_timeout_s` directly."""

AUTH_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
    # Prevent startup connections that bypass ANTHROPIC_BASE_URL in isolated
    # networks; only picked up when the caller has exported them.
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "CLAUDE_CODE_SKIP_FAST_MODE_NETWORK_ERRORS",
    # Flag Claude Code treats as "the host is already sandboxed, accept
    # --dangerously-skip-permissions even though euid==0". Required inside
    # containers that run as root; see anthropics/claude-code#9184, #3490.
    "IS_SANDBOX",
)


def load_auth_from_env(
    keys: Iterable[str] = AUTH_ENV_KEYS,
    *,
    require_token: bool = True,
) -> dict[str, str]:
    """Collect auth / gateway env vars currently set in the parent process.

    Args:
        keys: Which env keys to look up. Unset keys are silently skipped.
        require_token: When True (default), at least one of
            ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_AUTH_TOKEN`` must be set;
            otherwise a :class:`RuntimeError` is raised before any
            subprocess is spawned.

    Returns:
        Mapping of exported env keys to their current values.
    """
    picked: dict[str, str] = {}
    for key in keys:
        val = os.environ.get(key)
        if val:
            picked[key] = val
    if require_token and not (
        "ANTHROPIC_API_KEY" in picked or "ANTHROPIC_AUTH_TOKEN" in picked
    ):
        raise RuntimeError(
            "Claude Code auth env var missing: set ANTHROPIC_API_KEY "
            "or ANTHROPIC_AUTH_TOKEN before connecting "
            "(ANTHROPIC_BASE_URL optional for gateways)."
        )
    return picked


def _is_root() -> bool:
    """True when the current process euid is 0.

    ``os.geteuid`` does not exist on Windows; there's no root-vs-user split
    for the CLI's sandbox heuristic either, so we return False and let the
    native Claude Code binary decide.
    """
    getter = getattr(os, "geteuid", None)
    if getter is None:
        return False
    try:
        return getter() == 0
    except OSError:
        return False


def _needs_sandbox_flag(
    options: ClaudeAgentOptions, merged_env: dict[str, str]
) -> bool:
    """Decide whether to auto-inject ``IS_SANDBOX=1`` into the CLI subprocess.

    Triggered only when:

    * the caller asked for ``permission_mode="bypassPermissions"`` — that's
      the mode Claude Code translates to ``--dangerously-skip-permissions``,
      which is the flag the root-check rejects;
    * the current process is root (``euid == 0``), i.e. the typical
      container default user;
    * the caller hasn't explicitly opted out by exporting
      ``IS_SANDBOX=0``. Any non-empty value other than ``0`` is treated as
      already-configured and left alone.

    Rationale: Anthropic's own guidance for containers is exactly this env
    var — see anthropics/claude-code issues #9184, #3490, #927. The CLI is
    sandbox-agnostic; it trusts the caller's ``IS_SANDBOX`` declaration.
    """
    if options.permission_mode != "bypassPermissions":
        return False
    if not _is_root():
        return False
    existing = merged_env.get("IS_SANDBOX")
    if existing is None:
        return True
    return existing.strip() == ""


def _format_message(msg: object) -> str | None:
    """Render a message from the SDK into a single-line string for printing."""
    if isinstance(msg, AssistantMessage):
        parts: list[str] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"[tool_use:{block.name} input={block.input}]")
        return f"Claude: {' '.join(parts)}" if parts else None
    if isinstance(msg, UserMessage):
        texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
        return f"User: {' '.join(texts)}" if texts else None
    if isinstance(msg, ResultMessage):
        cost = f" cost=${msg.total_cost_usd:.4f}" if msg.total_cost_usd else ""
        return f"[result session_id={msg.session_id}{cost}]"
    if isinstance(msg, SystemMessage):
        return None
    return None


class ClaudeCodeConnector:
    """Stateful wrapper around ``ClaudeSDKClient`` with session id persistence.

    Construction options:
        ``session_id``: if provided, the session is resumed on connect via
            ``ClaudeAgentOptions.resume``. Takes precedence over ``session_file``.
        ``session_file``: optional path. When present and no explicit
            ``session_id`` was passed, the last saved id is loaded from it on
            construction. The current id is written back on disconnect.
        ``options``: base ``ClaudeAgentOptions``; ``resume`` will be injected
            into a copy when a session id is known.

    Access the active id via ``session_id`` at any time after the first response.
    """

    def __init__(
        self,
        session_id: str | None = None,
        session_file: str | os.PathLike[str] | None = None,
        options: ClaudeAgentOptions | None = None,
        load_auth: bool = True,
    ) -> None:
        self._session_file: Path | None = Path(session_file) if session_file else None

        if session_id is None and self._session_file is not None:
            session_id = self._load_session_id(self._session_file)

        self._session_id: str | None = session_id
        base = options or ClaudeAgentOptions()

        merged_env: dict[str, str] = dict(base.env)
        if load_auth:
            for k, v in load_auth_from_env().items():
                merged_env.setdefault(k, v)

        if _needs_sandbox_flag(base, merged_env):
            merged_env["IS_SANDBOX"] = "1"

        changes: dict[str, object] = {"env": merged_env}
        if session_id is not None:
            changes["resume"] = session_id
        self._options = replace(base, **changes)
        self._client: ClaudeSDKClient | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @staticmethod
    def _load_session_id(path: Path) -> str | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        sid = data.get("session_id") if isinstance(data, dict) else None
        return sid if isinstance(sid, str) and sid else None

    def _save_session_id(self) -> None:
        if self._session_file is None or self._session_id is None:
            return
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"session_id": self._session_id}, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(
            prefix=self._session_file.name + ".",
            dir=str(self._session_file.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, self._session_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    DISCONNECT_TIMEOUT_S: float = 15.0
    """Bounded wait on :py:meth:`ClaudeSDKClient.disconnect`.

    Level-2 fallback from ``docs/issue.md`` §8.2: after ``interrupt()``
    times out or an idle timeout fires, ``__aexit__`` must still return
    within a predictable window so the phase retry loop keeps moving.
    Exceeding this budget logs a warning and swallows the error (we
    drop the client reference and rely on process-exit cleanup rather
    than blocking the orchestrator).
    """

    async def __aenter__(self) -> "ClaudeCodeConnector":
        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        assert self._client is not None
        client = self._client
        try:
            try:
                await asyncio.wait_for(
                    client.disconnect(), timeout=self.DISCONNECT_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                # Level-2 fallback: disconnect itself is wedged. Don't
                # re-raise — a wedged disconnect must not mask the
                # upstream exception (idle_timeout / wall_timeout /
                # PhaseError) that the orchestrator is handling.
                import logging

                logging.getLogger(__name__).warning(
                    "ClaudeSDKClient.disconnect exceeded %.1fs; "
                    "proceeding without clean shutdown",
                    self.DISCONNECT_TIMEOUT_S,
                )
            except Exception as disc_exc:  # noqa: BLE001 - best-effort cleanup
                import logging

                logging.getLogger(__name__).warning(
                    "ClaudeSDKClient.disconnect failed: %r",
                    disc_exc,
                )
        finally:
            self._client = None
            self._save_session_id()
        return False

    async def ask(
        self,
        prompt: str,
        *,
        idle_timeout_s: float | None = None,
    ) -> AsyncIterator[object]:
        """Send ``prompt`` and yield each message until the response completes.

        Updates ``session_id`` whenever a ``ResultMessage`` arrives so that
        resuming after the first turn works even before the connector exits.

        When ``idle_timeout_s`` is a positive float, each ``__anext__`` on
        the SDK message stream is wrapped in :func:`asyncio.wait_for`.
        A stall longer than ``idle_timeout_s`` seconds surfaces as
        :class:`asyncio.TimeoutError` so the caller can decide whether to
        retry (see :func:`turbo_optimize.orchestrator.run_phase._execute_phase`).
        ``idle_timeout_s=None`` preserves the original unbounded behaviour
        and is the default — callers that did not opt into the new path
        are unaffected.

        Long-running tool calls are a legitimate source of stream
        silence: between the ``tool_use`` emission and the eventual
        ``tool_result`` the SDK genuinely has nothing to deliver.  When
        the most recent :class:`AssistantMessage` declared a ``Bash`` /
        ``BashOutput`` / ``TaskOutput`` tool_use with an explicit
        ``timeout`` (see :data:`_LONG_RUNNING_TOOLS`), the idle budget
        for subsequent ``__anext__`` calls is raised to
        ``max(idle_timeout_s, tool_timeout_s) + LONG_RUNNING_IDLE_SLACK_S``
        and reset whenever a new ``AssistantMessage`` arrives.  This
        keeps the stall guard honest while eliminating two
        false-positives observed on real campaigns:

        * 2026-04-23 ``VALIDATE (full)`` — a 498s ``pytest`` dominated
          a 360s idle budget;
        * 2026-04-25 ``VALIDATE (quick)`` — a ``TaskOutput
          timeout=600000`` (CLI hard-cap) ran to its full 600s native
          limit and the bare 600s idle fired at elapsed=600.1s, which
          the older ``Bash``-only extension did not cover.
        """
        if self._client is None:
            raise RuntimeError("Connector is not active; use 'async with'.")
        await self._client.query(prompt)
        agen = self._client.receive_response().__aiter__()
        pending_long_running_timeout_s: float = 0.0
        while True:
            if idle_timeout_s is None or idle_timeout_s <= 0:
                effective_idle: float | None = None
            elif pending_long_running_timeout_s > 0:
                effective_idle = (
                    max(idle_timeout_s, pending_long_running_timeout_s)
                    + LONG_RUNNING_IDLE_SLACK_S
                )
            else:
                effective_idle = idle_timeout_s
            try:
                if effective_idle is None:
                    msg = await agen.__anext__()
                else:
                    msg = await asyncio.wait_for(
                        agen.__anext__(), timeout=effective_idle
                    )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                # Deliberately propagate so the orchestrator can tag the
                # failure with phase / last_event_kind context.  We do
                # NOT try to aclose() the generator here because the
                # containing connector is about to be torn down via
                # ``__aexit__`` anyway, and aclose on a stuck stream is
                # itself subject to the hang.
                raise
            if isinstance(msg, ResultMessage) and msg.session_id:
                self._session_id = msg.session_id
            if isinstance(msg, AssistantMessage):
                # Recompute on every assistant turn: the previous turn's
                # extended budget expires once the agent regains control,
                # and the new turn may or may not declare its own
                # long-running tool call.
                pending_long_running_timeout_s = (
                    _extract_pending_long_running_timeout_s(msg)
                )
            yield msg


async def _run_turn(conn: ClaudeCodeConnector, prompts: Iterable[str]) -> None:
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        async for msg in conn.ask(prompt):
            line = _format_message(msg)
            if line is not None:
                print(line)


async def demo_persistence(session_file: Path = DEFAULT_SESSION_FILE) -> None:
    """Two sequential connections sharing context through ``session_file``.

    Round 1 opens a fresh session; the id is saved on disconnect.
    Round 2 reads the saved id, reconnects with ``resume=<id>`` and asks a
    follow-up that only makes sense if the previous turn is still in context.
    """
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful coding assistant.",
        allowed_tools=["Read"],
    )

    print("=== Round 1: new session ===")
    async with ClaudeCodeConnector(
        session_file=session_file, options=options
    ) as conn:
        await _run_turn(conn, ["Remember the number 42 for later. Just acknowledge."])
        print(f"[persisted session_id={conn.session_id} -> {session_file}]")

    print("\n=== Round 2: resume from saved session_id ===")
    async with ClaudeCodeConnector(
        session_file=session_file, options=options
    ) as conn:
        print(f"[resumed session_id={conn.session_id}]")
        await _run_turn(conn, ["What number did I ask you to remember?"])


def main() -> None:
    asyncio.run(demo_persistence())


if __name__ == "__main__":
    main()
