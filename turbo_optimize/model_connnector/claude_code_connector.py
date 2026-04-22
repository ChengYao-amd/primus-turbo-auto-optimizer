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

    async def __aenter__(self) -> "ClaudeCodeConnector":
        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        assert self._client is not None
        try:
            await self._client.disconnect()
        finally:
            self._client = None
            self._save_session_id()
        return False

    async def ask(self, prompt: str) -> AsyncIterator[object]:
        """Send ``prompt`` and yield each message until the response completes.

        Updates ``session_id`` whenever a ``ResultMessage`` arrives so that
        resuming after the first turn works even before the connector exits.
        """
        if self._client is None:
            raise RuntimeError("Connector is not active; use 'async with'.")
        await self._client.query(prompt)
        async for msg in self._client.receive_response():
            if isinstance(msg, ResultMessage) and msg.session_id:
                self._session_id = msg.session_id
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
