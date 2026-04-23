"""Generic phase runner used by every orchestrator step.

Contract: each phase is a *fresh* `ClaudeSDKClient` connection bounded by
`async with ClaudeCodeConnector(...)` (plan decision (a)). The orchestrator
calls `run_phase(...)`, which:

1. Assembles a `ClaudeAgentOptions` with the phase-specific system prompt,
   allowed tool whitelist, optional MCP servers, optional `agents` /
   `extra_tools` extension points.
2. Renders the phase prompt (markdown template + f-string variables).
3. Streams the conversation, records every message to a per-phase log,
   and checks the SIGINT stop flag on every message. On stop, calls
   `client.interrupt()` and raises `GracefulStop`.
4. Loads the structured JSON the phase was required to emit and returns
   it to the orchestrator.

`v1` always runs on the Python-driven state machine (agents=None). The
`agents`, `extra_tools`, `mcp_servers` kwargs are explicit extension
seams so upgrading a single phase to plan (c) (Task-driven sub-agents) is
a one-line change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    McpServerConfig,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from turbo_optimize.config import CampaignParams, get_phase_timeouts
from turbo_optimize.errors import PhaseIdleTimeout, PhaseWallTimeout
from turbo_optimize.logs import append_cost_row
from turbo_optimize.model_connnector.claude_code_connector import ClaudeCodeConnector
from turbo_optimize.signals import GracefulStop, stop_requested


log = logging.getLogger(__name__)


INTERRUPT_TIMEOUT_S: float = 10.0
"""Bounded wait for ``ClaudeSDKClient.interrupt``.

Level-1 of the fallback chain described in ``docs/issue.md`` §8.2.
Level-2 lives inside
:py:meth:`turbo_optimize.model_connnector.claude_code_connector.ClaudeCodeConnector.__aexit__`
where ``disconnect`` is wrapped in ``asyncio.wait_for``.
"""


@dataclass
class PhaseInvocation:
    phase: str
    prompt: str
    allowed_tools: list[str]
    system_prompt: str
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    agents: dict[str, AgentDefinition] | None = None
    extra_tools: list[str] = field(default_factory=list)
    expected_output: Path | None = None
    max_turns: int | None = None
    cwd: Path | None = None
    setting_sources: list[str] = field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    round_n: int | None = None
    phase_variant: str | None = None
    campaign_dir: Path | None = None
    idle_timeout_s: float | None = None
    """Stream-level idle budget. ``None`` keeps the pre-timeout (unbounded) behaviour."""
    wall_timeout_s: float | None = None
    """Total phase wall budget across all retries. ``None`` disables the guard."""
    max_retries: int = 0
    """Extra attempts after the first one. ``0`` preserves the original no-retry contract."""
    retriable: bool = False
    """Opt-in flag per phase. When ``False``, idle timeouts always abort without retry."""


@dataclass
class PhaseOutcome:
    phase: str
    messages_log: Path
    structured: dict[str, Any] | None
    stopped: bool = False


@dataclass
class _AttemptOutcome:
    """Single attempt inside :func:`_execute_phase`.

    ``last_event_kind`` is threaded back out so that when the outer
    wall timeout fires mid-attempt we can still emit a useful
    ``last_event_kind`` field in the transcript event.
    """

    cost_usd: float = 0.0
    turns: int = 0
    sdk_duration_ms: int | None = None
    stopped: bool = False
    last_event_kind: str | None = None


async def run_phase(
    phase: str,
    *,
    campaign_dir: Path,
    params: CampaignParams,
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    mcp_servers: dict[str, McpServerConfig] | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    extra_tools: list[str] | None = None,
    expected_output: Path | None = None,
    max_turns: int | None = None,
    setting_sources: list[str] | None = None,
    force: bool = False,
    round_n: int | None = None,
    phase_variant: str | None = None,
    idle_timeout_s: float | None = None,
    wall_timeout_s: float | None = None,
    max_retries: int | None = None,
    retriable: bool | None = None,
) -> PhaseOutcome:
    """Run one phase, possibly reusing a cached structured result.

    ``force=True`` skips the cache-reuse shortcut so the caller can
    re-invoke Claude even when ``expected_output`` already exists. Used
    by OPTIMIZE retry (the previous attempt left a stale JSON) and by
    ANALYZE retry-hint (the first attempt produced a duplicate
    hypothesis that we explicitly want to re-prompt).

    ``round_n`` and ``phase_variant`` flow through to ``logs/cost.md``
    (they do not affect the prompt or the Claude session). Both are
    optional: phases that run once per campaign leave ``round_n=None``
    and the row renders ``-``.

    The four timeout kwargs (``idle_timeout_s`` / ``wall_timeout_s`` /
    ``max_retries`` / ``retriable``) default to ``None``; unset values are
    resolved from
    :data:`turbo_optimize.config.PHASE_TIMEOUT_DEFAULTS` keyed by ``phase``.
    Callers that need to opt out can pass ``idle_timeout_s=0``.
    """
    defaults = get_phase_timeouts(phase, phase_variant)
    if idle_timeout_s is None:
        idle_timeout_s = float(defaults["idle"]) if defaults.get("idle") else None
    elif idle_timeout_s <= 0:
        idle_timeout_s = None
    if wall_timeout_s is None:
        wall_timeout_s = float(defaults["wall"]) if defaults.get("wall") else None
    elif wall_timeout_s <= 0:
        wall_timeout_s = None
    if max_retries is None:
        max_retries = int(defaults.get("retries", 0) or 0)
    if retriable is None:
        retriable = bool(defaults.get("retriable", False))

    invocation = PhaseInvocation(
        phase=phase,
        prompt=prompt,
        allowed_tools=list(allowed_tools),
        system_prompt=system_prompt,
        mcp_servers=mcp_servers or {},
        agents=agents,
        extra_tools=list(extra_tools or []),
        expected_output=expected_output,
        max_turns=max_turns,
        cwd=params.workspace_root,
        setting_sources=list(setting_sources or []),
        model=params.model,
        effort=params.effort,
        round_n=round_n,
        phase_variant=phase_variant,
        campaign_dir=campaign_dir,
        idle_timeout_s=idle_timeout_s,
        wall_timeout_s=wall_timeout_s,
        max_retries=max_retries,
        retriable=retriable,
    )
    messages_log = campaign_dir / "profiles" / f"_transcript_{phase.lower()}.jsonl"
    messages_log.parent.mkdir(parents=True, exist_ok=True)
    if params.dry_run:
        log.info("[dry-run] %s: skipping ClaudeSDKClient connection", phase)
        return PhaseOutcome(
            phase=phase,
            messages_log=messages_log,
            structured={"dry_run": True, "phase": phase},
            stopped=False,
        )

    if not force and expected_output is not None and expected_output.exists():
        cache_start = time.perf_counter()
        try:
            structured = _load_expected_output(expected_output)
        except (FileNotFoundError, ValueError) as exc:
            log.warning(
                "[%s] cached output at %s unusable (%s); re-running phase",
                phase,
                expected_output,
                exc,
            )
        else:
            wall_s = time.perf_counter() - cache_start
            log.info(
                "[%s] reusing cached output at %s (skipping Claude session)",
                phase,
                expected_output,
            )
            _record_cost(invocation, status="cached", wall_s=wall_s)
            return PhaseOutcome(
                phase=phase,
                messages_log=messages_log,
                structured=structured,
                stopped=False,
            )

    if force and expected_output is not None and expected_output.exists():
        try:
            expected_output.unlink()
        except OSError as exc:
            log.warning(
                "[%s] could not unlink stale %s before re-run: %s",
                phase,
                expected_output,
                exc,
            )

    return await _execute_phase(invocation, messages_log)


def _record_cost(
    invocation: PhaseInvocation,
    *,
    status: str,
    wall_s: float,
    sdk_s: float | None = None,
    turns: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Best-effort append to ``logs/cost.md``.

    Swallows any IO exception (bad permissions, disk full, racing
    writer, …) so the phase itself never fails just because the log
    couldn't be written. The log is append-only and idempotent at row
    granularity.
    """
    if invocation.campaign_dir is None:
        return
    try:
        append_cost_row(
            invocation.campaign_dir,
            phase=invocation.phase,
            round_n=invocation.round_n,
            status=status,
            wall_s=wall_s,
            sdk_s=sdk_s,
            turns=turns,
            cost_usd=cost_usd,
            phase_variant=invocation.phase_variant,
        )
    except (OSError, ValueError) as exc:
        log.warning("could not append cost.md row for %s: %s", invocation.phase, exc)


def _build_options(invocation: PhaseInvocation) -> ClaudeAgentOptions:
    option_kwargs: dict[str, Any] = {
        "system_prompt": invocation.system_prompt,
        "allowed_tools": list(invocation.allowed_tools) + list(invocation.extra_tools),
        "permission_mode": "bypassPermissions",
        "include_partial_messages": False,
    }
    if invocation.cwd:
        option_kwargs["cwd"] = str(invocation.cwd)
    if invocation.mcp_servers:
        option_kwargs["mcp_servers"] = dict(invocation.mcp_servers)
    if invocation.agents:
        option_kwargs["agents"] = dict(invocation.agents)
    if invocation.max_turns is not None:
        option_kwargs["max_turns"] = invocation.max_turns
    if invocation.setting_sources:
        option_kwargs["setting_sources"] = list(invocation.setting_sources)
    if invocation.model:
        option_kwargs["model"] = invocation.model
    if invocation.effort:
        option_kwargs["effort"] = invocation.effort
    return ClaudeAgentOptions(**option_kwargs)


async def _execute_phase(
    invocation: PhaseInvocation,
    transcript_path: Path,
) -> PhaseOutcome:
    """Run one phase end-to-end with idle- and wall-timeout guards.

    Each attempt is a fresh ``ClaudeCodeConnector`` session executed by
    :func:`_run_single_attempt`. An :class:`~turbo_optimize.errors.PhaseIdleTimeout`
    from the attempt triggers a retry when ``retriable`` is set and
    ``attempt < max_retries``; otherwise it propagates. The whole
    attempt loop is wrapped in :func:`asyncio.wait_for` so a
    :class:`~turbo_optimize.errors.PhaseWallTimeout` can cap total
    spend even when individual attempts never go idle.

    Cost, turn count and SDK-reported duration accumulate across every
    attempt: the ``logs/cost.md`` row reflects the full work, not just
    the successful final attempt.
    """
    log.info(
        "[%s] phase begin (tools=%d mcp=%d agents=%d transcript=%s idle=%s wall=%s retries=%d retriable=%s)",
        invocation.phase,
        len(invocation.allowed_tools) + len(invocation.extra_tools),
        len(invocation.mcp_servers),
        len(invocation.agents or {}),
        transcript_path,
        _fmt_timeout(invocation.idle_timeout_s),
        _fmt_timeout(invocation.wall_timeout_s),
        invocation.max_retries,
        invocation.retriable,
    )

    with transcript_path.open("a", encoding="utf-8") as transcript:
        transcript.write(
            _json_line(
                kind="phase_begin",
                phase=invocation.phase,
                prompt_sha=_short_sha(invocation.prompt),
                allowed_tools=invocation.allowed_tools + invocation.extra_tools,
                has_mcp=bool(invocation.mcp_servers),
                has_agents=bool(invocation.agents),
                idle_timeout_s=invocation.idle_timeout_s,
                wall_timeout_s=invocation.wall_timeout_s,
                max_retries=invocation.max_retries,
                retriable=invocation.retriable,
            )
        )

    start_wall = time.perf_counter()
    totals = _AttemptOutcome()
    status = "ok"
    stopped = False
    error_to_raise: BaseException | None = None

    try:
        if invocation.wall_timeout_s is None:
            status, stopped = await _attempt_loop(
                invocation, transcript_path, totals=totals
            )
        else:
            try:
                status, stopped = await asyncio.wait_for(
                    _attempt_loop(
                        invocation, transcript_path, totals=totals
                    ),
                    timeout=invocation.wall_timeout_s,
                )
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start_wall
                _record_transcript_event(
                    transcript_path,
                    kind="wall_timeout",
                    phase=invocation.phase,
                    elapsed_s=round(elapsed, 3),
                    wall_timeout_s=invocation.wall_timeout_s,
                    last_event_kind=totals.last_event_kind,
                )
                status = "wall_timeout"
                error_to_raise = PhaseWallTimeout(
                    phase=invocation.phase,
                    elapsed_s=elapsed,
                )
    except PhaseIdleTimeout:
        status = "idle_timeout_exhausted"
        raise
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        _record_transcript_event(
            transcript_path,
            kind="phase_error",
            phase=invocation.phase,
            error=repr(exc),
        )
        raise
    finally:
        wall_dt = time.perf_counter() - start_wall
        _record_transcript_event(
            transcript_path,
            kind="phase_end",
            phase=invocation.phase,
            status=status,
            wall_s=round(wall_dt, 3),
        )
        sdk_s = (
            None
            if totals.sdk_duration_ms is None
            else totals.sdk_duration_ms / 1000.0
        )
        sdk_dt = "n/a" if sdk_s is None else f"{sdk_s:.1f}s"
        log.info(
            "[%s] phase end status=%s wall=%.1fs sdk=%s turns=%d cost=$%.4f",
            invocation.phase,
            status,
            wall_dt,
            sdk_dt,
            totals.turns,
            totals.cost_usd,
        )
        _record_cost(
            invocation,
            status=status,
            wall_s=wall_dt,
            sdk_s=sdk_s,
            turns=totals.turns,
            cost_usd=totals.cost_usd,
        )

    if error_to_raise is not None:
        raise error_to_raise

    if stopped:
        raise GracefulStop(f"phase {invocation.phase} interrupted by SIGINT")

    structured: dict[str, Any] | None = None
    if invocation.expected_output is not None:
        structured = _load_expected_output(invocation.expected_output)
    return PhaseOutcome(
        phase=invocation.phase,
        messages_log=transcript_path,
        structured=structured,
        stopped=stopped,
    )


async def _attempt_loop(
    invocation: PhaseInvocation,
    transcript_path: Path,
    *,
    totals: _AttemptOutcome,
) -> tuple[str, bool]:
    """Run up to ``max_retries+1`` connector sessions for one phase.

    Accumulates cost / turns / sdk duration into ``totals``. Returns
    ``(status, stopped)`` so the outer ``finally`` can log a single
    aggregated row. On unrecoverable idle timeout, raises
    :class:`~turbo_optimize.errors.PhaseIdleTimeout` so the outer
    handler can tag the status as ``idle_timeout_exhausted``.
    """
    status = "ok"
    stopped = False
    last_idle_error: PhaseIdleTimeout | None = None

    for attempt in range(invocation.max_retries + 1):
        if attempt > 0:
            _record_transcript_event(
                transcript_path,
                kind="retry_attempt",
                phase=invocation.phase,
                attempt=attempt,
                reason="idle_timeout",
                max_retries=invocation.max_retries,
            )

        try:
            attempt_outcome = await _run_single_attempt(
                invocation,
                transcript_path,
                attempt=attempt,
                totals=totals,
            )
        except PhaseIdleTimeout as exc:
            totals.last_event_kind = exc.last_event_kind
            _record_transcript_event(
                transcript_path,
                kind="idle_timeout",
                phase=invocation.phase,
                elapsed_s=round(exc.elapsed_s, 3),
                last_event_kind=exc.last_event_kind,
                attempt=attempt,
                idle_timeout_s=invocation.idle_timeout_s,
            )
            last_idle_error = exc
            if invocation.retriable and attempt < invocation.max_retries:
                log.warning(
                    "[%s] idle_timeout after %.1fs; retrying (attempt %d/%d)",
                    invocation.phase,
                    exc.elapsed_s,
                    attempt + 1,
                    invocation.max_retries,
                )
                status = "idle_timeout_retrying"
                continue
            # Retries exhausted or phase not marked retriable.
            raise

        stopped = attempt_outcome.stopped
        if stopped:
            status = "interrupted"
        elif attempt > 0:
            status = "idle_timeout_retry_ok"
        else:
            status = "ok"
        return status, stopped

    # Defensive: loop either returned, raised, or continued until
    # exhaustion. If control reaches here it means ``max_retries < 0``
    # or a logic bug; surface the last error rather than silently
    # claiming success.
    if last_idle_error is not None:
        raise last_idle_error
    return status, stopped


async def _run_single_attempt(
    invocation: PhaseInvocation,
    transcript_path: Path,
    *,
    attempt: int,
    totals: _AttemptOutcome,
) -> _AttemptOutcome:
    """Open one connector, drain the response stream, and return totals.

    A fresh ``ClaudeCodeConnector`` per attempt is deliberate: retrying
    after an idle timeout should not try to ``resume=`` a session that
    almost certainly has mismatched tool_use / tool_result pairs on the
    remote side (see ``docs/issue.md`` §11.5).

    ``totals`` is updated *live* (as each ``ResultMessage`` arrives) so
    a wall-timeout cancellation mid-attempt still records the partial
    spend to ``logs/cost.md`` instead of silently dropping it.
    """
    options = _build_options(invocation)
    outcome = _AttemptOutcome()
    attempt_start = time.monotonic()

    with transcript_path.open("a", encoding="utf-8") as transcript:
        _record_transcript_event_to(
            transcript,
            kind="attempt_begin",
            phase=invocation.phase,
            attempt=attempt,
        )
        try:
            async with ClaudeCodeConnector(options=options) as conn:
                assert conn._client is not None
                last_event_at = time.monotonic()
                try:
                    async for msg in conn.ask(
                        invocation.prompt,
                        idle_timeout_s=invocation.idle_timeout_s,
                    ):
                        last_event_at = time.monotonic()
                        event = _summarize_message(msg)
                        if event is not None:
                            outcome.last_event_kind = event.get("kind")
                            totals.last_event_kind = outcome.last_event_kind
                            transcript.write(_json_line(**event))
                        if isinstance(msg, ResultMessage):
                            cost = getattr(msg, "total_cost_usd", None)
                            if cost:
                                outcome.cost_usd += float(cost)
                                totals.cost_usd += float(cost)
                            turns = getattr(msg, "num_turns", None)
                            if turns:
                                outcome.turns += int(turns)
                                totals.turns += int(turns)
                            dur = getattr(msg, "duration_ms", None)
                            if dur is not None:
                                outcome.sdk_duration_ms = int(dur)
                                totals.sdk_duration_ms = int(dur)
                        if stop_requested():
                            outcome.stopped = True
                            log.warning(
                                "SIGINT during phase %s: interrupting client",
                                invocation.phase,
                            )
                            await _interrupt_with_timeout(conn)
                            break
                except asyncio.TimeoutError as exc:
                    elapsed = time.monotonic() - last_event_at
                    log.warning(
                        "[%s] idle stall detected after %.1fs (last=%s); "
                        "interrupting attempt %d",
                        invocation.phase,
                        elapsed,
                        outcome.last_event_kind,
                        attempt,
                    )
                    await _interrupt_with_timeout(conn)
                    raise PhaseIdleTimeout(
                        phase=invocation.phase,
                        elapsed_s=elapsed,
                        last_event_kind=outcome.last_event_kind,
                    ) from exc
        finally:
            attempt_wall = time.monotonic() - attempt_start
            _record_transcript_event_to(
                transcript,
                kind="attempt_end",
                phase=invocation.phase,
                attempt=attempt,
                wall_s=round(attempt_wall, 3),
                cost_usd=round(outcome.cost_usd, 6),
                turns=outcome.turns,
                stopped=outcome.stopped,
            )

    return outcome


async def _interrupt_with_timeout(
    conn: ClaudeCodeConnector,
    *,
    timeout_s: float = INTERRUPT_TIMEOUT_S,
) -> None:
    """Best-effort :py:meth:`ClaudeSDKClient.interrupt` wrapped in ``wait_for``.

    Level-1 of the three-stage fallback chain from ``docs/issue.md``
    §8.2. A stuck interrupt must not block the orchestrator; the
    connector's ``__aexit__`` will still run afterwards with its own
    bounded ``disconnect`` (Level-2). Any error surfaced here is logged
    and swallowed.
    """
    client = getattr(conn, "_client", None)
    if client is None:
        return
    try:
        await asyncio.wait_for(client.interrupt(), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.warning(
            "interrupt did not complete within %.1fs; continuing to disconnect",
            timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
        log.warning("interrupt failed: %r", exc)


def _record_transcript_event(
    transcript_path: Path, *, kind: str, **fields: Any
) -> None:
    """Append a synthetic (non-SDK) event to the transcript jsonl.

    Used for ``phase_begin`` / ``phase_end`` / ``phase_error`` and the
    new timeout- / retry- related events. Swallows IO errors so a
    transcript write failure never masks the underlying phase outcome.
    """
    try:
        with transcript_path.open("a", encoding="utf-8") as f:
            _record_transcript_event_to(f, kind=kind, **fields)
    except OSError as exc:
        log.warning("could not write transcript event %s: %s", kind, exc)


def _record_transcript_event_to(handle, *, kind: str, **fields: Any) -> None:
    """Write one synthetic event to an already-open transcript handle."""
    payload = {"kind": kind, "ts": datetime.now().isoformat(timespec="seconds")}
    payload.update(fields)
    handle.write(_json_line(**payload))


def _fmt_timeout(value: float | None) -> str:
    if value is None:
        return "off"
    return f"{value:.0f}s"


def _summarize_message(msg: Any) -> dict[str, Any] | None:
    now = datetime.now().isoformat(timespec="seconds")
    if isinstance(msg, AssistantMessage):
        parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                parts.append(
                    {"type": "tool_use", "name": block.name, "input": block.input}
                )
        return {"kind": "assistant", "ts": now, "blocks": parts}
    if isinstance(msg, UserMessage):
        parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolResultBlock):
                parts.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": str(block.content)[:4000],
                    }
                )
        return {"kind": "user", "ts": now, "blocks": parts}
    if isinstance(msg, ResultMessage):
        return {
            "kind": "result",
            "ts": now,
            "session_id": msg.session_id,
            "duration_ms": getattr(msg, "duration_ms", None),
            "total_cost_usd": getattr(msg, "total_cost_usd", None),
            "is_error": getattr(msg, "is_error", False),
            "num_turns": getattr(msg, "num_turns", None),
        }
    if isinstance(msg, SystemMessage):
        return {"kind": "system", "ts": now, "subtype": getattr(msg, "subtype", None)}
    return {"kind": "other", "ts": now, "type": type(msg).__name__}


def _load_expected_output(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        raise FileNotFoundError(
            f"expected phase output missing at {path}; "
            "did the phase prompt emit the Write call?"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"phase output {path} is not valid JSON: {exc}"
        ) from exc


def _json_line(**kwargs: Any) -> str:
    return json.dumps(kwargs, ensure_ascii=False) + "\n"


def _short_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
