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

from turbo_optimize.config import CampaignParams
from turbo_optimize.logs import append_cost_row
from turbo_optimize.model_connnector.claude_code_connector import ClaudeCodeConnector
from turbo_optimize.signals import GracefulStop, stop_requested


log = logging.getLogger(__name__)


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


@dataclass
class PhaseOutcome:
    phase: str
    messages_log: Path
    structured: dict[str, Any] | None
    stopped: bool = False


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
    """
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
    """Run one phase end-to-end, emitting INFO progress lines on entry/exit.

    Cost, turn count and SDK-reported duration come from ``ResultMessage``
    events; we accumulate across all results observed in the stream (a
    single ``ask`` normally yields one, but the SDK does not guarantee
    uniqueness for multi-turn phases).
    """
    options = _build_options(invocation)
    structured: dict[str, Any] | None = None
    stopped = False
    total_cost_usd: float = 0.0
    total_turns: int = 0
    sdk_duration_ms: int | None = None

    log.info(
        "[%s] phase begin (tools=%d mcp=%d agents=%d transcript=%s)",
        invocation.phase,
        len(invocation.allowed_tools) + len(invocation.extra_tools),
        len(invocation.mcp_servers),
        len(invocation.agents or {}),
        transcript_path,
    )
    start_wall = time.perf_counter()
    status = "ok"

    try:
        with transcript_path.open("a", encoding="utf-8") as transcript:
            transcript.write(
                _json_line(
                    kind="phase_begin",
                    phase=invocation.phase,
                    prompt_sha=_short_sha(invocation.prompt),
                    allowed_tools=invocation.allowed_tools + invocation.extra_tools,
                    has_mcp=bool(invocation.mcp_servers),
                    has_agents=bool(invocation.agents),
                )
            )
            try:
                async with ClaudeCodeConnector(options=options) as conn:
                    assert conn._client is not None
                    try:
                        async for msg in conn.ask(invocation.prompt):
                            _record_message(transcript, msg)
                            if isinstance(msg, ResultMessage):
                                cost = getattr(msg, "total_cost_usd", None)
                                if cost:
                                    total_cost_usd += float(cost)
                                turns = getattr(msg, "num_turns", None)
                                if turns:
                                    total_turns += int(turns)
                                dur = getattr(msg, "duration_ms", None)
                                if dur is not None:
                                    sdk_duration_ms = int(dur)
                            if stop_requested():
                                stopped = True
                                log.warning(
                                    "SIGINT during phase %s: interrupting client",
                                    invocation.phase,
                                )
                                try:
                                    await conn._client.interrupt()
                                except Exception as exc:  # noqa: BLE001
                                    log.warning("interrupt failed: %s", exc)
                                break
                    finally:
                        transcript.write(
                            _json_line(kind="phase_end", phase=invocation.phase)
                        )
            except Exception as exc:
                transcript.write(
                    _json_line(
                        kind="phase_error",
                        phase=invocation.phase,
                        error=repr(exc),
                    )
                )
                status = f"error:{type(exc).__name__}"
                raise

        if stopped:
            status = "interrupted"
    finally:
        wall_dt = time.perf_counter() - start_wall
        sdk_s = None if sdk_duration_ms is None else sdk_duration_ms / 1000.0
        sdk_dt = "n/a" if sdk_s is None else f"{sdk_s:.1f}s"
        log.info(
            "[%s] phase end status=%s wall=%.1fs sdk=%s turns=%d cost=$%.4f",
            invocation.phase,
            status,
            wall_dt,
            sdk_dt,
            total_turns,
            total_cost_usd,
        )
        _record_cost(
            invocation,
            status=status,
            wall_s=wall_dt,
            sdk_s=sdk_s,
            turns=total_turns,
            cost_usd=total_cost_usd,
        )

    if stopped:
        raise GracefulStop(f"phase {invocation.phase} interrupted by SIGINT")

    if invocation.expected_output is not None:
        structured = _load_expected_output(invocation.expected_output)
    return PhaseOutcome(
        phase=invocation.phase,
        messages_log=transcript_path,
        structured=structured,
        stopped=stopped,
    )


def _record_message(transcript, msg: Any) -> None:
    event = _summarize_message(msg)
    if event is None:
        return
    transcript.write(_json_line(**event))


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
