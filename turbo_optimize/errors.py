"""Phase-level exceptions raised by the orchestrator.

``docs/issue.md`` tracks a silent-hang incident (2026-04-23) where the
SDK message loop blocked 55 minutes on ``epoll_wait`` because the TCP
stream never reported EOF. To make that failure mode visible and
recoverable, :mod:`turbo_optimize.orchestrator.run_phase` now enforces
two independent timeouts:

* **idle** — max seconds between two successive SDK messages. Captures
  streams that are silently stuck mid-response.
* **wall** — max seconds for the whole phase (including retries).
  Captures runaway sessions whose messages keep flowing but never
  reach a ``ResultMessage``.

These exceptions are the only ones that ``run_phase`` converts into
``cost.md`` status strings (``wall_timeout`` / ``idle_timeout_*``); all
other SDK-originated errors surface as ``error:<ExcName>`` as before.
"""

from __future__ import annotations


class PhaseTimeout(Exception):
    """Base class for every orchestrator timeout.

    ``elapsed_s`` is the wall-clock duration the callsite wants to
    surface. Callers should prefer the two concrete subclasses below —
    this base is here so ``except PhaseTimeout`` can catch either.
    """

    def __init__(
        self,
        phase: str,
        elapsed_s: float,
        message: str | None = None,
    ) -> None:
        self.phase = phase
        self.elapsed_s = float(elapsed_s)
        super().__init__(
            message
            or f"{phase} timed out after {self.elapsed_s:.1f}s"
        )


class PhaseIdleTimeout(PhaseTimeout):
    """Raised when the SDK stream produced no message for ``idle_timeout_s``.

    ``last_event_kind`` records the ``kind`` of the most recent
    ``_summarize_message`` payload before the stall so retry decisions
    and post-hoc transcript inspection can answer "stuck after which
    message type".
    """

    def __init__(
        self,
        phase: str,
        elapsed_s: float,
        last_event_kind: str | None,
    ) -> None:
        self.last_event_kind = last_event_kind
        super().__init__(
            phase=phase,
            elapsed_s=elapsed_s,
            message=(
                f"{phase} idle {elapsed_s:.1f}s since last event "
                f"(kind={last_event_kind!r})"
            ),
        )


class PhaseWallTimeout(PhaseTimeout):
    """Raised when total phase wall time exceeded ``wall_timeout_s``.

    Never retried: wall expiration means the whole budget for the
    phase is spent, including any retries that already ran.
    """


class PhaseExpectedOutputMissing(Exception):
    """Raised when a phase session finishes without writing the expected JSON.

    Symptom observed in round-7 OPTIMIZE (2026-04-23): the Claude
    session closed cleanly (either on ``max_turns`` exhaustion or
    because the model decided it was "done"), every other side-effect
    of the round was on disk (kernel edits, rebuild, libraries), but
    the structured ``phase_result/*.json`` expected by
    :func:`_load_expected_output` was absent. The old behaviour was
    to throw :class:`FileNotFoundError` and kill the campaign, burning
    every dollar the session cost.

    This exception is the recoverable variant: the wrap-up recovery
    layer in :func:`turbo_optimize.orchestrator.run_phase._execute_phase`
    raises it only after its own bounded recovery attempts also failed
    to produce the file. Callers may catch it to mark the round as a
    rollback instead of aborting the whole campaign.
    """

    def __init__(
        self,
        phase: str,
        expected_output: object,
        recovery_attempts: int,
    ) -> None:
        self.phase = phase
        self.expected_output = expected_output
        self.recovery_attempts = int(recovery_attempts)
        super().__init__(
            f"{phase} produced no output at {expected_output} after "
            f"{self.recovery_attempts} wrap-up recovery attempt(s); "
            "the session ended cleanly but never emitted the required "
            "JSON Write. Inspect the phase transcript to see whether "
            "the model stopped on its own or was cut off."
        )
