"""Campaign orchestrator: drives the state machine across phases.

Entry point: :func:`run_campaign`. Called from `turbo_optimize.cli.main`
after argument parsing. Assumes the process is already running inside
``asyncio.run`` so that SIGINT handlers can be attached to the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turbo_optimize.config import CampaignParams, make_campaign_id
from turbo_optimize.logs import (
    append_baseline,
    append_final_report,
    append_round_entry,
    append_trend_row,
    append_verified_ineffective,
    extract_history,
    init_cost_log,
    init_optimize_log,
    init_performance_trend,
)
from turbo_optimize.manifest import (
    ManifestError,
    confirm_interactively,
    is_already_confirmed,
    mark_confirmed,
    read_manifest,
    validate_manifest,
)
from turbo_optimize.orchestrator import warm_restart
from turbo_optimize.orchestrator.phases import (
    analyze,
    baseline,
    define_target,
    optimize as optimize_phase,
    prepare_environment,
    read_historical_tips,
    report as report_phase,
    stagnation_review,
    survey_related_work,
    validate as validate_phase,
)
from turbo_optimize.scoring import (
    BenchmarkParse,
    DecisionResult,
    ScoreVector,
    ShapeResult,
    check_hypothesis_duplicate,
    decide_accept_rollback,
    split_primary_metric,
)
from turbo_optimize.signals import (
    GracefulStop,
    get_stop_event,
    install_sigint_handler,
    stop_requested,
)
from turbo_optimize.state import (
    RunState,
    advance_phase,
    load_or_init_run,
    record_round_event,
    save_run_state,
)


log = logging.getLogger(__name__)


class StagnationError(Exception):
    """Raised when ANALYZE dedup retries are exhausted."""


def _absolutize_paths(params: CampaignParams) -> None:
    """Normalize every filesystem path on ``params`` to an absolute form.

    The Claude sessions run with ``ClaudeAgentOptions.cwd =
    params.workspace_root``, while the Python orchestrator runs from the
    user's shell cwd. If we hand Claude a relative path like
    ``state/phase_result/prepare_environment.json``, the SDK resolves it
    against Claude's cwd (``<workspace_root>/state/...``), but Python then
    looks up the same literal string against its own cwd
    (``<shell_cwd>/state/...``) and raises ``FileNotFoundError`` — exactly
    the "phase end status=ok" + crash we saw in the wild.

    Resolving once at campaign start removes the ambiguity: every prompt
    substitution, every disk probe, both sides agree on one absolute path.

    State-dir namespacing (``state/<campaign_id>/``) and stray-phase-
    result migration happen in separate helpers; see ``run_campaign``.
    """
    params.workspace_root = params.workspace_root.resolve()
    params.skills_root = params.skills_root.resolve()
    params.state_dir = params.state_dir.resolve()
    if params.campaign_dir is not None:
        params.campaign_dir = params.campaign_dir.resolve()


def _resolve_campaign_id(params: CampaignParams) -> None:
    """Ensure ``params.campaign_id`` is set before any state write.

    For ``-s <id>`` resumes the CLI has already populated it. For fresh
    ``-p`` runs we generate it now from the prompt so the first
    ``save_run_state`` can land in the namespaced ``state/<id>/``
    location. ``make_campaign_id`` is deterministic for the same
    (prompt, rounded-timestamp) pair, so launching the same prompt
    twice within a minute produces the same id — aligned with resume
    semantics.
    """
    if params.campaign_id is None:
        params.campaign_id = make_campaign_id(params.prompt or "")


def _namespace_state_dir(params: CampaignParams) -> None:
    """Nest ``state_dir`` under ``campaign_id`` to isolate concurrent runs.

    Before: every campaign shared ``state/run.json`` + ``state/phase_result/``,
    so running two campaigns back-to-back clobbered each other's state.
    After: each campaign owns ``state/<campaign_id>/run.json`` +
    ``state/<campaign_id>/phase_result/``.

    Idempotent: if ``state_dir.name`` already equals ``campaign_id`` we
    return unchanged. ``warm_restart.sh`` burns the *already-namespaced*
    ``state_dir`` into its body, so a warm restart calls the CLI with
    ``--state-dir state/<id>``; without this guard we'd end up with
    ``state/<id>/<id>/``.
    """
    assert params.campaign_id is not None
    if params.state_dir.name == params.campaign_id:
        return
    params.state_dir = params.state_dir / params.campaign_id


def _migrate_legacy_state(params: CampaignParams) -> None:
    """One-shot migration from the pre-namespace layout.

    Campaigns that started before state-dir namespacing keep their
    ``run.json`` + ``phase_result/*.json`` at the *parent* of the new
    ``state_dir``. On the next resume we detect that shape and move the
    files in. Only migrates when the legacy ``run.json`` actually
    belongs to ``params.campaign_id``; an unrelated campaign's state is
    left untouched so its own resume still works.

    Called after ``_resolve_campaign_id`` but before
    ``_namespace_state_dir`` so ``params.state_dir`` still points at the
    parent.
    """
    assert params.campaign_id is not None
    if params.state_dir.name == params.campaign_id:
        return
    legacy_run = params.state_dir / "run.json"
    if not legacy_run.exists():
        return
    try:
        data = json.loads(legacy_run.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if data.get("campaign_id") != params.campaign_id:
        return

    dest = params.state_dir / params.campaign_id
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "run.json").exists():
        try:
            shutil.move(str(legacy_run), str(dest / "run.json"))
        except OSError as exc:
            log.warning("failed to migrate %s: %s", legacy_run, exc)
            return
    legacy_phase = params.state_dir / "phase_result"
    if legacy_phase.is_dir():
        dest_phase = dest / "phase_result"
        dest_phase.mkdir(parents=True, exist_ok=True)
        for src in legacy_phase.glob("*.json"):
            dst = dest_phase / src.name
            if dst.exists():
                continue
            try:
                shutil.move(str(src), str(dst))
            except OSError as exc:
                log.warning("failed to migrate %s: %s", src, exc)
        try:
            next(legacy_phase.iterdir())
        except StopIteration:
            try:
                legacy_phase.rmdir()
            except OSError:
                pass
    log.info(
        "migrated legacy top-level state to %s (campaign_id=%s)",
        dest,
        params.campaign_id,
    )


def _migrate_stray_phase_results(params: CampaignParams) -> None:
    """Rescue phase_result JSONs that an older run put under workspace_root.

    Before path normalisation, Claude (cwd = workspace_root) would write
    ``state/phase_result/<phase>.json`` to ``workspace_root/state/...``,
    orphaning the files from the orchestrator. On resume we detect that
    layout and move the files into the correct absolute ``state_dir`` so
    ``-s <campaign>`` can pick up where the crashed run left off instead of
    re-running the phase (and re-paying for it).
    """
    stray_root = params.workspace_root / "state" / "phase_result"
    if not stray_root.is_dir():
        return
    dest_root = params.state_dir / "phase_result"
    dest_root.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for src in stray_root.glob("*.json"):
        dst = dest_root / src.name
        if dst.exists():
            continue
        try:
            shutil.move(str(src), str(dst))
            moved.append(src.name)
        except OSError as exc:
            log.warning("could not migrate %s -> %s: %s", src, dst, exc)
    if moved:
        log.warning(
            "migrated %d stray phase_result file(s) from %s (Claude cwd) "
            "to %s: %s",
            len(moved),
            stray_root,
            dest_root,
            ", ".join(moved),
        )
        try:
            next(stray_root.iterdir())
        except StopIteration:
            try:
                stray_root.rmdir()
                stray_root.parent.rmdir()
            except OSError:
                pass


@dataclass
class ValidationOutputs:
    correctness_ok: bool
    build_ok: bool
    aggregate_score: dict[str, float]
    score_vector: ScoreVector | None
    raw: dict


async def run_campaign(params: CampaignParams) -> int:
    install_sigint_handler()
    get_stop_event()

    _absolutize_paths(params)
    _resolve_campaign_id(params)
    _migrate_legacy_state(params)
    _namespace_state_dir(params)
    _migrate_stray_phase_results(params)
    state, resumed = load_or_init_run(params)
    _rewind_if_needed(state)
    _resolve_model_settings(params, state, resumed)
    warm_restart.write_script(params)
    if params.campaign_dir is not None:
        init_cost_log(params.campaign_dir)
    log.info(
        "campaign start (resumed=%s phase=%s model=%s effort=%s)",
        resumed,
        state.current_phase,
        params.model,
        params.effort,
    )

    try:
        await _execute(params, state, resumed=resumed)
    except GracefulStop:
        log.warning("graceful stop: jumping to REPORT phase")
        await _final_report(params, state, reason="SIGINT")
    except StagnationError as exc:
        log.error("stagnation unresolved: %s", exc)
        await _final_report(params, state, reason=f"stagnation: {exc}")
        return 3
    except ManifestError as exc:
        log.error("manifest phase failed: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        log.exception("campaign crashed: %s", exc)
        return 1

    return 0


async def _execute(params: CampaignParams, state: RunState, *, resumed: bool) -> None:
    phase = state.current_phase
    if params.dry_run:
        await _dry_run_plan(params, state)
        return

    if phase in ("DEFINE_TARGET",):
        await _phase_define_target(params, state)
        phase = state.current_phase

    if phase in ("USER_CONFIRM_MANIFEST",):
        await _phase_confirm_manifest(params, state)
        phase = state.current_phase

    if phase in ("PREPARE_ENVIRONMENT",):
        await _phase_prepare_environment(params, state)
        phase = state.current_phase

    if phase in ("SURVEY_RELATED_WORK",):
        await _phase_survey(params, state)
        phase = state.current_phase

    if phase in ("READ_HISTORICAL_TIPS",):
        await _phase_tips(params, state)
        phase = state.current_phase

    if phase in ("BASELINE",):
        await _phase_baseline(params, state)
        phase = state.current_phase

    while phase in ("ANALYZE", "STAGNATION_REVIEW", "TERMINATION_CHECK"):
        if stop_requested():
            raise GracefulStop("orchestrator loop observed stop flag")

        if phase == "ANALYZE":
            await _run_round(params, state)
            phase = state.current_phase
            continue

        if phase == "STAGNATION_REVIEW":
            await _phase_stagnation(params, state)
            phase = state.current_phase
            continue

        if phase == "TERMINATION_CHECK":
            terminate, reason = _termination_check(params, state)
            if terminate:
                await _final_report(params, state, reason=reason)
                return
            state.current_round += 1
            advance_phase(state, "ANALYZE")
            save_run_state(params.state_dir, state)
            phase = state.current_phase

    if phase == "REPORT":
        await _final_report(params, state, reason="phase-marker")


# --- phase wrappers ---------------------------------------------------


async def _phase_define_target(params: CampaignParams, state: RunState) -> None:
    if params.campaign_id is None:
        params.campaign_id = make_campaign_id(params.prompt or "")
    campaign_dir = define_target.ensure_campaign_dir(params)
    state.campaign_id = params.campaign_id
    state.campaign_dir = str(campaign_dir)
    save_run_state(params.state_dir, state)
    init_cost_log(campaign_dir)

    outcome = await define_target.run(params)
    result = outcome.structured or {}
    _merge_result_into_params(params, result)
    state.params = params.to_dict()
    advance_phase(state, "USER_CONFIRM_MANIFEST")
    save_run_state(params.state_dir, state)
    warm_restart.write_script(params)


async def _phase_confirm_manifest(params: CampaignParams, state: RunState) -> None:
    assert params.campaign_dir is not None
    if not is_already_confirmed(params.campaign_dir):
        manifest = await confirm_interactively(params.campaign_dir)
        mark_confirmed(params.campaign_dir)
    else:
        manifest = read_manifest(params.campaign_dir)

    validation = validate_manifest(manifest)
    if validation.missing:
        raise ManifestError(
            f"manifest missing required fields: {validation.missing}"
        )
    if manifest.get("execution_mode") == "workspace":
        raise ManifestError(
            "execution_mode=workspace is not supported in v1; rerun "
            "DEFINE_TARGET with execution_mode=repo"
        )

    params.merge_manifest(manifest)
    state.params = params.to_dict()
    advance_phase(state, "PREPARE_ENVIRONMENT")
    save_run_state(params.state_dir, state)


async def _phase_prepare_environment(params: CampaignParams, state: RunState) -> None:
    await prepare_environment.run(params)
    advance_phase(state, "SURVEY_RELATED_WORK")
    save_run_state(params.state_dir, state)


async def _phase_survey(params: CampaignParams, state: RunState) -> None:
    await survey_related_work.run(params)
    advance_phase(state, "READ_HISTORICAL_TIPS")
    save_run_state(params.state_dir, state)


async def _phase_tips(params: CampaignParams, state: RunState) -> None:
    await read_historical_tips.run(params)
    advance_phase(state, "BASELINE")
    save_run_state(params.state_dir, state)


async def _phase_baseline(params: CampaignParams, state: RunState) -> None:
    outcome = await baseline.run(params)
    result = outcome.structured or {}
    aggregate = _coerce_score_dict(result.get("aggregate_score", {}))
    state.best_round = 1
    state.best_score = aggregate
    state.current_round = 1
    state.rollback_streak = 0
    record_round_event(
        state,
        round_n=1,
        decision="BASELINE",
        score=aggregate,
        description="Baseline round",
    )
    assert params.campaign_dir is not None
    init_optimize_log(params.campaign_dir, read_manifest(params.campaign_dir))
    init_performance_trend(params.campaign_dir)
    append_baseline(
        params.campaign_dir,
        backend=params.target_backend or "",
        gpu=params.target_gpu or "",
        commit=_current_git_commit(params.workspace_root),
        aggregate_score=aggregate,
        all_check_pass=bool(result.get("test_pass", True)),
        quick_baseline_log=_coerce_quick_baseline_log(result),
    )
    fwd_avg, fwd_peak, bwd_avg, bwd_peak, step_geo = _trend_metrics(result, aggregate)
    append_trend_row(
        params.campaign_dir,
        round_n=1,
        status="BASELINE",
        description="Baseline persistent kernel",
        fwd_avg=fwd_avg,
        fwd_peak=fwd_peak,
        bwd_avg=bwd_avg,
        bwd_peak=bwd_peak,
        step_geomean=step_geo,
        vs_baseline=None,
        key_finding="starting point",
    )
    state.current_round = 2
    advance_phase(state, "ANALYZE")
    save_run_state(params.state_dir, state)


async def _phase_stagnation(params: CampaignParams, state: RunState) -> None:
    await stagnation_review.run(params, rollback_streak=state.rollback_streak)
    advance_phase(state, "TERMINATION_CHECK")
    save_run_state(params.state_dir, state)


# --- round execution --------------------------------------------------


async def _run_round(params: CampaignParams, state: RunState) -> None:
    """Drive one ANALYZE → OPTIMIZE → VALIDATE → DECIDE round.

    Two enhancements over the straight-line version documented in
    ``optimize-loop.md``:

    1. **Debug retry** (``params.debug_retry``): when OPTIMIZE builds
       produce a broken kernel (``build failed``) or VALIDATE reports
       ``correctness failed``, the hypothesis is assumed to still be
       sound and we replay OPTIMIZE+VALIDATE with a structured
       ``retry_context`` that points Claude at the failing log. After
       ``debug_retry`` attempts, the round is rolled back verbatim.
       Semantic failures (score regression, shape check=FAIL on the
       aggregate gate, "no metric improved") are *not* retried — those
       indicate the hypothesis itself is ineffective, so we drop into
       the usual ROLLBACK path immediately.

    2. **Full validation gate** (question 3 from the campaign review):
       ``quick`` can only propose an ACCEPT. Any ACCEPTED /
       ACCEPT_PENDING_NOISE candidate is re-validated with
       ``validation_level="full"`` and the full run's decision is the
       authoritative one written to ``logs/optimize.md`` and to the
       verified_ineffective list. If the full re-run disagrees, the
       round is rolled back instead of accepted.
    """
    round_n = state.current_round
    hypothesis = await _analyze_with_dedup(params, round_n, state)

    opt_result, val_result, quick_decision = await _run_optimize_validate_with_retry(
        params, state, round_n, hypothesis
    )

    final_val_result = val_result
    final_decision = quick_decision
    if quick_decision.decision in ("ACCEPTED", "ACCEPT_PENDING_NOISE"):
        log.info(
            "round-%d quick decision=%s -> escalating to full validation",
            round_n,
            quick_decision.decision,
        )
        advance_phase(state, "VALIDATE")
        save_run_state(params.state_dir, state)
        outcome = await validate_phase.run(
            params, round_n=round_n, validation_level="full"
        )
        full_val_result = outcome.structured or {}
        final_val_result = full_val_result
        final_decision = _apply_decision(
            params, state, round_n, hypothesis, opt_result, full_val_result
        )
        if final_decision.decision != quick_decision.decision:
            log.warning(
                "round-%d full validation overturned quick result: %s -> %s "
                "(reason: %s)",
                round_n,
                quick_decision.decision,
                final_decision.decision,
                final_decision.reason,
            )

    _after_decision(
        params, state, round_n, hypothesis, final_val_result, final_decision
    )


async def _run_optimize_validate_with_retry(
    params: CampaignParams,
    state: RunState,
    round_n: int,
    hypothesis: dict,
) -> tuple[dict, dict, DecisionResult]:
    """Inner OPTIMIZE+VALIDATE micro-loop with ``debug_retry`` attempts.

    Returns ``(opt_result, val_result, quick_decision)`` from the last
    attempt, whether or not the retries were exhausted. Callers should
    only invoke the full-validation gate when ``quick_decision`` is
    ACCEPT-flavoured; otherwise the returned decision already encodes
    why the round is rolling back.
    """
    max_attempts = max(1, 1 + int(params.debug_retry or 0))
    retry_context: str | None = None
    opt_result: dict = {}
    val_result: dict = {}
    quick_decision: DecisionResult | None = None

    for attempt in range(1, max_attempts + 1):
        advance_phase(state, "OPTIMIZE")
        save_run_state(params.state_dir, state)
        outcome = await optimize_phase.run(
            params,
            round_n=round_n,
            hypothesis=hypothesis,
            rebuild_required=_rebuild_required(params),
            retry_context=retry_context,
        )
        opt_result = outcome.structured or {}

        advance_phase(state, "VALIDATE")
        save_run_state(params.state_dir, state)
        outcome = await validate_phase.run(
            params, round_n=round_n, validation_level="quick"
        )
        val_result = outcome.structured or {}

        quick_decision = _apply_decision(
            params, state, round_n, hypothesis, opt_result, val_result
        )

        if not _is_retryable_bug(quick_decision):
            return opt_result, val_result, quick_decision

        if attempt >= max_attempts:
            log.warning(
                "round-%d debug-retry exhausted (%d/%d): last reason=%s",
                round_n,
                attempt,
                max_attempts,
                quick_decision.reason,
            )
            return opt_result, val_result, quick_decision

        log.info(
            "round-%d attempt %d/%d failed (%s); preparing retry_context",
            round_n,
            attempt,
            max_attempts,
            quick_decision.reason,
        )
        retry_context = _build_retry_context(
            attempt=attempt,
            decision=quick_decision,
            opt_result=opt_result,
            val_result=val_result,
        )

    assert quick_decision is not None
    return opt_result, val_result, quick_decision


_RETRYABLE_REASON_PREFIXES: tuple[str, ...] = (
    "build failed",
    "correctness failed",
    "benchmark Check=FAIL",
)


def _is_retryable_bug(decision: DecisionResult) -> bool:
    """True when the rollback reason is a fixable implementation error.

    Score regressions, "no improvement" verdicts, and per-shape
    performance regressions are NOT retryable — those mean the
    hypothesis is ineffective, not that the code has a bug we can
    plaster over. Only the three reasons listed in
    :data:`_RETRYABLE_REASON_PREFIXES` trigger a retry-with-fix loop.
    """
    if decision.decision != "ROLLBACK":
        return False
    reason = decision.reason or ""
    return any(reason.startswith(prefix) for prefix in _RETRYABLE_REASON_PREFIXES)


def _build_retry_context(
    *,
    attempt: int,
    decision: DecisionResult,
    opt_result: dict,
    val_result: dict,
) -> str:
    """Produce the markdown block Claude sees at the start of the next
    OPTIMIZE attempt.

    Focused on pointers (log paths, failing shapes) rather than
    freeform prose. The prompt wrapper around this block already tells
    Claude to keep the hypothesis unchanged.
    """
    build_ok = bool(opt_result.get("build_ok", True))
    lines = [f"## Previous attempt #{attempt} failed"]
    lines.append(f"- orchestrator reason: {decision.reason}")
    if not build_ok:
        notes = opt_result.get("notes") or opt_result.get("diff_summary")
        if notes:
            lines.append(f"- OPTIMIZE notes: {notes}")
        build_log = opt_result.get("build_log")
        if build_log:
            lines.append(f"- build_log: {build_log}")
    else:
        val_notes = val_result.get("notes")
        if val_notes:
            lines.append(f"- VALIDATE notes: {val_notes}")
        failing: list[dict] = []
        for entry in val_result.get("score_vector") or []:
            if not isinstance(entry, dict):
                continue
            check = str(entry.get("check", "")).upper()
            if check and check != "PASS":
                failing.append(entry)
        if failing:
            lines.append("- failing shapes:")
            for entry in failing[:6]:
                shape = entry.get("shape") or {}
                lines.append(f"  - shape={shape} check={entry.get('check')}")
        bench = val_result.get("benchmark_csv")
        if bench:
            lines.append(f"- benchmark_csv: {bench}")
    lines.append(
        "Fix the concrete implementation bug (syntax error, API "
        "mismatch, off-by-one, wrong dtype, missing import, ...) and "
        "re-emit the OPTIMIZE phase_result JSON. Keep the "
        "primary_hypothesis text identical; rewrites that change the "
        "direction will be rejected by ANALYZE dedup next round."
    )
    return "\n".join(lines)


def _rewind_if_needed(state: RunState) -> None:
    """On resume, rewind mid-round phases back to ANALYZE for the same round_n.

    Round numbers must stay stable (SKILL requires monotonic rounds),
    so we keep current_round untouched and only reset the sub-phase.
    OPTIMIZE / VALIDATE side effects are idempotent at the file layer
    (summary.md / kernel_snapshot are overwritten on re-run).
    """
    mid_round = {"OPTIMIZE", "VALIDATE", "DECIDE"}
    if state.current_phase in mid_round and state.current_round > 0:
        log.info(
            "resume: rewind %s (round-%d) back to ANALYZE",
            state.current_phase,
            state.current_round,
        )
        state.current_phase = "ANALYZE"


def _resolve_model_settings(
    params: CampaignParams, state: RunState, resumed: bool
) -> None:
    """Decide the final ``params.model`` / ``params.effort`` for this run.

    Priority (highest wins):
    1. CLI flag (non-None ``params.model`` / ``params.effort``)
    2. On resume, values saved in ``state.params``
    3. Module-level defaults (:func:`CampaignParams.resolve_runtime_defaults`)

    The resolved values are written back into ``state.params`` so subsequent
    resumes remember the choice even if the user invokes the CLI without
    ``--model`` / ``--effort`` next time.
    """
    if resumed:
        saved = state.params or {}
        if params.model in (None, "") and saved.get("model"):
            params.model = saved["model"]
        if params.effort in (None, "") and saved.get("effort"):
            params.effort = saved["effort"]

    params.resolve_runtime_defaults()

    state.params = params.to_dict()
    save_run_state(params.state_dir, state)


async def _analyze_with_dedup(
    params: CampaignParams, round_n: int, state: RunState
) -> dict:
    retry_hint: str | None = None
    for attempt in range(3):
        outcome = await analyze.run(
            params, round_n=round_n, retry_hint=retry_hint
        )
        result = outcome.structured or {}
        history = extract_history(params.campaign_dir)  # type: ignore[arg-type]
        match = check_hypothesis_duplicate(
            result.get("primary_hypothesis", ""),
            history.verified_ineffective,
        )
        if match is None:
            return result
        log.warning(
            "ANALYZE retry %d: hypothesis matches verified_ineffective "
            "round-%s reason=%s similarity=%.2f",
            attempt + 1,
            match.round,
            match.reason,
            match.similarity,
        )
        retry_hint = (
            f"previous hypothesis '{result.get('primary_hypothesis')}' "
            f"overlaps with verified-ineffective entry "
            f"(round-{match.round}: {match.direction}; reason: {match.reason}). "
            "pick a substantially different direction."
        )
    raise StagnationError(
        "ANALYZE kept proposing duplicates against verified_ineffective; "
        "trigger STAGNATION_REVIEW"
    )


def _apply_decision(
    params: CampaignParams,
    state: RunState,
    round_n: int,
    hypothesis: dict,
    opt_result: dict,
    val_result: dict,
) -> DecisionResult:
    correctness_ok = bool(val_result.get("correctness_ok", False))
    build_ok = bool(opt_result.get("build_ok", True))
    aggregate = _coerce_score_dict(val_result.get("aggregate_score", {}))
    score_vector = _build_score_vector(
        val_result.get("score_vector", []),
        params.primary_metric or "",
    )
    best_vector = _build_score_vector(
        _history_best_score_vector(params),
        params.primary_metric or "",
    )
    candidate = ScoreVector(per_shape=score_vector, aggregate=aggregate)
    best = None
    if state.best_round is not None and state.best_score:
        best = ScoreVector(per_shape=best_vector, aggregate=state.best_score)

    decision = decide_accept_rollback(
        candidate,
        best,
        params.primary_metric or "Forward TFLOPS",
        correctness_ok=correctness_ok,
        build_ok=build_ok,
    )
    log.info(
        "round-%d decision=%s reason=%s improvement=%s",
        round_n,
        decision.decision,
        decision.reason,
        decision.improvement_pct,
    )
    return decision


def _after_decision(
    params: CampaignParams,
    state: RunState,
    round_n: int,
    hypothesis: dict,
    val_result: dict,
    decision: DecisionResult,
) -> None:
    assert params.campaign_dir is not None
    aggregate = _coerce_score_dict(val_result.get("aggregate_score", {}))
    description = hypothesis.get("primary_hypothesis", "n/a")

    if decision.decision == "ACCEPTED":
        state.best_round = round_n
        state.best_score = aggregate
        state.rollback_streak = 0
        record_round_event(
            state,
            round_n=round_n,
            decision="ACCEPTED",
            score=aggregate,
            description=description,
        )
        append_round_entry(
            params.campaign_dir,
            round_n=round_n,
            description=description[:60],
            validation_level=val_result.get("validation_level", "quick"),
            hypothesis=description,
            changes=str(val_result.get("changes_summary", "see OPTIMIZE output")),
            aggregate_score_delta=_format_improvement(decision.improvement_pct),
            test_result="PASS",
            decision="accept",
        )
        fwd_avg, fwd_peak, bwd_avg, bwd_peak, step_geo = _trend_metrics(val_result, aggregate)
        append_trend_row(
            params.campaign_dir,
            round_n=round_n,
            status="ACCEPTED",
            description=description[:60],
            fwd_avg=fwd_avg,
            fwd_peak=fwd_peak,
            bwd_avg=bwd_avg,
            bwd_peak=bwd_peak,
            step_geomean=step_geo,
            vs_baseline=_baseline_deltas(state, fwd_avg, bwd_avg, step_geo),
            key_finding=hypothesis.get("verification_signal", ""),
        )
        _git_commit_if_enabled(params, round_n, hypothesis, decision)
        advance_phase(state, "TERMINATION_CHECK")
    elif decision.decision == "ACCEPT_PENDING_NOISE":
        log.info(
            "round-%d triggered noise gate (<%.1f%% improvement). "
            "Python-side re-measurement not wired in v1; accepting with a note.",
            round_n,
            2.0,
        )
        state.best_round = round_n
        state.best_score = aggregate
        state.rollback_streak = 0
        record_round_event(
            state,
            round_n=round_n,
            decision="ACCEPTED (noise-bounded)",
            score=aggregate,
            description=description,
        )
        append_round_entry(
            params.campaign_dir,
            round_n=round_n,
            description=description[:60],
            validation_level=val_result.get("validation_level", "quick"),
            hypothesis=description,
            changes=str(val_result.get("changes_summary", "see OPTIMIZE output")),
            aggregate_score_delta=_format_improvement(decision.improvement_pct)
            + " (noise-bounded)",
            test_result="PASS",
            decision="accept",
            notes="improvement under noise threshold; confirm with full validation later",
        )
        fwd_avg, fwd_peak, bwd_avg, bwd_peak, step_geo = _trend_metrics(val_result, aggregate)
        append_trend_row(
            params.campaign_dir,
            round_n=round_n,
            status="ACCEPTED",
            description=description[:60],
            fwd_avg=fwd_avg,
            fwd_peak=fwd_peak,
            bwd_avg=bwd_avg,
            bwd_peak=bwd_peak,
            step_geomean=step_geo,
            vs_baseline=_baseline_deltas(state, fwd_avg, bwd_avg, step_geo),
            key_finding="noise-bounded accept",
        )
        _git_commit_if_enabled(params, round_n, hypothesis, decision)
        advance_phase(state, "TERMINATION_CHECK")
    else:
        state.rollback_streak += 1
        record_round_event(
            state,
            round_n=round_n,
            decision="ROLLED BACK",
            score=aggregate,
            description=description,
        )
        append_round_entry(
            params.campaign_dir,
            round_n=round_n,
            description=description[:60],
            validation_level=val_result.get("validation_level", "quick"),
            hypothesis=description,
            changes=str(val_result.get("changes_summary", "see OPTIMIZE output")),
            aggregate_score_delta=_format_improvement(decision.improvement_pct),
            test_result="PASS" if val_result.get("correctness_ok", False) else "FAIL",
            decision="rollback",
            notes=decision.reason,
        )
        fwd_avg, fwd_peak, bwd_avg, bwd_peak, step_geo = _trend_metrics(val_result, aggregate)
        append_trend_row(
            params.campaign_dir,
            round_n=round_n,
            status="ROLLBACK",
            description=description[:60],
            fwd_avg=fwd_avg,
            fwd_peak=fwd_peak,
            bwd_avg=bwd_avg,
            bwd_peak=bwd_peak,
            step_geomean=step_geo,
            vs_baseline=_baseline_deltas(state, fwd_avg, bwd_avg, step_geo),
            key_finding=decision.reason[:60],
        )
        append_verified_ineffective(
            params.campaign_dir,
            round_n=round_n,
            direction=description,
            reason=decision.reason,
        )
        _rollback_kernel(params, state, round_n)
        if state.rollback_streak >= 2:
            advance_phase(state, "STAGNATION_REVIEW")
        else:
            advance_phase(state, "TERMINATION_CHECK")
    save_run_state(params.state_dir, state)


def _rollback_kernel(
    params: CampaignParams, state: RunState, round_n: int
) -> None:
    assert params.campaign_dir is not None
    best = state.best_round
    if best is None or params.kernel_source in (None, ""):
        return
    src = (
        params.campaign_dir
        / "rounds"
        / f"round-{best}"
        / "kernel_snapshot"
    )
    if not src.exists():
        log.warning("rollback skipped: best-round snapshot %s missing", src)
        return
    dst_root = params.workspace_root
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    log.info("rolled kernel back to round-%d snapshot at %s", best, src)


def _git_commit_if_enabled(
    params: CampaignParams,
    round_n: int,
    hypothesis: dict,
    decision: DecisionResult,
) -> None:
    if not params.git_commit:
        return
    first = _format_improvement(decision.improvement_pct)
    msg = (
        f"[optimize] {params.target_op} {params.target_backend} "
        f"round-{round_n}: {hypothesis.get('primary_hypothesis', '')[:60]}\n\n"
        f"Hypothesis: {hypothesis.get('primary_hypothesis', '')}\n"
        f"Result: {first}\n"
        f"Details: {params.campaign_dir}/logs/optimize.md\n"
    )
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(params.workspace_root),
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(params.workspace_root),
            check=True,
        )
        log.info("git commit recorded for round-%d", round_n)
    except subprocess.CalledProcessError as exc:
        log.warning("git commit failed (non-fatal): %s", exc)


# --- termination + report ---------------------------------------------


def _termination_check(params: CampaignParams, state: RunState) -> tuple[bool, str]:
    checks = {"T1": False, "T2": False, "T3": False, "T4": False, "T5": False}

    if params.performance_target and state.best_score:
        target_val = _parse_target_value(params.performance_target)
        if target_val is not None:
            best_first = _first_metric(params, state.best_score) or 0
            if best_first >= target_val:
                checks["T1"] = True
    if (
        params.max_iterations is not None
        and state.current_round >= params.max_iterations
    ):
        checks["T3"] = True
    if params.max_duration and state.started_at:
        dur_s = _parse_duration_s(params.max_duration)
        if dur_s is not None and _started_elapsed_s(state.started_at) >= dur_s:
            checks["T4"] = True
    if stop_requested():
        checks["T5"] = True

    passed = [k for k, v in checks.items() if v]
    if not passed:
        return False, ""
    if params.campaign_dir is not None:
        _append_termination_block(params.campaign_dir, checks, passed)
    return True, f"conditions: {passed}"


def _append_termination_block(
    campaign_dir: Path, checks: dict[str, bool], passed: list[str]
) -> None:
    from turbo_optimize.logs import optimize_log_path

    path = optimize_log_path(campaign_dir)
    if not path.exists():
        return
    lines = ["", "### Termination Check"]
    mapping = {
        "T1": "performance_target",
        "T2": "hardware efficiency",
        "T3": "max_iterations reached",
        "T4": "max_duration reached",
        "T5": "user requested stop",
    }
    for key in ("T1", "T2", "T3", "T4", "T5"):
        marker = "PASS" if checks[key] else "no"
        lines.append(f"- {key} {mapping[key]}: {marker}")
    lines.append(f"-> Satisfied condition(s): {', '.join(passed)}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


async def _final_report(params: CampaignParams, state: RunState, *, reason: str) -> None:
    if params.campaign_dir is None:
        log.warning("no campaign_dir; skipping REPORT")
        return
    termination = {
        "reason": reason,
        "best_round": state.best_round,
        "best_score": state.best_score,
        "total_rounds": state.current_round,
        "rollback_streak": state.rollback_streak,
    }
    try:
        outcome = await report_phase.run(params, termination=termination)
        result = outcome.structured or {}
    except Exception as exc:  # noqa: BLE001
        log.exception("REPORT phase failed: %s", exc)
        result = {}
    append_final_report(
        params.campaign_dir,
        body=json.dumps(
            {"reason": reason, "state": state.to_dict(), "result": result},
            ensure_ascii=False,
            indent=2,
        ),
    )
    advance_phase(state, "DONE")
    save_run_state(params.state_dir, state)


# --- helpers ----------------------------------------------------------


def _merge_result_into_params(params: CampaignParams, result: dict) -> None:
    manifest_like = {
        k: v for k, v in result.items() if k in {
            "target_op", "target_backend", "target_lang", "target_gpu",
            "execution_mode", "project_skill", "primary_metric",
            "performance_target", "target_shapes", "kernel_source",
            "test_command", "benchmark_command", "quick_command",
            "git_commit", "git_branch", "max_iterations", "max_duration",
        }
    }
    if manifest_like:
        params.merge_manifest(manifest_like)


def _coerce_score_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _first_metric(params: CampaignParams, scores: dict[str, float]) -> float | None:
    if not scores:
        return None
    metrics = split_primary_metric(params.primary_metric or "")
    for metric in metrics:
        if metric in scores:
            return scores[metric]
    return next(iter(scores.values()), None)


def _trend_metrics(
    result: dict, aggregate: dict[str, float]
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Return `(fwd_avg, fwd_peak, bwd_avg, bwd_peak, step_geomean)`.

    Source order:
    1. `result["trend_row"]` if the phase runner (baseline / validate)
       already emitted the numbers explicitly.
    2. Fall back to `aggregate["Forward TFLOPS"]` / `aggregate["Backward TFLOPS"]`
       for the avg columns, and `result["score_vector"]` per-shape max for peak.
    Missing direction → `None` (rendered as `-`). `step_geomean` is computed
    from `fwd_avg` and `bwd_avg` if the runner did not provide it explicitly.
    """
    fwd_avg = fwd_peak = bwd_avg = bwd_peak = step_geomean = None
    explicit = result.get("trend_row") if isinstance(result, dict) else None
    if isinstance(explicit, dict):
        fwd_avg = _safe_float(explicit.get("fwd_avg"))
        fwd_peak = _safe_float(explicit.get("fwd_peak"))
        bwd_avg = _safe_float(explicit.get("bwd_avg"))
        bwd_peak = _safe_float(explicit.get("bwd_peak"))
        step_geomean = _safe_float(explicit.get("step_geomean"))

    if fwd_avg is None:
        fwd_avg = _safe_float(aggregate.get("Forward TFLOPS"))
    if bwd_avg is None:
        bwd_avg = _safe_float(aggregate.get("Backward TFLOPS"))
    if fwd_peak is None:
        fwd_peak = _peak_from_vector(result.get("score_vector"), "Forward TFLOPS")
    if bwd_peak is None:
        bwd_peak = _peak_from_vector(result.get("score_vector"), "Backward TFLOPS")
    if fwd_peak is None and fwd_avg is not None:
        fwd_peak = fwd_avg
    if bwd_peak is None and bwd_avg is not None:
        bwd_peak = bwd_avg

    if step_geomean is None:
        step_geomean = _compute_step_geomean(fwd_avg, bwd_avg)
    return fwd_avg, fwd_peak, bwd_avg, bwd_peak, step_geomean


def _compute_step_geomean(
    fwd_avg: float | None, bwd_avg: float | None
) -> float | None:
    """`step = sqrt(fwd_avg * bwd_avg)`; degrade to fwd_avg when bwd is absent."""
    if fwd_avg is None and bwd_avg is None:
        return None
    if bwd_avg is None:
        return fwd_avg
    if fwd_avg is None:
        return bwd_avg
    product = fwd_avg * bwd_avg
    if product <= 0:
        return None
    return product ** 0.5


def _baseline_deltas(
    state: RunState,
    candidate_fwd: float | None,
    candidate_bwd: float | None,
    candidate_step: float | None,
) -> dict[str, float | None] | None:
    """Build the three-part vs-baseline dict for `append_trend_row`.

    Returns None for the baseline row (it renders the cell as `—`). Each
    component can independently be None when the baseline lacks that direction.
    """
    baseline = next((h for h in state.history if h["decision"] == "BASELINE"), None)
    if not baseline:
        return None
    base = baseline.get("score") or {}
    base_fwd = _safe_float(base.get("Forward TFLOPS"))
    base_bwd = _safe_float(base.get("Backward TFLOPS"))
    base_step = _compute_step_geomean(base_fwd, base_bwd)
    return {
        "step": _pct_delta(candidate_step, base_step),
        "fwd": _pct_delta(candidate_fwd, base_fwd),
        "bwd": _pct_delta(candidate_bwd, base_bwd),
    }


def _pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return (candidate - baseline) / baseline * 100.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_quick_baseline_log(result: dict) -> str | None:
    """Normalize the ``quick_baseline_log`` field from the BASELINE JSON.

    The BASELINE prompt instructs Claude to emit
    ``"rounds/round-1/artifacts/quick_baseline.log"`` as a campaign-relative
    path. This helper:

    1. Rejects non-string values and empty strings (returns ``None``).
    2. Strips whitespace.
    3. Does **not** verify the file exists — a missing log should be a
       warning later, not a silent drop of the path from ``optimize.md``.
       The ``append_baseline`` helper omits the log line entirely when this
       returns ``None``, which is the intended behaviour when the BASELINE
       phase failed before running ``quick_command``.
    """
    raw = result.get("quick_baseline_log") if isinstance(result, dict) else None
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    return raw


def _peak_from_vector(vec: Any, metric: str) -> float | None:
    if not isinstance(vec, list):
        return None
    values: list[float] = []
    for entry in vec:
        if not isinstance(entry, dict):
            continue
        if entry.get("check") != "PASS":
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        v = _safe_float(metrics.get(metric))
        if v is not None:
            values.append(v)
    return max(values) if values else None


def _format_improvement(improvement_pct: dict[str, float]) -> str:
    if not improvement_pct:
        return "n/a"
    parts = [f"{k}: {v:+.2f}%" for k, v in improvement_pct.items()]
    return ", ".join(parts)


def _build_score_vector(rows: Any, primary_metric: str) -> list[ShapeResult]:
    if not isinstance(rows, list):
        return []
    out: list[ShapeResult] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        shape = entry.get("shape", {}) or {}
        check = str(entry.get("check", "UNKNOWN")).upper()
        metrics_raw = entry.get("metrics", {}) or {}
        metrics: dict[str, float] = {}
        for k, v in metrics_raw.items():
            try:
                metrics[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        out.append(ShapeResult(shape=shape, check=check, metrics=metrics))
    return out


def _history_best_score_vector(params: CampaignParams) -> list[dict]:
    if params.campaign_dir is None:
        return []
    path = params.campaign_dir / "rounds" / "round-1" / "artifacts" / "benchmark.csv"
    if not path.exists():
        return []
    try:
        import csv as _csv
    except ImportError:
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for raw in reader:
            rows.append({"shape": {}, "check": raw.get("Check", "PASS"), "metrics": raw})
    return rows


def _rebuild_required(params: CampaignParams) -> bool:
    backend = (params.target_backend or "").upper()
    if backend in ("TRITON",):
        return False
    return True


def _current_git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _parse_target_value(spec: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", spec)
    if not m:
        return None
    return float(m.group(1))


def _parse_duration_s(spec: str) -> int | None:
    m = re.match(r"(\d+)\s*([hmHM])?", spec.strip())
    if not m:
        return None
    value = int(m.group(1))
    unit = (m.group(2) or "h").lower()
    return value * (3600 if unit == "h" else 60)


def _started_elapsed_s(started_at: str) -> float:
    from datetime import datetime as _dt

    try:
        t0 = _dt.strptime(started_at, "%Y-%m-%d %H:%M")
    except ValueError:
        return 0.0
    return (_dt.now() - t0).total_seconds()


async def _dry_run_plan(params: CampaignParams, state: RunState) -> None:
    if params.campaign_id is None:
        params.campaign_id = make_campaign_id(params.prompt or "")
    campaign_dir = define_target.ensure_campaign_dir(params)
    state.campaign_id = params.campaign_id
    state.campaign_dir = str(campaign_dir)
    save_run_state(params.state_dir, state)

    plan = [
        ("DEFINE_TARGET", "write manifest draft"),
        ("USER_CONFIRM_MANIFEST", "block until y/e/n (or sentinel file)"),
        ("PREPARE_ENVIRONMENT", "scaffold campaign dir, snapshot kernel"),
        ("SURVEY_RELATED_WORK", "produce related_work.md"),
        ("READ_HISTORICAL_TIPS", "read agent/historical_experience/<...>/tips.md"),
        ("BASELINE", "round-1: full test+benchmark, pick representative shapes"),
        ("ANALYZE → OPTIMIZE → VALIDATE", "per-round loop with ACCEPT/ROLLBACK"),
        ("STAGNATION_REVIEW", "triggered when rollback_streak >= 2"),
        ("TERMINATION_CHECK", "T1-T5 gates"),
        ("REPORT", "final summary appended to logs/optimize.md"),
    ]
    print("\n=== primus-turbo-optimize dry-run plan ===", flush=True)
    for name, desc in plan:
        print(f"  [{name:<25}] {desc}", flush=True)
    print(f"\nCampaign dir: {campaign_dir}")
    print(f"State dir:    {params.state_dir}")
    print(f"Skills root:  {params.skills_root}")
    print(f"Workspace:    {params.workspace_root}\n")
