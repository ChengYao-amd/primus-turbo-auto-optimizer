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
    append_termination_block,
    append_trend_row,
    append_verified_ineffective,
    extract_history,
    init_cost_log,
    init_optimize_log,
    init_performance_trend,
    upsert_current_best,
    upsert_directions_to_try,
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
    profile as profile_phase,
    read_historical_tips,
    report as report_phase,
    review as review_phase,
    stagnation_review,
    survey_related_work,
    validate as validate_phase,
)
from turbo_optimize.scoring import (
    BenchmarkParse,
    DecisionResult,
    REVIEW_VERDICT_AGREE,
    REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND,
    REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK,
    REVIEW_VERDICT_ESCALATE_HUMAN,
    ReviewBundle,
    ScoreVector,
    ScoringError,
    ShapeResult,
    check_hypothesis_duplicate,
    compute_review_signals,
    compute_score_vector,
    decide_accept_rollback,
    parse_bench_csv,
    split_primary_metric,
    verify_shape_consistency,
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
    phase_result_path,
    record_round_event,
    save_run_state,
)


log = logging.getLogger(__name__)


# Git integration is module-level policy, not user-configurable:
#   * every ACCEPTED round commits so rollback can do
#     ``git reset --hard HEAD`` + ``git clean -fd`` and recover the
#     full workspace (see :func:`_git_rollback`). The pre-existing
#     file-copy path in :func:`_rollback_kernel` only ever restored
#     the single file at ``params.kernel_source`` and even wrote it
#     to the wrong path (workspace root instead of the nested
#     subdir), so it is now only used as a fallback when git is
#     unavailable.
#   * experiments run on a dedicated ``optimize/<campaign_id>`` branch
#     so the user's source branch never accumulates throwaway commits.
FORCED_GIT_COMMIT: bool = True
FORCED_GIT_BRANCH: str = "auto"

# REVIEW phase is currently hardcoded to tolerant mode: only the three
# hard rules (hypothesis-metric alignment, off-target gain,
# correctness-bit-identity) can downgrade an ACCEPT to a ROLLBACK.
# Strict mode (where quick-vs-full and noise-band warnings also
# downgrade) is intentionally not implemented yet — the user requested
# review_tolerant only. Changing this constant to ``"strict"`` requires
# the matching branch in :func:`turbo_optimize.scoring.compute_review_signals`.
REVIEW_MODE: str = "tolerant"

GIT_POLICY_EXPLANATION: str = (
    "Git policy (forced, not user-configurable):\n"
    f"  git_commit = {str(FORCED_GIT_COMMIT).lower()} — every ACCEPTED round "
    "commits; rollback is `git reset --hard HEAD` + `git clean -fd` + "
    "`git submodule update --recursive`, which reverts all modified "
    "tracked files, deletes untracked files at any depth, and pulls "
    "submodule pointers back to HEAD. The legacy snapshot-copy path "
    "is kept only as a fallback when git is unavailable.\n"
    f"  git_branch = {FORCED_GIT_BRANCH} — a dedicated branch "
    "`optimize/<campaign_id>` is created off `base_branch` so "
    "experiments never land on the user's source branch.\n"
    "Any `git_commit` / `git_branch` entries in manifest.yaml are "
    "ignored by the orchestrator."
)


def log_git_policy(campaign_stage: str) -> None:
    """Emit the forced-git-policy explanation at a named pipeline stage.

    Called from ``_phase_confirm_manifest`` and from the start of
    ``_phase_prepare_environment`` so the reasoning for the policy is
    reproducible in every campaign log.
    """
    for line in GIT_POLICY_EXPLANATION.splitlines():
        log.info("[%s] %s", campaign_stage, line)


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
    _rewind_if_needed(state, params)
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
    log_git_policy("MANIFEST_CONFIRM")
    advance_phase(state, "PREPARE_ENVIRONMENT")
    save_run_state(params.state_dir, state)


async def _phase_prepare_environment(params: CampaignParams, state: RunState) -> None:
    log_git_policy("PREPARE_ENVIRONMENT")
    outcome = await prepare_environment.run(params)
    result = outcome.structured or {}
    _enforce_base_branch_gate(params, result)
    advance_phase(state, "SURVEY_RELATED_WORK")
    save_run_state(params.state_dir, state)


def _enforce_base_branch_gate(
    params: CampaignParams, prepare_result: dict[str, Any]
) -> None:
    """Block the campaign when base_branch / workspace-clean gates fail.

    The PREPARE_ENVIRONMENT prompt is responsible for detecting these
    conditions and emitting ``base_branch_confirmed`` / ``workspace_clean``
    in the structured JSON. The orchestrator then enforces them so a
    hand-edited workspace can't silently produce commits on top of
    uncontrolled base commits. Because :data:`FORCED_GIT_COMMIT` is
    always ``True``, the ``workspace_clean`` requirement is unconditional
    — there is no opt-out path.
    """
    confirmed = prepare_result.get("base_branch_confirmed")
    expected = prepare_result.get("base_branch_expected")
    observed = prepare_result.get("base_branch_observed")
    if confirmed is False:
        raise ManifestError(
            f"base_branch gate failed: manifest expects '{expected}', "
            f"HEAD is on '{observed}'. Fix the manifest or checkout "
            "the correct base before resuming."
        )
    if prepare_result.get("workspace_clean") is False:
        raise ManifestError(
            "git_commit is forced on (see FORCED_GIT_COMMIT) but "
            "PREPARE_ENVIRONMENT reported workspace_clean=false. "
            "Stash or commit local edits and resume."
        )


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
    await _run_profile_phase(params, state, round_n=1, trigger="post_baseline")


async def _phase_stagnation(params: CampaignParams, state: RunState) -> None:
    await _run_profile_phase(
        params, state, round_n=state.current_round, trigger="pre_stagnation"
    )
    outcome = await stagnation_review.run(
        params,
        rollback_streak=state.rollback_streak,
        current_round=state.current_round,
    )
    result = outcome.structured or {}
    directions = result.get("new_directions") or []
    if params.campaign_dir is not None and isinstance(directions, list):
        upsert_directions_to_try(
            params.campaign_dir,
            round_n=state.current_round,
            directions=directions,
        )
    advance_phase(state, "TERMINATION_CHECK")
    save_run_state(params.state_dir, state)


async def _run_profile_phase(
    params: CampaignParams,
    state: RunState,
    *,
    round_n: int,
    trigger: str,
) -> None:
    """Invoke PROFILE at ``trigger`` and tolerate missing rocprof tools.

    The phase is advisory: even when it reports ``skipped=true`` (for
    instance because ``rocprofv3`` is not installed on the host) the
    orchestrator keeps going. We only log the situation so later
    ANALYZE runs can see whether the profile was available.
    """
    previous_phase = state.current_phase
    advance_phase(state, "PROFILE")
    save_run_state(params.state_dir, state)
    try:
        outcome = await profile_phase.run(
            params, round_n=round_n, trigger=trigger
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "PROFILE (%s, round-%d) raised %s; continuing without profile",
            trigger,
            round_n,
            exc,
        )
    else:
        result = outcome.structured or {}
        if result.get("skipped"):
            log.info(
                "PROFILE (%s, round-%d) skipped: %s",
                trigger,
                round_n,
                result.get("skip_reason") or "no reason given",
            )
        else:
            log.info(
                "PROFILE (%s, round-%d) wrote artifacts to %s",
                trigger,
                round_n,
                result.get("artifacts_dir"),
            )
    advance_phase(state, previous_phase)
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
    full_val_result: dict[str, Any] = {}
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

    if final_decision.decision in ("ACCEPTED", "ACCEPT_PENDING_NOISE"):
        final_decision = await _run_review_phase(
            params,
            state,
            round_n=round_n,
            hypothesis=hypothesis,
            opt_result=opt_result,
            quick_val_result=val_result,
            full_val_result=full_val_result or final_val_result,
            decision=final_decision,
        )

    _after_decision(
        params,
        state,
        round_n,
        hypothesis,
        opt_result,
        final_val_result,
        final_decision,
    )
    if final_decision.decision in ("ACCEPTED", "ACCEPT_PENDING_NOISE"):
        await _run_profile_phase(
            params, state, round_n=round_n, trigger="post_accept"
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
    failure_history: list[dict] = _load_failure_ledger(params, round_n)

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
            params,
            round_n=round_n,
            validation_level="quick",
            force=bool(retry_context),
        )
        val_result = outcome.structured or {}

        quick_decision = _apply_decision(
            params, state, round_n, hypothesis, opt_result, val_result
        )

        if not _is_retryable_bug(quick_decision):
            return opt_result, val_result, quick_decision

        entry = _append_failure_ledger(
            params,
            round_n=round_n,
            attempt=attempt,
            val_result=val_result,
            decision=quick_decision,
        )
        failure_history.append(entry)

        if attempt >= max_attempts:
            log.warning(
                "round-%d debug-retry exhausted (%d/%d): last reason=%s "
                "last_category=%s",
                round_n,
                attempt,
                max_attempts,
                quick_decision.reason,
                val_result.get("failure_category"),
            )
            return opt_result, val_result, quick_decision

        log.info(
            "round-%d attempt %d/%d failed (%s, category=%s); "
            "preparing retry_context",
            round_n,
            attempt,
            max_attempts,
            quick_decision.reason,
            val_result.get("failure_category"),
        )
        retry_context = _build_retry_context(
            attempt=attempt,
            decision=quick_decision,
            opt_result=opt_result,
            val_result=val_result,
            previous_failures=failure_history,
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
    previous_failures: list[dict] | None = None,
) -> str:
    """Produce the markdown block Claude sees at the start of the next
    OPTIMIZE attempt.

    Focused on pointers (log paths, failing shapes, failure category)
    rather than freeform prose. The VALIDATE phase now emits
    ``failure_category`` + ``failure_summary`` on the phase_result JSON;
    we surface both to the OPTIMIZE retry so the next attempt knows the
    classification instead of re-deriving it from the raw log.

    ``previous_failures`` is the accumulated history of attempt-level
    classifications for this round. Including it lets the agent notice
    that the same category keeps recurring and avoid the same edit
    twice.
    """
    build_ok = bool(opt_result.get("build_ok", True))
    category = val_result.get("failure_category")
    summary = val_result.get("failure_summary")
    fail_log = val_result.get("failure_log_path")

    lines = [f"## Previous attempt #{attempt} failed"]
    lines.append(f"- orchestrator reason: {decision.reason}")
    if category:
        lines.append(f"- failure_category: `{category}`")
    if summary:
        lines.append(f"- failure_summary: {summary}")
    if fail_log:
        lines.append(f"- failure_log_path: {fail_log}")

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

    if previous_failures:
        lines.append("")
        lines.append("## Failure history in this round")
        for fail in previous_failures[-3:]:
            a = fail.get("attempt")
            c = fail.get("category") or "?"
            s = fail.get("summary") or ""
            lines.append(f"- attempt-{a} category={c} — {s}")
        seen = {fail.get("category") for fail in previous_failures if fail.get("category")}
        if category and category in seen:
            lines.append(
                f"- NOTE: category `{category}` repeated across attempts; "
                "change the edit direction instead of re-applying the "
                "same fix verbatim."
            )

    lines.append("")
    lines.append(
        "Fix the concrete implementation bug (syntax error, API "
        "mismatch, off-by-one, wrong dtype, missing import, ...) and "
        "re-emit the OPTIMIZE phase_result JSON. Keep the "
        "primary_hypothesis text identical; rewrites that change the "
        "direction will be rejected by ANALYZE dedup next round."
    )
    return "\n".join(lines)


def _append_failure_ledger(
    params: CampaignParams,
    *,
    round_n: int,
    attempt: int,
    val_result: dict,
    decision: DecisionResult,
) -> dict:
    """Persist a structured failure record under ``state_dir/failures/``.

    The ledger survives resume — on warm restart the next retry can read
    the whole history of attempts so it doesn't propose the same bad
    fix Claude already tried three attempts ago.

    Returns the appended entry so callers can also thread it into the
    ``retry_context`` without re-reading from disk.
    """
    assert params.state_dir is not None
    ledger_dir = params.state_dir / "failures"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / f"round{round_n}.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    entry = {
        "attempt": attempt,
        "category": val_result.get("failure_category"),
        "summary": val_result.get("failure_summary"),
        "log_path": val_result.get("failure_log_path"),
        "orchestrator_reason": decision.reason,
        "build_ok": bool(val_result.get("build_ok", True)),
        "correctness_ok": bool(val_result.get("correctness_ok", False)),
    }
    existing.append(entry)
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entry


def _load_failure_ledger(
    params: CampaignParams, round_n: int
) -> list[dict]:
    """Read the round's accumulated failure history (empty if absent)."""
    path = params.state_dir / "failures" / f"round{round_n}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _rewind_if_needed(state: RunState, params: CampaignParams) -> None:
    """On resume, rewind mid-round phases back to ANALYZE for the same round_n.

    Round numbers must stay stable (SKILL requires monotonic rounds),
    so we keep current_round untouched and only reset the sub-phase.
    OPTIMIZE / VALIDATE side effects are idempotent at the file layer
    (summary.md / kernel_snapshot are overwritten on re-run). PROFILE
    is an advisory sibling of ANALYZE; if we crashed inside PROFILE the
    safest resume point is ANALYZE too — the re-run of PROFILE through
    the round trigger overwrites the partial artifacts.

    A campaign that already reached ``DONE`` is also rewindable when the
    user explicitly raised the budget on the new invocation (e.g.
    ``warm_restart.sh -i 100`` after the previous run terminated at
    ``round=50`` with ``max_iterations=50``).  In that case we drop back
    to ``TERMINATION_CHECK``: the next loop iteration re-evaluates T1/T3/T4
    against the new ``params`` and either advances to a fresh
    ``ANALYZE`` round or terminates again with the same on-disk
    artifacts (``final_report.md`` / ``optimize.md`` get overwritten on
    the next ``_final_report``).
    """
    mid_round = {"OPTIMIZE", "VALIDATE", "DECIDE", "PROFILE", "REVIEW"}
    if state.current_phase in mid_round and state.current_round > 0:
        log.info(
            "resume: rewind %s (round-%d) back to ANALYZE",
            state.current_phase,
            state.current_round,
        )
        state.current_phase = "ANALYZE"
        return

    if state.current_phase == "DONE" and _can_extend_after_done(state, params):
        log.info(
            "resume: campaign was DONE at round=%d but new budgets "
            "(max_iterations=%s, max_duration=%s) leave room; "
            "rewinding to TERMINATION_CHECK",
            state.current_round,
            params.max_iterations,
            params.max_duration,
        )
        state.current_phase = "TERMINATION_CHECK"


def _can_extend_after_done(state: RunState, params: CampaignParams) -> bool:
    """Whether the new ``params`` would NOT immediately re-trigger T3 / T4.

    Only ``max_iterations`` (T3) and ``max_duration`` (T4) are
    user-resettable on a warm restart; the other termination predicates
    (target met / stagnation / SIGINT) live entirely in ``state`` and
    cannot be overridden from the CLI.  This helper mirrors the T3/T4
    branches of :func:`_termination_check` so the rewind decision stays
    consistent with the next TERMINATION_CHECK pass.
    """
    if (
        params.max_iterations is not None
        and state.current_round >= params.max_iterations
    ):
        return False
    if params.max_duration and state.started_at:
        dur_s = _parse_duration_s(params.max_duration)
        if dur_s is not None and _started_elapsed_s(state.started_at) >= dur_s:
            return False
    if params.max_iterations is None and not params.max_duration:
        # Neither knob set: no signal that the user wants to extend; keep
        # the campaign DONE rather than silently looping.
        return False
    return True


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
        planned_files = _coerce_planned_files(result.get("planned_modified_files"))
        match = check_hypothesis_duplicate(
            result.get("primary_hypothesis", ""),
            history.verified_ineffective,
            planned_modified_files=planned_files,
        )
        if match is None:
            return result
        log.warning(
            "ANALYZE retry %d: hypothesis matches verified_ineffective "
            "round-%s signal=%s text_sim=%.2f file_overlap=%.2f reason=%s",
            attempt + 1,
            match.round,
            match.signal,
            match.similarity,
            match.file_overlap,
            match.reason,
        )
        hint_parts = [
            f"previous hypothesis '{result.get('primary_hypothesis')}' "
            f"overlaps with verified-ineffective entry "
            f"(round-{match.round}: {match.direction}; reason: {match.reason})."
        ]
        if match.signal in ("files", "both") and match.file_overlap > 0:
            hint_parts.append(
                f"Modified-file overlap is {match.file_overlap:.2f} Jaccard "
                "— the previous round touched the same files. Propose edits "
                "in a different file or a different subsystem (e.g. grid "
                "vs. tile vs. dtype) instead."
            )
        else:
            hint_parts.append(
                "Textual similarity is too high; rephrase the mechanism, not "
                "just the wording."
            )
        hint_parts.append("Pick a substantially different direction.")
        retry_hint = " ".join(hint_parts)
    raise StagnationError(
        "ANALYZE kept proposing duplicates against verified_ineffective; "
        "trigger STAGNATION_REVIEW"
    )


def _coerce_planned_files(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


async def _run_review_phase(
    params: CampaignParams,
    state: RunState,
    *,
    round_n: int,
    hypothesis: dict[str, Any],
    opt_result: dict[str, Any],
    quick_val_result: dict[str, Any],
    full_val_result: dict[str, Any],
    decision: DecisionResult,
) -> DecisionResult:
    """Run REVIEW, merge its verdict into ``decision``.

    Runs only after full-validation pronounced an ACCEPT-flavoured
    decision. Builds the Python review bundle from the quick and full
    CSVs, passes it into the phase runner, and then translates the
    returned verdict via :func:`_apply_review_verdict`.

    The phase is soft-wrapped in try/except: a failed REVIEW is
    surfaced as an ERROR log but never crashes the round — we keep the
    numeric decision intact because silently rejecting a good result
    because the review plumbing broke would be worse than skipping the
    check.
    """
    advance_phase(state, "REVIEW")
    save_run_state(params.state_dir, state)

    review_bundle = _build_review_bundle(
        params,
        hypothesis=hypothesis,
        opt_result=opt_result,
        quick_val_result=quick_val_result,
        full_val_result=full_val_result,
    )
    decision_payload = {
        "decision": decision.decision,
        "reason": decision.reason,
        "improvement_pct": dict(decision.improvement_pct),
        "regressions": list(decision.regressions),
        "noise_check_required": decision.noise_check_required,
    }
    try:
        outcome = await review_phase.run(
            params,
            round_n=round_n,
            hypothesis=hypothesis,
            opt_result=opt_result,
            quick_val_result=quick_val_result,
            full_val_result=full_val_result,
            decision=decision_payload,
            review_bundle=review_bundle,
        )
        review_result = outcome.structured or {}
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "REVIEW phase failed for round-%d (%s); keeping numeric "
            "decision %s as-is",
            round_n,
            exc,
            decision.decision,
        )
        return decision

    merged = _apply_review_verdict(decision, review_result, review_bundle)
    log.info(
        "round-%d REVIEW verdict=%s -> decision %s (reason: %s)",
        round_n,
        review_result.get("review_verdict"),
        merged.decision,
        merged.reason,
    )
    return merged


def _build_review_bundle(
    params: CampaignParams,
    *,
    hypothesis: dict[str, Any],
    opt_result: dict[str, Any],
    quick_val_result: dict[str, Any],
    full_val_result: dict[str, Any],
) -> ReviewBundle:
    """Parse the quick / full CSVs and call :func:`compute_review_signals`.

    The score vectors produced here feed both the REVIEW prompt (so
    the LLM sees the same aggregated numbers the Python verdict was
    built from) and the fallback payload when the LLM fails to emit a
    valid verdict. Missing CSVs short-circuit cleanly: the relevant
    signal is marked as "not available" rather than crashing the
    phase.
    """
    primary_metric = params.primary_metric or "Forward TFLOPS"
    quick_candidate = _parse_val_csv_to_score_vector(
        params, quick_val_result.get("benchmark_csv"), primary_metric
    )
    full_candidate = _parse_val_csv_to_score_vector(
        params, full_val_result.get("benchmark_csv"), primary_metric
    )
    baseline_best = _baseline_score_vector(params, primary_metric)
    return compute_review_signals(
        hypothesis=hypothesis,
        opt_result=opt_result,
        quick_val_result=quick_val_result,
        full_val_result=full_val_result,
        quick_candidate=quick_candidate,
        full_candidate=full_candidate,
        best=baseline_best,
        primary_metric=primary_metric,
        mode=REVIEW_MODE,
    )


def _parse_val_csv_to_score_vector(
    params: CampaignParams,
    csv_path_hint: Any,
    primary_metric: str,
) -> ScoreVector | None:
    """Resolve a VALIDATE ``benchmark_csv`` hint to a ScoreVector.

    Paths may arrive absolute (older runs) or relative to the campaign
    directory (current prompt template). Missing / unparseable CSVs
    return ``None`` so callers can continue with partial signal
    coverage instead of forcing the phase to error out.
    """
    if not csv_path_hint or params.campaign_dir is None:
        return None
    raw = str(csv_path_hint)
    path = Path(raw)
    if not path.is_absolute():
        path = params.campaign_dir / raw
    if not path.exists():
        log.debug("REVIEW: benchmark_csv %s does not exist on disk", path)
        return None
    try:
        parse = parse_bench_csv(path, primary_metric)
    except ScoringError as exc:
        log.warning("REVIEW: could not parse %s: %s", path, exc)
        return None
    try:
        return compute_score_vector(parse)
    except ScoringError as exc:
        log.warning("REVIEW: could not aggregate %s: %s", path, exc)
        return None


def _baseline_score_vector(
    params: CampaignParams, primary_metric: str
) -> ScoreVector | None:
    """Build the score vector for BASELINE (round-1) in canonical form."""
    if params.campaign_dir is None:
        return None
    path = params.campaign_dir / "rounds" / "round-1" / "artifacts" / "benchmark.csv"
    if not path.exists():
        return None
    try:
        parse = parse_bench_csv(path, primary_metric)
        return compute_score_vector(parse)
    except ScoringError as exc:
        log.warning("REVIEW: could not build baseline score vector: %s", exc)
        return None


def _apply_review_verdict(
    decision: DecisionResult,
    review_result: dict[str, Any],
    fallback: ReviewBundle,
) -> DecisionResult:
    """Translate the REVIEW verdict into a mutated ``DecisionResult``.

    * ``AGREE`` — decision is kept verbatim, reason is left untouched.
    * ``DOWNGRADE_TO_NOISE_BOUND`` — in tolerant mode the numeric
      ACCEPT is preserved; we only annotate the reason so operators
      see that REVIEW flagged at least one soft signal.
    * ``DOWNGRADE_TO_ROLLBACK`` — force the decision to ROLLBACK with
      the REVIEW reason prefix. No per-shape regressions are added;
      the Python noise-band gate will still re-fire on the next
      round's comparison against the (unchanged) baseline best.
    * ``ESCALATE_HUMAN`` — also forces ROLLBACK but with a more
      prominent reason so the operator can spot the correctness
      concern in the optimize log.
    """
    verdict = review_result.get("review_verdict") or fallback.tolerant_verdict
    review_reason = review_result.get("review_reason") or fallback.tolerant_reason
    if verdict == REVIEW_VERDICT_AGREE:
        return decision
    if verdict == REVIEW_VERDICT_DOWNGRADE_TO_NOISE_BOUND:
        return DecisionResult(
            decision=decision.decision,
            reason=(
                f"{decision.reason} | REVIEW(tolerant)=DOWNGRADE_TO_NOISE_BOUND: "
                f"{review_reason}"
            ),
            improvement_pct=decision.improvement_pct,
            regressions=decision.regressions,
            noise_check_required=True,
        )
    if verdict == REVIEW_VERDICT_DOWNGRADE_TO_ROLLBACK:
        return DecisionResult(
            decision="ROLLBACK",
            reason=(
                f"REVIEW(tolerant)=DOWNGRADE_TO_ROLLBACK: {review_reason} "
                f"(prior numeric reason: {decision.reason})"
            ),
            improvement_pct=decision.improvement_pct,
            regressions=decision.regressions,
            noise_check_required=decision.noise_check_required,
        )
    if verdict == REVIEW_VERDICT_ESCALATE_HUMAN:
        log.error(
            "REVIEW escalated round to human: %s (decision rolled back)",
            review_reason,
        )
        return DecisionResult(
            decision="ROLLBACK",
            reason=(
                f"REVIEW(tolerant)=ESCALATE_HUMAN: {review_reason} "
                "(correctness claim not verified; rolled back and logged)"
            ),
            improvement_pct=decision.improvement_pct,
            regressions=decision.regressions,
            noise_check_required=decision.noise_check_required,
        )
    log.warning("REVIEW returned unknown verdict %r; keeping decision", verdict)
    return decision


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

    report = verify_shape_consistency(candidate, best)
    if not report.consistent:
        log.warning(
            "round-%d shape-consistency warning: %s; candidate-only=%s "
            "baseline-only=%s. The per-shape regression gate only considers "
            "overlapping shapes — audit the bench harness if this persists.",
            round_n,
            report.mismatch_reason,
            report.candidate_only[:5],
            report.baseline_only[:5],
        )
    elif report.baseline_only and candidate.per_shape:
        log.info(
            "round-%d baseline measured %d extra shape(s) not in candidate: "
            "%s (acceptable — baseline may run a wider sweep).",
            round_n,
            len(report.baseline_only),
            report.baseline_only[:5],
        )

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
    opt_result: dict,
    val_result: dict,
    decision: DecisionResult,
) -> None:
    assert params.campaign_dir is not None
    aggregate = _coerce_score_dict(val_result.get("aggregate_score", {}))
    modified_files = _coerce_planned_files(opt_result.get("modified_files"))
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
        upsert_current_best(
            params.campaign_dir,
            best_round=round_n,
            best_score=aggregate,
            baseline_score=_baseline_score_from_state(state),
        )
        _git_commit_round(params, round_n, hypothesis, decision)
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
        upsert_current_best(
            params.campaign_dir,
            best_round=round_n,
            best_score=aggregate,
            baseline_score=_baseline_score_from_state(state),
        )
        _git_commit_round(params, round_n, hypothesis, decision)
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
            modified_files=modified_files,
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
    """Revert the workspace to the last ACCEPTED commit.

    Canonical path (git): ``FORCED_GIT_COMMIT=True`` guarantees every
    accepted round lands as a commit on the ``optimize/<campaign_id>``
    branch, so ``HEAD`` on that branch is always the last accepted
    state. The rollback is then just::

        git reset --hard HEAD
        git clean -fd
        git submodule update --recursive

    which reverts *all* modified tracked files (not only
    ``kernel_source``), deletes *all* untracked files the failed round
    emitted (including ones at unexpected paths like the repo root),
    and pulls the submodule pointers back to their HEAD-pinned
    commits. Before this change, the rollback path was a file-copy
    from ``rounds/round-{best}/kernel_snapshot/``; that path only
    restored the single file at ``params.kernel_source`` and wrote it
    to ``workspace_root/<basename>`` (i.e. the repo root rather than
    the original nested directory). The result was a slow-motion
    working-tree corruption across consecutive rollbacks — exactly
    the residue we found on campaign
    ``optimize_grouped_gemm_fp8_tensorwise_triton_back_202604231519``
    after rounds 4/5/6 rolled back and round 7 crashed without
    rolling back. See ``docs/issue.md`` entries on "rollback leaves
    foreign files" for the failure history.

    Fallback path (file copy): kept for the pathological case where
    ``workspace_root`` is not a git working tree (e.g. a test fixture
    or a user who manually de-initialised git). A WARNING makes it
    very visible because the fallback cannot restore nested dirs or
    remove added files.
    """
    assert params.campaign_dir is not None
    workspace = params.workspace_root
    if workspace is None:
        log.warning("rollback skipped: workspace_root is unset")
        return

    if _git_rollback(workspace, state=state, round_n=round_n):
        return

    # --- fallback: snapshot-copy (known-broken; last resort only) ---
    best = state.best_round
    if best is None or params.kernel_source in (None, ""):
        log.warning(
            "rollback fell back to snapshot-copy but best_round=%s / "
            "kernel_source=%r; nothing to do",
            best,
            params.kernel_source,
        )
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
    log.warning(
        "rollback falling back to snapshot-copy for round-%d; "
        "this cannot restore non-kernel files (C++, build artefacts, "
        "untracked garbage). Fix by ensuring `%s` is a git working tree.",
        round_n,
        workspace,
    )
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    log.info("rolled kernel back to round-%d snapshot at %s", best, src)


def _git_rollback(
    workspace: Path,
    *,
    state: RunState,
    round_n: int,
) -> bool:
    """Run ``git reset --hard HEAD`` + ``git clean -fd`` inside ``workspace``.

    Returns ``True`` if every git step succeeded, ``False`` if git is
    unavailable or any command failed (so the caller can fall back to
    the snapshot-copy path). Each subprocess is bounded by a short
    timeout so a stuck git process cannot hang the phase.

    Submodules are refreshed with ``git submodule update --recursive``
    and then scrubbed with ``git submodule foreach --recursive
    'git reset --hard && git clean -fd'`` so that 3rdparty checkouts
    (the Primus-Turbo repo ships ``3rdparty/composable_kernel`` as a
    submodule) both return to their HEAD-pinned commit AND drop any
    untracked files a failed round may have dumped inside them. The
    foreach step is what caught the real-world incident on the
    2026-04-23 campaign where rounds 4-6 left 40+ generated
    ``*_hip.hpp`` headers inside ``3rdparty/composable_kernel`` that
    the plain ``submodule update`` step would otherwise preserve.
    """
    if not (workspace / ".git").exists():
        log.warning(
            "rollback: %s has no .git directory; falling back to snapshot-copy",
            workspace,
        )
        return False

    best = state.best_round
    best_str = f"round-{best}" if best is not None else "baseline"
    log.info(
        "rollback: git-reset workspace %s back to HEAD (= %s, last ACCEPTED)",
        workspace,
        best_str,
    )
    steps: list[tuple[list[str], float]] = [
        (["git", "reset", "--hard", "HEAD"], 30.0),
        (["git", "clean", "-fd"], 30.0),
        (["git", "submodule", "update", "--recursive"], 120.0),
        (
            [
                "git",
                "submodule",
                "foreach",
                "--recursive",
                "git reset --hard && git clean -fd",
            ],
            120.0,
        ),
    ]
    for argv, timeout_s in steps:
        try:
            subprocess.run(
                argv,
                cwd=str(workspace),
                check=True,
                timeout=timeout_s,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            log.warning(
                "rollback: `git` executable not on PATH; falling back to "
                "snapshot-copy (fix PATH to enable the canonical rollback)"
            )
            return False
        except subprocess.TimeoutExpired:
            log.warning(
                "rollback: `%s` exceeded %.0fs timeout in %s; "
                "falling back to snapshot-copy",
                " ".join(argv),
                timeout_s,
                workspace,
            )
            return False
        except subprocess.CalledProcessError as exc:
            log.warning(
                "rollback: `%s` failed (rc=%s) in %s: %s; "
                "falling back to snapshot-copy",
                " ".join(argv),
                exc.returncode,
                workspace,
                (exc.stderr or "").strip().splitlines()[:3],
            )
            return False
    log.info(
        "rollback: workspace %s clean at HEAD (round-%d changes discarded)",
        workspace,
        round_n,
    )
    return True


def _git_commit_round(
    params: CampaignParams,
    round_n: int,
    hypothesis: dict,
    decision: DecisionResult,
) -> None:
    """Commit the accepted round to the optimize branch.

    Git commit is forced on (:data:`FORCED_GIT_COMMIT`); there is no
    opt-out. The commit is annotated with the hypothesis and the
    measured improvement so the branch history doubles as an audit log.
    """
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

    if not optimize_log_path(campaign_dir).exists():
        return
    append_termination_block(campaign_dir, checks=checks, passed=passed)


def _invalidate_stale_report_cache(state_dir: Path, state: RunState) -> None:
    """Drop the cached REPORT ``phase_result/report.json`` when the
    on-disk summary no longer matches the current campaign state.

    Why this exists. ``run_phase`` reuses ``phase_result/<phase>.json``
    whenever the file is loadable, which means a warm-restart that
    hits REPORT again will normally skip the LLM session entirely
    (``cost.md`` row labelled ``cached``). For most phases that is the
    right choice — the phase output is purely a function of the prompt
    inputs. REPORT is the exception: its only side effect besides
    writing ``report.json`` is appending entries to the cross-campaign
    ``tips.md`` knowledge base via ``mcp__turbo__append_tip``. If the
    cache wins, **no new tips are written** even when the resumed
    campaign produced more ACCEPT / ROLLBACK rounds with fresh
    lessons. The
    ``optimize_grouped_gemm_fp8_tensorwise_triton_back_202604231519``
    incident lost the R56 / R57 / R69 / R77 success tips this way: the
    second REPORT (after warm restart) ran in ``cached`` mode and
    distilled nothing.

    Staleness rule. We force a rerun when either of:

    * ``final_best_aggregate.round`` in the cached report disagrees
      with :attr:`RunState.best_round` — a different best round was
      accepted on the resumed run, so the headline numbers and the
      "Key Effective Optimizations" tip pool both moved.
    * ``total_rounds`` in the cached report disagrees with
      :attr:`RunState.current_round` — more rounds happened (even if
      the best stayed the same), and any new ROLLBACKs may carry
      reusable failure tips that the previous REPORT could not see.

    Cache misses (file absent) and corrupt / old-schema files (JSON
    load error) are no-ops — ``run_phase`` already handles those by
    re-running the phase, so this helper only acts on the genuinely
    stale case.
    """
    report_path = phase_result_path(state_dir, "REPORT")
    if not report_path.exists():
        return
    try:
        prev = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning(
            "REPORT cache check: cannot parse %s (%s); leaving as-is "
            "(run_phase will re-run on its own when the cache is "
            "unloadable)",
            report_path,
            exc,
        )
        return
    cached_best = (prev.get("final_best_aggregate") or {}).get("round")
    cached_total = prev.get("total_rounds")
    if cached_best == state.best_round and cached_total == state.current_round:
        return
    log.info(
        "REPORT cache stale (cached best_round=%s total_rounds=%s vs "
        "state best_round=%s current_round=%s); unlinking %s so the "
        "phase re-runs and new ACCEPT / ROLLBACK rounds get distilled "
        "into tips.md",
        cached_best,
        cached_total,
        state.best_round,
        state.current_round,
        report_path,
    )
    try:
        report_path.unlink()
    except OSError as exc:
        log.warning(
            "REPORT cache invalidation: cannot unlink %s (%s); falling "
            "back to cached output, new tips may be lost this round",
            report_path,
            exc,
        )


async def _final_report(params: CampaignParams, state: RunState, *, reason: str) -> None:
    if params.campaign_dir is None:
        log.warning("no campaign_dir; skipping REPORT")
        return
    _invalidate_stale_report_cache(params.state_dir, state)
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
            "base_branch", "max_iterations", "max_duration",
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
    base = _baseline_score_from_state(state)
    if not base:
        return None
    base_fwd = _safe_float(base.get("Forward TFLOPS"))
    base_bwd = _safe_float(base.get("Backward TFLOPS"))
    base_step = _compute_step_geomean(base_fwd, base_bwd)
    return {
        "step": _pct_delta(candidate_step, base_step),
        "fwd": _pct_delta(candidate_fwd, base_fwd),
        "bwd": _pct_delta(candidate_bwd, base_bwd),
    }


def _baseline_score_from_state(state: RunState) -> dict[str, float] | None:
    """Return the aggregate score recorded by the BASELINE history event.

    Shared by :func:`_baseline_deltas` (trend rendering) and the new
    ``upsert_current_best`` call so both references use the same
    baseline dict rather than duplicating history-scan logic.
    """
    baseline = next(
        (h for h in state.history if h.get("decision") == "BASELINE"), None
    )
    if not baseline:
        return None
    score = baseline.get("score") or {}
    return {k: float(v) for k, v in score.items() if isinstance(v, (int, float))}


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
    """Return the BASELINE per-shape score vector, parsed from round-1's CSV.

    The previous implementation built ``{"shape": {}, ...}`` entries
    directly from the raw row dict, which collapsed every shape into
    the same empty geometry key and exposed shape columns (``B, M, N,
    K``) as if they were metrics. That broke the per-shape regression
    gate silently: ``find_per_shape_regressions`` saw one lookup entry
    for the whole baseline, so no candidate shape ever matched and no
    regression could ever be detected.

    We now route the read through :func:`parse_bench_csv` so the
    returned rows honour the canonical schema (``Forward TFLOPS`` /
    ``Backward TFLOPS``, PASS / FAIL, geometry-only shape dict) and
    the alias normaliser — meaning a BASELINE CSV produced by either
    the full ``benchmark_command`` harness or the quick
    ``quick_command`` harness is picked up identically. This is the
    guarantee that makes the per-round comparison methodologically
    consistent with the rules documented in
    ``docs/performance-measurement-confidence.md``.
    """
    if params.campaign_dir is None:
        return []
    path = params.campaign_dir / "rounds" / "round-1" / "artifacts" / "benchmark.csv"
    if not path.exists():
        return []
    try:
        parse = parse_bench_csv(path, params.primary_metric or "")
    except ScoringError as exc:
        log.warning("could not parse baseline CSV %s: %s", path, exc)
        return []
    rows: list[dict] = []
    for row in parse.rows:
        rows.append(
            {
                "shape": dict(row.shape),
                "check": row.check,
                "metrics": dict(row.metrics),
            }
        )
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
