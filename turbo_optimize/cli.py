"""`primus-turbo-optimize` entry point.

Usage:
    primus-turbo-optimize -p "optimize gemm fp8 blockwise with triton backend"
    primus-turbo-optimize -s gemm_fp8_blockwise_triton_gfx942_20260412

The CLI is intentionally small: parse flags, build a partial
:class:`CampaignParams`, hand control to the orchestrator. Everything else
(state persistence, manifest confirmation, phase execution, signal handling)
lives in the modules imported below.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from turbo_optimize import __version__
from turbo_optimize.config import (
    EFFORT_CHOICES,
    EFFORT_FALLBACK,
    MODEL_FALLBACK,
    CampaignParams,
    default_campaign_root,
    validate_campaign_id,
)


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Exposed separately from :func:`_parse_args` so unit tests can introspect
    default values / flag behaviour without going through ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        prog="primus-turbo-optimize",
        description=(
            "Drive the kernel-optimize skill loop as a long-running, "
            "mostly unattended campaign."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-p",
        "--prompt",
        metavar="TEXT",
        help="Natural-language optimization goal (starts a new campaign).",
    )
    mode.add_argument(
        "-s",
        "--campaign",
        metavar="ID",
        help="Resume an existing campaign (directory name under agent/workspace/).",
    )
    mode.add_argument(
        "--cleanup-stray",
        metavar="ID",
        help=(
            "Move top-level untracked files in --workspace-root into "
            "<campaign_dir>/_stray/<timestamp>/ for the given campaign. "
            "Dry-run by default; pass --apply to actually move files."
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Together with --cleanup-stray: actually move files (instead "
            "of just listing them)."
        ),
    )

    parser.add_argument(
        "--skills-root",
        type=Path,
        default=Path("agent_workspace/Primus-Turbo/agent"),
        help="Root of the agent/ skill tree (default: %(default)s).",
    )
    parser.add_argument(
        "--project-skill",
        default="primus-turbo-develop",
        help="Project skill name (default: %(default)s).",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("agent_workspace/Primus-Turbo"),
        help=(
            "Root passed to ClaudeAgentOptions.cwd; also the parent of "
            "agent/workspace/<campaign>/. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("state"),
        help=(
            "Parent directory for per-campaign state. The orchestrator "
            "nests every campaign under <state_dir>/<campaign_id>/ so "
            "run.json and phase_result/* from different runs never clash. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="ID",
        help=(
            f"Anthropic model id passed to ClaudeAgentOptions.model "
            f"(e.g. 'claude-opus-4-7', 'claude-sonnet-4-5'). Default for "
            f"new campaigns: '{MODEL_FALLBACK}'. On `-s <campaign>` resume, "
            f"the value saved in run.json is reused unless this flag "
            f"overrides it."
        ),
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=EFFORT_CHOICES,
        help=(
            f"Extended-thinking depth passed to ClaudeAgentOptions.effort. "
            f"Default for new campaigns: '{EFFORT_FALLBACK}'. Resume "
            f"behaviour matches --model."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override manifest.max_iterations (must be < 120).",
    )
    parser.add_argument(
        "--max-duration",
        default=None,
        help='Override manifest.max_duration, e.g. "4h" or "90m".',
    )
    parser.add_argument(
        "--debug-retry",
        type=int,
        default=3,
        metavar="N",
        help=(
            "How many times OPTIMIZE+VALIDATE may retry the same hypothesis "
            "with a build/correctness error hint before rolling the round "
            "back. 0 disables retry (instant ROLLBACK on first failure). "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--base-branch",
        default=None,
        metavar="BRANCH",
        help=(
            "Branch that every OPTIMIZE commit descends from. Overrides "
            "the `base_branch` value Claude writes into manifest.yaml. "
            "The PREPARE_ENVIRONMENT gate rejects the campaign when the "
            "working tree is not on this branch. Default: read from manifest "
            "(which itself defaults to 'main')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan-only mode: print phase plan without connecting to Claude.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Lower log level (default INFO; -v -> DEBUG).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _configure_logging(verbosity: int) -> None:
    """Default level is INFO so phase begin / end lines are visible. ``-v``
    drops to DEBUG (and saturates there)."""
    level = logging.INFO - 10 * verbosity
    level = max(level, logging.DEBUG)
    logging.basicConfig(format=LOG_FORMAT, level=level)


def _build_params(args: argparse.Namespace) -> CampaignParams:
    if args.max_iterations is not None:
        if not (0 < args.max_iterations < 120):
            raise SystemExit(
                "ERROR: --max-iterations must be in (0, 120) to stay inside "
                "the iteration_rules hard limit."
            )
    if args.debug_retry < 0:
        raise SystemExit("ERROR: --debug-retry must be >= 0")

    campaign_id: str | None = None
    campaign_dir: Path | None = None
    if args.campaign is not None:
        campaign_id = validate_campaign_id(args.campaign)
        campaign_dir = default_campaign_root(args.workspace_root) / campaign_id
        if not campaign_dir.exists():
            raise SystemExit(
                f"ERROR: campaign '{campaign_id}' not found at {campaign_dir}"
            )

    return CampaignParams(
        prompt=args.prompt,
        campaign_id=campaign_id,
        campaign_dir=campaign_dir,
        workspace_root=args.workspace_root,
        skills_root=args.skills_root,
        project_skill=args.project_skill,
        state_dir=args.state_dir,
        model=args.model,
        effort=args.effort,
        max_iterations=args.max_iterations,
        max_duration=args.max_duration,
        debug_retry=args.debug_retry,
        base_branch=args.base_branch,
        dry_run=args.dry_run,
    )


def _run_cleanup(args: argparse.Namespace) -> int:
    """Handle ``--cleanup-stray <id>`` without ever touching Claude.

    Short-circuits before the ``CampaignParams`` builder because
    ``--prompt`` and ``--campaign`` are not set in this mode (they share
    the mutually-exclusive group with ``--cleanup-stray``).
    """
    from turbo_optimize.config import default_campaign_root, validate_campaign_id
    from turbo_optimize.orchestrator.cleanup import (
        cleanup_stray_files,
        format_report,
    )

    campaign_id = validate_campaign_id(args.cleanup_stray)
    campaign_dir = default_campaign_root(args.workspace_root) / campaign_id
    if not campaign_dir.exists():
        print(f"ERROR: campaign '{campaign_id}' not found at {campaign_dir}")
        return 2
    report = cleanup_stray_files(
        campaign_dir,
        args.workspace_root,
        apply=args.apply,
    )
    print(format_report(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    if args.cleanup_stray is not None:
        return _run_cleanup(args)
    params = _build_params(args)

    from turbo_optimize.orchestrator.campaign import run_campaign

    try:
        return asyncio.run(run_campaign(params))
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning(
            "hard stop on second SIGINT; campaign state left as-is for resume"
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
