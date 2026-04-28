"""``primus-turbo-view`` command-line entry point.

::

    primus-turbo-view <path> [-o OUT_DIR] [-v]
                              [--multi] [--single]
                              [--watch] [--port PORT] [--host HOST] [--no-open]

Mode selection:

* Default: auto-detect. If ``<path>/state/<id>/run.json`` exists the
  path is treated as a single campaign; otherwise we run
  ``discover_campaigns`` and use overview mode when ≥2 campaigns
  are found.
* ``--single`` forces single-campaign mode and errors out if the
  path lacks a run.json.
* ``--multi`` forces overview mode even when only one campaign is
  visible (useful for debugging / staged workspaces).

Watch flags (``--watch`` / ``--port`` / ``--host`` / ``--no-open``)
start an HTTP server (default ``127.0.0.1:8765``), open the browser,
and re-render whenever the campaign on disk changes (debounced 500
ms). The server pushes a ``reload`` event over Server-Sent Events;
the front-end refetches ``data.json`` and re-bootstraps without a
full page reload.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from turbo_view import __version__
from turbo_view.discover import _is_valid_campaign, discover_campaigns
from turbo_view.io.loader import load_campaign
from turbo_view.render.build import write_detail
from turbo_view.render.overview import write_overview


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="primus-turbo-view",
        description=(
            "Render primus-turbo-optimize campaigns into a self-contained "
            "HTML dashboard. Auto-detects single vs multi-campaign mode."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Campaign directory or workspace root.",
    )
    parser.add_argument(
        "-o", "--out-dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <path>/view/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--multi",
        action="store_true",
        help="Force overview/multi-campaign mode.",
    )
    mode.add_argument(
        "--single",
        action="store_true",
        help="Force single-campaign mode.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Serve the dashboard and rebuild on file changes.",
    )
    parser.add_argument("--port", type=int, default=8765,
                        help="Watch-mode HTTP port (0 = OS-chosen).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Watch-mode bind address.")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open the browser on start.")
    parser.add_argument(
        "-v", "--verbose",
        action="count", default=0,
        help="Increase logging verbosity (repeat for more).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args(argv)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _looks_like_single_campaign(path: Path) -> bool:
    """Heuristic: ``path`` itself looks like one campaign root.

    Matches when ``state/<id>/run.json`` exists, OR the directory has
    a ``manifest.yaml`` / ``rounds/`` / ``profiles/`` -- the layout
    real workspaces produce even before the first ``run.json`` is
    written.
    """
    if (path / "manifest.yaml").is_file():
        return True
    if (path / "rounds").is_dir() or (path / "profiles").is_dir():
        return True
    state_dir = path / "state"
    if state_dir.is_dir() and any(
        rj.is_file() for rj in state_dir.glob("*/run.json")
    ):
        return True
    return False


def _resolve_mode(args: argparse.Namespace, log: logging.Logger) -> str:
    if args.single:
        return "single"
    if args.multi:
        return "multi"
    if _looks_like_single_campaign(args.path):
        return "single"
    handles = discover_campaigns(args.path)
    if len(handles) <= 1:
        if not handles:
            log.warning("no campaigns discovered under %s", args.path)
            return "single"
        return "single"
    return "multi"


def _run_watch(
    *,
    mode: str,
    path: Path,
    out_dir: Path,
    host: str,
    port: int,
    open_browser: bool,
    log: logging.Logger,
) -> int:
    # Local import keeps watchdog out of the import path for the
    # plain ``primus-turbo-view`` command, so users without the
    # ``view`` extra installed still get the static build.
    from turbo_view.watch.server import serve

    stop_event = threading.Event()

    def _on_started(url: str) -> None:
        print(f"serving {url}")
        print(f"  out_dir: {out_dir}")
        print(f"  watching: {path}")
        print("Ctrl-C to stop")

    server = serve(
        mode=mode,
        root=path,
        out_dir=out_dir,
        host=host,
        port=port,
        open_browser=open_browser,
        on_started=_on_started,
    )

    def _shutdown(signum, frame):  # type: ignore[no-untyped-def]
        log.info("signal %s -> shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        stop_event.wait()
    finally:
        server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    log = logging.getLogger("turbo_view.cli")

    path: Path = args.path
    if not path.is_dir():
        log.error("not a directory: %s", path)
        return 2

    mode = _resolve_mode(args, log)
    out_dir: Path = args.out_dir or (path / "view")
    log.info("rendering %s -> %s [mode=%s]", path, out_dir, mode)

    if args.watch:
        # Single-mode watch needs a campaign-shaped directory but
        # NOT necessarily a ``run.json``; many real workspaces have
        # only ``manifest.yaml`` + ``rounds/`` + ``profiles/``.
        # ``_is_valid_campaign`` matches what discover/load tolerate.
        if mode == "single" and not _is_valid_campaign(path):
            log.error(
                "watch mode in single-campaign needs a manifest.yaml or "
                "a parseable state/<id>/run.json under %s; pass --multi "
                "or use a campaign directory.",
                path,
            )
            return 2
        return _run_watch(
            mode=mode,
            path=path,
            out_dir=out_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            log=log,
        )

    if mode == "multi":
        try:
            index_path = write_overview(path, out_dir)
        except RuntimeError as exc:
            log.error("%s", exc)
            return 2
    else:
        bundle = load_campaign(path)
        if bundle.state is None:
            log.warning(
                "no run.json under %s/state/*/ — dashboard will show N/A "
                "for sticky-bar fields", path,
            )
        index_path = write_detail(bundle, out_dir)

    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
