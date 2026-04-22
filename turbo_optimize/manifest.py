"""Read / validate / confirm the DEFINE_TARGET manifest.

The SKILL template for `manifest.yaml` lives at SKILL.md lines 174-199.
Confirmation is the single human-in-the-loop point in the entire
campaign. Two modes:

* tty: print a summary, accept `y` / `e` / `n` on stdin
* non-tty: block until `<campaign_dir>/manifest.confirmed` exists

After confirmation, the values are mirrored into `state/run.json.params`
so later phases never need to reparse the yaml.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


log = logging.getLogger(__name__)


REQUIRED_FIELDS: tuple[str, ...] = (
    "target_op",
    "target_backend",
    "target_lang",
    "target_gpu",
    "execution_mode",
    "project_skill",
    "primary_metric",
    "target_shapes",
    "kernel_source",
    "test_command",
    "benchmark_command",
    "quick_command",
    "git_commit",
    "git_branch",
)


CONFIRM_FILE_NAME = "manifest.confirmed"
CANCEL_FILE_NAME = "manifest.canceled"


class ManifestError(Exception):
    """Raised when the manifest cannot be parsed or fails validation."""


@dataclass
class ManifestValidation:
    manifest: dict[str, Any]
    missing: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing


def manifest_path(campaign_dir: Path) -> Path:
    return campaign_dir / "manifest.yaml"


def read_manifest(campaign_dir: Path) -> dict[str, Any]:
    path = manifest_path(campaign_dir)
    if not path.exists():
        raise ManifestError(f"manifest.yaml not found at {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest.yaml parse error: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest.yaml must be a YAML mapping")
    return data


def write_manifest(campaign_dir: Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def validate_manifest(manifest: dict[str, Any]) -> ManifestValidation:
    missing = [k for k in REQUIRED_FIELDS if manifest.get(k) in (None, "")]
    warnings: list[str] = []

    if manifest.get("execution_mode") not in ("repo", "workspace", None):
        warnings.append(
            "execution_mode should be 'repo' or 'workspace'; v1 only supports 'repo'."
        )
    if manifest.get("execution_mode") == "workspace":
        warnings.append(
            "v1 only supports execution_mode=repo; workspace mode will be rejected."
        )
    max_iter = manifest.get("max_iterations")
    if max_iter is not None:
        try:
            if int(max_iter) >= 120:
                warnings.append("max_iterations must be < 120 per SKILL rules.")
        except (TypeError, ValueError):
            warnings.append("max_iterations must be an integer or null.")

    return ManifestValidation(manifest=manifest, missing=missing, warnings=warnings)


def summarize_for_prompt(manifest: dict[str, Any]) -> str:
    lines = ["Draft manifest.yaml (key fields):"]
    keys = [
        "target_op",
        "target_backend",
        "target_gpu",
        "execution_mode",
        "primary_metric",
        "performance_target",
        "target_shapes",
        "git_commit",
        "git_branch",
        "max_iterations",
        "max_duration",
    ]
    for key in keys:
        lines.append(f"  {key}: {manifest.get(key)!r}")
    return "\n".join(lines)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


async def confirm_interactively(
    campaign_dir: Path,
    *,
    poll_interval: float = 2.0,
    initial_print: bool = True,
) -> dict[str, Any]:
    """Block until the user confirms the manifest, return the confirmed dict.

    `y` / `e` / `n` prompt in tty mode. In non-tty mode, block on the
    sentinel file `<campaign_dir>/manifest.confirmed` (content `ok`)
    or `manifest.canceled` for an abort signal.
    """
    confirm_file = campaign_dir / CONFIRM_FILE_NAME
    cancel_file = campaign_dir / CANCEL_FILE_NAME

    if _is_interactive():
        return await asyncio.to_thread(_confirm_tty, campaign_dir, initial_print)

    if initial_print:
        manifest = read_manifest(campaign_dir)
        print(summarize_for_prompt(manifest), flush=True)
        print(
            f"[non-tty] waiting for confirmation: "
            f"`echo ok > {confirm_file}` to accept, "
            f"`echo no > {cancel_file}` to abort.",
            flush=True,
        )

    while True:
        if confirm_file.exists():
            payload = confirm_file.read_text(encoding="utf-8").strip().lower()
            if payload in ("", "ok", "y", "yes"):
                log.info("manifest confirmed via sentinel file")
                return read_manifest(campaign_dir)
            log.warning("ignoring unknown sentinel payload: %r", payload)
            confirm_file.unlink(missing_ok=True)
        if cancel_file.exists():
            raise ManifestError("manifest confirmation canceled by sentinel file")
        await asyncio.sleep(poll_interval)


def _confirm_tty(campaign_dir: Path, initial_print: bool) -> dict[str, Any]:
    while True:
        manifest = read_manifest(campaign_dir)
        validation = validate_manifest(manifest)
        if initial_print or True:
            print("\n" + "=" * 60, flush=True)
            print(summarize_for_prompt(manifest), flush=True)
            if validation.missing:
                print(f"MISSING FIELDS: {validation.missing}", flush=True)
            for warning in validation.warnings:
                print(f"WARN: {warning}", flush=True)
            print(
                "Confirm manifest: [y] accept / [e] edit in $EDITOR / [n] abort",
                flush=True,
            )
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            raise ManifestError("stdin closed before manifest confirmation") from None

        if choice in ("y", "yes"):
            if not validation.ok:
                print(
                    f"cannot accept: missing {validation.missing}. "
                    "press 'e' to edit first.",
                    flush=True,
                )
                continue
            return manifest
        if choice in ("e", "edit"):
            _spawn_editor(manifest_path(campaign_dir))
            continue
        if choice in ("n", "no"):
            raise ManifestError("manifest rejected by user")
        print("please answer y / e / n", flush=True)


def _spawn_editor(path: Path) -> None:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    subprocess.run([editor, str(path)], check=False)


def is_already_confirmed(campaign_dir: Path) -> bool:
    """Campaign resume check — skip confirmation if sentinel already exists."""
    confirm_file = campaign_dir / CONFIRM_FILE_NAME
    return confirm_file.exists()


def mark_confirmed(campaign_dir: Path, payload: str = "ok") -> None:
    (campaign_dir / CONFIRM_FILE_NAME).write_text(payload + "\n", encoding="utf-8")


def wait_for_manifest_draft(
    campaign_dir: Path,
    *,
    timeout_s: float | None = None,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Poll until `manifest.yaml` appears (written by DEFINE_TARGET)."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    path = manifest_path(campaign_dir)
    while True:
        if path.exists():
            return read_manifest(campaign_dir)
        if deadline is not None and time.monotonic() > deadline:
            raise ManifestError(f"timed out waiting for manifest draft at {path}")
        time.sleep(poll_interval)
