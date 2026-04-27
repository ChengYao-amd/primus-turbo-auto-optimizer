"""Campaign-scoped configuration objects.

`CampaignParams` is the single source of truth consumed by every phase runner
after `DEFINE_TARGET` has populated `manifest.yaml`. The CLI layer builds a
partial instance from command-line flags; `state.load_or_init_run()` then
merges the manifest-confirmed fields once the user approves the draft.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

MODEL_FALLBACK = "claude-opus-4-7"
"""Hardcoded default for ``--model``. Anthropic ``claude-opus-4-7``
(released 2026-04-16). Not read from any environment variable: the
orchestrator picks this value iff the CLI flag and saved ``run.json``
are both silent, so a single slug lives in exactly one place."""

EFFORT_FALLBACK = "max"
"""Hardcoded default for ``--effort``. See :data:`MODEL_FALLBACK` for
the reasoning."""

EFFORT_CHOICES: tuple[str, ...] = ("low", "medium", "high", "max")


PHASE_TIMEOUT_DEFAULTS: dict[str, dict[str, object]] = {
    # Sizing model (updated 2026-04-23 after the
    # ``optimize_grouped_gemm_fp8_tensorwise_triton_back_202604231027``
    # false-positive where ``VALIDATE (full)`` idle=360 fired mid-pytest):
    #
    #   idle >= 1.5 * worst_single_silent_subtask + stream_keepalive_slack
    #
    # During a long Bash tool call the SDK stream is genuinely silent
    # between the ``tool_use`` emission and the eventual ``tool_result``;
    # idle must cover that full interval, not the average turn gap.
    # Observed worst cases on the primus-turbo benchmark rigs:
    #
    #   * ``benchmark_command`` 288-shape sweep   ≈ 1000s (≈17 min)
    #   * ``test_command`` pytest + Triton autotune cache miss ≈ 500s
    #   * ``quick_command`` autotune warm-up       ≈ 200s
    #   * ``rocprof`` capture on MI300             ≈ 300s
    #   * ``pip install`` of project with compile  ≈ 400s
    #   * ``WebFetch`` large doc + retries         ≈ 90s
    #
    # ``wall`` still follows ``P95 * 1.7`` across all retries.
    #
    # Keys may be either ``"PHASE"`` or ``"PHASE (variant)"``; see
    # :func:`get_phase_timeouts` for lookup precedence. ``VALIDATE``
    # is the main reason per-variant entries exist: quick and full
    # have ~5x different work envelopes.
    #
    # ``idle``   — seconds between two adjacent SDK messages.
    # ``wall``   — hard upper bound for the whole phase including retries.
    # ``retries`` — how many extra attempts after idle timeout.
    # ``retriable`` — whether a retry is safe (idempotent side-effects).
    "DEFINE_TARGET":        {"idle": 300,  "wall": 900,  "retries": 1, "retriable": True},
    "PREPARE_ENVIRONMENT":  {"idle": 900,  "wall": 5400, "retries": 0, "retriable": False},
    "SURVEY_RELATED_WORK":  {"idle": 300,  "wall": 1800, "retries": 1, "retriable": True},
    "READ_HISTORICAL_TIPS": {"idle": 180,  "wall": 600,  "retries": 1, "retriable": True},
    "BASELINE":             {"idle": 1500, "wall": 7200, "retries": 0, "retriable": False},
    "PROFILE":              {"idle": 600,  "wall": 2400, "retries": 1, "retriable": True},
    "ANALYZE":              {"idle": 420,  "wall": 2400, "retries": 1, "retriable": True},
    "OPTIMIZE":             {"idle": 600,  "wall": 3600, "retries": 0, "retriable": False},
    "VALIDATE":             {"idle": 1500, "wall": 7200, "retries": 0, "retriable": False},
    "VALIDATE (quick)":     {"idle": 600,  "wall": 2400, "retries": 0, "retriable": False},
    "VALIDATE (full)":      {"idle": 1500, "wall": 7200, "retries": 0, "retriable": False},
    # REVIEW is read-only (no bench/test/build). Idle covers the
    # worst-case LLM latency when it has to cross-reference the quick
    # + full CSVs; wall stays short because no subprocess can legitimately
    # stretch this phase.
    "REVIEW":               {"idle": 300,  "wall": 900,  "retries": 1, "retriable": True},
    "STAGNATION_REVIEW":    {"idle": 300,  "wall": 1200, "retries": 1, "retriable": True},
    "REPORT":               {"idle": 300,  "wall": 1200, "retries": 1, "retriable": True},
}

PHASE_TIMEOUT_FALLBACK: dict[str, object] = {
    # Used when a phase name is not registered above. Keep generous so
    # new / experimental phases never regress into a silent hang, but
    # not so large that a real stall sits for hours.
    "idle": 600,
    "wall": 3600,
    "retries": 0,
    "retriable": False,
}


def get_phase_timeouts(
    phase: str, phase_variant: str | None = None
) -> dict[str, object]:
    """Return the default idle/wall/retries/retriable tuple for ``phase``.

    Lookup precedence (first match wins):

    1. ``"PHASE (variant)"`` — variant-scoped entry (e.g.
       ``"VALIDATE (quick)"`` vs ``"VALIDATE (full)"``).
    2. ``"PHASE"`` — phase-only entry, used when no variant-specific
       tuning is needed.
    3. :data:`PHASE_TIMEOUT_FALLBACK` — conservative defaults for
       unregistered phases.

    Callers of :func:`turbo_optimize.orchestrator.run_phase.run_phase`
    may still override any field via explicit kwargs; the table is
    only consulted when the caller leaves a slot as ``None``.
    """
    if phase_variant:
        variant_key = f"{phase} ({phase_variant})"
        if variant_key in PHASE_TIMEOUT_DEFAULTS:
            return dict(PHASE_TIMEOUT_DEFAULTS[variant_key])
    return dict(PHASE_TIMEOUT_DEFAULTS.get(phase, PHASE_TIMEOUT_FALLBACK))


@dataclass
class CampaignParams:
    """Runtime parameters for a single optimization campaign."""

    prompt: str | None = None
    campaign_id: str | None = None
    campaign_dir: Path | None = None

    workspace_root: Path = Path("agent_workspace/Primus-Turbo")
    skills_root: Path = Path("agent_workspace/Primus-Turbo/agent")
    project_skill: str = "primus-turbo-develop"

    model: str | None = None
    effort: str | None = None

    target_op: str | None = None
    target_backend: str | None = None
    target_lang: str | None = None
    target_gpu: str | None = None
    execution_mode: str | None = None
    primary_metric: str | None = None
    performance_target: str | None = None
    target_shapes: str | None = None
    representative_shapes: str | None = None
    kernel_source: str | None = None
    test_command: str | None = None
    benchmark_command: str | None = None
    quick_command: str | None = None
    profile_command: str | None = None
    related_work_file: str | None = None

    # Git integration is non-negotiable for this campaign runner:
    #   * every ACCEPTED round is committed so rollback can use
    #     ``git reset --hard`` instead of the flat file-copy path
    #     (:func:`turbo_optimize.orchestrator.campaign._rollback_kernel`
    #     cannot restore nested subdirs, delete newly-added files, or
    #     clear Triton/pycache artefacts);
    #   * experiments always run on a dedicated ``optimize/<campaign_id>``
    #     branch so the user's source branch stays untouched.
    # ``git_commit`` and ``git_branch`` are therefore force-applied as
    # module-level constants in :mod:`turbo_optimize.orchestrator.campaign`
    # (:data:`FORCED_GIT_COMMIT` / :data:`FORCED_GIT_BRANCH`) and are not
    # exposed as :class:`CampaignParams` fields. ``base_branch`` remains
    # user-configurable — it names the upstream branch the optimize
    # branch descends from.
    base_branch: str | None = None
    max_iterations: int | None = None
    max_duration: str | None = None
    debug_retry: int = 3

    dry_run: bool = False
    state_dir: Path = Path("state")

    # Historical-tips knowledge base root. ``None`` resolves at access
    # time via :func:`default_tips_root` so the chosen location follows
    # the current tool checkout / ``TURBO_TIPS_ROOT`` env override even
    # when an old ``run.json`` (which never wrote this key) is resumed.
    # The path is intentionally OUTSIDE ``workspace_root``: keeping tips
    # under the optimized project meant
    # ``_git_rollback`` -> ``git clean -fd`` would erase ``tips.md``
    # along with every untracked file the failed round dropped, which is
    # exactly how the
    # ``optimize_grouped_gemm_fp8_tensorwise_triton_back_202604231519``
    # run lost all 5 tips written at R50 REPORT.
    tips_root: Path | None = None

    manifest_fields: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CampaignParams":
        kwargs = dict(data)
        for key in (
            "workspace_root",
            "skills_root",
            "state_dir",
            "campaign_dir",
            "tips_root",
        ):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        return cls(**kwargs)

    def resolved_tips_root(self) -> Path:
        """Return the effective historical-tips root (always absolute).

        Resolves :attr:`tips_root` against :func:`default_tips_root` when
        the field is unset, so callers never have to special-case
        ``None``. Result is ``Path``-resolved (symlinks collapsed) to
        avoid cross-mount comparison surprises.
        """
        root = self.tips_root if self.tips_root is not None else default_tips_root()
        return Path(root).expanduser().resolve()

    def resolve_runtime_defaults(self) -> None:
        """Fill ``model`` / ``effort`` with the hardcoded fallbacks if unset.

        Called after the CLI + resume-state merge. Priority (highest wins):

        1. CLI flag (non-None ``self.model`` / ``self.effort``)
        2. On resume, whatever the previous run wrote into ``run.json``
           (already copied into ``self`` by the orchestrator)
        3. :data:`MODEL_FALLBACK` / :data:`EFFORT_FALLBACK`

        ``effort`` is validated against :data:`EFFORT_CHOICES` so a typo in
        the saved ``run.json`` surfaces loudly instead of being silently
        dropped by the SDK.
        """
        if self.model in (None, ""):
            self.model = MODEL_FALLBACK
        if self.effort in (None, ""):
            self.effort = EFFORT_FALLBACK
        if self.effort not in EFFORT_CHOICES:
            raise ValueError(
                f"invalid effort '{self.effort}': expected one of "
                f"{', '.join(EFFORT_CHOICES)}"
            )

    def merge_manifest(self, manifest: dict) -> None:
        """Copy manifest.yaml fields into this params instance.

        Called after the user confirms the draft manifest. Fields that the CLI
        already supplied (e.g. `--max-iterations` override) win over the
        manifest; everything else is taken verbatim from the yaml.

        ``base_branch`` is CLI-authoritative: the CLI value always wins
        when provided, otherwise the manifest value is used.
        ``git_commit`` / ``git_branch`` are not in the mapping at all —
        they are module-level constants (see :class:`CampaignParams`
        docstring) and any such keys in manifest.yaml are silently
        ignored.
        """
        mapping = {
            "target_op": "target_op",
            "target_backend": "target_backend",
            "target_lang": "target_lang",
            "target_gpu": "target_gpu",
            "execution_mode": "execution_mode",
            "primary_metric": "primary_metric",
            "performance_target": "performance_target",
            "target_shapes": "target_shapes",
            "representative_shapes": "representative_shapes",
            "kernel_source": "kernel_source",
            "test_command": "test_command",
            "benchmark_command": "benchmark_command",
            "quick_command": "quick_command",
            "profile_command": "profile_command",
            "related_work_file": "related_work_file",
            "base_branch": "base_branch",
            "project_skill": "project_skill",
        }
        for attr, key in mapping.items():
            if key in manifest and manifest[key] is not None:
                current = getattr(self, attr)
                if current in (None, "", False) or attr in (
                    "target_op",
                    "target_backend",
                    "target_gpu",
                    "execution_mode",
                    "primary_metric",
                    "kernel_source",
                    "test_command",
                    "benchmark_command",
                ):
                    setattr(self, attr, manifest[key])

        if self.max_iterations is None and manifest.get("max_iterations") is not None:
            self.max_iterations = int(manifest["max_iterations"])
        if self.max_duration is None and manifest.get("max_duration"):
            self.max_duration = manifest["max_duration"]

        self.manifest_fields = {
            k: v for k, v in manifest.items() if not k.startswith("_")
        }

        self._expand_campaign_vars()

    def _expand_campaign_vars(self) -> None:
        """Replace ``${CAMPAIGN_DIR}`` in command/path fields with the
        actual campaign directory.

        The manifest stores these fields as portable templates so that
        ``manifest.yaml`` never embeds an absolute path.  Expansion
        happens once, right after ``merge_manifest``, so every downstream
        consumer sees fully-resolved paths.
        """
        if self.campaign_dir is None:
            return
        campaign_dir_str = str(self.campaign_dir)
        for attr in ("quick_command", "profile_command", "related_work_file"):
            value = getattr(self, attr)
            if value and "${CAMPAIGN_DIR}" in value:
                setattr(self, attr, value.replace("${CAMPAIGN_DIR}", campaign_dir_str))


def make_campaign_id(prompt: str, *, now: datetime | None = None) -> str:
    """Heuristic campaign id derived from a prompt.

    `DEFINE_TARGET` later writes the canonical id into the manifest;
    this function only produces a temporary placeholder used for the
    campaign directory during the very first phase.
    """
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d%H%M")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", prompt.strip().lower()).strip("_")
    if not slug:
        slug = "campaign"
    return f"{slug[:48]}_{stamp}"


def validate_campaign_id(cid: str) -> str:
    if not CAMPAIGN_ID_RE.match(cid):
        raise ValueError(
            f"invalid campaign id '{cid}': "
            "expect [A-Za-z0-9_.-] only (no slashes, no whitespace)"
        )
    return cid


def default_campaign_root(workspace_root: Path) -> Path:
    return workspace_root / "agent" / "workspace"


def default_tips_root() -> Path:
    """Return the default historical-tips root path.

    Resolution order (first match wins):

    1. ``TURBO_TIPS_ROOT`` environment variable. Useful when several
       worktrees / virtualenvs of this repo should share one knowledge
       base, or when tips live on a network volume.
    2. ``<tool_repo>/agent_data/historical_experience``, where
       ``<tool_repo>`` is the parent of the ``turbo_optimize`` package
       (i.e. ``primus-turbo-auto-optimizer/`` for this checkout).

    The default location is intentionally inside the orchestrator
    repo rather than under ``workspace_root``: the ``_git_rollback``
    step run after every failed round executes ``git clean -fd``
    inside ``workspace_root``, which deletes any untracked file or
    directory there — including the tips knowledge base. Keeping
    ``tips.md`` outside of that tree makes the file survive every
    rollback regardless of how the optimized project's ``.gitignore``
    is configured.
    """
    env = os.environ.get("TURBO_TIPS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    pkg_dir = Path(__file__).resolve().parent
    tool_repo = pkg_dir.parent
    return tool_repo / "agent_data" / "historical_experience"
