"""Campaign-scoped configuration objects.

`CampaignParams` is the single source of truth consumed by every phase runner
after `DEFINE_TARGET` has populated `manifest.yaml`. The CLI layer builds a
partial instance from command-line flags; `state.load_or_init_run()` then
merges the manifest-confirmed fields once the user approves the draft.
"""

from __future__ import annotations

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
    # Empirical basis: 80+ phase executions across
    # ``optimize_gemm_fp8_blockwise_kernel_with_triton_b_202604220415``
    # (10 rounds) and ``...221559`` (10 rounds, with PROFILE enabled).
    # For each phase we take the observed max wall and aim for
    # ``wall >= P95 * 1.7`` (rule of thumb); idle >= ``avg-turn-gap * 1.5``
    # plus slack for the longest single silent sub-step (autotune,
    # pip install, cmake configure, rocprof capture, WebFetch).
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
    "DEFINE_TARGET":        {"idle": 90,  "wall": 600,  "retries": 1, "retriable": True},
    "PREPARE_ENVIRONMENT":  {"idle": 300, "wall": 3600, "retries": 0, "retriable": False},
    "SURVEY_RELATED_WORK":  {"idle": 180, "wall": 1500, "retries": 1, "retriable": True},
    "READ_HISTORICAL_TIPS": {"idle": 60,  "wall": 300,  "retries": 1, "retriable": True},
    "BASELINE":             {"idle": 420, "wall": 5400, "retries": 0, "retriable": False},
    "PROFILE":              {"idle": 150, "wall": 1200, "retries": 1, "retriable": True},
    "ANALYZE":              {"idle": 240, "wall": 1800, "retries": 1, "retriable": True},
    "OPTIMIZE":             {"idle": 300, "wall": 3600, "retries": 0, "retriable": False},
    "VALIDATE":             {"idle": 300, "wall": 5400, "retries": 0, "retriable": False},
    "VALIDATE (quick)":     {"idle": 300, "wall": 1800, "retries": 0, "retriable": False},
    "VALIDATE (full)":      {"idle": 360, "wall": 5400, "retries": 0, "retriable": False},
    "STAGNATION_REVIEW":    {"idle": 180, "wall": 900,  "retries": 1, "retriable": True},
    "REPORT":               {"idle": 150, "wall": 900,  "retries": 1, "retriable": True},
}

PHASE_TIMEOUT_FALLBACK: dict[str, object] = {
    # Used when a phase name is not registered above. Keep generous so
    # new / experimental phases never regress into a silent hang, but
    # not so large that a real stall sits for hours.
    "idle": 300,
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

    git_commit: bool = False
    git_branch: str = "auto"
    base_branch: str | None = None
    max_iterations: int | None = None
    max_duration: str | None = None
    debug_retry: int = 3

    dry_run: bool = False
    state_dir: Path = Path("state")

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
        ):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        return cls(**kwargs)

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

        ``git_commit`` and ``base_branch`` are CLI-authoritative: the CLI
        value always wins when provided.  ``git_commit`` is excluded from
        the mapping entirely (default ``False``); ``base_branch`` is in
        the mapping but is skipped when the CLI already set it to a
        non-empty value.
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
            "git_branch": "git_branch",
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
