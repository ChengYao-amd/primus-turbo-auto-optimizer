"""Read targeted sections out of kernel-optimize SKILL / workflow markdown.

Every phase prompt is built from two inputs: (a) the Python f-string
template in `prompts/<phase>.md`, and (b) an authoritative excerpt from
the SKILL / workflow / rules markdown loaded here. Keeping the excerpt
verbatim avoids drifting from the single source of truth when the SKILL
repo evolves.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


log = logging.getLogger(__name__)


KERNEL_OPTIMIZE_REL = "skills/kernel-optimize/SKILL.md"
WORKFLOW_REL = "skills/kernel-optimize/workflow/optimize-loop.md"
RULES_REL = "rules/iteration_rules.mdc"


@dataclass
class SkillLocation:
    skills_root: Path

    def skill_md(self) -> Path:
        return self.skills_root / KERNEL_OPTIMIZE_REL

    def workflow_md(self) -> Path:
        return self.skills_root / WORKFLOW_REL

    def rules_md(self) -> Path:
        return self.skills_root / RULES_REL


@lru_cache(maxsize=16)
def _read_text_cached(path_str: str) -> str:
    path = Path(path_str)
    return path.read_text(encoding="utf-8")


def _read_text(path: Path) -> str:
    return _read_text_cached(str(path))


def load_section(path: Path, heading: str) -> str:
    """Return the markdown section whose H2/H3 heading matches `heading`.

    Matches both `## DEFINE_TARGET` and `### 1. ENVIRONMENT_BASELINE` styles.
    Returns everything until the next H2/H3 of equal or lower depth.
    """
    text = _read_text(path)
    header_pat = re.compile(r"^(#{2,3})\s+(.*?)\s*$", re.MULTILINE)
    headers: list[tuple[int, int, int, str]] = []
    for m in header_pat.finditer(text):
        depth = len(m.group(1))
        headers.append((m.start(), m.end(), depth, m.group(2).strip()))

    needle_norm = _normalize(heading)
    start: int | None = None
    start_depth = 0
    for i, (s, e, depth, title) in enumerate(headers):
        if _normalize(title).find(needle_norm) != -1:
            start = s
            start_depth = depth
            next_i = i + 1
            break
    else:
        raise KeyError(f"heading '{heading}' not found in {path}")

    end = len(text)
    for s, _e, depth, _title in headers[next_i:]:
        if depth <= start_depth:
            end = s
            break
    return text[start:end].rstrip() + "\n"


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def load_skill_section(
    skills_root: Path, phase: str, *, aliases: dict[str, tuple[Path, str]] | None = None
) -> str:
    """Load the authoritative skill excerpt for a given phase name.

    Phase names follow the state-machine vocabulary (uppercase with
    underscores). The mapping below points at the canonical section in
    SKILL.md / workflow/optimize-loop.md / iteration_rules.mdc.
    """
    loc = SkillLocation(skills_root=skills_root)
    default_map: dict[str, tuple[Path, str]] = {
        "DEFINE_TARGET": (loc.skill_md(), "DEFINE_TARGET"),
        "PREPARE_ENVIRONMENT": (loc.skill_md(), "PREPARE_ENVIRONMENT"),
        "SURVEY_RELATED_WORK": (loc.skill_md(), "SURVEY_RELATED_WORK"),
        "READ_HISTORICAL_TIPS": (loc.skill_md(), "READ_HISTORICAL_TIPS"),
        "BASELINE": (loc.workflow_md(), "ENVIRONMENT_BASELINE"),
        "ANALYZE": (loc.workflow_md(), "ANALYZE"),
        "OPTIMIZE": (loc.workflow_md(), "OPTIMIZE"),
        "VALIDATE": (loc.workflow_md(), "VALIDATE"),
        "ACCEPT": (loc.workflow_md(), "ACCEPT / REPORT"),
        "STAGNATION_REVIEW": (
            loc.workflow_md(),
            "Stagnation Detection and Conditional Intervention",
        ),
        "TERMINATION_CHECK": (loc.workflow_md(), "Termination Conditions"),
        "REPORT": (loc.workflow_md(), "ACCEPT / REPORT"),
        "ITERATION_RULES": (loc.rules_md(), "Core Principle"),
        "SCORING": (loc.workflow_md(), "Scoring Operations Specification"),
        "ROLLBACK_RULES": (loc.workflow_md(), "Rollback Rules"),
        "GIT_INTEGRATION": (loc.workflow_md(), "Git Integration Specification"),
        "LOG_TEMPLATE": (loc.workflow_md(), "Optimization Log Template"),
        "ROUND_SUMMARY_TEMPLATE": (loc.workflow_md(), "Round Summary Template"),
    }
    if aliases:
        default_map.update(aliases)

    if phase not in default_map:
        raise KeyError(f"no skill excerpt mapping for phase '{phase}'")
    path, heading = default_map[phase]
    return load_section(path, heading)


def render_workspace_hygiene(workspace_root: Path, campaign_dir: Path) -> str:
    """Build the authoritative output-path-discipline block.

    Injected into every phase prompt that might create files (BASELINE,
    OPTIMIZE, VALIDATE, PREPARE_ENVIRONMENT). The block is the single
    source of truth for "stray files in ``workspace_root`` are a bug";
    when the rule changes, update it here and every phase picks up the
    new wording on the next run.
    """
    return (
        "<workspace_hygiene>\n"
        "Output-path discipline (non-negotiable):\n"
        f"- All files you CREATE must live under `{campaign_dir}/`. Never "
        f"write new files directly into `{workspace_root}/` (the repo "
        "top-level).\n"
        "- In-place edits to tracked source files under the project tree "
        "(e.g. `primus_turbo/`, `benchmarks/`, `tests/`) are allowed — "
        "those stay where they are so the framework actually uses your "
        "change. Do NOT create *new* files next to them.\n"
        "- Benchmark / test commands often dump auxiliary files (CSV, "
        "PNG, JSON, log) to the current working directory. After the "
        "command finishes, `mv` (NOT `cp`) every such artifact into the "
        "current round's `artifacts/` folder under the campaign "
        "directory. Using `cp` is a bug because the original copy would "
        "stay in the repo root.\n"
        "- When snapshotting a kernel file into "
        "`rounds/round-N/kernel_snapshot/`, always finish the destination "
        "path with a trailing `/` (e.g. "
        f"`cp src.py {campaign_dir}/rounds/round-N/kernel_snapshot/`) so "
        "`cp` treats it as a directory. Forgetting the slash creates a "
        "stray file named `kernel_snapshot` in the parent directory.\n"
        "- Before emitting the structured phase result, run "
        f"`ls {workspace_root}` and verify the output contains no new "
        "top-level files that you created during this phase. If any "
        "stray artifact is found, `mv` it under the campaign directory "
        "and note the move in the structured result's `notes` field.\n"
        "</workspace_hygiene>\n"
    )


@lru_cache(maxsize=32)
def load_prompt_template(name: str) -> str:
    root = Path(__file__).parent / "prompts"
    path = root / f"{name}.md"
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, variables: dict) -> str:
    template = load_prompt_template(name)
    try:
        return template.format(**variables)
    except KeyError as exc:
        raise KeyError(
            f"prompt template '{name}.md' needs variable {exc!s}; "
            f"provided keys: {sorted(variables.keys())}"
        ) from exc
