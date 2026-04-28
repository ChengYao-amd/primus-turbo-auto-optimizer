"""Render a ``CampaignBundle`` into a self-contained ``out_dir/``.

Output layout (per spec §2.3, PR-1 subset):

::

    <out_dir>/
      index.html        single-page entry, payload inlined as JSON
      data.json         same payload, separate file (watch mode reuses)
      assets/
        app.js          panel renderers (vanilla)
        app.css
        markdown.css
        chart.umd.min.js
        chartjs-plugin-annotation.min.js
        VENDOR_NOTES.md

The template + assets are package-data inside ``turbo_view.render``;
we never read from the source tree at runtime.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from turbo_view.model import CampaignBundle
from turbo_view.render.payload import SCHEMA_VERSION, bundle_to_payload

log = logging.getLogger(__name__)

_ASSET_FILENAMES = (
    "app.js",
    "app.css",
    "markdown.css",
    "chart.umd.min.js",
    "chartjs-plugin-annotation.min.js",
    "VENDOR_NOTES.md",
)


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("turbo_view.render", "templates"),
        autoescape=select_autoescape(["html"]),
    )


# overview.py and watch/server.py both need the same Jinja env + asset
# copy, so expose a public alias to avoid duplicating the loader path.
_template_env = _env


def _sanitize(value: Any) -> Any:
    """Replace NaN / ±Infinity with ``None`` recursively.

    Python's ``json.dumps`` emits the JS-unfriendly literals ``NaN`` /
    ``Infinity`` by default, which makes ``JSON.parse`` throw and
    silently kills the whole front-end bootstrap. Heatmap rows for
    rolled-back rounds are the typical source (mean over zero samples
    yields NaN). Sanitizing once at the boundary keeps the produced
    JSON RFC-8259 conformant.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    return value


def _payload_json_for_inline(payload: dict[str, Any]) -> str:
    """Return JSON safe to embed inside ``<script id="data">``.

    ``</script>`` inside string literals would close the host script
    tag prematurely; escape the slash to neutralise that.
    """
    raw = json.dumps(
        _sanitize(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return raw.replace("</", "<\\/")


def _copy_assets(dst_assets_dir: Path) -> None:
    """Copy vendored + first-party assets into ``dst_assets_dir``.

    Source: ``turbo_view/render/assets/`` (resolved relative to the
    installed package). Sourcing via package paths keeps editable
    installs and wheel installs equivalent.

    Callers pass the *full* destination directory (not its parent),
    so multi-page builds can share a single ``assets/`` at the top
    of the output tree.
    """
    assets_root = Path(__file__).resolve().parent / "assets"
    dst_assets_dir.mkdir(parents=True, exist_ok=True)
    for name in _ASSET_FILENAMES:
        src = assets_root / name
        if not src.is_file():
            log.warning("missing asset %s; skipping", src)
            continue
        shutil.copyfile(src, dst_assets_dir / name)


def render_detail(
    bundle: CampaignBundle,
    asset_prefix: str = "assets/",
    watch_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Pure function: ``CampaignBundle`` -> ``(index_html, payload)``.

    Returned separately so callers (CLI, future watch server) can
    decide whether to write to disk, stream over HTTP, or unit test.

    ``watch_mode=True`` emits ``<meta name="turbo-view-watch">`` so
    the front-end opens an EventSource and the live-tail panel shows
    itself.
    """
    payload = bundle_to_payload(bundle)
    title = _title_for(bundle)
    template = _env().get_template("detail.html")
    html = template.render(
        title=title,
        rendered_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
        payload_json=_payload_json_for_inline(payload),
        asset_prefix=asset_prefix,
        watch_mode=watch_mode,
    )
    return html, payload


def _title_for(bundle: CampaignBundle) -> str:
    if bundle.state is not None and bundle.state.campaign_id:
        return f"primus-turbo-view · {bundle.state.campaign_id}"
    return "primus-turbo-view"


def write_detail(
    bundle: CampaignBundle,
    out_dir: Path,
    copy_assets: bool = True,
    asset_prefix: str = "assets/",
    watch_mode: bool = False,
) -> Path:
    """Render ``bundle`` and write ``index.html`` + ``data.json`` + assets.

    ``copy_assets=False`` skips the per-page asset copy — useful for
    multi-campaign overview where a shared ``assets/`` lives one
    level up. ``asset_prefix`` rewrites every ``./assets/`` in the
    rendered HTML to point at the shared location. ``watch_mode``
    forwards through to :func:`render_detail`.

    Returns the path of ``index.html``.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html, payload = render_detail(
        bundle, asset_prefix=asset_prefix, watch_mode=watch_mode
    )
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    (out_dir / "data.json").write_text(
        json.dumps(
            _sanitize(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if copy_assets:
        _copy_assets(out_dir / "assets")
    return index_path
