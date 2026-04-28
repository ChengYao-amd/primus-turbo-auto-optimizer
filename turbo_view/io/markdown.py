"""Markdown -> sanitized HTML for inline rendering in dashboard panels.

Pipeline: ``markdown-it-py`` (CommonMark + table) -> ``bleach.clean``
with an explicit allow-list. ``bleach`` is the trust boundary —
everything Claude writes into ``summary.md`` flows through here.
"""

from __future__ import annotations

from functools import lru_cache

import bleach
from markdown_it import MarkdownIt

_ALLOWED_TAGS = frozenset({
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "code", "pre", "blockquote", "br", "hr",
    "strong", "em", "del", "ins",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "a",
})

_ALLOWED_ATTRS = {
    "a": ["href", "title"],
    "code": ["class"],
    "pre": ["class"],
}

_ALLOWED_PROTOCOLS = frozenset({"http", "https", "mailto"})


@lru_cache(maxsize=1)
def _md() -> MarkdownIt:
    md = MarkdownIt(
        "commonmark",
        {"html": True, "linkify": False, "typographer": False},
    ).enable("table")
    md.validateLink = lambda url: True
    return md


def render_markdown(text: str | None) -> str:
    if not text:
        return ""
    raw = _md().render(text)
    return bleach.clean(
        raw,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )
