"""``turbo_view.io.markdown.render_markdown`` produces XSS-safe HTML.

We need to render Claude-generated ``summary.md`` / ``optimize.md``
snippets inline in the dashboard. Anything off the ``allowed_tags``
list (script, style, iframe, on*-attributes, javascript: URLs) must
be stripped or escaped.
"""

from __future__ import annotations

from turbo_view.io.markdown import render_markdown


def test_render_markdown_emits_paragraph():
    html = render_markdown("hello **world**")
    assert "<p>" in html
    assert "<strong>world</strong>" in html


def test_render_markdown_keeps_headings_lists_code_tables():
    md = (
        "# Title\n\n"
        "## Sub\n\n"
        "- one\n- two\n\n"
        "```python\nprint('x')\n```\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    html = render_markdown(md)
    assert "<h1>Title" in html
    assert "<h2>Sub" in html
    assert "<ul>" in html and "<li>one</li>" in html
    assert "<pre><code" in html and "print" in html
    assert "<table>" in html and "<td>1</td>" in html


def test_render_markdown_strips_script_tag():
    html = render_markdown("hi <script>alert(1)</script>")
    assert "<script" not in html
    assert "alert" in html


def test_render_markdown_strips_event_handlers_and_js_urls():
    html = render_markdown('[click](javascript:alert(1) "x")\n\n<a href="https://ok" onclick="x">y</a>')
    assert "javascript:" not in html
    assert "onclick" not in html


def test_render_markdown_returns_empty_string_for_none_or_empty():
    assert render_markdown(None) == ""
    assert render_markdown("") == ""
