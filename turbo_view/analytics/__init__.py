"""Analytics layer: dataclass bundle -> chart-ready dict.

Each module here owns one panel's data shape and is independent of
the front-end framework — the dicts are designed so a tiny vanilla
JS function can hand them straight to Chart.js (or render a CSS-grid
heatmap) without further transformation.
"""
