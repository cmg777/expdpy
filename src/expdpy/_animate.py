"""Shared animation vocabulary: play/pause controls, a time slider, and range pinning.

Every animated expdpy figure — the hand-built ones (the Gapminder bubble scatter, the animated
histogram/violin, the animated treemap/sunburst) and the ``plotly.express`` ones (the animated
box/strip plots) — draws its controls from here, so they all get the same "▶ Play" / "⏸ Pause"
buttons, the same frame speed, and the same time-labelled slider.

The hand-built figures assemble their own ``go.Frame`` list and call :func:`play_controls`
+ :func:`time_slider`; the ``plotly.express`` figures let px build the frames and then call
:func:`retheme_px_animation`, which swaps px's terse auto-controls for the expdpy ones.

This module imports only :mod:`numpy` / :mod:`pandas` / :mod:`plotly` plus two intra-package
leaves (``_common``, ``_theme``), so it stays free of import cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._common import sorted_levels
from expdpy._theme import apply_default_layout

__all__ = [
    "FRAME_MS",
    "TRANS_MS",
    "fmt_period",
    "padded_range",
    "global_range",
    "global_color_range",
    "category_orders",
    "play_controls",
    "time_slider",
    "retheme_px_animation",
]

FRAME_MS = 600
TRANS_MS = 300


def fmt_period(value: object) -> str:
    """Format a period label, rendering integral floats (years) without a decimal point."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def padded_range(values: np.ndarray) -> list[float]:
    """Return a min/max range with 5% padding (fixed across frames so the view holds still)."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (abs(hi) * 0.05 or 1.0) if hi == lo else (hi - lo) * 0.05
    return [lo - pad, hi + pad]


def global_range(series: pd.Series, *, log: bool = False) -> list[float]:
    """Return a padded numeric range over the whole (complete-case) column.

    Feeding this to px ``range_x`` / ``range_y`` pins the axis so the view does not jump between
    animation frames. When ``log=True`` the range is returned in **log10 units** (what Plotly's
    log axis expects) and non-positive values are dropped first.
    """
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if log:
        vals = vals[vals > 0]
    if vals.size == 0:
        return [0.0, 1.0]
    if log:
        return padded_range(np.log10(vals))
    return padded_range(vals)


def global_color_range(series: pd.Series) -> list[float]:
    """``[min, max]`` for a continuous color encoding, so ``range_color`` holds across frames."""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [0.0, 1.0]
    return [float(np.nanmin(vals)), float(np.nanmax(vals))]


def category_orders(*specs: tuple[str, pd.Series]) -> dict[str, list[str]]:
    """Build ``{col: sorted_levels(series)}`` for each ``(col, series)`` spec.

    Numeric-aware (via :func:`expdpy._common.sorted_levels`), so group/color categories keep a
    stable order across animation frames and on the axis even when a level is absent in some
    periods. The series should already be string-typed (px matches ``category_orders`` values
    against the column's values).
    """
    return {col: sorted_levels(series) for col, series in specs}


def play_controls(*, frame_ms: int = FRAME_MS, trans_ms: int = TRANS_MS) -> list[dict]:
    """Return the shared single ▶ play button, tucked in the bottom-left, left of the slider.

    Plotly ``updatemenus`` buttons are stateless — a genuine one-button play↔pause toggle needs
    JavaScript that static / notebook / Streamlit-rendered figures don't have. So this is a
    play-only control placed unobtrusively at the bottom-left (matching Plotly Express's default
    geometry, with the time slider to its right); to pause, grab the slider.
    """
    play_args = {
        "frame": {"duration": frame_ms, "redraw": True},
        "fromcurrent": True,
        "transition": {"duration": trans_ms},
    }
    return [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top",
            "pad": {"r": 10, "t": 70},
            "buttons": [
                {"label": "▶", "method": "animate", "args": [None, play_args]},
            ],
        }
    ]


def time_slider(
    period_labels: list[str],
    *,
    time_label: str,
    frame_ms: int = FRAME_MS,
    trans_ms: int = TRANS_MS,
    active: int = 0,
) -> list[dict]:
    """Return the shared time slider — one step per period, prefixed with the time label.

    Each ``period_labels`` entry must equal the ``name`` of the corresponding ``go.Frame`` so the
    step animates to it.
    """
    return [
        {
            "active": active,
            "x": 0.1,
            "y": 0,
            "xanchor": "left",
            "yanchor": "top",
            "len": 0.9,
            "currentvalue": {"prefix": f"{time_label}: "},
            "pad": {"b": 10, "t": 50},
            "steps": [
                {
                    "method": "animate",
                    "label": lbl,
                    "args": [
                        [lbl],
                        {
                            "frame": {"duration": frame_ms, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": trans_ms},
                        },
                    ],
                }
                for lbl in period_labels
            ],
        }
    ]


def retheme_px_animation(
    fig: go.Figure,
    *,
    time_label: str,
    dark: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    frame_ms: int = FRAME_MS,
    trans_ms: int = TRANS_MS,
) -> go.Figure:
    """Re-theme a ``plotly.express`` animated figure to expdpy standards, in place.

    Applies :func:`expdpy._theme.apply_default_layout`, then — when the figure is animated —
    swaps px's terse auto-controls (a ``▶`` / ``■`` menu and a ``"col="`` slider) for the shared
    single ▶ :func:`play_controls` button and the time-labelled :func:`time_slider`. The
    replacement reuses px's own frame names (read off its slider steps) so the steps stay wired to
    the right frames regardless of how px stringified the period values.

    A **no-op on the controls when the figure is static** (``fig.frames`` empty), which is the
    graceful fallback path when no time id is available.
    """
    apply_default_layout(fig, dark=dark, title=title, subtitle=subtitle)
    if not fig.frames:
        return fig
    if fig.layout.sliders:
        labels = [str(step.label) for step in fig.layout.sliders[0].steps]
    else:  # pragma: no cover - px always builds a slider for an animation
        labels = [str(frame.name) for frame in fig.frames]
    # Direct assignment REPLACES px's auto controls; update_layout would merge element-wise
    # (leaving px's second ■ button behind), so assign the tuples outright.
    fig.layout.updatemenus = play_controls(frame_ms=frame_ms, trans_ms=trans_ms)
    fig.layout.sliders = time_slider(
        labels, time_label=time_label, frame_ms=frame_ms, trans_ms=trans_ms
    )
    return fig
