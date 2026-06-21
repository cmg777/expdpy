"""Event studies, staggered difference-in-differences, and panel-structure visualization.

``analyze_event_study`` wraps pyfixest's modern DiD estimators (Gardner's two-stage
``did2s``, Sun-Abraham ``saturated``, local-projections ``lpdid`` and the classic two-way
fixed-effects ``twfe``) behind one beginner-friendly signature and returns a themed Plotly
event-study plot. ``analyze_panel_view`` reimplements pyfixest's ``panelview`` in Plotly to
show the treatment structure (which units are treated when) — and derives the binary
treatment indicator from a first-treatment ``cohort`` column for you.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfixest as pf

from expdpy._estimation import SSC
from expdpy._theme import COLOR_SEQUENCE, apply_default_layout, color_for
from expdpy._types import EventStudyResult, PanelViewResult
from expdpy._validation import ensure_dataframe

__all__ = ["analyze_event_study", "analyze_panel_view"]

_UNTREATED_COLOR = "#BAB0AC"  # Tableau gray
_TREATED_COLOR = COLOR_SEQUENCE[0]  # Tableau blue


def _ci_cols(frame: pd.DataFrame) -> tuple[str, str]:
    """Return the two confidence-interval column names (those ending in ``%``)."""
    cols = [c for c in frame.columns if str(c).endswith("%")]
    return cols[0], cols[1]


def _parse_path(tidy: pd.DataFrame) -> pd.DataFrame:
    """Parse a ``prefix::<event_time>`` tidy frame into a clean event-time path.

    Drops non-finite event times (the binned ``-inf`` never-treated reference row).
    """
    t = tidy.reset_index()
    t = t.rename(columns={t.columns[0]: "Coefficient"})
    lo, hi = _ci_cols(t)
    event_time = pd.to_numeric(
        t["Coefficient"].astype(str).str.split("::").str[-1], errors="coerce"
    )
    t = t.assign(event_time=event_time)
    t = t[np.isfinite(t["event_time"])]
    return pd.DataFrame(
        {
            "event_time": t["event_time"].astype(float),
            "estimate": t["Estimate"].astype(float),
            "se": t["Std. Error"].astype(float),
            "ci_lower": t[lo].astype(float),
            "ci_upper": t[hi].astype(float),
            "cohort": None,
        }
    )


def _add_reference(path: pd.DataFrame) -> pd.DataFrame:
    """Insert the t = -1 reference period (estimate 0) if it is missing, then sort."""
    if -1.0 not in set(path["event_time"]):
        ref = pd.DataFrame(
            [
                {
                    "event_time": -1.0,
                    "estimate": 0.0,
                    "se": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "cohort": None,
                }
            ]
        )
        path = pd.concat([path, ref], ignore_index=True)
    return path.sort_values("event_time").reset_index(drop=True)


def analyze_event_study(
    df: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    cohort: str,
    estimator: Literal["did2s", "twfe", "saturated", "lpdid"] = "did2s",
    cluster: str | None = None,
    pre_window: int | None = None,
    post_window: int | None = None,
    never_treated_value: int = 0,
    title: str | None = None,
) -> EventStudyResult:
    """Estimate and plot an event study for staggered treatment adoption.

    Parameters
    ----------
    df
        Long panel data frame.
    outcome
        Outcome variable name.
    unit
        Unit (cross-section) identifier.
    time
        Time identifier.
    cohort
        First-treated period for each unit; ``never_treated_value`` marks never-treated units.
    estimator
        ``"did2s"`` (Gardner two-stage, the default and robust to heterogeneity), ``"twfe"``
        (classic two-way fixed effects — shown for comparison, biased under heterogeneous
        effects), ``"saturated"`` (Sun-Abraham, one curve per cohort) or ``"lpdid"``
        (local-projections DiD).
    cluster
        Cluster variable for standard errors (defaults to ``unit``).
    pre_window, post_window
        Event-time window for ``"lpdid"`` (ignored by the other estimators).
    never_treated_value
        The value of ``cohort`` that marks never-treated units (default ``0``).
    title
        Optional figure title.

    Returns
    -------
    EventStudyResult
        ``df`` (event-time path), ``fig`` (Plotly), ``model`` (fitted pyfixest object) and
        ``estimator``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_staggered_did

    df = load_staggered_did()
    ex.analyze_event_study(
        df, outcome="outcome", unit="unit", time="year", cohort="cohort"
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    for col in (outcome, unit, time, cohort):
        if col not in df.columns:
            raise KeyError(f"column not found in df: {col!r}")

    cl = cluster or unit
    work = df[[outcome, unit, time, cohort]].dropna().copy()
    never = work[cohort] == never_treated_value
    work["_rel"] = np.where(never, np.inf, work[time] - work[cohort])
    work["_treated"] = ((~never) & (work[time] >= work[cohort])).astype(int)

    if estimator == "did2s":
        model = pf.did2s(
            work,
            yname=outcome,
            first_stage=f"~0 | {unit} + {time}",
            second_stage="~i(_rel, ref=-1.0)",
            treatment="_treated",
            cluster=cl,
        )
        path = _add_reference(_parse_path(model.tidy()))
    elif estimator == "twfe":
        model = pf.feols(
            f"{outcome} ~ i(_rel, ref=-1.0) | {unit} + {time}",
            data=work,
            vcov={"CRV1": cl},
            ssc=SSC,
        )
        path = _add_reference(_parse_path(model.tidy()))
    elif estimator == "lpdid":
        model = pf.lpdid(
            work,
            yname=outcome,
            idname=unit,
            tname=time,
            gname=cohort,
            pre_window=pre_window,
            post_window=post_window,
            never_treated=never_treated_value,
            att=False,
        )
        path = _add_reference(_parse_path(model.tidy()))
    elif estimator == "saturated":
        model = pf.event_study(
            work,
            yname=outcome,
            idname=unit,
            tname=time,
            gname=cohort,
            estimator="saturated",
            cluster=cl,
        )
        path = _saturated_path(model)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown estimator {estimator!r}")

    fig = _event_study_fig(path, estimator, title)
    return EventStudyResult(df=path, fig=fig, model=model, estimator=estimator)


def _saturated_path(model: Any) -> pd.DataFrame:
    """Build the per-cohort event-time path from a Sun-Abraham saturated model."""
    cohort_dict = getattr(model, "_res_cohort_eventtime_dict", None)
    if not cohort_dict:  # pragma: no cover - defensive
        raise RuntimeError("saturated event study returned no per-cohort estimates")
    frames = []
    for coh, payload in cohort_dict.items():
        est = payload["est"]
        lo, hi = _ci_cols(est)
        frames.append(
            pd.DataFrame(
                {
                    "event_time": est["time"].astype(float),
                    "estimate": est["Estimate"].astype(float),
                    "se": est["Std. Error"].astype(float),
                    "ci_lower": est[lo].astype(float),
                    "ci_upper": est[hi].astype(float),
                    "cohort": str(coh),
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["cohort", "event_time"])
        .reset_index(drop=True)
    )


def _event_study_fig(
    path: pd.DataFrame, estimator: str, title: str | None
) -> go.Figure:
    """Assemble the themed event-study Plotly figure (CI whiskers, zero + t=-1 guides)."""
    has_cohorts = path["cohort"].notna().any()
    groups = list(path["cohort"].dropna().unique()) if has_cohorts else [None]

    fig = go.Figure()
    for j, coh in enumerate(groups):
        sub = (path if coh is None else path[path["cohort"] == coh]).sort_values(
            "event_time"
        )
        err = {
            "type": "data",
            "symmetric": False,
            "array": (sub["ci_upper"] - sub["estimate"]).to_numpy(),
            "arrayminus": (sub["estimate"] - sub["ci_lower"]).to_numpy(),
            "thickness": 1.2,
            "color": color_for(j),
        }
        fig.add_trace(
            go.Scatter(
                x=sub["event_time"],
                y=sub["estimate"],
                mode="lines+markers",
                error_y=err,
                line={"color": color_for(j)},
                marker={"color": color_for(j), "size": 7},
                name=(f"cohort {coh}" if coh is not None else "estimate"),
                showlegend=coh is not None,
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.4)")
    fig.add_vline(x=-1, line_dash="dash", line_color="rgba(0,0,0,0.35)")
    apply_default_layout(
        fig,
        xaxis={"title": "Event time (periods relative to treatment)"},
        yaxis={"title": "Estimated effect on the outcome"},
    )
    fig.update_layout(title=title or f"Event study ({estimator})")
    return fig


def analyze_panel_view(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    treat: str | None = None,
    cohort: str | None = None,
    outcome: str | None = None,
    never_treated_value: int = 0,
    sort_by_timing: bool = True,
    max_units: int | None = 200,
    title: str | None = None,
) -> PanelViewResult:
    """Visualize the treatment structure of a panel (a themed ``panelview``).

    Provide either a binary ``treat`` column or a first-treatment ``cohort`` column (from
    which the indicator is derived). With ``outcome`` given, plots each unit's outcome over
    time instead of the treatment quilt.

    Parameters
    ----------
    df
        Long panel data frame.
    unit, time
        Unit and time identifiers.
    treat
        Binary (0/1) treatment-status column.
    cohort
        First-treated period column (used to derive ``treat`` when ``treat`` is omitted).
    outcome
        If given, draw an outcome-over-time line per unit instead of the treatment quilt.
    never_treated_value
        Value of ``cohort`` marking never-treated units (default ``0``).
    sort_by_timing
        Order units by their first treated period (clearest staggered-adoption picture).
    max_units
        Cap the number of units shown (an even spread is sampled when there are more).
    title
        Optional figure title.

    Returns
    -------
    PanelViewResult
        ``df`` (the treatment quilt, or the tidy outcome frame) and ``fig``.
    """
    df = ensure_dataframe(df)
    for col in (unit, time):
        if col not in df.columns:
            raise KeyError(f"column not found in df: {col!r}")
    if treat is None and cohort is None:
        raise ValueError(
            "provide either 'treat' (binary) or 'cohort' (first-treated period)"
        )

    work = df.copy()
    if treat is None:
        never = work[cohort] == never_treated_value
        work["_treat"] = ((~never) & (work[time] >= work[cohort])).astype(int)
        treat_col = "_treat"
    else:
        treat_col = treat
        uniq = set(pd.unique(work[treat_col].dropna()))
        if not uniq <= {0, 1}:
            raise ValueError(
                f"'treat' must be binary (0/1); found values {sorted(map(str, uniq))}"
            )

    if outcome is not None:
        sub = work[[unit, time, outcome]].dropna()
        units = list(pd.unique(sub[unit]))
        if max_units is not None and len(units) > max_units:
            keep = {
                units[i] for i in np.linspace(0, len(units) - 1, max_units).astype(int)
            }
            sub = sub[sub[unit].isin(keep)]
        fig = go.Figure()
        for _u, g in sub.groupby(unit, sort=False):
            gg = g.sort_values(time)
            fig.add_trace(
                go.Scatter(
                    x=gg[time],
                    y=gg[outcome],
                    mode="lines",
                    line={"color": "rgba(78,121,167,0.25)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        apply_default_layout(fig, xaxis={"title": time}, yaxis={"title": outcome})
        fig.update_layout(title=title or f"{outcome} by {unit} over {time}")
        return PanelViewResult(df=sub.reset_index(drop=True), fig=fig)

    quilt = work.pivot_table(
        index=unit, columns=time, values=treat_col, aggfunc="first"
    )
    if sort_by_timing:
        cols = list(quilt.columns)

        def first_treated(row: pd.Series) -> float:
            hits = [c for c in cols if row[c] == 1]
            return float(min(hits)) if hits else float("inf")

        order = quilt.apply(first_treated, axis=1).sort_values().index
        quilt = quilt.loc[order]
    if max_units is not None and len(quilt) > max_units:
        idx = np.linspace(0, len(quilt) - 1, max_units).astype(int)
        quilt = quilt.iloc[idx]

    fig = go.Figure(
        go.Heatmap(
            z=quilt.to_numpy(),
            x=[str(c) for c in quilt.columns],
            y=[str(i) for i in quilt.index],
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, _UNTREATED_COLOR],
                [0.5, _UNTREATED_COLOR],
                [0.5, _TREATED_COLOR],
                [1.0, _TREATED_COLOR],
            ],
            colorbar={
                "tickvals": [0.25, 0.75],
                "ticktext": ["untreated", "treated"],
                "title": "",
            },
            xgap=1,
            ygap=1,
            hovertemplate=f"{unit}=%{{y}}<br>{time}=%{{x}}<br>treated=%{{z}}<extra></extra>",
        )
    )
    apply_default_layout(fig, xaxis={"title": time}, yaxis={"title": unit})
    fig.update_layout(title=title or "Treatment structure")
    return PanelViewResult(df=quilt, fig=fig)
