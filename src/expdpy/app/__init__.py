"""The ExPdPy interactive app (Shiny for Python).

``ExPdPy`` builds a config-driven, no-code exploration UI on top of the library's
``prepare_*`` functions: a reactive sample pipeline (subset / outlier treatment) feeds an
ordered set of analysis components (descriptive table, histogram, correlations, trends,
scatter, regression, ...), each rendered with Plotly or Great Tables. It also supports
in-app data upload, save/load of the analysis configuration, and export of a reproducible
notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from expdpy.app import _components as comp
from expdpy.app._components import COMPONENT_KIND, COMPONENT_ORDER, TS_COMPONENTS
from expdpy.app._config_io import dump_config, load_config
from expdpy.app._export_nb import build_export_zip
from expdpy.app._sample import apply_user_vars, build_analysis_sample
from expdpy.app._state import parse_config
from expdpy.app._varcat import create_var_categories

if TYPE_CHECKING:
    from shiny import App

__all__ = ["ExPdPy"]

_OUTLIER_CHOICES = {
    "1": "None",
    "2": "Winsorize 1%",
    "3": "Winsorize 5%",
    "4": "Truncate 1%",
    "5": "Truncate 5%",
}


def _normalize_samples(
    df: Any, df_name: str | Sequence[str] | None
) -> dict[str, pd.DataFrame]:
    if df is None:
        return {}
    if isinstance(df, pd.DataFrame):
        name = df_name if isinstance(df_name, str) else "Sample"
        return {name: df}
    if isinstance(df, Mapping):
        return dict(df)
    return {f"Sample {i + 1}": d for i, d in enumerate(df)}


def _resolve_ids(
    df_def: pd.DataFrame | None, cs_id: Sequence[str] | str | None, ts_id: str | None
) -> tuple[list[str], str | None]:
    if df_def is not None:
        cs = list(df_def.loc[df_def["type"] == "cs_id", "var_name"])
        ts_rows = list(df_def.loc[df_def["type"] == "ts_id", "var_name"])
        return cs, (ts_rows[0] if ts_rows else None)
    if isinstance(cs_id, str):
        cs_id = [cs_id]
    return (list(cs_id) if cs_id else []), ts_id


def _active_components(components: Any, ts_id: str | None) -> list[str]:
    if isinstance(components, Mapping):
        selected = [c for c in COMPONENT_ORDER if components.get(c)]
    elif isinstance(components, (list, tuple)):
        selected = [c for c in components if c in COMPONENT_ORDER]
    else:
        selected = list(COMPONENT_ORDER)
    renderable = [c for c in selected if c in COMPONENT_KIND]
    if not ts_id:
        renderable = [c for c in renderable if c not in TS_COMPONENTS]
    return renderable


def _g(inp: Any, key: str, default: Any = None) -> Any:
    """Read an input value safely, returning ``default`` if absent/unset."""
    try:
        val = inp[key]()
    except Exception:
        return default
    return default if val in (None, "") else val


def ExPdPy(
    df: Any = None,
    cs_id: Sequence[str] | str | None = None,
    ts_id: str | None = None,
    df_def: pd.DataFrame | None = None,
    var_def: pd.DataFrame | None = None,
    config_list: dict | None = None,
    *,
    title: str = "ExPdPy - Explore your data!",
    df_name: str | Sequence[str] | None = None,
    components: Any = None,
    factor_cutoff: int = 10,
    export_nb_option: bool = True,
    save_settings_option: bool = True,
    store_encrypted: bool = False,
    key_phrase: str = "What a wonderful key",
    run: bool = True,
    **run_kwargs: Any,
) -> App:
    """Launch (or build) the interactive ExPdPy app.

    Parameters
    ----------
    df
        A :class:`pandas.DataFrame`, a mapping of name->DataFrame, a list of DataFrames, or
        ``None`` to start with an upload dialog.
    cs_id, ts_id
        Cross-sectional / time-series identifier column name(s). Overridden by ``df_def``.
    df_def
        Optional variable-definition frame (columns ``var_name``, ``var_def``, ``type``)
        used to identify the panel dimensions.
    var_def
        Optional analysis-sample variable definitions (advanced mode). Each ``var_def`` is a
        safe expression evaluated to build the analysis sample.
    config_list
        Optional startup configuration (see :func:`expdpy.data.get_config`).
    title
        App title.
    df_name
        Display name(s) for the provided sample(s).
    components
        Ordered list (or ``{name: bool}`` mapping) selecting which components to show.
    factor_cutoff
        Numeric columns with at most this many unique values are treated as factors.
    export_nb_option, save_settings_option
        Enable the notebook-export and config save/load controls.
    store_encrypted, key_phrase
        Encrypt saved configurations with a Fernet key derived from ``key_phrase``.
    run
        If ``True`` (default), start the app server; otherwise return the :class:`shiny.App`.
    **run_kwargs
        Forwarded to :meth:`shiny.App.run` (e.g. ``port``, ``host``, ``launch_browser``).

    Returns
    -------
    shiny.App
        The constructed app (also returned when ``run=True`` after the server stops).
    """
    from shiny import App, reactive, render, req, ui
    from shinywidgets import render_plotly

    samples = _normalize_samples(df, df_name)
    cs_list, ts = _resolve_ids(df_def, cs_id, ts_id)
    active = _active_components(components, ts)
    base_cfg = parse_config(config_list)

    # ------------------------------------------------------------------ UI ---
    sidebar_items: list[Any] = [ui.h4("ExPdPy")]
    if len(samples) > 1:
        sidebar_items.append(
            ui.input_select(
                "sample", "Sample", choices=list(samples), selected=next(iter(samples))
            )
        )
    sidebar_items.append(
        ui.input_file(
            "upload",
            "Upload data (CSV/Excel/parquet)",
            accept=[".csv", ".xlsx", ".xls", ".parquet"],
        )
    )
    sidebar_items.append(ui.output_ui("sidebar_controls"))
    if save_settings_option:
        sidebar_items += [
            ui.hr(),
            ui.download_button("download_config", "Save config"),
            ui.input_file("upload_config", "Load config", accept=[".json", ".cfg"]),
        ]
    if export_nb_option:
        sidebar_items += [
            ui.hr(),
            ui.download_button("download_nb", "Export notebook + data"),
        ]

    app_ui = ui.page_sidebar(
        ui.sidebar(*sidebar_items, width=320),
        ui.output_ui("main_ui"),
        title=title,
    )

    # -------------------------------------------------------------- server ---
    def server(inp: Any, output: Any, session: Any) -> None:
        cfg_state = reactive.value(dict(base_cfg))
        uploaded = reactive.value(None)

        @reactive.calc
        def base_df() -> pd.DataFrame | None:
            up = uploaded()
            if up is not None:
                return up
            if not samples:
                return None
            if len(samples) > 1:
                return samples.get(
                    _g(inp, "sample", next(iter(samples))), next(iter(samples.values()))
                )
            return next(iter(samples.values()))

        @reactive.calc
        def analysis_sample() -> pd.DataFrame | None:
            data = base_df()
            if data is None:
                return None
            if var_def is not None:
                data = apply_user_vars(data, var_def, cs_list, ts)
            cfg = dict(cfg_state())
            cfg["subset_factor"] = _g(inp, "subset_factor", "Full Sample")
            cfg["subset_value"] = _g(inp, "subset_value", "All")
            cfg["outlier_treatment"] = _g(inp, "outlier_treatment", "1")
            return build_analysis_sample(data, cs_list, ts, cfg)

        @reactive.calc
        def var_cats():
            sample = analysis_sample()
            if sample is None:
                return create_var_categories(pd.DataFrame())
            return create_var_categories(
                sample, cs_list, ts, factor_cutoff=factor_cutoff
            )

        # --- in-app upload ---------------------------------------------------
        @reactive.effect
        @reactive.event(lambda: inp.upload())
        def _on_upload():
            from expdpy.app._upload import read_uploaded

            files = inp.upload()
            if files:
                f = files[0]
                uploaded.set(read_uploaded(f["datapath"], f["name"]))

        # --- config load -----------------------------------------------------
        @reactive.effect
        @reactive.event(lambda: inp.upload_config())
        def _on_config():
            files = inp.upload_config()
            if files:
                with open(files[0]["datapath"], "rb") as fh:
                    raw = fh.read()
                cfg = load_config(raw, key_phrase if store_encrypted else None)
                cfg_state.set(parse_config(cfg))

        # --- sidebar controls (subset + outlier) -----------------------------
        @render.ui
        def sidebar_controls():
            vc = var_cats()
            cfg = cfg_state()
            factors = ["Full Sample", *vc.grouping]
            controls = [
                ui.input_select(
                    "subset_factor",
                    "Subset by",
                    choices=factors,
                    selected=cfg.get("subset_factor", "Full Sample"),
                ),
                ui.output_ui("subset_value_ui"),
                ui.input_select(
                    "outlier_treatment",
                    "Outlier treatment",
                    choices=_OUTLIER_CHOICES,
                    selected=str(cfg.get("outlier_treatment", "1")),
                ),
            ]
            return ui.TagList(*controls)

        @render.ui
        def subset_value_ui():
            sample = analysis_sample()
            sf = _g(inp, "subset_factor", "Full Sample")
            if (
                sample is None
                or sf in (None, "Full Sample")
                or sf not in sample.columns
            ):
                return ui.TagList()
            levels = [
                "All",
                *[str(v) for v in sorted(sample[sf].dropna().unique(), key=str)],
            ]
            return ui.input_select(
                "subset_value", "Value", choices=levels, selected="All"
            )

        @render.ui
        def fwl_focal_ui():
            # Focal-variable choices track the live regression regressors (reg_x); the
            # other regressors become the FWL controls. Resets if the focal is removed.
            xs = _g(inp, "reg_x", []) or []
            xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
            xs = [x for x in xs if x not in (None, "", "None")]
            if not xs:
                return ui.help_text(
                    "Select one or more independent variables in the regression card above."
                )
            sel = cfg_state().get("fwl_focal")
            if sel not in xs:
                sel = xs[0]
            return ui.input_select(
                "fwl_focal", "Focal variable", choices=xs, selected=sel
            )

        # --- main component area --------------------------------------------
        @render.ui
        def main_ui():
            if base_df() is None:
                return ui.div(
                    ui.h3("Welcome to ExPdPy"),
                    ui.p(
                        "Upload a data file (sidebar) with at least two numeric variables to begin."
                    ),
                )
            vc = var_cats()
            cfg = cfg_state()
            cards = [_component_card(name, vc, cfg, ts) for name in active]
            return ui.TagList(*[c for c in cards if c is not None])

        # --- component renderers (defined for every component; render only
        #     when their output placeholder is present in the DOM) -----------
        @render.ui
        def t_descriptive_table():
            return ui.HTML(comp.descriptive(analysis_sample()) or "")

        @render.ui
        def t_ext_obs():
            return ui.HTML(
                comp.ext_obs(analysis_sample(), _g(inp, "ext_obs_var")) or ""
            )

        @render.ui
        def t_regression():
            xs = _g(inp, "reg_x", []) or []
            xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
            html = comp.regression(
                analysis_sample(),
                _g(inp, "reg_y"),
                xs,
                [_g(inp, "reg_fe1"), _g(inp, "reg_fe2")],
                _cluster_vars(
                    _g(inp, "cluster", 1), _g(inp, "reg_fe1"), _g(inp, "reg_fe2")
                ),
            )
            return ui.HTML(html or "")

        @render_plotly
        def w_corrplot():
            fig = comp.corrplot(analysis_sample())
            req(fig is not None)
            return fig

        @render_plotly
        def w_histogram():
            fig = comp.histogram(
                analysis_sample(),
                _g(inp, "hist_var"),
                int(_g(inp, "hist_nr_of_breaks", 20)),
            )
            req(fig is not None)
            return fig

        @render_plotly
        def w_bar_chart():
            fig = comp.bar_chart(analysis_sample(), _g(inp, "bar_chart_var1"))
            req(fig is not None)
            return fig

        @render_plotly
        def w_missing_values():
            fig = comp.missing(analysis_sample(), ts)
            req(fig is not None)
            return fig

        @render_plotly
        def w_scatter_plot():
            fig = comp.scatter(
                analysis_sample(),
                _g(inp, "scatter_x"),
                _g(inp, "scatter_y"),
                _g(inp, "scatter_color"),
                _g(inp, "scatter_size"),
                bool(_g(inp, "scatter_loess", True)),
            )
            req(fig is not None)
            return fig

        @render_plotly
        def w_fwl_plot():
            xs = _g(inp, "reg_x", []) or []
            xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
            fig = comp.fwl_plot(
                analysis_sample(),
                _g(inp, "reg_y"),
                xs,
                _g(inp, "fwl_focal"),
                [_g(inp, "reg_fe1"), _g(inp, "reg_fe2")],
                _cluster_vars(
                    _g(inp, "cluster", 1), _g(inp, "reg_fe1"), _g(inp, "reg_fe2")
                ),
            )
            req(fig is not None)
            return fig

        @render_plotly
        def w_trend_graph():
            variables = [_g(inp, f"trend_graph_var{i}") for i in (1, 2, 3)]
            fig = comp.trend(analysis_sample(), ts, variables)
            req(fig is not None)
            return fig

        @render_plotly
        def w_quantile_trend_graph():
            fig = comp.quantile_trend(
                analysis_sample(), ts, _g(inp, "quantile_trend_graph_var")
            )
            req(fig is not None)
            return fig

        @render_plotly
        def w_by_group_bar_graph():
            fig = comp.by_group_bar(
                analysis_sample(), _g(inp, "bgbg_byvar"), _g(inp, "bgbg_var")
            )
            req(fig is not None)
            return fig

        @render_plotly
        def w_by_group_violin_graph():
            fig = comp.by_group_violin(
                analysis_sample(), _g(inp, "bgvg_byvar"), _g(inp, "bgvg_var")
            )
            req(fig is not None)
            return fig

        @render_plotly
        def w_by_group_trend_graph():
            fig = comp.by_group_trend(
                analysis_sample(), ts, _g(inp, "bgtg_byvar"), _g(inp, "bgtg_var")
            )
            req(fig is not None)
            return fig

        # --- downloads -------------------------------------------------------
        def _current_config() -> dict:
            cfg = dict(cfg_state())
            for key in _CONFIG_INPUT_KEYS:
                val = _g(inp, key, None)
                if val is not None:
                    cfg[key] = list(val) if isinstance(val, tuple) else val
            return cfg

        @render.download(filename="expdpy_config.json")
        def download_config():
            yield dump_config(
                _current_config(), key_phrase if store_encrypted else None
            )

        @render.download(filename="ExPdPy_analysis.zip")
        def download_nb():
            sample = analysis_sample()
            if sample is not None:
                yield build_export_zip(_current_config(), active, sample, ts)

    app = App(app_ui, server)
    if run:
        app.run(**run_kwargs)
    return app


# Input ids that participate in saved configurations.
_CONFIG_INPUT_KEYS = [
    "subset_factor",
    "subset_value",
    "outlier_treatment",
    "hist_var",
    "hist_nr_of_breaks",
    "ext_obs_var",
    "bar_chart_var1",
    "scatter_x",
    "scatter_y",
    "scatter_color",
    "scatter_size",
    "scatter_loess",
    "trend_graph_var1",
    "trend_graph_var2",
    "trend_graph_var3",
    "quantile_trend_graph_var",
    "bgbg_byvar",
    "bgbg_var",
    "bgvg_byvar",
    "bgvg_var",
    "bgtg_byvar",
    "bgtg_var",
    "reg_y",
    "reg_x",
    "reg_fe1",
    "reg_fe2",
    "cluster",
    "fwl_focal",
]


def _cluster_vars(choice: Any, fe1: str | None, fe2: str | None) -> list[str]:
    """Translate the cluster radio (1-4) into a list of cluster variables."""
    fes = [f for f in (fe1, fe2) if f and f != "None"]
    try:
        choice = int(choice)
    except (TypeError, ValueError):
        choice = 1
    if choice <= 1:
        return []
    return fes[: choice - 1]


def _sel(
    id_: str, label: str, choices: list[str], cfg: dict, *, none: bool = False
) -> Any:
    from shiny import ui

    opts = (["None", *choices] if none else choices) or ["None"]
    selected = cfg.get(id_)
    if selected not in opts:
        selected = opts[0]
    return ui.input_select(id_, label, choices=opts, selected=selected)


def _component_card(name: str, vc, cfg: dict, ts: str | None) -> Any:
    """Build the controls + output card for a single component."""
    from shiny import ui
    from shinywidgets import output_widget

    numeric = vc.numeric_logical or ["None"]
    factors = vc.grouping or ["None"]
    fe_choices = vc.fe_choices or ["None"]
    kind = COMPONENT_KIND[name]
    out = output_widget(f"w_{name}") if kind == "plotly" else ui.output_ui(f"t_{name}")

    controls: list[Any] = []
    if name == "histogram":
        controls = [
            _sel("hist_var", "Variable", numeric, cfg),
            ui.input_slider(
                "hist_nr_of_breaks",
                "Bins",
                5,
                100,
                int(cfg.get("hist_nr_of_breaks", 20)),
            ),
        ]
    elif name == "ext_obs":
        controls = [_sel("ext_obs_var", "Variable", numeric, cfg)]
    elif name == "bar_chart":
        controls = [_sel("bar_chart_var1", "Variable", factors, cfg)]
    elif name == "scatter_plot":
        controls = [
            _sel("scatter_x", "X", numeric, cfg),
            _sel("scatter_y", "Y", numeric, cfg),
            _sel("scatter_color", "Color", factors + numeric, cfg, none=True),
            _sel("scatter_size", "Size", numeric, cfg, none=True),
            ui.input_checkbox(
                "scatter_loess", "LOESS", bool(cfg.get("scatter_loess", True))
            ),
        ]
    elif name == "trend_graph":
        controls = [
            _sel(f"trend_graph_var{i}", f"Variable {i}", numeric, cfg, none=(i > 1))
            for i in (1, 2, 3)
        ]
    elif name == "quantile_trend_graph":
        controls = [_sel("quantile_trend_graph_var", "Variable", numeric, cfg)]
    elif name == "by_group_bar_graph":
        controls = [
            _sel("bgbg_byvar", "Group by", factors, cfg),
            _sel("bgbg_var", "Variable", numeric, cfg),
        ]
    elif name == "by_group_violin_graph":
        controls = [
            _sel("bgvg_byvar", "Group by", factors, cfg),
            _sel("bgvg_var", "Variable", numeric, cfg),
        ]
    elif name == "by_group_trend_graph":
        controls = [
            _sel("bgtg_byvar", "Group by", factors, cfg),
            _sel("bgtg_var", "Variable", numeric, cfg),
        ]
    elif name == "regression":
        controls = [
            _sel("reg_y", "Dependent", numeric, cfg),
            ui.input_selectize(
                "reg_x",
                "Independent",
                choices=numeric,
                multiple=True,
                selected=[c for c in (cfg.get("reg_x") or []) if c in numeric],
            ),
            _sel("reg_fe1", "Fixed effect 1", fe_choices, cfg, none=True),
            _sel("reg_fe2", "Fixed effect 2", fe_choices, cfg, none=True),
            ui.input_select(
                "cluster",
                "Cluster SE",
                choices={"1": "None", "2": "FE 1", "3": "FE 1 + FE 2"},
                selected=str(cfg.get("cluster", 1)),
            ),
        ]
    elif name == "fwl_plot":
        controls = [
            ui.markdown(
                "**Frisch-Waugh-Lovell plot.** Residualizes the dependent variable and "
                "the focal regressor on the *other* regressors **and** the fixed effects "
                "chosen in the regression above, then plots the two residuals. The fitted "
                "slope equals the focal coefficient in that regression."
            ),
            ui.output_ui("fwl_focal_ui"),
        ]
    # descriptive_table, corrplot, missing_values need no selectors.
    return ui.card(ui.card_header(name.replace("_", " ").title()), *controls, out)
