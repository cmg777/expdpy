"""Generate a reproducible notebook/script of the current ExPdPy analysis."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence

import pandas as pd

__all__ = ["build_blocks", "build_script", "build_notebook", "build_export_zip"]

# Control-only components have no standalone output cell.
_SKIP = {"sample_selection", "subset_factor", "grouping", "udvars", "html_block"}

_SAMPLE_FILE = "expdpy_sample.parquet"
_DDEF_FILE = "expdpy_data_def.csv"
_LOAD_HEADING = "Load the analysis sample"


def _q(value: object) -> str:
    return repr(value)


def _emit_setup() -> str:
    return "import pandas as pd\nimport expdpy as ex"


def _emit_load() -> str:
    # Load the data *and* its dictionary, then re-attach the variable labels and declare the
    # panel — so figures and tables read exactly as they did in the app.
    return (
        f"df = pd.read_parquet({_q(_SAMPLE_FILE)})\n"
        f"data_def = pd.read_csv({_q(_DDEF_FILE)})\n"
        "df = ex.set_labels(df, data_def, set_panel=True)\n"
        "df.head()"
    )


def _emit_descriptive(cfg: dict) -> str:
    return "res = ex.explore_descriptive_table(df.select_dtypes('number'))\nres.gt"


def _emit_histogram(cfg: dict) -> str:
    return f"ex.explore_histogram(df, var={_q(cfg.get('hist_var'))}, bins={int(cfg.get('hist_nr_of_breaks', 20))}).fig"


def _emit_bar_chart(cfg: dict) -> str:
    return f"ex.explore_bar_plot(df, var={_q(cfg.get('bar_chart_var1'))}).fig"


def _emit_ext_obs(cfg: dict) -> str:
    return f"ex.explore_ext_obs_table(df, n=5, var={_q(cfg.get('ext_obs_var'))}).gt"


def _emit_missing(cfg: dict, time: str | None) -> str:
    return f"ex.explore_missing_values_plot(df, time={_q(time)}).fig"


def _emit_by_group_bar(cfg: dict) -> str:
    return (
        f"ex.explore_bar_plot_by_group(df, by_var={_q(cfg.get('bgbg_byvar'))}, "
        f"var={_q(cfg.get('bgbg_var'))}).fig"
    )


def _emit_by_group_violin(cfg: dict) -> str:
    return (
        f"ex.explore_violin_plot_by_group(df, by_var={_q(cfg.get('bgvg_byvar'))}, "
        f"var={_q(cfg.get('bgvg_var'))}).fig"
    )


def _emit_by_group_trend(cfg: dict, time: str | None) -> str:
    return (
        f"ex.explore_trend_plot_by_group(df, group_var={_q(cfg.get('bgtg_byvar'))}, "
        f"var={_q(cfg.get('bgtg_var'))}, time={_q(time)}).fig"
    )


def _emit_trend(cfg: dict, time: str | None) -> str:
    variables = [
        cfg.get(k)
        for k in ("trend_graph_var1", "trend_graph_var2", "trend_graph_var3")
        if cfg.get(k) not in (None, "None")
    ]
    return f"ex.explore_trend_plot(df, var={_q(variables)}, time={_q(time)}).fig"


def _emit_quantile_trend(cfg: dict, time: str | None) -> str:
    qs = [float(q) for q in cfg.get("quantile_trend_graph_quantiles", [])]
    return (
        f"ex.explore_quantile_trend_plot(df, "
        f"var={_q(cfg.get('quantile_trend_graph_var'))}, quantiles={_q(qs)}, "
        f"time={_q(time)}).fig"
    )


def _emit_corr(cfg: dict) -> str:
    return "ex.explore_correlation_plot(df.select_dtypes('number')).fig"


def _emit_scatter(cfg: dict) -> str:
    color = cfg.get("scatter_color")
    size = cfg.get("scatter_size")
    return (
        f"ex.explore_scatter_plot(df, x={_q(cfg.get('scatter_x'))}, "
        f"y={_q(cfg.get('scatter_y'))}, "
        f"color={_q(None if color in (None, 'None') else color)}, "
        f"size={_q(None if size in (None, 'None') else size)}, "
        f"loess={1 if cfg.get('scatter_loess') else 0}).fig"
    )


def _emit_regression(cfg: dict) -> str:
    reg_x = cfg.get("reg_x")
    idvs = reg_x if isinstance(reg_x, list) else [reg_x]
    fes = [
        cfg.get(k) for k in ("reg_fe1", "reg_fe2") if cfg.get(k) not in (None, "None")
    ]
    cluster_choice = int(cfg.get("cluster", 1))
    clusters = fes[: max(0, cluster_choice - 1)] if cluster_choice > 1 else []
    return (
        f"res = ex.analyze_regression_table(df, dvs={_q(cfg.get('reg_y'))}, "
        f"idvs={_q(idvs)}, feffects={_q(fes)}, clusters={_q(clusters)})\nres.etable"
    )


def _emit_fwl_plot(cfg: dict) -> str | None:
    reg_x = cfg.get("reg_x")
    xs = reg_x if isinstance(reg_x, list) else [reg_x]
    xs = [x for x in xs if x not in (None, "None")]
    if not xs:
        return None  # mirrors the live no-op when no regressors are chosen
    focal = cfg.get("fwl_focal")
    if focal not in xs:  # mirror the UI auto-selecting the first regressor as focal
        focal = xs[0]
    controls = [x for x in xs if x != focal]
    fes = [
        cfg.get(k) for k in ("reg_fe1", "reg_fe2") if cfg.get(k) not in (None, "None")
    ]
    cluster_choice = int(cfg.get("cluster", 1))
    clusters = fes[: max(0, cluster_choice - 1)] if cluster_choice > 1 else []
    return (
        f"ex.analyze_fwl_plot(df, dv={_q(cfg.get('reg_y'))}, var={_q(focal)}, "
        f"controls={_q(controls)}, feffects={_q(fes)}, clusters={_q(clusters)}).fig"
    )


_EMITTERS = {
    "descriptive_table": lambda c, t: _emit_descriptive(c),
    "histogram": lambda c, t: _emit_histogram(c),
    "bar_chart": lambda c, t: _emit_bar_chart(c),
    "ext_obs": lambda c, t: _emit_ext_obs(c),
    "missing_values": lambda c, t: _emit_missing(c, t),
    "by_group_bar_graph": lambda c, t: _emit_by_group_bar(c),
    "by_group_violin_graph": lambda c, t: _emit_by_group_violin(c),
    "by_group_trend_graph": lambda c, t: _emit_by_group_trend(c, t),
    "trend_graph": lambda c, t: _emit_trend(c, t),
    "quantile_trend_graph": lambda c, t: _emit_quantile_trend(c, t),
    "corrplot": lambda c, t: _emit_corr(c),
    "scatter_plot": lambda c, t: _emit_scatter(c),
    "regression": lambda c, t: _emit_regression(c),
    "fwl_plot": lambda c, t: _emit_fwl_plot(c),
}

_HEADINGS = {
    "descriptive_table": "Descriptive statistics",
    "histogram": "Histogram",
    "bar_chart": "Bar chart",
    "ext_obs": "Extreme observations",
    "missing_values": "Missing values",
    "by_group_bar_graph": "By-group bar graph",
    "by_group_violin_graph": "By-group violin graph",
    "by_group_trend_graph": "By-group trend graph",
    "trend_graph": "Trend graph",
    "quantile_trend_graph": "Quantile trend graph",
    "corrplot": "Correlations",
    "scatter_plot": "Scatter plot",
    "regression": "Regression",
    "fwl_plot": "Frisch-Waugh-Lovell plot",
}


_SUBSET_HEADING = "Subset the sample"
_OUTLIER_HEADING = "Outlier treatment"


def _emit_subset(config: dict, time: str | None) -> str | None:
    """Reproduce the period / category / range filters as a pandas boolean mask, or ``None``."""
    conds: list[str] = []
    pr = config.get("period_range")
    if pr and time:
        conds.append(f"df[{_q(time)}].between({pr[0]!r}, {pr[1]!r})")
    for var, vals in (config.get("cat_filters") or {}).items():
        if vals:
            conds.append(f"df[{_q(var)}].astype(str).isin({[str(v) for v in vals]!r})")
    for var, bounds in (config.get("range_filters") or {}).items():
        if bounds is not None:
            conds.append(
                f"df[{_q(var)}].between({float(bounds[0])!r}, {float(bounds[1])!r})"
            )
    if not conds:
        return None
    mask = "\n    & ".join(conds)
    return f"df = df[\n    {mask}\n].copy()\ndf.shape"


def _emit_outliers(
    config: dict, entities: Sequence[str] | None, time: str | None
) -> str | None:
    """Reproduce the winsorize/truncate outlier treatment (matches the app pipeline), or ``None``."""
    spec = {
        "2": (0.01, False),
        "3": (0.05, False),
        "4": (0.01, True),
        "5": (0.05, True),
    }.get(str(config.get("outlier_treatment", "1")))
    if spec is None:
        return None
    pct, truncate = spec
    ids = [c for c in [*(entities or []), time] if c]
    of = config.get("outlier_factor")
    by = f", by=df[{_q(of)}].to_numpy()" if of not in (None, "None") else ""
    return (
        f"_ids = {ids!r}\n"
        "_num = [c for c in df.columns if c not in _ids "
        "and pd.api.types.is_numeric_dtype(df[c]) "
        "and not pd.api.types.is_bool_dtype(df[c])]\n"
        f"df[_num] = ex.treat_outliers(df[_num], {pct!r}, truncate={truncate!r}{by})"
    )


def build_blocks(
    config: dict,
    components: Sequence[str],
    time: str | None = None,
    entities: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(heading, code)`` blocks reproducing the displayed components."""
    blocks: list[tuple[str, str]] = [
        ("Setup", _emit_setup()),
        (_LOAD_HEADING, _emit_load()),
    ]
    subset = _emit_subset(config, time)
    if subset:
        blocks.append((_SUBSET_HEADING, subset))
    outliers = _emit_outliers(config, entities, time)
    if outliers:
        blocks.append((_OUTLIER_HEADING, outliers))
    for name in components:
        if name in _SKIP or name not in _EMITTERS:
            continue
        code = _EMITTERS[name](config, time)
        if code:  # emitters may return None when their selection is incomplete
            blocks.append((_HEADINGS.get(name, name), code))
    return blocks


def build_script(
    config: dict,
    components: Sequence[str],
    time: str | None = None,
    entities: Sequence[str] | None = None,
) -> str:
    """Return a runnable ``.py`` script reproducing the analysis (plain, local-run)."""
    parts = ["# Generated by expdpy ExPdPy — reproducible analysis script", ""]
    for heading, code in build_blocks(config, components, time, entities):
        parts.append(f"# --- {heading} ---")
        parts.append(code)
        parts.append("")
    return "\n".join(parts)


def _colab_header_md() -> str:
    """Return the header markdown explaining this is a Colab-ready notebook."""
    return (
        "# ExPdPy analysis — Google Colab notebook\n"
        "\n"
        "_Generated by the **ExPdPy** app for [Google Colab](https://colab.research.google.com)._\n"
        "\n"
        "Some setup cells were added so this notebook runs smoothly on Colab:\n"
        "\n"
        "- The **first code cell** installs the exact `expdpy` version that produced this "
        "notebook (pinned), upgrades NumPy/Numba, and then **restarts the runtime once** so the "
        "upgraded NumPy loads cleanly. When the runtime reconnects, just run the cells again "
        "(Runtime ▸ Run all) — the install cell skips the restart the second time.\n"
        "- Before the **Load** cell, add the two data files from the exported zip "
        "(`expdpy_sample.parquet` and `expdpy_data_def.csv`) to this session — see that cell."
    )


def _install_cell(version: str) -> str:
    """Return the Colab install + one-time runtime-restart cell, pinned to ``version``.

    NumPy/Numba are upgraded first because Colab pre-installs an older NumPy that ``expdpy``
    needs above; the runtime is then restarted ONCE (guarded by a ``/tmp`` flag) so the kernel
    reloads the upgraded NumPy instead of the one it imported at startup.
    """
    return (
        "import importlib.util\n"
        "import os\n"
        "\n"
        f'!pip install -q "numpy>=2.1" "numba>=0.61" "expdpy=={version}"\n'
        "\n"
        '_RESTART_FLAG = "/tmp/.expdpy_runtime_restarted"\n'
        '_ON_COLAB = importlib.util.find_spec("google.colab") is not None\n'
        "if _ON_COLAB and not os.path.exists(_RESTART_FLAG):\n"
        '    with open(_RESTART_FLAG, "w"):\n'
        "        pass\n"
        '    print("Install complete - restarting the runtime once so NumPy loads cleanly.")\n'
        '    print("After it reconnects, run the cells again (Runtime > Run all).")\n'
        "    os.kill(os.getpid(), 9)"
    )


# Force Plotly's Colab renderer so figures returned as the last cell expression draw there.
_SETUP_CELL = (
    "# Ensure Plotly figures render in Google Colab (a no-op in other notebook frontends).\n"
    "import plotly.io as pio\n"
    "\n"
    "try:\n"
    "    import google.colab  # noqa: F401  (present only on Colab)\n"
    "\n"
    '    pio.renderers.default = "colab"\n'
    "except ImportError:\n"
    "    pass"
)

_LOAD_INSTRUCTION_MD = (
    "## Load your data\n"
    "\n"
    "Add the two files from the exported zip — **`expdpy_sample.parquet`** and "
    "**`expdpy_data_def.csv`** — to this Colab session (drag them into the file panel on the "
    "left, or use the upload button there), then run the cell below. "
    "`set_labels(..., set_panel=True)` re-attaches the variable labels and declares the panel "
    "from the dictionary."
)


def build_notebook(
    config: dict,
    components: Sequence[str],
    time: str | None = None,
    entities: Sequence[str] | None = None,
) -> bytes:
    """Return a Colab-ready Jupyter notebook (``.ipynb`` bytes) reproducing the analysis."""
    import nbformat

    import expdpy

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell(_colab_header_md()))
    nb.cells.append(nbformat.v4.new_code_cell(_install_cell(expdpy.__version__)))
    nb.cells.append(nbformat.v4.new_code_cell(_SETUP_CELL))
    for heading, code in build_blocks(config, components, time, entities):
        if heading == _LOAD_HEADING:
            nb.cells.append(nbformat.v4.new_markdown_cell(_LOAD_INSTRUCTION_MD))
        else:
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## {heading}"))
        nb.cells.append(nbformat.v4.new_code_cell(code))
    return nbformat.writes(nb).encode()


def _export_def(
    sample: pd.DataFrame, df_def: pd.DataFrame | None, time: str | None
) -> pd.DataFrame:
    """Return the data dictionary to export for ``sample``.

    Uses the active ``df_def`` (filtered to the columns actually present in ``sample``) when it
    covers them; otherwise — e.g. when user-defined variables renamed columns, or no dictionary
    was available — infers a fresh one so the export always ships a dictionary.
    """
    from expdpy._data_def import build_data_def

    if df_def is not None and "var_name" in df_def.columns:
        cols = list(sample.columns)
        if set(cols) <= set(df_def["var_name"]):
            kept = df_def[df_def["var_name"].isin(cols)]
            order = {name: i for i, name in enumerate(cols)}
            return kept.sort_values("var_name", key=lambda s: s.map(order)).reset_index(
                drop=True
            )
    return build_data_def(sample, time=time)


def _narrowed_filters(config: dict, df: pd.DataFrame, time: str | None) -> dict:
    """Return ``config`` with only the filters that actually narrow ``df`` (drop the no-ops).

    ``df`` is the exported (unfiltered) working frame, so a category selection equal to all
    levels, or a range/period equal to the full span, is dropped — keeping the emitted subset
    cell meaningful.
    """
    out = dict(config)
    pr = config.get("period_range")
    if pr and time and time in df.columns:
        full = (df[time].min(), df[time].max())
        out["period_range"] = list(pr) if tuple(pr) != full else None
    cats = {}
    for var, vals in (config.get("cat_filters") or {}).items():
        if var in df.columns and vals:
            allv = {str(v) for v in df[var].dropna().unique()}
            if {str(v) for v in vals} != allv:
                cats[var] = vals
    out["cat_filters"] = cats
    rngs = {}
    for var, bounds in (config.get("range_filters") or {}).items():
        if var in df.columns and bounds is not None:
            full_r = (float(df[var].min()), float(df[var].max()))
            if (float(bounds[0]), float(bounds[1])) != full_r:
                rngs[var] = bounds
    out["range_filters"] = rngs
    return out


def build_export_zip(
    config: dict,
    components: Sequence[str],
    sample: pd.DataFrame,
    time: str | None = None,
    df_def: pd.DataFrame | None = None,
    entities: Sequence[str] | None = None,
) -> bytes:
    """Bundle the working frame (parquet), its data dictionary (csv), a notebook and a script.

    ``sample`` is the *unfiltered* working frame; the notebook reproduces the active filters and
    outlier treatment as code so the analysis rebuilds from it. The data dictionary is always
    written — from ``df_def`` when available, otherwise inferred — so the notebook can re-attach
    labels and declare the panel via :func:`~expdpy.set_labels`.
    """
    cfg = _narrowed_filters(config, sample, time)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        sample_bytes = io.BytesIO()
        sample.to_parquet(sample_bytes, index=False)
        zf.writestr(_SAMPLE_FILE, sample_bytes.getvalue())
        zf.writestr(_DDEF_FILE, _export_def(sample, df_def, time).to_csv(index=False))
        zf.writestr(
            "ExPdPy_analysis.ipynb", build_notebook(cfg, components, time, entities)
        )
        zf.writestr("ExPdPy_analysis.py", build_script(cfg, components, time, entities))
    return buf.getvalue()
