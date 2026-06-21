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


def _q(value: object) -> str:
    return repr(value)


def _emit_setup() -> str:
    return "import pandas as pd\nimport expdpy as ex"


def _emit_load() -> str:
    return f"df = pd.read_parquet({_q(_SAMPLE_FILE)})\ndf.head()"


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


def build_blocks(
    config: dict, components: Sequence[str], time: str | None = None
) -> list[tuple[str, str]]:
    """Return ``(heading, code)`` blocks reproducing the displayed components."""
    blocks: list[tuple[str, str]] = [
        ("Setup", _emit_setup()),
        ("Load the analysis sample", _emit_load()),
    ]
    for name in components:
        if name in _SKIP or name not in _EMITTERS:
            continue
        code = _EMITTERS[name](config, time)
        if code:  # emitters may return None when their selection is incomplete
            blocks.append((_HEADINGS.get(name, name), code))
    return blocks


def build_script(
    config: dict, components: Sequence[str], time: str | None = None
) -> str:
    """Return a runnable ``.py`` script reproducing the analysis."""
    parts = ["# Generated by expdpy ExPdPy — reproducible analysis script", ""]
    for heading, code in build_blocks(config, components, time):
        parts.append(f"# --- {heading} ---")
        parts.append(code)
        parts.append("")
    return "\n".join(parts)


def build_notebook(
    config: dict, components: Sequence[str], time: str | None = None
) -> bytes:
    """Return a Jupyter notebook (``.ipynb`` bytes) reproducing the analysis."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells.append(
        nbformat.v4.new_markdown_cell("# ExPdPy analysis\n\nGenerated by `expdpy`.")
    )
    for heading, code in build_blocks(config, components, time):
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## {heading}"))
        nb.cells.append(nbformat.v4.new_code_cell(code))
    return nbformat.writes(nb).encode()


def build_export_zip(
    config: dict,
    components: Sequence[str],
    sample: pd.DataFrame,
    time: str | None = None,
) -> bytes:
    """Bundle the prepared sample (parquet) plus a notebook and script into a zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        sample_bytes = io.BytesIO()
        sample.to_parquet(sample_bytes, index=False)
        zf.writestr(_SAMPLE_FILE, sample_bytes.getvalue())
        zf.writestr("ExPdPy_analysis.ipynb", build_notebook(config, components, time))
        zf.writestr("ExPdPy_analysis.py", build_script(config, components, time))
    return buf.getvalue()
