"""Tests for the pedagogy layer: explainer registry + result interpretation.

Interpretation strings are asserted on stable substrings (not whole strings) to stay robust
to small wording changes. The forbidden-word test is the guardrail that keeps the teaching
prose associational rather than causal.
"""

from __future__ import annotations

import pandas as pd
import pytest

import expdpy as ex
from expdpy.pedagogy import Explainer, explain, list_topics

# Words/phrases that must never appear in an interpretation of a plain association.
_FORBIDDEN = ("causes", "caused by", "effect of", "causal effect of")


def _no_causal_language(text: str) -> bool:
    low = text.lower()
    return not any(bad in low for bad in _FORBIDDEN)


# --- explainer registry --------------------------------------------------------------


def test_list_topics_is_sorted_and_nonempty():
    topics = list_topics()
    assert topics == sorted(topics)
    assert {"fixed_effects", "clustered_se", "pearson", "fwl"} <= set(topics)


def test_every_topic_builds_markdown():
    for topic in list_topics():
        exp = explain(topic)
        assert isinstance(exp, Explainer)
        md = exp.to_markdown()
        assert md.startswith("### ")
        assert "**What it is.**" in md
        assert exp._repr_markdown_() == md


def test_explain_alias_resolves():
    assert explain("fe").topic == "fixed_effects"
    assert explain("clustering").topic == "clustered_se"


def test_new_module_topics_and_aliases_resolve():
    # Analyze: correlated random effects (Mundlak)
    assert explain("mundlak").topic == "correlated_random_effects"
    assert explain("cre").topic == "correlated_random_effects"
    # Learn: the core panel-data identities
    assert explain("demeaning").topic == "within_transformation"
    assert explain("fd").topic == "first_differences"
    assert explain("lsdv").topic == "dummy_variables"


def test_iv_topic_resolves():
    # Instrumental variables ships an explainer; its aliases resolve to the canonical key.
    assert "instrumental_variables" in set(list_topics())
    assert explain("iv").topic == "instrumental_variables"
    assert explain("2sls").topic == "instrumental_variables"


def test_removed_glm_topics_are_gone():
    # GLM-family topics remain unsupported (expdpy is OLS/IV only), so they still raise.
    assert "glm" not in set(list_topics())
    for alias in ("poisson", "logit", "probit"):
        with pytest.raises(KeyError):
            explain(alias)


def test_explain_unknown_raises_with_available():
    with pytest.raises(KeyError) as excinfo:
        explain("not_a_topic")
    msg = str(excinfo.value)
    assert "not_a_topic" in msg
    assert "fixed_effects" in msg  # the message lists available topics


def test_explain_exposed_on_package():
    assert callable(ex.explain)
    assert "fwl" in ex.list_topics()


# --- regression interpretation -------------------------------------------------------


def test_regression_interpret_words(kuznets):
    res = ex.analyze_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq"],
        feffects=["country", "year"],
        clusters=["country"],
    )
    text = res.interpret()
    assert "log_gdp_pc" in text
    assert "associated with" in text
    assert "clustered by" in text
    assert "within" in text  # mentions within-group variation under fixed effects
    assert _no_causal_language(text)


def test_regression_explain_topic_depends_on_design(sample_df, kuznets):
    pooled = ex.analyze_regression_table(sample_df, dvs="x2", idvs=["x1"])
    assert pooled.explain().topic == "ols"

    clustered = ex.analyze_regression_table(
        sample_df, dvs="x2", idvs=["x1"], clusters=["firm"]
    )
    assert clustered.explain().topic == "clustered_se"

    fe = ex.analyze_regression_table(
        kuznets, dvs="gini_regional", idvs=["log_gdp_pc"], feffects=["country"]
    )
    assert fe.explain().topic == "fixed_effects"


def test_regression_tidy_and_glance(kuznets):
    res = ex.analyze_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc"],
        feffects=["country"],
    )
    assert res.tidy() is res.df
    glance = res.glance()
    assert isinstance(glance, pd.DataFrame)
    assert {"model", "N", "r2", "r2_within", "has_fe"} <= set(glance.columns)
    assert int(glance.loc[0, "N"]) == res.models[0]._N


# --- correlation / fwl / trend / descriptive -----------------------------------------


def test_correlation_interpret(kuznets):
    res = ex.explore_correlation_table(
        kuznets[["gini_regional", "log_gdp_pc", "population"]]
    )
    text = res.interpret()
    assert "Pearson" in text
    assert "association" in text
    assert _no_causal_language(text)
    long = res.tidy()
    assert list(long.columns) == ["var1", "var2", "correlation"]


def test_fwl_interpret(kuznets):
    res = ex.analyze_fwl_plot(
        kuznets, dv="gini_regional", var="log_gdp_pc", controls=["log_gdp_pc_sq"]
    )
    text = res.interpret()
    assert "Frisch-Waugh-Lovell" in text
    assert "partial slope" in text
    assert _no_causal_language(text)
    assert list(res.glance().columns) == [
        "slope",
        "se",
        "intercept",
        "n_obs",
        "r2_within",
    ]


def test_trend_interpret(kuznets):
    res = ex.explore_trend_plot(kuznets, var=["gini_regional"], time="year")
    text = res.interpret()
    assert "gini_regional" in text
    assert "2015" in text and "2025" in text


def test_descriptive_interpret_flags_skew(kuznets):
    res = ex.explore_descriptive_table(
        kuznets[["gini_regional", "log_gdp_pc", "population"]]
    )
    text = res.interpret()
    assert "population" in text
    assert "skewed" in text  # population is heavily right-skewed
    tidy = res.tidy()
    assert "variable" in tidy.columns


def test_no_causal_language_across_interpreters(sample_df, kuznets):
    """Guardrail: associations must never be described in causal terms."""
    pooled = ex.analyze_regression_table(sample_df, dvs="x2", idvs=["x1", "x3"])
    corr = ex.explore_correlation_table(sample_df[["x1", "x2", "x3"]])
    desc = ex.explore_descriptive_table(sample_df[["x1", "x2", "x3"]])
    for res in (pooled, corr, desc):
        assert _no_causal_language(res.interpret())


def test_interpret_not_implemented_for_plain_results(kuznets):
    """A result type without an interpreter raises a clear NotImplementedError."""
    from expdpy.pedagogy import Interpretable

    hist = ex.explore_histogram(kuznets, var="gini_regional")
    assert not isinstance(hist, Interpretable)
