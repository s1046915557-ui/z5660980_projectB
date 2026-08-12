"""Plotly and Streamlit smoke tests for the first AssetFund views."""

import pathlib
import sys

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import allocation, app_charts, app_data

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "streamlit_app.py"


def test_app_chart_payloads_cover_the_selected_funds():
    """Pure chart builders expose the expected fund and holdings traces."""
    artifacts = app_data.load_app_artifacts()
    catalog = app_data.fund_catalog(artifacts)
    display_names = dict(
        zip(catalog["fund_id"], catalog["display_name"], strict=True)
    )

    risk_return = app_charts.risk_return_figure(catalog)
    assert sum(len(trace.x) for trace in risk_return.data) == 13

    selected = ["combined_equal_weight", "combined_hierarchical_risk_parity"]
    comparison = app_charts.comparison_growth_figure(
        artifacts.fund_returns,
        display_names,
        selected,
    )
    assert len(comparison.data) == 2

    sheet = app_data.fund_fact_sheet(artifacts, selected[-1])
    holdings = app_charts.holdings_figure(sheet.current_holdings)
    assert len(holdings.data[0].x) == 10

    allocation_weights = {
        "combined_hierarchical_risk_parity": 0.4,
        "equity_minimum_variance": 0.4,
        "crypto_equal_weight": 0.2,
    }
    history = allocation.build_allocation_history(
        artifacts.fund_returns,
        allocation_weights,
    )
    allocation_growth = app_charts.allocation_growth_figure(history)
    assert len(allocation_growth.data) == 2

    _, lookthrough = allocation.latest_lookthrough_holdings(
        artifacts.fund_weights,
        allocation_weights,
    )
    lookthrough_chart = app_charts.lookthrough_holdings_figure(lookthrough)
    assert len(lookthrough_chart.data[0].x) == 12

    sentiment = app_data.sector_sentiment_view(
        artifacts,
        ["Tech", "Financials", "Energy"],
    )
    sentiment_chart = app_charts.sector_sentiment_figure(sentiment)
    assert len(sentiment_chart.data) == 3

    validation = app_data.sentiment_validation_summary(artifacts)
    validation_chart = app_charts.sentiment_validation_figure(validation)
    assert len(validation_chart.data) == 2
    assert tuple(validation_chart.data[0].x[0]) == (
        "Development",
        "Development",
        "Locked holdout",
        "Locked holdout",
    )
    assert tuple(validation_chart.data[0].x[1]) == (
        "Macro F1",
        "Balanced accuracy",
        "Macro F1",
        "Balanced accuracy",
    )

    _, costs = app_data.fusion_evidence(
        artifacts,
        "combined_hierarchical_risk_parity",
    )
    cost_chart = app_charts.fusion_cost_figure(costs)
    assert len(cost_chart.data) == 3


def test_default_compare_view_renders_without_exception():
    """The default URL loads the fund shelf and both comparison charts."""
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "AssetFund"
    assert any(header.value == "Choose a fund for the job" for header in app.header)
    assert len(app.dataframe) == 1
    assert app.sidebar.radio[0].value == "Compare funds"


def test_factsheet_view_is_url_shareable_and_complete():
    """A direct fund URL renders metrics, charts, holdings, and downloads."""
    app = AppTest.from_file(str(APP_PATH))
    app.query_params["view"] = "factsheet"
    app.query_params["fund"] = "combined_hierarchical_risk_parity"
    app.run(timeout=30)

    assert not app.exception
    assert app.sidebar.radio[0].value == "Fund factsheet"
    assert app.sidebar.selectbox[0].value == "combined_hierarchical_risk_parity"
    assert any(
        header.value == "Combined · Two-stage Sleeve Risk Parity"
        for header in app.header
    )
    assert len(app.metric) == 6
    assert len(app.dataframe) == 1
    assert len(app.get("download_button")) == 2


def test_sidebar_navigation_uses_one_click_after_a_shared_url():
    """After URL initialisation, session state owns later sidebar clicks."""
    app = AppTest.from_file(str(APP_PATH))
    app.query_params["view"] = "factsheet"
    app.query_params["fund"] = "combined_hierarchical_risk_parity"
    app.run(timeout=30)

    app.sidebar.radio[0].set_value("Build allocation")
    app.run(timeout=30)

    assert not app.exception
    assert app.sidebar.radio[0].value == "Build allocation"
    assert app.query_params["view"] == ["allocation"]
    assert any(
        header.value == "Build an AssetFund allocation"
        for header in app.header
    )

    app.run(timeout=30)
    assert app.sidebar.radio[0].value == "Build allocation"


def test_allocation_view_is_url_shareable_and_complete():
    """A direct allocation URL renders valid defaults and investor outcomes."""
    app = AppTest.from_file(str(APP_PATH))
    app.query_params["view"] = "allocation"
    app.run(timeout=30)

    assert not app.exception
    assert app.sidebar.radio[0].value == "Build allocation"
    assert any(
        header.value == "Build an AssetFund allocation"
        for header in app.header
    )
    assert len(app.multiselect[0].value) == 3
    assert len(app.number_input) == 4
    assert app.slider[0].value == 0.5
    assert len(app.metric) == 7
    assert app.metric[0].delta == ""
    assert len(app.get("plotly_chart")) == 2
    assert len(app.dataframe) == 1
    lookthrough_table = app.dataframe[0].value
    assert lookthrough_table["Source funds"].str.contains(
        " · ", regex=False
    ).all()
    assert not lookthrough_table["Source funds"].str.contains("_").any()


def test_sentiment_view_discloses_validation_and_cost_failure():
    """The research URL shows standalone, locked, and cost-aware evidence."""
    app = AppTest.from_file(str(APP_PATH))
    app.query_params["view"] = "sentiment"
    app.run(timeout=30)

    assert not app.exception
    assert app.sidebar.radio[0].value == "Sentiment research"
    assert any(
        header.value == "Read the market mood, then test it"
        for header in app.header
    )
    assert len(app.multiselect[0].value) == 3
    assert app.selectbox[0].value == "combined_hierarchical_risk_parity"
    assert len(app.metric) == 7
    assert app.metric[0].value == "146,836"
    assert app.metric[1].value == "146,830"
    assert app.metric[3].value == "T+1"
    assert len(app.get("plotly_chart")) == 3
    assert len(app.dataframe) == 2
    assert len(app.warning) == 1
    assert "does not survive" in app.warning[0].value
