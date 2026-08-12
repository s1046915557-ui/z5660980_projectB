"""Streamlit views for the AssetFund fund shelf and fund factsheets."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from src import allocation, app_charts, app_data

VIEW_SLUGS = {
    "Compare funds": "compare",
    "Fund factsheet": "factsheet",
    "Build allocation": "allocation",
    "Sentiment research": "sentiment",
}
VIEW_STATE_KEY = "assetfund_view"
PRIMARY_FUND_ID = "combined_hierarchical_risk_parity"
DEFAULT_ALLOCATION = {
    PRIMARY_FUND_ID: 40.0,
    "equity_minimum_variance": 40.0,
    "crypto_equal_weight": 20.0,
}


def _apply_assetfund_style() -> None:
    """Add a small standalone brand layer without external assets."""
    st.markdown(
        """
        <style>
        :root { --assetfund-navy: #203246; --assetfund-orange: #c85a0a; }
        .assetfund-kicker {
            color: var(--assetfund-orange); font-size: .78rem;
            font-weight: 750; letter-spacing: .10em; text-transform: uppercase;
        }
        .assetfund-note {
            border-left: 3px solid var(--assetfund-orange); padding: .2rem .8rem;
            color: #536273; font-size: .92rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e7ec; border-radius: .55rem; padding: .65rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _query_value(key: str, default: str) -> str:
    """Read one query parameter as a scalar string."""
    value = st.query_params.get(key, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def _sync_query_params(**updates: str) -> None:
    """Persist validated navigation state without clearing other controls."""
    for key, value in updates.items():
        if _query_value(key, "") != value:
            st.query_params[key] = value


def _format_percent(value: object, decimals: int = 1) -> str:
    """Format a decimal return or risk measure as a percentage."""
    return f"{float(value) * 100.0:.{decimals}f}%"


def _render_header(artifacts: app_data.AppArtifacts) -> None:
    """Render brand and compact research coverage."""
    st.markdown(
        '<div class="assetfund-kicker">Systematic investment research</div>',
        unsafe_allow_html=True,
    )
    st.title("AssetFund")
    st.caption(
        "Systematic equity, crypto and multi-asset funds evaluated with "
        "walk-forward out-of-sample evidence."
    )

    dates = artifacts.fund_returns["date"]
    latest_rebalance = artifacts.fund_weights["effective_start_date"].max()
    columns = st.columns(4)
    coverage = [
        ("Investable funds", f"{artifacts.performance_metrics['fund_id'].nunique()}"),
        ("First live result", f"{dates.min():%d %b %Y}"),
        ("Latest result", f"{dates.max():%d %b %Y}"),
        ("Latest target weights", f"{latest_rebalance:%d %b %Y}"),
    ]
    for column, (label, value) in zip(columns, coverage, strict=True):
        column.caption(label)
        column.markdown(f"**{value}**")
    st.divider()


def _display_names(catalog: pd.DataFrame) -> dict[str, str]:
    """Return a stable fund-id to client-facing-label mapping."""
    return dict(zip(catalog["fund_id"], catalog["display_name"], strict=True))


def _render_catalog_table(catalog: pd.DataFrame) -> pd.DataFrame:
    """Render and return the presentation-ready fund shelf."""
    table = catalog[
        [
            "display_name",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "latest_effective_number_of_holdings",
        ]
    ].copy()
    table["annualized_return"] *= 100.0
    table["annualized_volatility"] *= 100.0
    table["max_drawdown"] *= 100.0
    table = table.rename(
        columns={
            "display_name": "Fund",
            "annualized_return": "Annualised return (%)",
            "annualized_volatility": "Annualised volatility (%)",
            "sharpe_ratio": "Sharpe ratio",
            "max_drawdown": "Maximum drawdown (%)",
            "latest_effective_number_of_holdings": "Effective holdings",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=min(520, 36 * (len(table) + 1)),
        column_config={
            "Annualised return (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Annualised volatility (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
            "Maximum drawdown (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Effective holdings": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    return table


def render_fund_comparison(
    artifacts: app_data.AppArtifacts,
    universe: str,
) -> None:
    """Render the full fund shelf and selected growth comparison."""
    st.header("Choose a fund for the job")
    st.write(
        "Compare realised return, risk and drawdown before opening a detailed "
        "fund factsheet."
    )

    catalog = app_data.fund_catalog(artifacts)
    if universe != "All funds":
        catalog = catalog.loc[catalog["universe"].astype(str).eq(universe)].copy()
    display_names = _display_names(catalog)
    options = catalog["fund_id"].tolist()
    preferred = [
        fund_id
        for fund_id in [
            PRIMARY_FUND_ID,
            "combined_equal_weight",
            "equity_equal_weight",
        ]
        if fund_id in options
    ]
    defaults = preferred or options[: min(3, len(options))]

    selected = st.multiselect(
        "Growth comparison",
        options,
        default=defaults,
        max_selections=5,
        format_func=lambda fund_id: display_names[fund_id],
        help="Select up to five funds. Each line is a gross OOS wealth path.",
    )
    if selected:
        st.plotly_chart(
            app_charts.comparison_growth_figure(
                artifacts.fund_returns,
                display_names,
                selected,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        st.info("Select at least one fund to compare its OOS growth path.")

    st.plotly_chart(
        app_charts.risk_return_figure(catalog),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "Marker shape identifies the investment universe: circles are equity, "
        "squares are crypto and diamonds are combined funds."
    )

    st.subheader("Fund shelf")
    table = _render_catalog_table(catalog)
    st.download_button(
        "Download fund comparison",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="assetfund_fund_comparison.csv",
        mime="text/csv",
    )
    st.markdown(
        '<div class="assetfund-note">Returns are gross of transaction costs and '
        "management fees. OOS results begin after the 2020 estimation period.</div>",
        unsafe_allow_html=True,
    )


def _render_metric_cards(metrics: pd.Series) -> None:
    """Render the required factsheet measures in compact rows."""
    cards = [
        ("Annualised return", _format_percent(metrics["annualized_return"])),
        ("Annualised volatility", _format_percent(metrics["annualized_volatility"])),
        ("Sharpe ratio", f"{float(metrics['sharpe_ratio']):.2f}"),
        ("Maximum drawdown", _format_percent(metrics["max_drawdown"])),
        ("Average one-way turnover", _format_percent(metrics["average_one_way_turnover"])),
        (
            "Effective holdings",
            f"{float(metrics['latest_effective_number_of_holdings']):.1f}",
        ),
    ]
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(
            columns,
            cards[start : start + 3],
            strict=True,
        ):
            column.metric(label, value)


def _render_holdings_table(holdings: pd.DataFrame) -> pd.DataFrame:
    """Render all current holdings with client-facing labels and units."""
    table = holdings[
        ["ticker", "asset_class", "weight_percent"]
    ].rename(
        columns={
            "ticker": "Ticker",
            "asset_class": "Asset class",
            "weight_percent": "Target weight (%)",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=min(520, 36 * (len(table) + 1)),
        column_config={
            "Target weight (%)": st.column_config.NumberColumn(format="%.2f%%")
        },
    )
    return table


def render_fund_factsheet(
    artifacts: app_data.AppArtifacts,
    fund_id: str,
) -> None:
    """Render one investor-facing fund factsheet."""
    sheet = app_data.fund_fact_sheet(artifacts, fund_id)
    st.header(sheet.display_name)
    st.caption(
        f"{sheet.universe} universe · Monthly walk-forward rebalancing · "
        "Gross out-of-sample evidence"
    )
    _render_metric_cards(sheet.metrics)

    st.subheader("Performance")
    st.plotly_chart(
        app_charts.fund_growth_figure(sheet.daily_returns),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        app_charts.fund_drawdown_figure(sheet.daily_returns),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.download_button(
        "Download OOS return history",
        data=sheet.daily_returns.to_csv(index=False).encode("utf-8"),
        file_name=f"{sheet.fund_id}_returns.csv",
        mime="text/csv",
    )

    st.subheader("Current holdings")
    st.caption(
        f"Target portfolio effective {sheet.current_holdings_date:%d %b %Y}. "
        "Weights total 100%."
    )
    st.plotly_chart(
        app_charts.holdings_figure(sheet.current_holdings),
        width="stretch",
        config={"displayModeBar": False},
    )
    holdings_table = _render_holdings_table(sheet.current_holdings)
    st.download_button(
        "Download current holdings",
        data=holdings_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{sheet.fund_id}_current_holdings.csv",
        mime="text/csv",
    )
    st.markdown(
        '<div class="assetfund-note">Historical OOS performance is not a '
        "forecast. Portfolio targets can change at the next monthly rebalance.</div>",
        unsafe_allow_html=True,
    )


def _allocation_inputs(
    catalog: pd.DataFrame,
    display_names: Mapping[str, str],
) -> tuple[dict[str, float], float, float] | None:
    """Collect and validate the investor's product-level allocation order."""
    fund_ids = catalog["fund_id"].tolist()
    default_funds = [
        fund_id for fund_id in DEFAULT_ALLOCATION if fund_id in fund_ids
    ]
    selected = st.multiselect(
        "Funds in this allocation",
        fund_ids,
        default=default_funds,
        max_selections=5,
        format_func=lambda fund_id: display_names[fund_id],
        help="Choose between two and five AssetFund products.",
    )

    st.subheader("Set the starting weights")
    weights_percent = {}
    for fund_id in selected:
        weights_percent[fund_id] = st.number_input(
            display_names[fund_id],
            min_value=0.0,
            max_value=100.0,
            value=DEFAULT_ALLOCATION.get(fund_id, 0.0),
            step=5.0,
            format="%.1f",
            key=f"allocation_weight_{fund_id}",
        )

    total = sum(weights_percent.values())
    allocation_complete = abs(total - 100.0) <= 1e-8
    st.metric(
        "Total allocation",
        f"{total:.1f}%",
        delta=(
            None
            if allocation_complete
            else f"{total - 100.0:+.1f}% versus required"
        ),
        delta_color="off" if allocation_complete else "inverse",
    )
    if not selected or len(selected) < 2:
        st.error("Select at least two funds.")
        return None
    if abs(total - 100.0) > 1e-8:
        st.error("Fund weights must total exactly 100% before calculation.")
        return None

    input_columns = st.columns(2)
    initial_investment = input_columns[0].number_input(
        "Initial investment ($)",
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=1_000.0,
        format="%.0f",
    )
    fee_percent = input_columns[1].slider(
        "Illustrative annual management fee",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.05,
        format="%.2f%%",
    )
    decimal_weights = {
        fund_id: weight / 100.0
        for fund_id, weight in weights_percent.items()
    }
    return decimal_weights, initial_investment, fee_percent / 100.0


def _render_allocation_metrics(metrics: pd.Series) -> None:
    """Render the investor outcomes that reconcile to the allocation path."""
    cards = [
        ("Ending gross value", f"${metrics['ending_gross_value']:,.0f}"),
        ("Ending net value", f"${metrics['ending_net_value']:,.0f}"),
        ("Management-fee drag", f"${metrics['cumulative_fee_drag']:,.0f}"),
        ("Net maximum drawdown", _format_percent(metrics["max_drawdown"])),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, cards, strict=True):
        column.metric(label, value)


def render_allocation_builder(
    artifacts: app_data.AppArtifacts,
    catalog: pd.DataFrame,
    display_names: Mapping[str, str],
) -> None:
    """Render the calendar-safe investor allocation journey."""
    st.header("Build an AssetFund allocation")
    st.write(
        "Blend the available funds, apply an illustrative management fee and "
        "inspect the securities held underneath the product labels."
    )
    inputs = _allocation_inputs(catalog, display_names)
    if inputs is None:
        return

    fund_allocations, initial_investment, annual_fee = inputs
    history = allocation.build_allocation_history(
        artifacts.fund_returns,
        fund_allocations,
        annual_management_fee=annual_fee,
        initial_investment=initial_investment,
    )
    metrics = allocation.allocation_metrics(history)
    as_of_date, lookthrough = allocation.latest_lookthrough_holdings(
        artifacts.fund_weights,
        fund_allocations,
    )

    st.subheader("Historical OOS illustration")
    st.caption(
        f"{history['date'].min():%d %b %Y} to "
        f"{history['date'].max():%d %b %Y}. Fund weights are starting "
        "allocations and can drift with performance."
    )
    _render_allocation_metrics(metrics)
    st.plotly_chart(
        app_charts.allocation_growth_figure(history),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.subheader("Look through the fund labels")
    exposure = lookthrough.groupby("asset_class", observed=True)[
        "lookthrough_weight"
    ].sum()
    exposure_columns = st.columns(2)
    exposure_columns[0].metric(
        "Equity exposure",
        _format_percent(exposure.get("Equity", 0.0)),
    )
    exposure_columns[1].metric(
        "Crypto exposure",
        _format_percent(exposure.get("Crypto", 0.0)),
    )
    overlap_count = int(lookthrough["fund_count"].gt(1).sum())
    st.caption(
        f"Latest targets available by {as_of_date:%d %b %Y}. "
        f"{overlap_count} securities are held through more than one selected "
        "fund and are aggregated below."
    )
    st.plotly_chart(
        app_charts.lookthrough_holdings_figure(lookthrough),
        width="stretch",
        config={"displayModeBar": False},
    )
    table = lookthrough.copy()
    table["source_funds"] = table["source_funds"].apply(
        lambda value: ", ".join(
            display_names.get(fund_id, fund_id)
            for fund_id in value.split(", ")
        )
    )
    table["lookthrough_weight"] *= 100.0
    table = table.rename(
        columns={
            "ticker": "Ticker",
            "asset_class": "Asset class",
            "lookthrough_weight": "Allocation weight (%)",
            "fund_count": "Funds holding security",
            "source_funds": "Source funds",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=min(520, 36 * (len(table) + 1)),
        column_config={
            "Allocation weight (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            )
        },
    )
    st.markdown(
        '<div class="assetfund-note">This is a historical OOS illustration, '
        "not a forecast or investment recommendation. The fee is illustrative "
        "and is accrued using actual elapsed calendar days.</div>",
        unsafe_allow_html=True,
    )


def _render_latest_sentiment_snapshot(sentiment: pd.DataFrame) -> None:
    """Render the latest selected-sector signal and its news coverage."""
    latest_date = sentiment["date"].max()
    latest = sentiment.loc[sentiment["date"].eq(latest_date)].copy()
    table = latest[
        [
            "sector",
            "sentiment_expanding_zscore",
            "sentiment_score_100",
            "coverage_rate",
            "headline_count",
        ]
    ]
    table["coverage_rate"] *= 100.0
    table = table.rename(
        columns={
            "sector": "Sector",
            "sentiment_expanding_zscore": "Past-only z-score",
            "sentiment_score_100": "Sentiment score (0-100)",
            "coverage_rate": "Ticker coverage (%)",
            "headline_count": "Headlines",
        }
    )
    st.caption(f"Latest available sector signals: {latest_date:%d %b %Y}.")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Past-only z-score": st.column_config.NumberColumn(format="%.2f"),
            "Sentiment score (0-100)": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "Ticker coverage (%)": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    st.markdown(
        '<div class="assetfund-note">A score of 50 with 0% coverage means no '
        "fresh news, not bearish news. Its z-score can be negative because "
        "neutral zero is below the sector's historically positive average.</div>",
        unsafe_allow_html=True,
    )


def _render_sentiment_validation(artifacts: app_data.AppArtifacts) -> None:
    """Render development versus frozen locked-holdout model evidence."""
    summary = app_data.sentiment_validation_summary(artifacts)
    baseline_name = "Unmodified NLTK VADER"
    extension_name = "NLTK VADER + conservative finance extension"

    def model_score(split: str, model_name: str, metric: str) -> float:
        row = summary.loc[
            summary["evaluation_split"].eq(split)
            & summary["model_name"].eq(model_name)
        ]
        return float(row.iloc[0][metric])

    development_delta = model_score(
        "Development", extension_name, "macro_f1"
    ) - model_score("Development", baseline_name, "macro_f1")
    holdout_delta = model_score(
        "Locked holdout", extension_name, "macro_f1"
    ) - model_score("Locked holdout", baseline_name, "macro_f1")
    holdout_observations = int(
        summary.loc[
            summary["evaluation_split"].eq("Locked holdout"),
            "observations",
        ].iloc[0]
    )

    columns = st.columns(3)
    columns[0].metric(
        "Development Macro-F1 uplift",
        f"{development_delta:+.3f}",
    )
    columns[1].metric(
        "Locked-holdout Macro-F1 uplift",
        f"{holdout_delta:+.3f}",
    )
    columns[2].metric("Locked labels", f"{holdout_observations}")
    st.plotly_chart(
        app_charts.sentiment_validation_figure(summary),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "The manually reviewed label sets are intentionally small: 60 items "
        "for development and 30 untouched items for the locked holdout."
    )


def _fusion_evidence_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation-ready locked-holdout comparison table."""
    table = comparison[
        [
            "variant_label",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "delta_total_return",
            "delta_sharpe_ratio",
        ]
    ].copy()
    for column in ["total_return", "max_drawdown", "delta_total_return"]:
        table[column] *= 100.0
    return table.rename(
        columns={
            "variant_label": "Variant",
            "total_return": "Total return (%)",
            "sharpe_ratio": "Sharpe ratio",
            "max_drawdown": "Maximum drawdown (%)",
            "delta_total_return": "Return difference (pp)",
            "delta_sharpe_ratio": "Sharpe difference",
        }
    )


def _render_fusion_evidence_gate(
    comparison: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    """State whether the primary fusion improvement survives realistic costs."""
    gross = comparison.loc[comparison["fusion_rule"].eq("coverage_rank")].iloc[0]
    after_cost = costs.loc[
        costs["fusion_rule"].eq("coverage_rank")
        & costs["cost_bps"].eq(50.0)
    ].iloc[0]
    gross_delta = float(gross["delta_sharpe_ratio"])
    cost_delta = float(after_cost["delta_sharpe_ratio"])
    message = (
        "Evidence gate: coverage-aware sentiment changes locked-holdout Sharpe "
        f"by {gross_delta:+.3f} before costs and {cost_delta:+.3f} at 50 bps. "
    )
    if cost_delta > 0.0:
        st.success(message + "The improvement survives this cost scenario.")
    else:
        st.warning(
            message
            + "The edge does not survive this cost scenario, so it remains a "
            "research extension rather than a promoted fund improvement."
        )


def render_sentiment_research(
    artifacts: app_data.AppArtifacts,
    catalog: pd.DataFrame,
    display_names: Mapping[str, str],
) -> None:
    """Render the standalone sentiment index and fusion evidence."""
    st.header("Read the market mood, then test it")
    st.write(
        "AssetFund separates the standalone news signal from the harder question "
        "of whether that signal improves an investable fund."
    )

    sectors = sorted(artifacts.sector_sentiment["sector"].unique())
    defaults = [sector for sector in ["Tech", "Financials", "Energy"] if sector in sectors]
    selected_sectors = st.multiselect(
        "Sectors",
        sectors,
        default=defaults,
        max_selections=5,
        help="Choose up to five sectors for a readable comparison.",
    )
    if selected_sectors:
        sentiment = app_data.sector_sentiment_view(
            artifacts,
            selected_sectors,
        )
        full_sentiment = artifacts.sector_sentiment
        diagnostic = artifacts.sentiment_model_diagnostics.iloc[0]
        scored_headlines = int(diagnostic["scored_headline_rows"])
        excluded_headlines = int(diagnostic["outside_equity_calendar_rows"])
        indexed_headlines = scored_headlines - excluded_headlines
        overview_columns = st.columns(4)
        overview_columns[0].metric(
            "Scored headlines",
            f"{scored_headlines:,}",
        )
        overview_columns[1].metric(
            "Indexed headlines",
            f"{indexed_headlines:,}",
        )
        overview_columns[2].metric(
            "Mean ticker coverage",
            _format_percent(full_sentiment["coverage_rate"].mean()),
        )
        overview_columns[3].metric("Signal timing", "T+1")
        st.caption(
            f"The index covers {full_sentiment['sector'].nunique()} equity "
            f"sectors. {excluded_headlines} scored headlines outside the equity "
            "trading calendar are excluded; fusion uses a one-trading-day lag."
        )
        st.plotly_chart(
            app_charts.sector_sentiment_figure(sentiment),
            width="stretch",
            config={"displayModeBar": False},
        )
        _render_latest_sentiment_snapshot(sentiment)
    else:
        st.info("Select at least one sector to inspect its sentiment history.")

    st.subheader("Validate the language model")
    _render_sentiment_validation(artifacts)

    st.subheader("Did sentiment improve a fund?")
    eligible_set = set(artifacts.fusion_comparison["base_fund_id"])
    eligible_funds = [
        fund_id
        for fund_id in catalog["fund_id"]
        if fund_id in eligible_set
    ]
    default_index = (
        eligible_funds.index(PRIMARY_FUND_ID)
        if PRIMARY_FUND_ID in eligible_funds
        else 0
    )
    selected_fund = st.selectbox(
        "Base fund for the fusion test",
        eligible_funds,
        index=default_index,
        format_func=lambda fund_id: display_names[fund_id],
    )
    comparison, costs = app_data.fusion_evidence(artifacts, selected_fund)
    fusion_table = _fusion_evidence_table(comparison)
    st.dataframe(
        fusion_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Total return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.3f"),
            "Maximum drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Return difference (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "Sharpe difference": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    st.plotly_chart(
        app_charts.fusion_cost_figure(costs),
        width="stretch",
        config={"displayModeBar": False},
    )
    _render_fusion_evidence_gate(comparison, costs)
    st.markdown(
        '<div class="assetfund-note">The fusion decision uses lagged sector '
        "signals and a locked 2023 holdout. A negative cost-adjusted result is "
        "reported rather than hidden.</div>",
        unsafe_allow_html=True,
    )


def _validated_view() -> str:
    """Resolve a shareable view query to a valid sidebar label."""
    requested = _query_value("view", "compare")
    reverse = {slug: label for label, slug in VIEW_SLUGS.items()}
    return reverse.get(requested, "Compare funds")


def _validated_fund_id(
    display_names: Mapping[str, str],
) -> str:
    """Resolve a shareable fund query to an available fund ID."""
    default = (
        PRIMARY_FUND_ID
        if PRIMARY_FUND_ID in display_names
        else next(iter(display_names))
    )
    requested = _query_value("fund", default)
    return requested if requested in display_names else default


def render_app(artifacts: app_data.AppArtifacts) -> None:
    """Render the first AssetFund investor journey."""
    _apply_assetfund_style()
    _render_header(artifacts)
    catalog = app_data.fund_catalog(artifacts)
    display_names = _display_names(catalog)

    st.sidebar.markdown("## AssetFund")
    view_labels = list(VIEW_SLUGS)
    if VIEW_STATE_KEY not in st.session_state:
        st.session_state[VIEW_STATE_KEY] = _validated_view()
    elif st.session_state[VIEW_STATE_KEY] not in view_labels:
        st.session_state[VIEW_STATE_KEY] = "Compare funds"
    selected_view = st.sidebar.radio(
        "Explore",
        view_labels,
        key=VIEW_STATE_KEY,
    )
    _sync_query_params(view=VIEW_SLUGS[selected_view])

    if selected_view == "Compare funds":
        universe = st.sidebar.selectbox(
            "Investment universe",
            ["All funds", "Equity", "Crypto", "Combined"],
        )
        render_fund_comparison(artifacts, universe)
        return

    if selected_view == "Build allocation":
        render_allocation_builder(
            artifacts,
            catalog,
            display_names,
        )
        return

    if selected_view == "Sentiment research":
        render_sentiment_research(
            artifacts,
            catalog,
            display_names,
        )
        return

    fund_ids = catalog["fund_id"].tolist()
    default_fund = _validated_fund_id(display_names)
    selected_fund = st.sidebar.selectbox(
        "Fund",
        fund_ids,
        index=fund_ids.index(default_fund),
        format_func=lambda fund_id: display_names[fund_id],
    )
    _sync_query_params(view="factsheet", fund=selected_fund)
    render_fund_factsheet(artifacts, selected_fund)
