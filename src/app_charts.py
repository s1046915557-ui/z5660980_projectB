"""Pure Plotly chart builders for the AssetFund investor interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

INK = "#203246"
NAVY = "#24557A"
ORANGE = "#C85A0A"
TEAL = "#2A9D8F"
GREY = "#8A97A5"
GRID = "#E2E7EC"

METHOD_COLORS = {
    "Equal Weight": GREY,
    "Minimum Variance": NAVY,
    "Maximum Sharpe": ORANGE,
    "Risk Parity": "#69A84F",
    "Sleeve HRP": "#7667A8",
}
UNIVERSE_SYMBOLS = {
    "Equity": "circle",
    "Crypto": "square",
    "Combined": "diamond",
}


def _apply_theme(
    figure: go.Figure,
    *,
    yaxis_title: str,
    height: int = 470,
    time_axis: bool = False,
) -> go.Figure:
    """Apply the compact AssetFund app theme."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 24, "r": 24, "t": 44, "b": 54},
        hovermode="x unified" if time_axis else "closest",
        font={"color": INK},
        legend={
            "orientation": "h",
            "y": 1.10,
            "x": 1.0,
            "xanchor": "right",
            "title": None,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    figure.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor=GRID,
        zerolinecolor=GRID,
        automargin=True,
    )
    figure.update_xaxes(showgrid=False, automargin=True)
    if time_axis:
        figure.update_xaxes(
            rangeslider={"visible": True, "thickness": 0.06},
            tickformatstops=[
                {"dtickrange": [None, "M12"], "value": "%b\n%Y"},
                {"dtickrange": ["M12", None], "value": "%Y"},
            ],
        )
    return figure


def risk_return_figure(catalog: pd.DataFrame) -> go.Figure:
    """Show the fund shelf in annualised return-volatility space."""
    required = {
        "display_name",
        "universe",
        "method_label",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
    }
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Fund catalog is missing chart fields: {sorted(missing)}")

    figure = go.Figure()
    for method_label, group in catalog.groupby("method_label", observed=True):
        colour = METHOD_COLORS.get(str(method_label), NAVY)
        figure.add_trace(
            go.Scatter(
                x=group["annualized_volatility"] * 100.0,
                y=group["annualized_return"] * 100.0,
                mode="markers",
                name=str(method_label),
                customdata=list(
                    zip(
                        group["display_name"],
                        group["universe"].astype(str),
                        group["sharpe_ratio"],
                        strict=True,
                    )
                ),
                marker={
                    "color": colour,
                    "size": 13,
                    "symbol": [
                        UNIVERSE_SYMBOLS.get(str(value), "circle")
                        for value in group["universe"]
                    ],
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Universe: %{customdata[1]}<br>"
                    "Annualised return: %{y:.1f}%<br>"
                    "Annualised volatility: %{x:.1f}%<br>"
                    "Sharpe ratio: %{customdata[2]:.2f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(title="Return earned for the risk taken")
    figure.update_xaxes(title="Annualised volatility (%)")
    return _apply_theme(
        figure,
        yaxis_title="Annualised return (%)",
        height=500,
    )


def comparison_growth_figure(
    fund_returns: pd.DataFrame,
    display_names: Mapping[str, str],
    fund_ids: Sequence[str],
) -> go.Figure:
    """Compare OOS growth paths for a selected set of funds."""
    figure = go.Figure()
    for index, fund_id in enumerate(fund_ids):
        data = fund_returns.loc[
            fund_returns["fund_id"].eq(fund_id)
        ].sort_values("date")
        if data.empty:
            raise KeyError(f"Unknown fund_id in comparison: {fund_id}")
        colour_cycle = [NAVY, ORANGE, TEAL, GREY, "#7667A8"]
        figure.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["growth_of_one"],
                mode="lines",
                name=display_names.get(fund_id, fund_id),
                line={"width": 2.1, "color": colour_cycle[index % 5]},
                hovertemplate=(
                    "%{x|%d %b %Y}<br>Growth of $1: $%{y:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(title="Out-of-sample growth of $1")
    figure.update_yaxes(tickprefix="$", type="log")
    return _apply_theme(
        figure,
        yaxis_title="Growth of $1 (log scale)",
        time_axis=True,
    )


def fund_growth_figure(daily_returns: pd.DataFrame) -> go.Figure:
    """Show one fund's complete OOS growth path."""
    data = daily_returns.sort_values("date")
    figure = go.Figure(
        go.Scatter(
            x=data["date"],
            y=data["growth_of_one"],
            mode="lines",
            name="Growth of $1",
            line={"width": 2.4, "color": NAVY},
            hovertemplate=(
                "%{x|%d %b %Y}<br>Growth of $1: $%{y:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(title="Out-of-sample growth")
    figure.update_yaxes(tickprefix="$", type="log")
    return _apply_theme(
        figure,
        yaxis_title="Growth of $1 (log scale)",
        time_axis=True,
    )


def fund_drawdown_figure(daily_returns: pd.DataFrame) -> go.Figure:
    """Show one fund's peak-to-trough loss through time."""
    data = daily_returns.sort_values("date")
    figure = go.Figure(
        go.Scatter(
            x=data["date"],
            y=data["drawdown"] * 100.0,
            mode="lines",
            name="Drawdown",
            line={"width": 2.0, "color": ORANGE},
            fill="tozeroy",
            fillcolor="rgba(200, 90, 10, 0.12)",
            hovertemplate=(
                "%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>"
            ),
        )
    )
    figure.update_layout(title="Loss from the previous wealth peak")
    return _apply_theme(
        figure,
        yaxis_title="Drawdown (%)",
        time_axis=True,
    )


def holdings_figure(holdings: pd.DataFrame, *, top_n: int = 10) -> go.Figure:
    """Rank the largest current holdings for one fund."""
    largest = holdings.nlargest(top_n, "target_weight").sort_values(
        "target_weight"
    )
    colours = [
        ORANGE if asset_class == "Crypto" else NAVY
        for asset_class in largest["asset_class"]
    ]
    figure = go.Figure(
        go.Bar(
            x=largest["weight_percent"],
            y=largest["ticker"],
            orientation="h",
            marker={"color": colours},
            customdata=largest["asset_class"],
            hovertemplate=(
                "<b>%{y}</b><br>Weight: %{x:.2f}%<br>"
                "Asset class: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.update_layout(title=f"Largest {min(top_n, len(largest))} holdings")
    figure.update_xaxes(title="Target weight (%)", ticksuffix="%")
    return _apply_theme(
        figure,
        yaxis_title="",
        height=430,
    )


def allocation_growth_figure(history: pd.DataFrame) -> go.Figure:
    """Compare gross and management-fee-adjusted investor wealth."""
    required = {"date", "gross_value", "net_value"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            f"Allocation history is missing chart fields: {sorted(missing)}"
        )

    data = history.sort_values("date")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["gross_value"],
            mode="lines",
            name="Gross value",
            line={"width": 2.2, "color": NAVY},
            hovertemplate=(
                "%{x|%d %b %Y}<br>Gross value: $%{y:,.0f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["net_value"],
            mode="lines",
            name="Net value after fee",
            line={"width": 2.2, "color": ORANGE},
            hovertemplate=(
                "%{x|%d %b %Y}<br>Net value: $%{y:,.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(title="Historical OOS value of the allocation")
    figure.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _apply_theme(
        figure,
        yaxis_title="Portfolio value",
        time_axis=True,
    )


def lookthrough_holdings_figure(
    holdings: pd.DataFrame,
    *,
    top_n: int = 12,
) -> go.Figure:
    """Rank the largest securities after looking through selected funds."""
    required = {"ticker", "asset_class", "lookthrough_weight", "fund_count"}
    missing = required.difference(holdings.columns)
    if missing:
        raise ValueError(
            f"Look-through holdings are missing chart fields: {sorted(missing)}"
        )

    largest = holdings.nlargest(top_n, "lookthrough_weight").sort_values(
        "lookthrough_weight"
    )
    colours = [
        ORANGE if asset_class == "Crypto" else NAVY
        for asset_class in largest["asset_class"]
    ]
    figure = go.Figure(
        go.Bar(
            x=largest["lookthrough_weight"] * 100.0,
            y=largest["ticker"],
            orientation="h",
            marker={"color": colours},
            customdata=list(
                zip(
                    largest["asset_class"],
                    largest["fund_count"],
                    strict=True,
                )
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Look-through weight: %{x:.2f}%<br>"
                "Asset class: %{customdata[0]}<br>"
                "Held through %{customdata[1]} fund(s)<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Largest {min(top_n, len(largest))} look-through holdings"
    )
    figure.update_xaxes(title="Allocation weight (%)", ticksuffix="%")
    return _apply_theme(
        figure,
        yaxis_title="",
        height=470,
    )


def sector_sentiment_figure(sentiment: pd.DataFrame) -> go.Figure:
    """Show past-only standardised sentiment for selected equity sectors."""
    required = {
        "date",
        "sector",
        "sentiment_expanding_zscore",
        "coverage_rate",
        "headline_count",
    }
    missing = required.difference(sentiment.columns)
    if missing:
        raise ValueError(
            f"Sector sentiment is missing chart fields: {sorted(missing)}"
        )

    colours = [NAVY, ORANGE, TEAL, "#7667A8", "#69A84F"]
    figure = go.Figure()
    for index, (sector, group) in enumerate(
        sentiment.groupby("sector", observed=True)
    ):
        data = group.sort_values("date")
        figure.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["sentiment_expanding_zscore"],
                mode="lines",
                name=str(sector),
                line={"width": 1.7, "color": colours[index % len(colours)]},
                customdata=list(
                    zip(
                        data["coverage_rate"] * 100.0,
                        data["headline_count"],
                        strict=True,
                    )
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>"
                    "Past-only z-score: %{y:.2f}<br>"
                    "Ticker coverage: %{customdata[0]:.0f}%<br>"
                    "Headlines: %{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(
        y=0.0,
        line={"color": GREY, "width": 1, "dash": "dot"},
    )
    figure.update_layout(title="Sector sentiment surprises through time")
    return _apply_theme(
        figure,
        yaxis_title="Past-only expanding z-score",
        height=520,
        time_axis=True,
    )


def sentiment_validation_figure(summary: pd.DataFrame) -> go.Figure:
    """Compare baseline and finance-aware VADER across both label splits."""
    required = {
        "evaluation_split",
        "model_name",
        "balanced_accuracy",
        "macro_f1",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"Sentiment validation is missing chart fields: {sorted(missing)}"
        )

    models = summary["model_name"].drop_duplicates().tolist()
    colours = [GREY, NAVY]
    figure = go.Figure()
    for index, model_name in enumerate(models):
        model = summary.loc[summary["model_name"].eq(model_name)]
        split_values = []
        metric_values = []
        y_values = []
        for split in ["Development", "Locked holdout"]:
            row = model.loc[model["evaluation_split"].eq(split)].iloc[0]
            split_values.extend([split, split])
            metric_values.extend(["Macro F1", "Balanced accuracy"])
            y_values.extend(
                [
                    row["macro_f1"] * 100.0,
                    row["balanced_accuracy"] * 100.0,
                ]
            )
        figure.add_trace(
            go.Bar(
                x=[split_values, metric_values],
                y=y_values,
                name=str(model_name),
                marker={"color": colours[index % len(colours)]},
                hovertemplate="%{x}<br>Score: %{y:.1f}%<extra></extra>",
            )
        )
    figure.update_layout(
        title="Finance language helps, but the locked sample is harder",
        barmode="group",
    )
    figure.update_yaxes(range=[0.0, 100.0], ticksuffix="%")
    return _apply_theme(
        figure,
        yaxis_title="Classification score",
        height=470,
    )


def fusion_cost_figure(costs: pd.DataFrame) -> go.Figure:
    """Show whether each sentiment variant survives trading costs."""
    required = {
        "cost_bps",
        "variant_label",
        "sharpe_ratio",
        "delta_sharpe_ratio",
    }
    missing = required.difference(costs.columns)
    if missing:
        raise ValueError(
            f"Fusion costs are missing chart fields: {sorted(missing)}"
        )

    variant_colours = {
        "Base Fund": GREY,
        "Coverage-Aware Rank Sentiment": NAVY,
        "Naive Sentiment": ORANGE,
    }
    figure = go.Figure()
    for variant_label, group in costs.groupby("variant_label", observed=True):
        data = group.sort_values("cost_bps")
        figure.add_trace(
            go.Scatter(
                x=data["cost_bps"],
                y=data["sharpe_ratio"],
                mode="lines+markers",
                name=str(variant_label),
                line={
                    "width": 2.2,
                    "color": variant_colours.get(str(variant_label), TEAL),
                },
                marker={"size": 8},
                customdata=data["delta_sharpe_ratio"],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Cost: %{x:.0f} bps<br>"
                    "Sharpe ratio: %{y:.3f}<br>"
                    "Difference from base: %{customdata:+.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(title="Locked-holdout Sharpe after trading costs")
    figure.update_xaxes(title="One-way trading cost (bps)")
    return _apply_theme(
        figure,
        yaxis_title="Sharpe ratio",
        height=450,
    )
