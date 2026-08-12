"""AssetFund report-ready figures from precomputed Project B artifacts.

The public plotting functions accept dataframes so they can be tested without
rerunning portfolio optimisation or sentiment scoring. ``build_all_figures``
is the single production entry point used by ``scripts/run_part_b.py``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, PercentFormatter

ASSETFUND_SOURCE = (
    "FINS5545 hosted project data; AssetFund calculations by the author"
)
PRIMARY_FUND_ID = "combined_hierarchical_risk_parity"
HOLDOUT_START = pd.Timestamp("2023-01-01")

NAVY = "#1F4E79"
ORANGE = "#C55A11"
GREEN = "#70AD47"
TEAL = "#2A9D8F"
PURPLE = "#7666A6"
GREY = "#7A8694"
PALE_BLUE = "#DCE6F1"
PALE_ORANGE = "#F7E4D8"
GRID = "#D9DEE5"
INK = "#263746"
MUTED = "#5D6874"

METHOD_ORDER = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "hierarchical_risk_parity",
)
METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "hierarchical_risk_parity": "Hierarchical Sleeve Risk Parity",
}
METHOD_SHORT_LABELS = {
    "equal_weight": "Equal wt.",
    "minimum_variance": "Min variance",
    "maximum_sharpe": "Max Sharpe",
    "risk_parity": "Risk parity",
    "hierarchical_risk_parity": "Sleeve HRP",
}
METHOD_COLORS = {
    "equal_weight": GREY,
    "minimum_variance": NAVY,
    "maximum_sharpe": ORANGE,
    "risk_parity": GREEN,
    "hierarchical_risk_parity": PURPLE,
}
VARIANT_ORDER = (
    "Base Fund",
    "Naive Sentiment",
    "Coverage-Aware Rank Sentiment",
)
VARIANT_LABELS = {
    "Base Fund": "Base Sleeve HRP",
    "Naive Sentiment": "Naive sentiment",
    "Coverage-Aware Rank Sentiment": "Coverage-aware rank",
}
VARIANT_COLORS = {
    "Base Fund": NAVY,
    "Naive Sentiment": ORANGE,
    "Coverage-Aware Rank Sentiment": TEAL,
}
SECTOR_ORDER = (
    "Tech",
    "Financials",
    "Energy",
    "Consumer",
    "Industrials",
    "Healthcare",
    "Comm",
    "Materials",
    "Utilities",
    "RealEstate",
)
SECTOR_LABELS = {
    "Tech": "Technology",
    "Comm": "Communication",
    "RealEstate": "Real Estate",
}
UNIVERSE_MARKERS = {
    "Equity": "o",
    "Crypto": "s",
    "Combined": "D",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Raise a useful error when a figure input is incomplete."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _normalise_dates(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> pd.DataFrame:
    """Return a copy with one validated, timezone-naive date column."""
    data = frame.copy()
    values = pd.to_datetime(data[column], errors="coerce", utc=True)
    if values.isna().any():
        raise ValueError(f"{name}.{column} contains an invalid date.")
    data[column] = values.dt.tz_localize(None)
    return data


def _date_span(values: pd.Series) -> str:
    """Format an inclusive date span for a self-contained figure note."""
    dates = pd.to_datetime(values)
    return f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}"


def _style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply the custom AssetFund axis treatment."""
    axis.set_facecolor("white")
    axis.grid(
        axis=grid_axis,
        color=GRID,
        linewidth=0.7,
        alpha=0.8,
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=7)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def _format_year_axis(axis: plt.Axes) -> None:
    """Use sparse, horizontal year labels on a date axis."""
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _format_growth_axis(axis: plt.Axes, values: pd.Series) -> None:
    """Use readable dollar ticks on a logarithmic wealth axis."""
    minimum = float(values.min())
    maximum = float(values.max())
    lower = minimum * 0.97
    upper = maximum * 1.03
    if upper <= 2.1:
        candidates = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
    else:
        candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0)
    ticks = [value for value in candidates if lower <= value <= upper]
    if len(ticks) < 2:
        ticks = np.geomspace(lower, upper, num=3).tolist()

    axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"${value:g}")
    )
    axis.yaxis.set_minor_formatter(NullFormatter())


def _add_header(
    figure: plt.Figure,
    title: str,
    subtitle: str,
) -> None:
    """Add a consistent AssetFund research header."""
    figure.add_artist(
        Line2D(
            [0.05, 0.96],
            [0.972, 0.972],
            transform=figure.transFigure,
            color=ORANGE,
            linewidth=2.2,
        )
    )
    figure.text(
        0.05,
        0.945,
        "ASSETFUND  |  RESEARCH",
        color=NAVY,
        fontsize=7,
        fontweight="bold",
    )
    figure.text(
        0.05,
        0.905,
        title,
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(0.05, 0.875, subtitle, color=MUTED, fontsize=7.5)


def _add_footer(
    figure: plt.Figure,
    *,
    note: str,
    sample: str,
    units: str,
) -> None:
    """Add note, sample, units, and source inside the exported canvas."""
    footer = (
        f"Note: {note}  Sample: {sample}.  Units: {units}.  "
        f"Source: {ASSETFUND_SOURCE}."
    )
    figure.text(
        0.05,
        0.018,
        textwrap.fill(footer, width=142),
        color=MUTED,
        fontsize=6.1,
        linespacing=1.25,
        va="bottom",
    )


def _save_figure(
    figure: plt.Figure,
    output_dir: str | Path,
    filename: str,
) -> Path:
    """Export one Word/A4-ready PNG and close its Matplotlib state."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        metadata={"Creator": "AssetFund"},
    )
    plt.close(figure)
    return output_path


def plot_fund_growth_comparison(
    fund_returns: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot growth of one dollar across methods within each universe."""
    required = {
        "date",
        "fund_id",
        "universe",
        "method",
        "growth_of_one",
    }
    _require_columns(fund_returns, required, "fund_returns")
    data = _normalise_dates(fund_returns, "date", "fund_returns")
    data = data.sort_values(["fund_id", "date"])

    growth = pd.to_numeric(data["growth_of_one"], errors="coerce")
    if growth.isna().any() or not np.isfinite(growth).all():
        raise ValueError("fund_returns.growth_of_one must be finite.")
    if growth.le(0).any():
        raise ValueError("Log growth figures require positive wealth values.")
    if data.duplicated(["fund_id", "date"]).any():
        raise ValueError("fund_returns contains duplicate fund-date rows.")

    universes = ("Equity", "Crypto", "Combined")
    missing_universes = set(universes).difference(data["universe"].unique())
    if missing_universes:
        raise ValueError(
            f"fund_returns is missing universes: {sorted(missing_universes)}"
        )

    figure, axes = plt.subplots(
        len(universes),
        1,
        figsize=(6.27, 7.35),
        sharex=True,
    )
    for axis, universe in zip(axes, universes, strict=True):
        panel = data.loc[data["universe"].eq(universe)]
        for method in METHOD_ORDER:
            series = panel.loc[panel["method"].eq(method)]
            if series.empty:
                continue
            axis.plot(
                series["date"],
                series["growth_of_one"],
                color=METHOD_COLORS[method],
                linewidth=1.35,
                alpha=0.92,
                zorder=3,
            )
        axis.axhline(1.0, color=GRID, linewidth=0.8, zorder=1)
        _format_growth_axis(axis, panel["growth_of_one"])
        axis.set_title(universe, loc="left", fontsize=9, fontweight="bold")
        axis.set_ylabel("Growth of $1\n(log scale)", fontsize=7)
        _style_axis(axis)
        _format_year_axis(axis)

    axes[-1].set_xlabel("Out-of-sample date", fontsize=7.5)
    methods_present = [
        method for method in METHOD_ORDER if method in set(data["method"])
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2,
            label=METHOD_SHORT_LABELS[method],
        )
        for method in methods_present
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.85),
        ncol=3,
        frameon=False,
        fontsize=6.8,
    )
    _add_header(
        figure,
        "How AssetFund strategies grew $1",
        "Walk-forward out-of-sample performance, separated by trading calendar",
    )
    _add_footer(
        figure,
        note=(
            "Gross returns before transaction costs; each panel compares "
            "methods only within the same asset universe"
        ),
        sample=_date_span(data["date"]),
        units="Dollar value on a logarithmic scale",
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.79,
        bottom=0.13,
        hspace=0.34,
    )
    return _save_figure(figure, output_dir, "fund_growth_comparison.png")


def _primary_fusion_drawdowns(
    fund_returns: pd.DataFrame,
    fusion_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the base, naive, and primary drawdown paths."""
    base_required = {"date", "fund_id", "drawdown"}
    fusion_required = {
        "date",
        "base_fund_id",
        "variant_label",
        "drawdown",
    }
    _require_columns(fund_returns, base_required, "fund_returns")
    _require_columns(fusion_returns, fusion_required, "fusion_returns")
    base = _normalise_dates(fund_returns, "date", "fund_returns")
    augmented = _normalise_dates(fusion_returns, "date", "fusion_returns")

    base = base.loc[
        base["fund_id"].eq(PRIMARY_FUND_ID),
        ["date", "drawdown"],
    ].copy()
    base["variant_label"] = "Base Fund"
    augmented = augmented.loc[
        augmented["base_fund_id"].eq(PRIMARY_FUND_ID),
        ["date", "drawdown", "variant_label"],
    ].copy()
    paths = pd.concat([base, augmented], ignore_index=True)
    paths["drawdown"] = pd.to_numeric(paths["drawdown"], errors="coerce")
    if paths["drawdown"].isna().any() or not np.isfinite(
        paths["drawdown"]
    ).all():
        raise ValueError("Primary fusion drawdowns must be finite.")
    observed = set(paths["variant_label"])
    if observed != set(VARIANT_ORDER):
        raise ValueError(
            "Primary fusion drawdown variants differ from the frozen design."
        )
    if paths.duplicated(["variant_label", "date"]).any():
        raise ValueError("Primary fusion drawdowns contain duplicate dates.")
    return paths.sort_values(["variant_label", "date"]).reset_index(drop=True)


def plot_primary_fusion_drawdown(
    fund_returns: pd.DataFrame,
    fusion_returns: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot full-OOS drawdowns for the primary fund and two overlays."""
    data = _primary_fusion_drawdowns(fund_returns, fusion_returns)
    figure, axis = plt.subplots(figsize=(6.27, 3.75))

    for variant in VARIANT_ORDER:
        series = data.loc[data["variant_label"].eq(variant)]
        axis.plot(
            series["date"],
            series["drawdown"] * 100,
            color=VARIANT_COLORS[variant],
            linewidth=1.45 if variant != "Base Fund" else 1.7,
            alpha=0.92,
            label=VARIANT_LABELS[variant],
        )
    axis.axhline(0, color=GRID, linewidth=0.8)
    axis.axvline(
        HOLDOUT_START,
        color=INK,
        linewidth=0.9,
        linestyle=(0, (3, 3)),
        alpha=0.75,
    )
    axis.text(
        HOLDOUT_START + pd.Timedelta(days=18),
        0.04,
        "Locked 2023 holdout",
        transform=axis.get_xaxis_transform(),
        fontsize=6.5,
        color=MUTED,
        va="bottom",
    )
    axis.set_xlabel("Out-of-sample date", fontsize=7.5)
    axis.set_ylabel("Drawdown (%)", fontsize=7.5)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    _style_axis(axis)
    _format_year_axis(axis)
    axis.legend(frameon=False, fontsize=7, loc="lower left")

    _add_header(
        figure,
        "Drawdown of the primary combined fund",
        "Base Sleeve HRP compared with naive and coverage-aware sentiment overlays",
    )
    _add_footer(
        figure,
        note=(
            "Drawdown is measured from the previous wealth peak; the vertical "
            "line separates development from the untouched holdout"
        ),
        sample=_date_span(data["date"]),
        units="Percentage below the previous peak",
    )
    figure.subplots_adjust(left=0.13, right=0.97, top=0.79, bottom=0.22)
    return _save_figure(figure, output_dir, "primary_fusion_drawdown.png")


def _combined_sleeve_weights(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate combined-fund asset weights into equity and crypto sleeves."""
    required = {
        "effective_start_date",
        "fund_id",
        "universe",
        "method",
        "ticker",
        "target_weight",
    }
    _require_columns(fund_weights, required, "fund_weights")
    data = _normalise_dates(
        fund_weights,
        "effective_start_date",
        "fund_weights",
    )
    data = data.loc[data["universe"].eq("Combined")].copy()
    if data.empty:
        raise ValueError("fund_weights has no Combined fund rows.")
    data["target_weight"] = pd.to_numeric(
        data["target_weight"], errors="coerce"
    )
    if data["target_weight"].isna().any() or data["target_weight"].lt(0).any():
        raise ValueError("Combined target weights must be finite and long-only.")
    data["asset_class"] = np.where(
        data["ticker"].astype(str).str.endswith("-USD"),
        "Crypto",
        "Equity",
    )
    sleeves = (
        data.groupby(
            ["fund_id", "method", "effective_start_date", "asset_class"],
            as_index=False,
        )["target_weight"]
        .sum()
        .pivot(
            index=["fund_id", "method", "effective_start_date"],
            columns="asset_class",
            values="target_weight",
        )
        .fillna(0.0)
        .reset_index()
    )
    sleeves.columns.name = None
    for asset_class in ("Equity", "Crypto"):
        if asset_class not in sleeves:
            sleeves[asset_class] = 0.0
    totals = sleeves["Equity"] + sleeves["Crypto"]
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-8):
        raise ValueError("Combined sleeve weights do not sum to one.")
    return sleeves.sort_values(
        ["method", "effective_start_date"]
    ).reset_index(drop=True)


def plot_combined_sleeve_weights(
    fund_weights: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot equity-versus-crypto target sleeves across combined methods."""
    data = _combined_sleeve_weights(fund_weights)
    missing_methods = set(METHOD_ORDER).difference(data["method"].unique())
    if missing_methods:
        raise ValueError(
            f"Combined weights are missing methods: {sorted(missing_methods)}"
        )

    figure, axes = plt.subplots(3, 2, figsize=(6.27, 7.35), sharey=True)
    plot_axes = list(axes.flat[:5])
    guide_axis = axes.flat[5]

    for index, (axis, method) in enumerate(
        zip(plot_axes, METHOD_ORDER, strict=True)
    ):
        panel = data.loc[data["method"].eq(method)].sort_values(
            "effective_start_date"
        )
        dates = panel["effective_start_date"]
        equity = panel["Equity"].to_numpy() * 100
        crypto = panel["Crypto"].to_numpy() * 100
        axis.stackplot(
            dates,
            equity,
            crypto,
            colors=(PALE_BLUE, PALE_ORANGE),
            edgecolor=(NAVY, ORANGE),
            linewidth=0.65,
            step="post",
            alpha=0.95,
        )
        axis.plot(dates, equity, color=NAVY, linewidth=1.0, drawstyle="steps-post")
        axis.set_ylim(0, 100)
        axis.set_yticks([0, 50, 100])
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.set_title(
            METHOD_SHORT_LABELS[method],
            loc="left",
            fontsize=8,
            fontweight="bold",
        )
        _format_year_axis(axis)
        _style_axis(axis)
        if index < 4:
            axis.tick_params(labelbottom=False)

    guide_axis.axis("off")
    guide_axis.legend(
        handles=[
            Patch(facecolor=PALE_BLUE, edgecolor=NAVY, label="Equity sleeve"),
            Patch(
                facecolor=PALE_ORANGE,
                edgecolor=ORANGE,
                label="Crypto sleeve",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    guide_axis.text(
        0.0,
        0.56,
        "Reading guide",
        fontsize=8,
        fontweight="bold",
        color=INK,
        transform=guide_axis.transAxes,
    )
    guide_axis.text(
        0.0,
        0.48,
        textwrap.fill(
            "Each panel is a 100% target allocation. The boundary shows the "
            "equity share; the remaining area is crypto.",
            width=37,
        ),
        fontsize=7,
        color=MUTED,
        va="top",
        linespacing=1.35,
        transform=guide_axis.transAxes,
    )
    figure.text(
        0.015,
        0.49,
        "Target portfolio weight (%)",
        rotation=90,
        va="center",
        fontsize=7.5,
        color=INK,
    )
    figure.text(
        0.50,
        0.105,
        "Effective rebalance date",
        ha="center",
        fontsize=7.5,
        color=INK,
    )
    _add_header(
        figure,
        "What drives the combined funds",
        "Monthly equity and crypto target sleeves across five portfolio methods",
    )
    _add_footer(
        figure,
        note=(
            "Weights are long-only monthly targets formed before the effective "
            "holding date; individual assets are aggregated by sleeve"
        ),
        sample=_date_span(data["effective_start_date"]),
        units="Percentage of the combined fund target allocation",
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.82,
        bottom=0.16,
        hspace=0.36,
        wspace=0.18,
    )
    return _save_figure(figure, output_dir, "combined_sleeve_weights.png")


def _scorecard_labels(metrics: pd.DataFrame) -> pd.Series:
    """Return compact display labels for the 13-fund Sharpe scorecard."""
    return metrics.apply(
        lambda row: (
            f"{row['universe']} · {METHOD_SHORT_LABELS[row['method']]}"
        ),
        axis=1,
    )


def plot_fund_risk_return_scorecard(
    performance_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot annualised risk-return positions and Sharpe bars for all funds."""
    required = {
        "fund_id",
        "universe",
        "method",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
    }
    _require_columns(performance_metrics, required, "performance_metrics")
    data = performance_metrics.copy()
    if data["fund_id"].duplicated().any():
        raise ValueError("performance_metrics contains duplicate funds.")
    numeric = ["annualized_return", "annualized_volatility", "sharpe_ratio"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[numeric].isna().any().any() or not np.isfinite(
        data[numeric].to_numpy()
    ).all():
        raise ValueError("Performance scorecard metrics must be finite.")
    unknown_methods = set(data["method"]).difference(METHOD_ORDER)
    unknown_universes = set(data["universe"]).difference(UNIVERSE_MARKERS)
    if unknown_methods or unknown_universes:
        raise ValueError("Performance metrics contain an unknown display category.")

    data["annualized_return_pct"] = data["annualized_return"] * 100
    data["annualized_volatility_pct"] = data["annualized_volatility"] * 100
    data["display_label"] = _scorecard_labels(data)

    figure, (scatter_axis, bar_axis) = plt.subplots(
        2,
        1,
        figsize=(6.27, 7.35),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    for row in data.itertuples(index=False):
        scatter_axis.scatter(
            row.annualized_volatility_pct,
            row.annualized_return_pct,
            s=55 if row.fund_id == PRIMARY_FUND_ID else 38,
            marker=UNIVERSE_MARKERS[row.universe],
            facecolor=METHOD_COLORS[row.method],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.94,
            zorder=3,
        )
    primary = data.loc[data["fund_id"].eq(PRIMARY_FUND_ID)].iloc[0]
    scatter_axis.annotate(
        "Sleeve HRP",
        xy=(
            primary["annualized_volatility_pct"],
            primary["annualized_return_pct"],
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=6.5,
        color=PURPLE,
        arrowprops={"arrowstyle": "-", "color": PURPLE, "linewidth": 0.7},
    )
    scatter_axis.set_title(
        "Annualised return versus volatility",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    scatter_axis.set_xlabel("Annualised volatility (%)", fontsize=7.5)
    scatter_axis.set_ylabel("Annualised arithmetic return (%)", fontsize=7.5)
    _style_axis(scatter_axis, grid_axis="both")

    bars = data.sort_values("sharpe_ratio").reset_index(drop=True)
    bar_colors = bars["method"].map(METHOD_COLORS)
    bar_objects = bar_axis.barh(
        bars["display_label"],
        bars["sharpe_ratio"],
        color=bar_colors,
        alpha=0.9,
        height=0.62,
        zorder=3,
    )
    bar_axis.set_title(
        "Sharpe ratio across all 13 funds",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    bar_axis.set_xlabel("Out-of-sample Sharpe ratio (risk-free rate = 0)", fontsize=7.5)
    bar_axis.set_ylabel("Fund", fontsize=7.5)
    bar_axis.set_xlim(0, bars["sharpe_ratio"].max() * 1.16)
    bar_axis.tick_params(axis="y", labelsize=6.2)
    _style_axis(bar_axis, grid_axis="x")
    for rectangle, value in zip(
        bar_objects,
        bars["sharpe_ratio"],
        strict=True,
    ):
        bar_axis.text(
            value + bars["sharpe_ratio"].max() * 0.015,
            rectangle.get_y() + rectangle.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=6.2,
            color=INK,
        )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markersize=5.5,
            label=METHOD_SHORT_LABELS[method],
        )
        for method in METHOD_ORDER
        if method in set(data["method"])
    ]
    universe_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=INK,
            markersize=5.5,
            label=universe,
        )
        for universe, marker in UNIVERSE_MARKERS.items()
    ]
    figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.85),
        ncol=5,
        frameon=False,
        fontsize=6.3,
    )
    figure.legend(
        handles=universe_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.825),
        ncol=3,
        frameon=False,
        fontsize=6.3,
    )
    _add_header(
        figure,
        "The 13-fund risk-return shelf",
        "Calendar-matched annualisation and a common zero risk-free-rate assumption",
    )
    _add_footer(
        figure,
        note=(
            "Colours identify portfolio methods and marker shapes identify asset "
            "universes; the highlighted diamond is the additional Sleeve HRP fund"
        ),
        sample="01 Jan 2021 to 31 Dec 2023 (calendar varies by universe)",
        units="Annualised percentage rates and unitless Sharpe ratio",
    )
    figure.subplots_adjust(
        left=0.24,
        right=0.96,
        top=0.77,
        bottom=0.12,
        hspace=0.47,
    )
    return _save_figure(figure, output_dir, "fund_risk_return_scorecard.png")


def plot_sector_sentiment_index(
    sector_sentiment: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot past-only standardised sector sentiment in small multiples."""
    zscore_column = "sentiment_expanding_zscore"
    required = {"date", "sector", zscore_column}
    _require_columns(sector_sentiment, required, "sector_sentiment_index")
    data = _normalise_dates(
        sector_sentiment,
        "date",
        "sector_sentiment_index",
    )
    data[zscore_column] = pd.to_numeric(data[zscore_column], errors="coerce")
    available = data[zscore_column].dropna()
    if available.empty or not np.isfinite(available).all():
        raise ValueError("Standardised sector sentiment must contain finite values.")
    if data.duplicated(["sector", "date"]).any():
        raise ValueError("Sector sentiment contains duplicate sector-date rows.")
    missing_sectors = set(SECTOR_ORDER).difference(data["sector"].unique())
    if missing_sectors:
        raise ValueError(
            f"Sector sentiment is missing sectors: {sorted(missing_sectors)}"
        )
    data = data.sort_values(["sector", "date"])
    unavailable_sectors = data.groupby("sector", observed=True)[
        zscore_column
    ].count()
    if unavailable_sectors.eq(0).any():
        missing = unavailable_sectors[unavailable_sectors.eq(0)].index.tolist()
        raise ValueError(
            f"Standardised sentiment is unavailable for sectors: {missing}"
        )
    data["sentiment_zscore_21d"] = data.groupby("sector")[
        zscore_column
    ].transform(lambda values: values.rolling(21, min_periods=5).mean())

    absolute_limit = float(available.abs().max())
    axis_limit = max(2.0, np.ceil(absolute_limit * 2.0) / 2.0)

    figure, axes = plt.subplots(
        5,
        2,
        figsize=(6.27, 7.35),
        sharex=True,
        sharey=True,
    )
    for index, (axis, sector) in enumerate(
        zip(axes.flat, SECTOR_ORDER, strict=True)
    ):
        panel = data.loc[data["sector"].eq(sector)]
        axis.plot(
            panel["date"],
            panel[zscore_column].clip(-axis_limit, axis_limit),
            color=GREY,
            linewidth=0.45,
            alpha=0.32,
            zorder=2,
        )
        axis.plot(
            panel["date"],
            panel["sentiment_zscore_21d"].clip(-axis_limit, axis_limit),
            color=NAVY,
            linewidth=1.2,
            alpha=0.95,
            zorder=3,
        )
        axis.axhline(0, color=ORANGE, linewidth=0.75, alpha=0.8, zorder=1)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.set_title(
            SECTOR_LABELS.get(sector, sector),
            loc="left",
            fontsize=7.8,
            fontweight="bold",
        )
        _style_axis(axis)
        _format_year_axis(axis)
        if index < 8:
            axis.tick_params(labelbottom=False)

    figure.text(
        0.012,
        0.49,
        "Past-only expanding z-score",
        rotation=90,
        va="center",
        fontsize=7.5,
        color=INK,
    )
    figure.text(
        0.50,
        0.105,
        "Equity trading date",
        ha="center",
        fontsize=7.5,
        color=INK,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=GREY, linewidth=1, alpha=0.45, label="Daily"),
            Line2D([0], [0], color=NAVY, linewidth=1.8, label="21-day average"),
            Line2D([0], [0], color=ORANGE, linewidth=1, label="Historical norm (0)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.815),
        ncol=3,
        frameon=False,
        fontsize=6.8,
    )
    _add_header(
        figure,
        "AssetFund sector sentiment surprises",
        "Finance-aware VADER relative to each sector's expanding historical baseline",
    )
    _add_footer(
        figure,
        note=(
            "Each day is standardised against earlier sector observations only; "
            "the first 21 observations are unavailable, no-news ticker-days enter "
            "as neutral before standardisation, and the dark line smooths 21 days"
        ),
        sample=_date_span(data["date"]),
        units="Standard deviations from the prior expanding mean",
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.76,
        bottom=0.15,
        hspace=0.34,
        wspace=0.17,
    )
    return _save_figure(figure, output_dir, "sector_sentiment_index.png")


def _primary_holdout_growth(
    fund_returns: pd.DataFrame,
    fusion_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build rebased 2023 growth paths for the frozen primary comparison."""
    base_required = {"date", "fund_id", "return"}
    fusion_required = {
        "date",
        "base_fund_id",
        "variant_label",
        "return",
    }
    _require_columns(fund_returns, base_required, "fund_returns")
    _require_columns(fusion_returns, fusion_required, "fusion_returns")
    base = _normalise_dates(fund_returns, "date", "fund_returns")
    augmented = _normalise_dates(fusion_returns, "date", "fusion_returns")

    base = base.loc[
        base["fund_id"].eq(PRIMARY_FUND_ID) & base["date"].ge(HOLDOUT_START),
        ["date", "return"],
    ].copy()
    base["variant_label"] = "Base Fund"
    augmented = augmented.loc[
        augmented["base_fund_id"].eq(PRIMARY_FUND_ID)
        & augmented["date"].ge(HOLDOUT_START),
        ["date", "return", "variant_label"],
    ].copy()
    data = pd.concat([base, augmented], ignore_index=True)
    data["return"] = pd.to_numeric(data["return"], errors="coerce")
    if data["return"].isna().any() or not np.isfinite(data["return"]).all():
        raise ValueError("Primary holdout returns must be finite.")
    observed = set(data["variant_label"])
    if observed != set(VARIANT_ORDER):
        raise ValueError("Primary holdout variants differ from the frozen design.")
    if data.duplicated(["variant_label", "date"]).any():
        raise ValueError("Primary holdout returns contain duplicate dates.")
    data = data.sort_values(["variant_label", "date"]).reset_index(drop=True)
    data["holdout_growth_of_one"] = data.groupby("variant_label")[
        "return"
    ].transform(lambda values: (1.0 + values).cumprod())
    return data


def _primary_holdout_cost_deltas(
    fusion_costs: pd.DataFrame,
) -> pd.DataFrame:
    """Select the locked primary Sharpe deltas at the three cost settings."""
    required = {
        "base_fund_id",
        "variant_label",
        "evaluation_period",
        "cost_bps",
        "delta_sharpe_ratio",
    }
    _require_columns(fusion_costs, required, "fusion_transaction_costs")
    data = fusion_costs.loc[
        fusion_costs["base_fund_id"].eq(PRIMARY_FUND_ID)
        & fusion_costs["evaluation_period"].eq("locked_holdout_2023")
        & fusion_costs["variant_label"].isin(VARIANT_ORDER[1:]),
        ["variant_label", "cost_bps", "delta_sharpe_ratio"],
    ].copy()
    data[["cost_bps", "delta_sharpe_ratio"]] = data[
        ["cost_bps", "delta_sharpe_ratio"]
    ].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any() or not np.isfinite(
        data[["cost_bps", "delta_sharpe_ratio"]].to_numpy()
    ).all():
        raise ValueError("Primary holdout cost deltas must be finite.")
    expected_costs = {0.0, 10.0, 50.0}
    if set(data["cost_bps"]) != expected_costs:
        raise ValueError("Primary holdout costs must be 0, 10, and 50 bps.")
    counts = data.groupby("variant_label")["cost_bps"].nunique()
    if set(counts.index) != set(VARIANT_ORDER[1:]) or not counts.eq(3).all():
        raise ValueError("Primary holdout cost comparison is incomplete.")
    if data.duplicated(["variant_label", "cost_bps"]).any():
        raise ValueError("Primary holdout cost comparison contains duplicates.")
    return data.sort_values(["cost_bps", "variant_label"]).reset_index(drop=True)


def plot_fusion_holdout_cost_gate(
    fund_returns: pd.DataFrame,
    fusion_returns: pd.DataFrame,
    fusion_costs: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot the frozen 2023 fusion comparison and its execution-cost gate."""
    growth = _primary_holdout_growth(fund_returns, fusion_returns)
    costs = _primary_holdout_cost_deltas(fusion_costs)
    figure, (growth_axis, cost_axis) = plt.subplots(
        1,
        2,
        figsize=(6.27, 4.20),
    )

    for variant in VARIANT_ORDER:
        series = growth.loc[growth["variant_label"].eq(variant)]
        growth_axis.plot(
            series["date"],
            series["holdout_growth_of_one"],
            color=VARIANT_COLORS[variant],
            linewidth=1.55,
            alpha=0.94,
            zorder=3,
        )
    growth_axis.axhline(1.0, color=GRID, linewidth=0.8)
    growth_axis.set_title(
        "Locked 2023 growth of $1",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
    )
    growth_axis.set_xlabel("Holdout date", fontsize=7.2)
    growth_axis.set_ylabel("Growth of $1", fontsize=7.2)
    growth_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"${value:.2f}")
    )
    growth_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    growth_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _style_axis(growth_axis)

    cost_levels = (0.0, 10.0, 50.0)
    positions = np.arange(len(cost_levels), dtype=float)
    width = 0.34
    for offset, variant in zip((-width / 2, width / 2), VARIANT_ORDER[1:]):
        panel = costs.loc[costs["variant_label"].eq(variant)].set_index(
            "cost_bps"
        )
        values = panel.loc[list(cost_levels), "delta_sharpe_ratio"].to_numpy()
        bars = cost_axis.bar(
            positions + offset,
            values,
            width=width,
            color=VARIANT_COLORS[variant],
            alpha=0.92,
            zorder=3,
        )
        for rectangle, value in zip(bars, values, strict=True):
            vertical_offset = 3 if value >= 0 else -9
            cost_axis.annotate(
                f"{value:+.3f}",
                xy=(
                    rectangle.get_x() + rectangle.get_width() / 2,
                    value,
                ),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=5.8,
                color=INK,
            )
    cost_axis.axhline(0, color=INK, linewidth=0.8)
    cost_axis.set_xticks(positions, ["0 bps", "10 bps", "50 bps"])
    cost_axis.set_title(
        "Sharpe change after costs",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
    )
    cost_axis.set_xlabel("One-way execution cost", fontsize=7.2)
    cost_axis.set_ylabel("Change in Sharpe vs base", fontsize=7.2)
    _style_axis(cost_axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=VARIANT_COLORS[variant],
            linewidth=2,
            label=VARIANT_LABELS[variant],
        )
        for variant in VARIANT_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.84),
        ncol=3,
        frameon=False,
        fontsize=6.5,
    )
    _add_header(
        figure,
        "Sentiment fusion must pass an execution-cost gate",
        "Frozen coverage-aware rule versus its base and a naive benchmark",
    )
    _add_footer(
        figure,
        note=(
            "The rule and 0.25 tilt were fixed before the 2023 holdout; costs "
            "apply to one-way turnover at later rebalances, excluding formation"
        ),
        sample=_date_span(growth["date"]),
        units="Dollar growth and change in annualised Sharpe ratio",
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.76,
        bottom=0.25,
        wspace=0.36,
    )
    return _save_figure(figure, output_dir, "fusion_holdout_cost_gate.png")


def _validate_manifest(manifest: pd.DataFrame, output_dir: Path) -> None:
    """Fail if the reproducible six-figure pack is incomplete."""
    required = {
        "figure_id",
        "file_name",
        "title",
        "brief_requirement",
        "sample",
        "units",
        "source",
        "generated_from",
        "file_size_bytes",
    }
    _require_columns(manifest, required, "figure_manifest")
    if len(manifest) != 6 or manifest["figure_id"].nunique() != 6:
        raise ValueError("The AssetFund figure pack must contain six figures.")
    if manifest["file_name"].duplicated().any():
        raise ValueError("The figure manifest contains duplicate filenames.")
    for row in manifest.itertuples(index=False):
        path = output_dir / row.file_name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Figure output is missing or empty: {path.name}")
        if int(row.file_size_bytes) != path.stat().st_size:
            raise ValueError(f"Figure size audit does not reconcile: {path.name}")


def build_all_figures(project_root: str | Path) -> pd.DataFrame:
    """Read saved artifacts, build six figures, and save their manifest."""
    root = Path(project_root)
    data_dir = root / "results" / "data"
    table_dir = root / "results" / "tables"
    output_dir = root / "results" / "figures"
    input_paths = {
        "fund_returns": data_dir / "fund_returns.csv",
        "fund_weights": data_dir / "fund_weights.csv",
        "sector_sentiment": data_dir / "sector_sentiment_index.csv",
        "fusion_returns": data_dir / "fusion_returns.csv",
        "performance_metrics": table_dir / "performance_metrics.csv",
        "fusion_costs": table_dir / "fusion_transaction_cost_robustness.csv",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Figure inputs are missing: {missing}")

    frames = {name: pd.read_csv(path) for name, path in input_paths.items()}
    paths = [
        plot_fund_growth_comparison(frames["fund_returns"], output_dir),
        plot_primary_fusion_drawdown(
            frames["fund_returns"],
            frames["fusion_returns"],
            output_dir,
        ),
        plot_combined_sleeve_weights(frames["fund_weights"], output_dir),
        plot_fund_risk_return_scorecard(
            frames["performance_metrics"],
            output_dir,
        ),
        plot_sector_sentiment_index(frames["sector_sentiment"], output_dir),
        plot_fusion_holdout_cost_gate(
            frames["fund_returns"],
            frames["fusion_returns"],
            frames["fusion_costs"],
            output_dir,
        ),
    ]

    fund_sample = _date_span(frames["fund_returns"]["date"])
    primary_fund_sample = _date_span(
        frames["fund_returns"].loc[
            frames["fund_returns"]["fund_id"].eq(PRIMARY_FUND_ID),
            "date",
        ]
    )
    combined_weight_sample = _date_span(
        frames["fund_weights"].loc[
            frames["fund_weights"]["universe"].eq("Combined"),
            "effective_start_date",
        ]
    )
    sentiment_sample = _date_span(frames["sector_sentiment"]["date"])
    metadata = [
        {
            "figure_id": "B01",
            "title": "How AssetFund strategies grew $1",
            "brief_requirement": "Growth-of-$1 comparison across methods",
            "sample": fund_sample,
            "units": "Dollar value on a logarithmic scale",
            "generated_from": "results/data/fund_returns.csv",
        },
        {
            "figure_id": "B02",
            "title": "Drawdown of the primary combined fund",
            "brief_requirement": "Drawdown figure for at least one fund",
            "sample": primary_fund_sample,
            "units": "Percentage below the previous peak",
            "generated_from": (
                "results/data/fund_returns.csv; "
                "results/data/fusion_returns.csv"
            ),
        },
        {
            "figure_id": "B03",
            "title": "What drives the combined funds",
            "brief_requirement": "Portfolio weights over time across methods",
            "sample": combined_weight_sample,
            "units": "Percentage of target portfolio allocation",
            "generated_from": "results/data/fund_weights.csv",
        },
        {
            "figure_id": "B04",
            "title": "The 13-fund risk-return shelf",
            "brief_requirement": "Sharpe or return-versus-risk barplot",
            "sample": fund_sample,
            "units": "Annualised percentages and Sharpe ratio",
            "generated_from": "results/tables/performance_metrics.csv",
        },
        {
            "figure_id": "B05",
            "title": "AssetFund sector sentiment surprises",
            "brief_requirement": "Equity-sector sentiment-index time series",
            "sample": sentiment_sample,
            "units": "Standard deviations from the prior expanding mean",
            "generated_from": "results/data/sector_sentiment_index.csv",
        },
        {
            "figure_id": "B06",
            "title": "Sentiment fusion execution-cost gate",
            "brief_requirement": "Fusion before-versus-after figure",
            "sample": "03 Jan 2023 to 29 Dec 2023",
            "units": "Dollar growth and change in Sharpe ratio",
            "generated_from": (
                "results/data/fund_returns.csv; "
                "results/data/fusion_returns.csv; "
                "results/tables/fusion_transaction_cost_robustness.csv"
            ),
        },
    ]
    for row, path in zip(metadata, paths, strict=True):
        row["file_name"] = path.name
        row["source"] = ASSETFUND_SOURCE
        row["file_size_bytes"] = path.stat().st_size

    columns = [
        "figure_id",
        "file_name",
        "title",
        "brief_requirement",
        "sample",
        "units",
        "source",
        "generated_from",
        "file_size_bytes",
    ]
    manifest = pd.DataFrame(metadata)[columns]
    _validate_manifest(manifest, output_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(table_dir / "figure_manifest.csv", index=False)
    return manifest
