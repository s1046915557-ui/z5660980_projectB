"""Investor allocation, fee, and look-through calculations for AssetFund."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

DEFAULT_ANNUAL_MANAGEMENT_FEE = 0.005
CALENDAR_DAYS_PER_YEAR = 365.25


def _validated_allocations(
    allocations: Mapping[str, float],
    available_fund_ids: set[str],
) -> pd.Series:
    """Return positive decimal allocations after strict contract checks."""
    if not allocations:
        raise ValueError("Select at least two funds for an allocation.")

    try:
        weights = pd.Series(dict(allocations), dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Fund allocations must be numeric.") from exc

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Fund allocations must be finite.")
    if (weights < 0.0).any():
        raise ValueError("Fund allocations cannot be negative.")

    unknown = set(weights.index).difference(available_fund_ids)
    if unknown:
        raise KeyError(f"Unknown fund IDs: {sorted(unknown)}")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Fund allocations must sum to 100%.")

    positive = weights.loc[weights.gt(0.0)]
    if len(positive) < 2:
        raise ValueError("Allocate a positive amount to at least two funds.")
    return positive


def build_allocation_history(
    fund_returns: pd.DataFrame,
    allocations: Mapping[str, float],
    *,
    annual_management_fee: float = DEFAULT_ANNUAL_MANAGEMENT_FEE,
    initial_investment: float = 10_000.0,
) -> pd.DataFrame:
    """Build a calendar-safe buy-and-hold blend with fee-adjusted wealth.

    Allocations are decimal initial weights. Each selected fund is bought once;
    its weight can then drift with performance. Missing observations inside the
    common live sample mean that fund's published value was unchanged that day.
    The annual fee is accrued by actual elapsed calendar days, not row count.
    """
    required = {"date", "fund_id", "return"}
    missing = required.difference(fund_returns.columns)
    if missing:
        raise ValueError(f"Fund returns are missing columns: {sorted(missing)}")
    if fund_returns.empty:
        raise ValueError("Fund returns cannot be empty.")
    if not np.isfinite(float(annual_management_fee)):
        raise ValueError("Annual management fee must be finite.")
    if not 0.0 <= annual_management_fee < 1.0:
        raise ValueError("Annual management fee must lie in [0, 1).")
    if not np.isfinite(float(initial_investment)) or initial_investment <= 0.0:
        raise ValueError("Initial investment must be positive and finite.")

    available = set(fund_returns["fund_id"].astype(str))
    weights = _validated_allocations(allocations, available)
    selected_ids = weights.index.tolist()

    selected = fund_returns.loc[
        fund_returns["fund_id"].astype(str).isin(selected_ids),
        ["date", "fund_id", "return"],
    ].copy()
    selected["fund_id"] = selected["fund_id"].astype(str)
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected["return"] = pd.to_numeric(selected["return"], errors="coerce")

    if selected["date"].isna().any():
        raise ValueError("Fund returns contain invalid dates.")
    if selected.duplicated(["fund_id", "date"]).any():
        raise ValueError("Fund returns contain duplicate fund-date rows.")
    values = selected["return"].to_numpy()
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("Fund returns must be finite and greater than -100%.")

    bounds = selected.groupby("fund_id", observed=True)["date"].agg(
        ["min", "max"]
    )
    common_start = bounds["min"].max()
    common_end = bounds["max"].min()
    if common_start > common_end:
        raise ValueError("Selected funds do not have a common live sample.")

    common = selected.loc[selected["date"].between(common_start, common_end)]
    panel = (
        common.pivot(index="date", columns="fund_id", values="return")
        .reindex(columns=selected_ids)
        .sort_index()
        .fillna(0.0)
    )
    if panel.empty or len(panel) < 2:
        raise ValueError("Allocation history needs at least two valuation dates.")

    fund_growth = (1.0 + panel).cumprod()
    gross_growth = fund_growth.mul(weights, axis="columns").sum(axis=1)
    elapsed_days = (gross_growth.index - gross_growth.index[0]).days
    fee_factor = (1.0 - annual_management_fee) ** (
        elapsed_days / CALENDAR_DAYS_PER_YEAR
    )
    net_growth = gross_growth * fee_factor

    history = pd.DataFrame(
        {
            "date": gross_growth.index,
            "gross_growth_of_one": gross_growth.to_numpy(),
            "net_growth_of_one": net_growth.to_numpy(),
            "fee_factor": fee_factor,
        }
    )
    history["gross_return"] = history["gross_growth_of_one"].pct_change()
    history["net_return"] = history["net_growth_of_one"].pct_change()
    history.loc[0, "gross_return"] = history.loc[0, "gross_growth_of_one"] - 1.0
    history.loc[0, "net_return"] = history.loc[0, "net_growth_of_one"] - 1.0
    history["gross_value"] = history["gross_growth_of_one"] * initial_investment
    history["net_value"] = history["net_growth_of_one"] * initial_investment
    history["cumulative_fee_drag"] = (
        history["gross_value"] - history["net_value"]
    )
    history["net_drawdown"] = (
        history["net_growth_of_one"]
        / history["net_growth_of_one"].cummax()
        - 1.0
    )
    return history


def allocation_metrics(history: pd.DataFrame) -> pd.Series:
    """Summarise one fee-adjusted allocation over its realised OOS sample."""
    required = {
        "date",
        "net_return",
        "net_growth_of_one",
        "net_value",
        "gross_value",
        "cumulative_fee_drag",
        "net_drawdown",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Allocation history is missing columns: {sorted(missing)}")
    if len(history) < 2:
        raise ValueError("Allocation metrics need at least two valuation dates.")

    dates = pd.to_datetime(history["date"], errors="coerce")
    elapsed_days = int((dates.iloc[-1] - dates.iloc[0]).days)
    if dates.isna().any() or elapsed_days <= 0:
        raise ValueError("Allocation history must have increasing valid dates.")

    elapsed_years = elapsed_days / CALENDAR_DAYS_PER_YEAR
    periods_per_year = len(history) / elapsed_years
    ending_growth = float(history["net_growth_of_one"].iloc[-1])
    annualized_return = ending_growth ** (1.0 / elapsed_years) - 1.0
    annualized_volatility = float(
        history["net_return"].std(ddof=1) * np.sqrt(periods_per_year)
    )
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0.0
        else np.nan
    )

    return pd.Series(
        {
            "total_return": ending_growth - 1.0,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": float(history["net_drawdown"].min()),
            "ending_gross_value": float(history["gross_value"].iloc[-1]),
            "ending_net_value": float(history["net_value"].iloc[-1]),
            "cumulative_fee_drag": float(
                history["cumulative_fee_drag"].iloc[-1]
            ),
            "elapsed_years": elapsed_years,
            "observations": len(history),
        }
    )


def latest_lookthrough_holdings(
    fund_weights: pd.DataFrame,
    allocations: Mapping[str, float],
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Aggregate the latest constituent targets behind a fund allocation."""
    required = {"effective_start_date", "fund_id", "ticker", "target_weight"}
    missing = required.difference(fund_weights.columns)
    if missing:
        raise ValueError(f"Fund weights are missing columns: {sorted(missing)}")
    if fund_weights.empty:
        raise ValueError("Fund weights cannot be empty.")

    data = fund_weights.loc[:, sorted(required)].copy()
    data["fund_id"] = data["fund_id"].astype(str)
    data["effective_start_date"] = pd.to_datetime(
        data["effective_start_date"], errors="coerce"
    )
    data["target_weight"] = pd.to_numeric(
        data["target_weight"], errors="coerce"
    )
    if data["effective_start_date"].isna().any():
        raise ValueError("Fund weights contain invalid effective dates.")
    if not np.isfinite(data["target_weight"].to_numpy()).all():
        raise ValueError("Fund target weights must be finite.")
    if (data["target_weight"] < 0.0).any():
        raise ValueError("Fund target weights cannot be negative.")

    weights = _validated_allocations(allocations, set(data["fund_id"]))
    latest_blocks = []
    latest_dates = []
    for fund_id, fund_allocation in weights.items():
        fund = data.loc[data["fund_id"].eq(fund_id)].copy()
        latest_date = fund["effective_start_date"].max()
        latest = fund.loc[fund["effective_start_date"].eq(latest_date)].copy()
        if latest.duplicated("ticker").any():
            raise ValueError(f"Latest holdings for {fund_id} contain duplicates.")
        if not np.isclose(latest["target_weight"].sum(), 1.0, atol=1e-8):
            raise ValueError(f"Latest holdings for {fund_id} do not sum to one.")
        latest["lookthrough_weight"] = (
            latest["target_weight"] * fund_allocation
        )
        latest_blocks.append(latest)
        latest_dates.append(latest_date)

    combined = pd.concat(latest_blocks, ignore_index=True)
    combined["asset_class"] = np.where(
        combined["ticker"].str.endswith("-USD"),
        "Crypto",
        "Equity",
    )
    lookthrough = (
        combined.groupby(["ticker", "asset_class"], observed=True)
        .agg(
            lookthrough_weight=("lookthrough_weight", "sum"),
            fund_count=("fund_id", "nunique"),
            source_funds=(
                "fund_id",
                lambda values: ", ".join(sorted(set(values))),
            ),
        )
        .reset_index()
        .sort_values("lookthrough_weight", ascending=False)
        .reset_index(drop=True)
    )
    if not np.isclose(lookthrough["lookthrough_weight"].sum(), 1.0, atol=1e-8):
        raise ValueError("Look-through holdings do not sum to one.")

    conservative_as_of = min(latest_dates)
    return pd.Timestamp(conservative_as_of), lookthrough
