"""Build Project B return inputs and align headlines to trading days.

The return functions reuse the student's verified Project A work. Headline
alignment is adapted for Project B so every raw headline remains a separate row
until after sentiment scoring.
"""

import numpy as np
import pandas as pd

SENTIMENT_STANDARDIZATION_MIN_PERIODS = 21
SENTIMENT_STANDARDIZATION_RULE = "past_only_expanding_mean_std_ddof_1"


def past_only_expanding_zscore(
    values: pd.Series,
    min_periods: int = SENTIMENT_STANDARDIZATION_MIN_PERIODS,
) -> pd.Series:
    """Standardise each observation against strictly earlier values.

    This is a reporting feature, not part of the frozen VADER model. A zero
    historical standard deviation remains undefined instead of being turned
    into a misleading signal.
    """
    if min_periods < 2:
        raise ValueError("min_periods must be at least two.")

    series = pd.to_numeric(pd.Series(values), errors="coerce")
    history = series.shift(1)
    expanding = history.expanding(min_periods=min_periods)
    historical_mean = expanding.mean()
    historical_std = expanding.std(ddof=1).replace(0.0, np.nan)
    standardised = (series - historical_mean) / historical_std
    return standardised.replace([np.inf, -np.inf], np.nan)


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> pd.DataFrame:
    """Calculate simple daily returns separately for each ticker."""
    required_columns = {
        "ticker",
        "date",
        price_col,
    }

    missing_columns = required_columns.difference(prices.columns)

    if missing_columns:
        raise ValueError(f"Price data is missing required columns: {sorted(missing_columns)}")

    returns = (
        prices[["ticker", "date", price_col]]
        .copy()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    returns["return"] = returns.groupby("ticker")[price_col].pct_change(fill_method=None)

    return returns


def screen_extreme_returns(
    returns: pd.DataFrame,
    asset_class: str,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Flag daily returns whose absolute value exceeds a threshold."""
    required_columns = {
        "ticker",
        "date",
        "return",
    }

    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        raise ValueError(f"Return data is missing required columns: {sorted(missing_columns)}")

    extremes = returns.loc[returns["return"].abs() > threshold].copy()

    extremes["asset_class"] = asset_class
    extremes["absolute_return"] = extremes["return"].abs()
    extremes["screen_threshold"] = threshold

    extremes = extremes.sort_values(
        "absolute_return",
        ascending=False,
    ).reset_index(drop=True)

    output_columns = [
        "asset_class",
        "ticker",
        "date",
        "return",
        "absolute_return",
        "screen_threshold",
    ]

    return extremes[output_columns]


def build_combined_returns_panel(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Align separately computed equity and crypto returns on equity dates."""
    required_columns = {"ticker", "date", "return"}

    for name, data in {
        "Equity": equity_returns,
        "Crypto": crypto_returns,
    }.items():
        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            raise ValueError(
                f"{name} returns are missing required columns: {sorted(missing_columns)}"
            )

        if data.duplicated(["ticker", "date"]).any():
            raise ValueError(f"{name} returns contain duplicate ticker-date rows.")

    equity_wide = equity_returns.pivot(
        index="date",
        columns="ticker",
        values="return",
    ).sort_index()

    crypto_wide = crypto_returns.pivot(
        index="date",
        columns="ticker",
        values="return",
    ).sort_index()

    combined = equity_wide.join(crypto_wide, how="left").reset_index()

    combined.columns.name = None

    return combined


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_dates: pd.Series,
) -> pd.DataFrame:
    """Map each headline forward to an equity trading day without aggregating.

    Project B scores each raw headline before calculating ticker-day sentiment.
    The returned frame therefore retains one row per cleaned headline, preserves
    ``text_raw``, and adds ``trading_date`` plus an alignment flag.
    """
    required_columns = {
        "date",
        "ticker",
        "sector",
        "text_raw",
    }

    missing_columns = required_columns.difference(headlines.columns)

    if missing_columns:
        raise ValueError(f"Headline data is missing required columns: {sorted(missing_columns)}")

    headline_data = headlines.copy()
    headline_data["date"] = (
        pd.to_datetime(headline_data["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    calendar = (
        pd.to_datetime(pd.Series(equity_dates), utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if calendar.empty:
        raise ValueError("The equity trading calendar is empty.")

    last_headline_date = headline_data["date"].max()
    last_equity_date = calendar.max()

    if last_headline_date > last_equity_date:
        known_sample_end = pd.Timestamp("2023-12-31")
        next_equity_trading_day = pd.Timestamp("2024-01-02")

        if (
            last_equity_date == pd.Timestamp("2023-12-29")
            and last_headline_date <= known_sample_end
        ):
            calendar = pd.concat(
                [calendar, pd.Series([next_equity_trading_day])],
                ignore_index=True,
            )
        else:
            raise ValueError("Headline dates extend beyond the available equity calendar.")

    calendar_frame = pd.DataFrame({"trading_date": calendar})
    mapped = pd.merge_asof(
        headline_data.sort_values("date"),
        calendar_frame,
        left_on="date",
        right_on="trading_date",
        direction="forward",
    )

    unmapped_count = mapped["trading_date"].isna().sum()

    if unmapped_count:
        raise ValueError(
            f"{unmapped_count} headlines occur after the final available equity trading date."
        )

    if len(mapped) != len(headline_data):
        raise ValueError("Headline alignment changed the number of rows.")

    mapped["moved_to_next_trading_day"] = mapped["date"] != mapped["trading_date"]

    if not mapped["text_raw"].equals(mapped["title"]):
        raise ValueError("Headline alignment changed the preserved raw text.")

    return (
        mapped.sort_values(["trading_date", "ticker", "date", "title"])
        .reset_index(drop=True)
    )
