"""Load and validate the Project B market and headline data.

These functions reuse the student's verified Project A cleaning rules. Raw data
always loads through :mod:`src.data_access` and is never written into the project.
"""

import pandas as pd

from src import data_access


def load_clean_equities():
    """Load, validate, and organise the equity price panel."""
    df = data_access.load_equity_prices().copy()

    required_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
        "sector",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Equity data is missing required columns: {missing_columns}"
        )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    duplicate_count = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Equity data contains {duplicate_count} duplicate ticker-date rows."
        )
    missing_counts = df[required_columns].isna().sum()
    columns_with_missing = missing_counts[
        missing_counts > 0
    ].to_dict()

    if columns_with_missing:
        raise ValueError(
            f"Equity data contains missing values: {columns_with_missing}"
        )

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjClose",
    ]

    nonpositive_price_rows = (
        df[price_columns] <= 0
    ).any(axis=1).sum()

    if nonpositive_price_rows > 0:
        raise ValueError(
            "Equity data contains "
            f"{nonpositive_price_rows} rows with non-positive prices."
        )

    invalid_ohlc_rows = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()

    if invalid_ohlc_rows > 0:
        raise ValueError(
            "Equity data contains "
            f"{invalid_ohlc_rows} rows with inconsistent OHLC values."
        )

    negative_volume_rows = (
        df["volume"] < 0
    ).sum()

    if negative_volume_rows > 0:
        raise ValueError(
            "Equity data contains "
            f"{negative_volume_rows} rows with negative volume."
        )

    equity_calendar = pd.Index(
        df["date"].drop_duplicates()
    )

    observations_per_ticker = (
        df.groupby("ticker")["date"]
        .nunique()
    )

    missing_dates_per_ticker = (
        len(equity_calendar)
        - observations_per_ticker
    )

    tickers_with_missing_dates = (
        missing_dates_per_ticker[
            missing_dates_per_ticker > 0
        ].to_dict()
    )

    if tickers_with_missing_dates:
        raise ValueError(
            "Equity data contains missing trading dates: "
            f"{tickers_with_missing_dates}"
        )

    df = (
        df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


def load_clean_crypto():
    """Load, validate, and organise the crypto price panel."""
    df = data_access.load_crypto_prices().copy()

    required_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Crypto data is missing required columns: {missing_columns}"
        )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    df = df.loc[
        df["date"] <= pd.Timestamp("2023-12-31")
    ].copy()

    duplicate_count = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Crypto data contains {duplicate_count} duplicate ticker-date rows."
        )

    missing_counts = df[required_columns].isna().sum()
    columns_with_missing = missing_counts[
        missing_counts > 0
    ].to_dict()

    if columns_with_missing:
        raise ValueError(
            f"Crypto data contains missing values: {columns_with_missing}"
        )

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjClose",
    ]

    nonpositive_price_rows = (
        df[price_columns] <= 0
    ).any(axis=1).sum()

    if nonpositive_price_rows > 0:
        raise ValueError(
            "Crypto data contains "
            f"{nonpositive_price_rows} rows with non-positive prices."
        )

    invalid_ohlc_rows = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()

    if invalid_ohlc_rows > 0:
        raise ValueError(
            "Crypto data contains "
            f"{invalid_ohlc_rows} rows with inconsistent OHLC values."
        )

    negative_volume_rows = (
        df["volume"] < 0
    ).sum()

    if negative_volume_rows > 0:
        raise ValueError(
            "Crypto data contains "
            f"{negative_volume_rows} rows with negative volume."
        )

    crypto_calendar = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )

    observations_per_ticker = (
        df.groupby("ticker")["date"]
        .nunique()
    )

    missing_dates_per_ticker = (
        len(crypto_calendar)
        - observations_per_ticker
    )

    tickers_with_missing_dates = (
        missing_dates_per_ticker[
            missing_dates_per_ticker > 0
        ].to_dict()
    )

    if tickers_with_missing_dates:
        raise ValueError(
            "Crypto data contains missing calendar dates: "
            f"{tickers_with_missing_dates}"
        )

    df = (
        df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


def load_clean_headlines():
    """Load, validate, deduplicate, and preserve the news headlines."""
    df = data_access.load_news_headlines().copy()

    required_columns = [
        "date",
        "ticker",
        "sector",
        "title",
        "url",
        "publisher",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"News data is missing required columns: {missing_columns}"
        )

    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    df = df.loc[
        df["date"].between(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2023-12-31"),
        )
    ].copy()

    core_columns = [
        "date",
        "ticker",
        "sector",
        "title",
        "url",
    ]

    missing_core_counts = df[core_columns].isna().sum()
    core_columns_with_missing = missing_core_counts[
        missing_core_counts > 0
    ].to_dict()

    if core_columns_with_missing:
        raise ValueError(
            "News data contains missing core values: "
            f"{core_columns_with_missing}"
        )

    empty_title_rows = (
        df["title"]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_title_rows > 0:
        raise ValueError(
            f"News data contains {empty_title_rows} empty headlines."
        )

    duplicate_key = [
        "ticker",
        "date",
        "title",
    ]

    df = df.drop_duplicates(
        subset=duplicate_key,
        keep="first",
    ).copy()

    df["text_raw"] = df["title"]

    if not df["text_raw"].equals(df["title"]):
        raise ValueError(
            "The preserved text_raw column does not match title."
        )

    df = (
        df.sort_values(["ticker", "date", "title"])
        .reset_index(drop=True)
    )

    return df
