"""Tests for the Project B data and feature foundation."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl, features


@pytest.fixture(scope="module")
def equities():
    """Load the verified equity panel once for this test module."""
    return etl.load_clean_equities()


@pytest.fixture(scope="module")
def crypto():
    """Load the verified crypto panel once for this test module."""
    return etl.load_clean_crypto()


@pytest.fixture(scope="module")
def headlines():
    """Load the verified headline panel once for this test module."""
    return etl.load_clean_headlines()


def test_clean_dataset_contracts(equities, crypto, headlines):
    """Clean row counts, coverage, dates, and keys match Project A evidence."""
    assert len(equities) == 50_300
    assert equities["ticker"].nunique() == 50
    assert equities["date"].nunique() == 1_006
    assert not equities.duplicated(["ticker", "date"]).any()

    assert len(crypto) == 14_610
    assert crypto["ticker"].nunique() == 10
    assert crypto["date"].max() == pd.Timestamp("2023-12-31")
    assert not crypto.duplicated(["ticker", "date"]).any()

    assert len(headlines) == 146_836
    assert headlines["ticker"].nunique() == 50
    assert headlines["sector"].nunique() == 10
    assert not headlines.duplicated(["ticker", "date", "title"]).any()
    assert headlines["text_raw"].equals(headlines["title"])


def test_returns_are_computed_within_ticker(equities, crypto):
    """Only the first observation of each ticker has an undefined return."""
    equity_returns = features.daily_returns(equities)
    crypto_returns = features.daily_returns(crypto)

    assert equity_returns["return"].isna().sum() == 50
    assert crypto_returns["return"].isna().sum() == 10

    equity_first = equity_returns.groupby("ticker", sort=False).head(1)
    crypto_first = crypto_returns.groupby("ticker", sort=False).head(1)
    assert equity_first["return"].isna().all()
    assert crypto_first["return"].isna().all()


def test_combined_returns_use_equity_calendar(equities, crypto):
    """The combined panel has 60 assets on 1,006 equity trading dates."""
    equity_returns = features.daily_returns(equities)
    crypto_returns = features.daily_returns(crypto)
    combined = features.build_combined_returns_panel(
        equity_returns,
        crypto_returns,
    )

    assert combined.shape == (1_006, 61)
    assert combined["date"].is_unique
    assert combined["date"].equals(
        equities["date"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    crypto_columns = sorted(crypto["ticker"].unique())
    assert not combined[crypto_columns].isna().any().any()


def test_headlines_remain_individual_before_scoring(equities, headlines):
    """Alignment preserves every headline and does not aggregate the text."""
    aligned = features.assemble_headline_panel(
        headlines,
        equities["date"],
    )

    assert len(aligned) == len(headlines)
    assert aligned["text_raw"].equals(aligned["title"])
    assert aligned["trading_date"].notna().all()
    assert (aligned["trading_date"] >= aligned["date"]).all()
    assert aligned["moved_to_next_trading_day"].sum() == 12_557

    year_end = aligned.loc[
        aligned["date"] > pd.Timestamp("2023-12-29")
    ]
    assert len(year_end) == 6
    assert year_end["trading_date"].eq(pd.Timestamp("2024-01-02")).all()


def test_weekend_headline_moves_to_next_trading_day():
    """A Saturday headline maps forward to Monday, never backward to Friday."""
    headlines = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-06"), pd.Timestamp("2023-01-07")],
            "ticker": ["TEST", "TEST"],
            "sector": ["Example", "Example"],
            "title": ["Friday headline", "Saturday headline"],
            "text_raw": ["Friday headline", "Saturday headline"],
        }
    )
    equity_dates = pd.Series(
        [pd.Timestamp("2023-01-06"), pd.Timestamp("2023-01-09")]
    )

    aligned = features.assemble_headline_panel(headlines, equity_dates)
    mapped_dates = aligned.set_index("title")["trading_date"]

    assert mapped_dates["Friday headline"] == pd.Timestamp("2023-01-06")
    assert mapped_dates["Saturday headline"] == pd.Timestamp("2023-01-09")
