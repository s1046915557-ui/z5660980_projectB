"""Tests for NLTK VADER scoring and the lagged sector index."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import features, sentiment


class FakeAnalyzer:
    """Deterministic scorer that also records the exact received strings."""

    def __init__(self):
        self.received = []

    def polarity_scores(self, text):
        self.received.append(text)
        if text == "GOOD!":
            return {"neg": 0.0, "neu": 0.2, "pos": 0.8, "compound": 0.75}
        return {"neg": 0.6, "neu": 0.4, "pos": 0.0, "compound": -0.5}


def test_score_headlines_preserves_text_and_vader_inputs(monkeypatch):
    """Scoring receives unchanged casing and punctuation and retains every row."""
    analyzer = FakeAnalyzer()
    monkeypatch.setattr(
        sentiment,
        "_build_vader_analyzer",
        lambda **_kwargs: analyzer,
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-02", "2021-01-03", "2021-01-03"]),
            "trading_date": pd.to_datetime(
                ["2021-01-04", "2021-01-04", "2021-01-04"]
            ),
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Finance", "Finance"],
            "text_raw": ["GOOD!", "Not good", "GOOD!"],
        }
    )

    scored = sentiment.score_headlines(panel)

    assert scored["text_raw"].tolist() == panel["text_raw"].tolist()
    assert analyzer.received == ["GOOD!", "Not good"]
    assert scored["vader_compound"].tolist() == [0.75, -0.5, 0.75]
    assert scored["vader_class"].tolist() == ["positive", "negative", "positive"]


class FakeVaderConstants:
    """Minimal isolated VADER constants container for lexicon tests."""

    def __init__(self):
        self.SPECIAL_CASE_IDIOMS = {"the shit": 3.0}


class FakeVaderAnalyzer:
    """Minimal analyser state required by the finance extension."""

    def __init__(self):
        self.lexicon = {"crude": -2.7, "money": 0.0}
        self.constants = FakeVaderConstants()


def test_finance_extension_is_opt_in_and_does_not_contaminate_baseline(monkeypatch):
    """Finance terms change only the extended analyser instance."""
    import nltk.sentiment.vader

    monkeypatch.setattr(
        nltk.sentiment.vader,
        "SentimentIntensityAnalyzer",
        FakeVaderAnalyzer,
    )

    finance = sentiment._build_vader_analyzer(finance_aware=True)
    baseline = sentiment._build_vader_analyzer(finance_aware=False)

    assert finance.lexicon["beat"] > 0
    assert finance.lexicon["halt"] < 0
    assert finance.lexicon["crude"] == 0.0
    assert finance.constants.SPECIAL_CASE_IDIOMS["starts new crude"] > 0
    assert "beat" not in baseline.lexicon
    assert baseline.lexicon["crude"] == -2.7
    assert "starts new crude" not in baseline.constants.SPECIAL_CASE_IDIOMS


def test_sector_index_equal_weights_tickers_and_neutral_fills_no_news():
    """Headline volume cannot overweight a ticker and missing names contribute zero."""
    calendar = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    universe = pd.DataFrame(
        {
            "ticker": ["T1", "T2", "F1", "F2"],
            "sector": ["Tech", "Tech", "Finance", "Finance"],
        }
    )
    scores = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(
                ["2021-01-04", "2021-01-04", "2021-01-04", "2021-01-05"]
            ),
            "ticker": ["T1", "T1", "T2", "T1"],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "vader_compound": [0.8, 0.4, -0.2, 0.2],
        }
    )

    index = sentiment.sector_sentiment_index(
        scores,
        trading_calendar=calendar,
        sector_universe=universe,
    )

    tech = index[index["sector"] == "Tech"].set_index("date")
    finance = index[index["sector"] == "Finance"].set_index("date")

    assert tech.loc[calendar[0], "sentiment_compound"] == pytest.approx(0.2)
    assert tech.loc[calendar[0], "headline_count"] == 3
    assert tech.loc[calendar[0], "coverage_rate"] == pytest.approx(1.0)
    assert tech.loc[calendar[1], "sentiment_compound"] == pytest.approx(0.1)
    assert tech.loc[calendar[1], "coverage_rate"] == pytest.approx(0.5)
    assert finance["sentiment_compound"].eq(0.0).all()
    assert finance["coverage_rate"].eq(0.0).all()


def test_sector_standardisation_uses_only_earlier_observations():
    """The expanding benchmark excludes both the current and future values."""
    compounds = np.linspace(-0.4, 0.4, 25)
    original = features.past_only_expanding_zscore(pd.Series(compounds))
    changed_compounds = compounds.copy()
    changed_compounds[-1] = -0.95
    changed = features.past_only_expanding_zscore(pd.Series(changed_compounds))

    assert original.iloc[:21].isna().all()
    expected = (compounds[21] - compounds[:21].mean()) / compounds[:21].std(
        ddof=1
    )
    assert original.iloc[21] == pytest.approx(expected)
    pd.testing.assert_series_equal(original.iloc[:-1], changed.iloc[:-1])

    constant = features.past_only_expanding_zscore(pd.Series(np.zeros(25)))
    assert constant.isna().all()


def test_ticker_signal_retains_no_news_flag_and_lags_each_name():
    """A missing ticker-day becomes zero while its prior signal remains auditable."""
    calendar = pd.to_datetime(["2021-01-04", "2021-01-05"])
    universe = pd.DataFrame(
        {"ticker": ["T1", "T2"], "sector": ["Tech", "Tech"]}
    )
    scores = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2021-01-04", "2021-01-04"]),
            "ticker": ["T1", "T2"],
            "sector": ["Tech", "Tech"],
            "vader_compound": [0.6, -0.2],
        }
    )

    signal = sentiment.ticker_sentiment_signal(
        scores,
        trading_calendar=calendar,
        sector_universe=universe,
    ).set_index(["date", "ticker"])

    assert not signal.loc[(calendar[1], "T1"), "has_news"]
    assert signal.loc[(calendar[1], "T1"), "ticker_sentiment_compound"] == 0.0
    assert signal.loc[
        (calendar[1], "T1"), "lagged_ticker_sentiment_compound"
    ] == pytest.approx(0.6)
    assert signal.loc[(calendar[1], "T1"), "signal_source_date"] == calendar[0]


def test_sector_signal_is_lagged_by_one_actual_trading_day():
    """A date's usable signal is sourced only from the prior calendar row."""
    calendar = pd.to_datetime(["2021-01-08", "2021-01-11", "2021-01-12"])
    universe = pd.DataFrame(
        {"ticker": ["T1", "T2"], "sector": ["Tech", "Tech"]}
    )
    scores = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2021-01-11", "2021-01-12"]),
            "ticker": ["T1", "T2"],
            "sector": ["Tech", "Tech"],
            "vader_compound": [0.6, -0.8],
        }
    )

    index = sentiment.sector_sentiment_index(
        scores,
        trading_calendar=calendar,
        sector_universe=universe,
    ).set_index("date")

    assert np.isnan(index.loc[calendar[0], "lagged_sentiment_compound"])
    assert index.loc[calendar[1], "lagged_sentiment_compound"] == pytest.approx(0.0)
    assert index.loc[calendar[1], "signal_source_date"] == calendar[0]
    assert index.loc[calendar[2], "lagged_sentiment_compound"] == pytest.approx(0.3)
    assert index.loc[calendar[2], "signal_source_date"] == calendar[1]
