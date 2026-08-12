"""NLTK VADER headline scoring and a lagged sector sentiment index."""

from __future__ import annotations

import numpy as np
import pandas as pd

VADER_THRESHOLD = 0.05
NO_NEWS_RULE = "neutral_zero_with_coverage_flag"

# A deliberately small finance extension. The development review showed that
# broad market words such as "rally" and "recovery" can create target-mismatch
# errors, so only high-precision terms are added here.
FINANCE_VADER_LEXICON = {
    "beat": 1.9,
    "beats": 1.9,
    "beating": 1.9,
    "outperform": 2.0,
    "outperforms": 2.0,
    "outperformed": 2.0,
    "underperform": -2.0,
    "underperforms": -2.0,
    "underperformed": -2.0,
    "upgrade": 1.8,
    "upgrades": 1.8,
    "upgraded": 1.8,
    "downgrade": -1.8,
    "downgrades": -1.8,
    "downgraded": -1.8,
    "bullish": 1.8,
    "bearish": -1.8,
    "buy": 2.0,
    "buys": 2.0,
    "halt": -1.5,
    "bet": 1.0,
    "crude": 0.0,
    "money": 0.0,
}
FINANCE_VADER_IDIOMS = {
    "starts new crude": 1.5,
}


def _build_vader_analyzer(*, finance_aware: bool = True):
    """Create NLTK VADER with an optional conservative finance extension."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        if finance_aware:
            analyzer.lexicon.update(FINANCE_VADER_LEXICON)
            # Assign an instance copy so a finance-aware analyser cannot
            # contaminate a later baseline analyser through NLTK class state.
            analyzer.constants.SPECIAL_CASE_IDIOMS = {
                **analyzer.constants.SPECIAL_CASE_IDIOMS,
                **FINANCE_VADER_IDIOMS,
            }
        return analyzer
    except LookupError as exc:
        raise RuntimeError(
            "NLTK's vader_lexicon is unavailable. Run "
            "nltk.download('vader_lexicon') once before the build."
        ) from exc


def _required_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error when an input table is missing required fields."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _normalise_dates(values: pd.Series) -> pd.Series:
    """Return timezone-naive, normalised dates for joins and grouping."""
    return (
        pd.to_datetime(pd.Series(values), utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )


def _classify_compound(compound: float) -> str:
    """Apply the stated conventional VADER direction thresholds."""
    if compound >= VADER_THRESHOLD:
        return "positive"
    if compound <= -VADER_THRESHOLD:
        return "negative"
    return "neutral"


def score_headlines(
    panel: pd.DataFrame,
    *,
    finance_aware: bool = True,
) -> pd.DataFrame:
    """Score every preserved raw headline with NLTK VADER.

    Casing, punctuation, negation, and the original text are left unchanged.
    Identical headline strings are scored once and mapped back to every row.
    Set ``finance_aware=False`` to reproduce the unmodified VADER benchmark.
    """
    required = {
        "date",
        "trading_date",
        "ticker",
        "sector",
        "text_raw",
    }
    _required_columns(panel, required, "Headline panel")
    if panel.empty:
        raise ValueError("Headline panel cannot be empty.")
    if panel[list(required)].isna().any().any():
        raise ValueError("Headline panel contains missing required values.")

    output = panel.copy().reset_index(drop=True)
    original_text = output["text_raw"].astype(str).copy()
    if original_text.str.strip().eq("").any():
        raise ValueError("Headline panel contains blank text.")
    output["text_raw"] = original_text
    output["date"] = _normalise_dates(output["date"])
    output["trading_date"] = _normalise_dates(output["trading_date"])

    analyzer = _build_vader_analyzer(finance_aware=finance_aware)
    unique_text = original_text.drop_duplicates().reset_index(drop=True)
    score_rows = []
    for text in unique_text:
        scores = analyzer.polarity_scores(text)
        score_rows.append(
            {
                "text_raw": text,
                "vader_neg": float(scores["neg"]),
                "vader_neu": float(scores["neu"]),
                "vader_pos": float(scores["pos"]),
                "vader_compound": float(scores["compound"]),
            }
        )

    output["_row_order"] = np.arange(len(output))
    output = output.merge(
        pd.DataFrame(score_rows),
        on="text_raw",
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order")
    output = output.drop(columns="_row_order").reset_index(drop=True)

    proportions = output[["vader_neg", "vader_neu", "vader_pos"]]
    if not proportions.ge(0).all().all() or not proportions.le(1).all().all():
        raise ValueError("VADER proportions fall outside [0, 1].")
    if not np.allclose(proportions.sum(axis=1), 1.0, atol=0.002):
        raise ValueError("VADER negative, neutral, and positive shares do not sum to one.")
    if not output["vader_compound"].between(-1.0, 1.0).all():
        raise ValueError("VADER compound score falls outside [-1, 1].")
    if not output["text_raw"].equals(original_text):
        raise ValueError("VADER scoring changed the preserved headline text.")

    output["vader_class"] = output["vader_compound"].map(_classify_compound)
    return output


def _ticker_day_sentiment(scores: pd.DataFrame) -> pd.DataFrame:
    """Average headlines within ticker-day before sector aggregation."""
    required = {"trading_date", "ticker", "sector", "vader_compound"}
    _required_columns(scores, required, "Headline scores")

    ticker_sector_counts = scores.groupby("ticker")["sector"].nunique()
    if ticker_sector_counts.gt(1).any():
        raise ValueError("A ticker maps to more than one sector in the scores.")

    data = scores.copy()
    data["trading_date"] = _normalise_dates(data["trading_date"])
    return (
        data.groupby(
            ["trading_date", "ticker", "sector"],
            as_index=False,
            observed=True,
        )
        .agg(
            ticker_day_compound=("vader_compound", "mean"),
            headline_count=("vader_compound", "size"),
        )
        .sort_values(["trading_date", "sector", "ticker"])
        .reset_index(drop=True)
    )


def _resolve_sector_universe(
    scores: pd.DataFrame,
    sector_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate or infer one sector assignment for every equity ticker."""
    if sector_universe is None:
        universe = scores[["ticker", "sector"]].drop_duplicates().copy()
    else:
        _required_columns(
            sector_universe,
            {"ticker", "sector"},
            "Sector universe",
        )
        universe = sector_universe[["ticker", "sector"]].drop_duplicates().copy()

    if universe.empty or universe[["ticker", "sector"]].isna().any().any():
        raise ValueError("Sector universe must contain ticker-sector mappings.")
    if universe["ticker"].duplicated().any():
        raise ValueError("Each ticker must appear exactly once in the sector universe.")

    score_mapping = scores[["ticker", "sector"]].drop_duplicates()
    mapping_check = score_mapping.merge(
        universe,
        on="ticker",
        how="left",
        suffixes=("_score", "_universe"),
        validate="many_to_one",
    )
    if mapping_check["sector_universe"].isna().any() or not (
        mapping_check["sector_score"] == mapping_check["sector_universe"]
    ).all():
        raise ValueError("Headline scores and sector universe do not agree.")

    return universe.sort_values(["sector", "ticker"]).reset_index(drop=True)


def _resolve_trading_calendar(
    scores: pd.DataFrame,
    trading_calendar: pd.Series | pd.Index | None,
) -> pd.Series:
    """Validate or infer a sorted equity trading calendar."""
    if trading_calendar is None:
        calendar = _normalise_dates(scores["trading_date"])
    else:
        calendar = _normalise_dates(pd.Series(trading_calendar))
    calendar = calendar.drop_duplicates().sort_values().reset_index(drop=True)
    if calendar.empty:
        raise ValueError("Trading calendar cannot be empty.")
    return calendar


def ticker_sentiment_signal(
    scores: pd.DataFrame,
    *,
    trading_calendar: pd.Series | pd.Index | None = None,
    sector_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build complete neutral-fill ticker signals and lag them one trading day."""
    ticker_day = _ticker_day_sentiment(scores)
    universe = _resolve_sector_universe(scores, sector_universe)
    calendar = _resolve_trading_calendar(scores, trading_calendar)

    grid = pd.MultiIndex.from_product(
        [calendar, universe["ticker"]],
        names=["date", "ticker"],
    ).to_frame(index=False)
    grid = grid.merge(universe, on="ticker", how="left", validate="many_to_one")

    ticker_day = ticker_day.rename(columns={"trading_date": "date"})
    ticker_day = ticker_day.loc[ticker_day["date"].isin(calendar)].copy()
    grid = grid.merge(
        ticker_day,
        on=["date", "ticker", "sector"],
        how="left",
        validate="one_to_one",
    )
    grid["has_news"] = grid["headline_count"].notna()
    grid["headline_count"] = grid["headline_count"].fillna(0).astype(int)
    grid["ticker_sentiment_compound"] = grid["ticker_day_compound"].fillna(0.0)
    grid["ticker_sentiment_score_100"] = (
        grid["ticker_sentiment_compound"] + 1.0
    ) * 50.0

    grid = grid.sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = grid.groupby("ticker", sort=False)
    grid["signal_source_date"] = by_ticker["date"].shift(1)
    grid["lagged_ticker_sentiment_compound"] = by_ticker[
        "ticker_sentiment_compound"
    ].shift(1)
    grid["lagged_ticker_sentiment_score_100"] = (
        grid["lagged_ticker_sentiment_compound"] + 1.0
    ) * 50.0
    grid["lagged_has_news"] = by_ticker["has_news"].shift(1)
    grid["signal_available"] = grid[
        "lagged_ticker_sentiment_compound"
    ].notna()
    grid["no_news_rule"] = NO_NEWS_RULE
    grid["lag_trading_days"] = 1

    if not grid["ticker_sentiment_compound"].between(-1.0, 1.0).all():
        raise ValueError("Ticker sentiment falls outside [-1, 1].")

    return grid.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)


def sector_sentiment_index(
    scores: pd.DataFrame,
    *,
    trading_calendar: pd.Series | pd.Index | None = None,
    sector_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a ticker-equal, neutral-fill sector index and lag it one day.

    The original one-argument starter call remains valid. The formal build passes
    the complete equity calendar and ticker-sector universe so days without any
    headlines remain observable rather than silently disappearing.
    """
    ticker_signal = ticker_sentiment_signal(
        scores,
        trading_calendar=trading_calendar,
        sector_universe=sector_universe,
    )

    index = (
        ticker_signal.groupby(["date", "sector"], as_index=False, observed=True)
        .agg(
            sentiment_compound=("ticker_sentiment_compound", "mean"),
            news_only_compound=("ticker_day_compound", "mean"),
            headline_count=("headline_count", "sum"),
            news_ticker_count=("has_news", "sum"),
            sector_ticker_count=("ticker", "nunique"),
            signal_source_date=("signal_source_date", "first"),
            lagged_sentiment_compound=(
                "lagged_ticker_sentiment_compound",
                "mean",
            ),
            lagged_coverage_rate=("lagged_has_news", "mean"),
        )
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )
    index["coverage_rate"] = (
        index["news_ticker_count"] / index["sector_ticker_count"]
    )
    index["sentiment_score_100"] = (index["sentiment_compound"] + 1.0) * 50.0
    index["lagged_sentiment_score_100"] = (
        index["lagged_sentiment_compound"] + 1.0
    ) * 50.0
    index["signal_available"] = index["lagged_sentiment_compound"].notna()
    index["no_news_rule"] = NO_NEWS_RULE
    index["lag_trading_days"] = 1

    if not index["sentiment_compound"].between(-1.0, 1.0).all():
        raise ValueError("Sector sentiment falls outside [-1, 1].")
    if not index["coverage_rate"].between(0.0, 1.0).all():
        raise ValueError("Sector news coverage falls outside [0, 1].")

    return index.sort_values(["date", "sector"]).reset_index(drop=True)
