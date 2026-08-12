"""Look-ahead-safe sentiment tilts for equity-containing funds.

The primary rule is fixed before inspecting real fusion returns. It converts
the lagged sector sentiment cross-section to a bounded rank signal, shrinks the
signal by news coverage, and preserves the base fund's equity and crypto sleeve
weights. A simpler raw-sentiment rule is retained as a transparent benchmark.
"""

import numpy as np
import pandas as pd

from . import portfolios

PRIMARY_FUSION_RULE = "coverage_rank"
PRIMARY_TILT_STRENGTH = 0.25
NAIVE_FUSION_RULE = "naive"
NAIVE_TILT_STRENGTH = 1.0
FUSION_HOLDOUT_START = "2023-01-01"
FUSION_RULES = (NAIVE_FUSION_RULE, PRIMARY_FUSION_RULE)

_WEIGHT_COLUMNS = {
    "ticker",
    "target_weight",
    "asset_class",
    "sector",
    "decision_date",
    "effective_start_date",
}
_SENTIMENT_COLUMNS = {
    "date",
    "sector",
    "signal_source_date",
    "lagged_sentiment_compound",
    "lagged_coverage_rate",
    "signal_available",
}
_ASSET_CLASSES = {"Equity", "Crypto"}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    frame_name: str,
) -> None:
    """Raise a clear error when an input does not meet the data contract."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing)}"
        )


def _single_date(values: pd.Series, column: str) -> pd.Timestamp:
    """Return one normalised non-missing date from an input column."""
    dates = pd.to_datetime(values, errors="raise").dt.normalize()
    if dates.isna().any() or dates.nunique() != 1:
        raise ValueError(f"{column} must contain exactly one non-missing date.")
    return pd.Timestamp(dates.iloc[0])


def _validate_weight_frame(weights: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Validate one base target-weight vector and return its equity sleeve."""
    _require_columns(weights, _WEIGHT_COLUMNS, "weights")
    if weights.empty:
        raise ValueError("weights cannot be empty.")

    frame = weights.copy()
    if frame["ticker"].isna().any() or frame["ticker"].duplicated().any():
        raise ValueError("weights must contain one non-missing row per ticker.")
    if not frame["asset_class"].isin(_ASSET_CLASSES).all():
        raise ValueError("asset_class must contain only Equity or Crypto.")

    frame["target_weight"] = pd.to_numeric(
        frame["target_weight"],
        errors="raise",
    ).astype(float)
    target = frame["target_weight"].to_numpy()
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("target_weight must be finite and non-negative.")
    if not np.isclose(target.sum(), 1.0, atol=1e-10):
        raise ValueError("Base target weights must sum to one.")

    decision_date = _single_date(frame["decision_date"], "decision_date")
    effective_date = _single_date(
        frame["effective_start_date"],
        "effective_start_date",
    )
    if decision_date >= effective_date:
        raise ValueError("decision_date must be before effective_start_date.")
    frame["decision_date"] = decision_date
    frame["effective_start_date"] = effective_date

    equity = frame["asset_class"].eq("Equity")
    if not equity.any():
        raise ValueError("Sentiment fusion requires positive equity exposure.")
    if frame.loc[equity, "sector"].isna().any():
        raise ValueError("Every equity ticker must have a sector.")
    if frame.loc[equity, "sector"].astype(str).str.strip().eq("").any():
        raise ValueError("Every equity ticker must have a non-empty sector.")

    equity_weight = float(frame.loc[equity, "target_weight"].sum())
    if equity_weight <= 0:
        raise ValueError("Sentiment fusion requires positive equity weight.")
    return frame, equity_weight


def _validate_sentiment_frame(
    sentiment: pd.DataFrame,
    equity_sectors: set[str],
    decision_date: pd.Timestamp,
) -> pd.DataFrame:
    """Validate the complete lagged sector cross-section for one decision."""
    _require_columns(sentiment, _SENTIMENT_COLUMNS, "sentiment")
    if sentiment.empty:
        raise ValueError("sentiment cannot be empty.")

    frame = sentiment.copy()
    if frame["sector"].isna().any() or frame["sector"].duplicated().any():
        raise ValueError("sentiment must contain one non-missing row per sector.")
    sentiment_sectors = set(frame["sector"].astype(str))
    if sentiment_sectors != equity_sectors:
        raise ValueError(
            "sentiment sectors must exactly match the equity-sector universe."
        )

    sentiment_date = _single_date(frame["date"], "sentiment date")
    if sentiment_date != decision_date:
        raise ValueError("Sentiment must be observed on the portfolio decision date.")
    frame["date"] = sentiment_date
    frame["signal_source_date"] = pd.to_datetime(
        frame["signal_source_date"],
        errors="raise",
    ).dt.normalize()
    if frame["signal_source_date"].isna().any():
        raise ValueError("Every sector must have a lagged signal source date.")
    if not frame["signal_source_date"].lt(sentiment_date).all():
        raise ValueError("Every signal_source_date must precede the decision date.")

    if not pd.api.types.is_bool_dtype(frame["signal_available"]):
        raise TypeError("signal_available must be boolean.")
    if not frame["signal_available"].all():
        raise ValueError("Every sector signal must be available at the decision date.")

    for column in ["lagged_sentiment_compound", "lagged_coverage_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{column} must contain finite values.")
    if not frame["lagged_sentiment_compound"].between(-1.0, 1.0).all():
        raise ValueError("lagged_sentiment_compound must lie in [-1, 1].")
    if not frame["lagged_coverage_rate"].between(0.0, 1.0).all():
        raise ValueError("lagged_coverage_rate must lie in [0, 1].")

    return frame


def _coverage_rank_signal(sentiment: pd.DataFrame) -> pd.Series:
    """Return a coverage-shrunk sector rank scaled to the interval [-1, 1]."""
    count = len(sentiment)
    if count == 1:
        relative_rank = pd.Series(0.0, index=sentiment.index)
    else:
        ranks = sentiment["lagged_sentiment_compound"].rank(method="average")
        relative_rank = 2.0 * (ranks - 1.0) / (count - 1.0) - 1.0
    return relative_rank * sentiment["lagged_coverage_rate"]


def _sector_signal(sentiment: pd.DataFrame, rule: str) -> pd.Series:
    """Calculate either the benchmark or primary sector-level signal."""
    if rule == NAIVE_FUSION_RULE:
        return sentiment["lagged_sentiment_compound"].copy()
    if rule == PRIMARY_FUSION_RULE:
        return _coverage_rank_signal(sentiment)
    raise ValueError(f"rule must be one of {FUSION_RULES}.")


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    *,
    rule: str = PRIMARY_FUSION_RULE,
    tilt_strength: float = PRIMARY_TILT_STRENGTH,
) -> pd.DataFrame:
    """Apply one lagged sector-sentiment tilt to one base weight vector.

    The primary rule ranks the complete sector sentiment cross-section, scales
    the rank to ``[-1, 1]``, and shrinks it by the share of sector tickers with
    news. The multiplier is ``1 + tilt_strength * sector_signal``. Equity
    weights are then renormalised back to the original equity sleeve, while
    crypto weights remain unchanged. The naive benchmark uses the raw lagged
    compound score as ``sector_signal``.

    Parameters
    ----------
    weights:
        One fully invested target vector with explicit asset class, equity
        sector, decision date, and effective start date.
    sentiment:
        One complete sector cross-section observed on the decision date. Every
        source date must be earlier than that decision date.
    rule:
        ``"coverage_rank"`` for the primary rule or ``"naive"`` for the raw
        sentiment benchmark.
    tilt_strength:
        A pre-specified value in ``[0, 1]``. The primary value is 0.25.
    """
    if rule not in FUSION_RULES:
        raise ValueError(f"rule must be one of {FUSION_RULES}.")
    if not np.isfinite(tilt_strength) or not 0.0 <= tilt_strength <= 1.0:
        raise ValueError("tilt_strength must be finite and lie in [0, 1].")

    base, equity_weight = _validate_weight_frame(weights)
    decision_date = pd.Timestamp(base["decision_date"].iloc[0])
    equity = base["asset_class"].eq("Equity")
    equity_sectors = set(base.loc[equity, "sector"].astype(str))
    sector_data = _validate_sentiment_frame(
        sentiment,
        equity_sectors,
        decision_date,
    )
    sector_data["sector_signal"] = _sector_signal(sector_data, rule)
    sector_data["sentiment_multiplier"] = (
        1.0 + tilt_strength * sector_data["sector_signal"]
    )
    if (sector_data["sentiment_multiplier"] < 0).any():
        raise ValueError("The sentiment rule produced a negative multiplier.")

    signal_columns = [
        "sector",
        "date",
        "signal_source_date",
        "lagged_sentiment_compound",
        "lagged_coverage_rate",
        "sector_signal",
        "sentiment_multiplier",
    ]
    result = base.rename(columns={"target_weight": "base_target_weight"})
    result = result.merge(
        sector_data[signal_columns].rename(columns={"date": "sentiment_date"}),
        on="sector",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result["target_weight"] = result["base_target_weight"]
    result["fusion_rule"] = rule
    result["tilt_strength"] = float(tilt_strength)

    equity_result = result["asset_class"].eq("Equity")
    crypto = result["asset_class"].eq("Crypto")
    result.loc[crypto, "sector_signal"] = 0.0
    result.loc[crypto, "sentiment_multiplier"] = 1.0

    if result.loc[equity_result, "sentiment_multiplier"].isna().any():
        raise ValueError("An equity ticker could not be matched to sector sentiment.")
    result.loc[equity_result, "target_weight"] = (
        result.loc[equity_result, "base_target_weight"]
        * result.loc[equity_result, "sentiment_multiplier"]
    )
    tilted_equity_weight = float(
        result.loc[equity_result, "target_weight"].sum()
    )
    if tilted_equity_weight <= 0:
        raise ValueError("The sentiment tilt eliminated the equity sleeve.")
    result.loc[equity_result, "target_weight"] *= (
        equity_weight / tilted_equity_weight
    )

    adjusted = result["target_weight"].to_numpy(dtype=float)
    if not np.isfinite(adjusted).all() or (adjusted < -1e-12).any():
        raise ValueError("Adjusted weights must be finite and non-negative.")
    if not np.isclose(adjusted.sum(), 1.0, atol=1e-10):
        raise ValueError("Adjusted target weights must sum to one.")
    if not np.isclose(
        result.loc[equity_result, "target_weight"].sum(),
        equity_weight,
        atol=1e-10,
    ):
        raise ValueError("The equity sleeve weight changed during fusion.")
    if not np.allclose(
        result.loc[crypto, "target_weight"],
        result.loc[crypto, "base_target_weight"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("Crypto weights changed during sentiment fusion.")

    return result


def _backtest_sentiment_overlay(
    asset_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
    ticker_sectors: pd.DataFrame,
    *,
    rule: str = PRIMARY_FUSION_RULE,
    tilt_strength: float = PRIMARY_TILT_STRENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest one frozen monthly overlay without re-estimating base weights."""
    required_base = {
        "decision_date",
        "effective_start_date",
        "fund_id",
        "universe",
        "method",
        "ticker",
        "target_weight",
    }
    _require_columns(base_weights, required_base, "base_weights")
    _require_columns(ticker_sectors, {"ticker", "sector"}, "ticker_sectors")
    if base_weights.empty:
        raise ValueError("base_weights cannot be empty.")

    panel = asset_returns.copy().sort_index()
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("asset_returns must use a pandas DatetimeIndex.")
    panel.index = panel.index.normalize()
    if panel.empty or panel.index.duplicated().any():
        raise ValueError("asset_returns must contain unique non-empty dates.")
    if panel.columns.duplicated().any() or panel.isna().any().any():
        raise ValueError("asset_returns must be a complete panel of unique tickers.")
    values = panel.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("asset_returns must be finite and greater than -100%.")

    base = base_weights.copy()
    base["decision_date"] = pd.to_datetime(
        base["decision_date"], errors="raise"
    ).dt.normalize()
    base["effective_start_date"] = pd.to_datetime(
        base["effective_start_date"], errors="raise"
    ).dt.normalize()
    identity_columns = ["fund_id", "universe", "method"]
    if any(base[column].nunique() != 1 for column in identity_columns):
        raise ValueError("base_weights must describe exactly one base fund.")
    base_fund_id = str(base["fund_id"].iloc[0])
    universe = str(base["universe"].iloc[0])
    method = str(base["method"].iloc[0])
    if universe == "Crypto":
        raise ValueError("Crypto-only funds are outside the sentiment-fusion scope.")

    mapping = ticker_sectors[["ticker", "sector"]].drop_duplicates().copy()
    if mapping["ticker"].duplicated().any() or mapping.isna().any().any():
        raise ValueError("ticker_sectors must map every equity ticker once.")
    sector_map = mapping.set_index("ticker")["sector"]

    schedule = (
        base[["decision_date", "effective_start_date"]]
        .drop_duplicates()
        .sort_values("effective_start_date")
        .reset_index(drop=True)
    )
    if schedule["effective_start_date"].duplicated().any():
        raise ValueError("Each holding period must have one effective start date.")
    if not (
        schedule["decision_date"] < schedule["effective_start_date"]
    ).all():
        raise ValueError("Every decision date must precede its holding period.")

    daily_frames = []
    weight_frames = []
    audit_rows = []
    previous_live_weights = None

    for position, block in schedule.iterrows():
        decision_date = pd.Timestamp(block["decision_date"])
        effective_start = pd.Timestamp(block["effective_start_date"])
        if position + 1 < len(schedule):
            next_start = pd.Timestamp(
                schedule.iloc[position + 1]["effective_start_date"]
            )
            holding = panel.loc[
                (panel.index >= effective_start) & (panel.index < next_start)
            ]
        else:
            holding = panel.loc[panel.index >= effective_start]
        if holding.empty:
            raise ValueError("A fusion holding period has no asset returns.")

        block_base = base.loc[
            base["effective_start_date"].eq(effective_start)
        ].copy()
        if set(block_base["ticker"]) != set(panel.columns):
            raise ValueError(
                "Every base target must contain every return-panel ticker."
            )
        block_base = block_base.set_index("ticker").reindex(panel.columns)
        block_base.index.name = "ticker"
        block_base = block_base.reset_index()
        block_base["sector"] = block_base["ticker"].map(sector_map)
        unmapped = block_base["sector"].isna()
        invalid_unmapped = unmapped & ~block_base["ticker"].str.endswith("-USD")
        if invalid_unmapped.any():
            missing = sorted(block_base.loc[invalid_unmapped, "ticker"])
            raise ValueError(f"Equity tickers lack a sector mapping: {missing}")
        block_base["asset_class"] = np.where(
            unmapped,
            "Crypto",
            "Equity",
        )

        block_sentiment = sector_sentiment.loc[
            pd.to_datetime(sector_sentiment["date"]).dt.normalize().eq(
                decision_date
            )
        ].copy()
        adjusted = apply_sentiment(
            block_base,
            block_sentiment,
            rule=rule,
            tilt_strength=tilt_strength,
        )
        adjusted = adjusted.set_index("ticker").reindex(panel.columns)
        adjusted.index.name = "ticker"
        adjusted = adjusted.reset_index()
        target = adjusted["target_weight"].to_numpy(dtype=float)
        base_target = adjusted["base_target_weight"].to_numpy(dtype=float)
        turnover = (
            np.nan
            if previous_live_weights is None
            else portfolios.one_way_turnover(previous_live_weights, target)
        )
        block_returns, previous_live_weights = (
            portfolios.buy_and_hold_block_returns(holding, target)
        )

        augmented_fund_id = f"{base_fund_id}_{rule}_sentiment"
        daily_frames.append(
            pd.DataFrame(
                {
                    "date": block_returns.index,
                    "decision_date": decision_date,
                    "effective_start_date": effective_start,
                    "base_fund_id": base_fund_id,
                    "fund_id": augmented_fund_id,
                    "universe": universe,
                    "method": method,
                    "fusion_rule": rule,
                    "tilt_strength": float(tilt_strength),
                    "return": block_returns.to_numpy(),
                }
            )
        )
        adjusted["base_fund_id"] = base_fund_id
        adjusted["fund_id"] = augmented_fund_id
        adjusted["universe"] = universe
        adjusted["method"] = method
        weight_frames.append(adjusted)

        equity = adjusted["asset_class"].eq("Equity")
        crypto = adjusted["asset_class"].eq("Crypto")
        audit_rows.append(
            {
                "decision_date": decision_date,
                "effective_start_date": effective_start,
                "effective_end_date": pd.Timestamp(holding.index.max()),
                "base_fund_id": base_fund_id,
                "fund_id": augmented_fund_id,
                "universe": universe,
                "method": method,
                "fusion_rule": rule,
                "tilt_strength": float(tilt_strength),
                "one_way_turnover": turnover,
                "one_way_active_shift_from_base": float(
                    0.5 * np.abs(target - base_target).sum()
                ),
                "largest_target_weight": float(target.max()),
                "effective_number_of_holdings": float(
                    1.0 / np.square(target).sum()
                ),
                "equity_sleeve_target_weight": float(
                    adjusted.loc[equity, "target_weight"].sum()
                ),
                "crypto_sleeve_target_weight": float(
                    adjusted.loc[crypto, "target_weight"].sum()
                ),
                "earliest_signal_source_date": pd.to_datetime(
                    block_sentiment["signal_source_date"]
                ).min(),
                "latest_signal_source_date": pd.to_datetime(
                    block_sentiment["signal_source_date"]
                ).max(),
            }
        )

    daily = pd.concat(daily_frames, ignore_index=True).sort_values("date")
    daily = daily.reset_index(drop=True)
    daily["growth_of_one"] = (1.0 + daily["return"]).cumprod()
    daily["drawdown"] = (
        daily["growth_of_one"] / daily["growth_of_one"].cummax() - 1.0
    )
    weights = pd.concat(weight_frames, ignore_index=True).sort_values(
        ["effective_start_date", "ticker"]
    )
    weights = weights.reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("effective_start_date")
    audit = audit.reset_index(drop=True)
    return daily, weights, audit
