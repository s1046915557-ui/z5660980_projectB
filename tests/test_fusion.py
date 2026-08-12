"""Synthetic tests for the frozen Project B sentiment-fusion rule."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import run_part_b
from src import fusion, portfolios


def combined_weights() -> pd.DataFrame:
    """Return one combined target with two equity sectors and crypto."""
    return pd.DataFrame(
        {
            "ticker": ["TECH_A", "TECH_B", "ENERGY_A", "BTC-USD"],
            "target_weight": [0.30, 0.20, 0.20, 0.30],
            "asset_class": ["Equity", "Equity", "Equity", "Crypto"],
            "sector": ["Tech", "Tech", "Energy", pd.NA],
            "decision_date": pd.to_datetime(["2022-12-30"] * 4),
            "effective_start_date": pd.to_datetime(["2023-01-03"] * 4),
        }
    )


def sector_sentiment() -> pd.DataFrame:
    """Return two available sector signals known before the decision."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-30"] * 2),
            "sector": ["Tech", "Energy"],
            "signal_source_date": pd.to_datetime(["2022-12-29"] * 2),
            "lagged_sentiment_compound": [0.40, -0.20],
            "lagged_coverage_rate": [1.0, 1.0],
            "signal_available": [True, True],
        }
    )


def test_primary_rule_is_long_only_and_fully_invested():
    """The coverage-rank overlay produces valid bounded target weights."""
    result = fusion.apply_sentiment(combined_weights(), sector_sentiment())

    assert np.isfinite(result["target_weight"]).all()
    assert result["target_weight"].ge(0).all()
    assert result["target_weight"].sum() == pytest.approx(1.0)
    multipliers = result.groupby("sector", dropna=True)[
        "sentiment_multiplier"
    ].first()
    assert multipliers["Tech"] == pytest.approx(1.25)
    assert multipliers["Energy"] == pytest.approx(0.75)


def test_combined_sleeves_and_crypto_weights_are_preserved():
    """Fusion changes equity selection without changing asset-class allocation."""
    base = combined_weights()
    result = fusion.apply_sentiment(base, sector_sentiment())
    equity = result["asset_class"].eq("Equity")
    crypto = result["asset_class"].eq("Crypto")

    assert result.loc[equity, "target_weight"].sum() == pytest.approx(0.70)
    assert result.loc[crypto, "target_weight"].sum() == pytest.approx(0.30)
    np.testing.assert_allclose(
        result.loc[crypto, "target_weight"],
        result.loc[crypto, "base_target_weight"],
        atol=1e-12,
        rtol=0.0,
    )


def test_within_sector_relative_weights_are_preserved():
    """A sector multiplier does not replace the base optimiser within sectors."""
    result = fusion.apply_sentiment(combined_weights(), sector_sentiment())
    tech = result.loc[result["sector"].eq("Tech")].set_index("ticker")

    base_ratio = tech.loc["TECH_A", "base_target_weight"] / tech.loc[
        "TECH_B", "base_target_weight"
    ]
    adjusted_ratio = tech.loc["TECH_A", "target_weight"] / tech.loc[
        "TECH_B", "target_weight"
    ]
    assert adjusted_ratio == pytest.approx(base_ratio)


@pytest.mark.parametrize("rule", fusion.FUSION_RULES)
def test_zero_strength_exactly_recovers_base_weights(rule):
    """A zero-strength overlay is an exact implementation benchmark."""
    result = fusion.apply_sentiment(
        combined_weights(),
        sector_sentiment(),
        rule=rule,
        tilt_strength=0.0,
    )

    np.testing.assert_allclose(
        result["target_weight"],
        result["base_target_weight"],
        atol=1e-12,
        rtol=0.0,
    )


def test_naive_rule_uses_raw_lagged_compound_score():
    """The transparent benchmark follows the lecture-style raw multiplier."""
    result = fusion.apply_sentiment(
        combined_weights(),
        sector_sentiment(),
        rule=fusion.NAIVE_FUSION_RULE,
        tilt_strength=fusion.NAIVE_TILT_STRENGTH,
    )
    multipliers = result.groupby("sector", dropna=True)[
        "sentiment_multiplier"
    ].first()

    assert multipliers["Tech"] == pytest.approx(1.40)
    assert multipliers["Energy"] == pytest.approx(0.80)


def test_zero_coverage_shrinks_primary_active_signal_to_zero():
    """An unobserved sector receives no active multiplier under the primary rule."""
    sentiment = sector_sentiment()
    sentiment.loc[sentiment["sector"].eq("Tech"), "lagged_coverage_rate"] = 0.0
    result = fusion.apply_sentiment(combined_weights(), sentiment)
    tech = result.loc[result["sector"].eq("Tech")]

    assert tech["sector_signal"].eq(0.0).all()
    assert tech["sentiment_multiplier"].eq(1.0).all()


def test_signal_source_must_precede_decision_date():
    """Same-day or future source dates are rejected as look-ahead unsafe."""
    sentiment = sector_sentiment()
    sentiment.loc[0, "signal_source_date"] = pd.Timestamp("2022-12-30")

    with pytest.raises(ValueError, match="must precede"):
        fusion.apply_sentiment(combined_weights(), sentiment)


def test_sentiment_date_must_equal_portfolio_decision_date():
    """Effective-date sentiment cannot be substituted for decision-date data."""
    sentiment = sector_sentiment()
    sentiment["date"] = pd.Timestamp("2023-01-03")

    with pytest.raises(ValueError, match="decision date"):
        fusion.apply_sentiment(combined_weights(), sentiment)


def test_decision_date_must_precede_holding_period():
    """A target cannot be formed on or after its effective start date."""
    weights = combined_weights()
    weights["decision_date"] = weights["effective_start_date"]

    with pytest.raises(ValueError, match="before effective_start_date"):
        fusion.apply_sentiment(weights, sector_sentiment())


def test_equity_only_fund_remains_fully_invested():
    """The same rule works for a 100% equity fund without a crypto sleeve."""
    weights = combined_weights().loc[
        lambda frame: frame["asset_class"].eq("Equity")
    ].copy()
    weights["target_weight"] = weights["target_weight"] / weights[
        "target_weight"
    ].sum()
    result = fusion.apply_sentiment(weights, sector_sentiment())

    assert result["target_weight"].sum() == pytest.approx(1.0)
    assert result["asset_class"].eq("Equity").all()


def test_crypto_only_fund_is_outside_the_fusion_scope():
    """Price-only crypto funds cannot receive an equity sentiment overlay."""
    weights = combined_weights().loc[
        lambda frame: frame["asset_class"].eq("Crypto")
    ].copy()
    weights["target_weight"] = 1.0

    with pytest.raises(ValueError, match="equity exposure"):
        fusion.apply_sentiment(weights, sector_sentiment())


def test_sector_cross_section_must_be_complete():
    """Dropping a sector is rejected because it changes the rank definition."""
    incomplete = sector_sentiment().loc[lambda frame: frame["sector"].eq("Tech")]

    with pytest.raises(ValueError, match="exactly match"):
        fusion.apply_sentiment(combined_weights(), incomplete)


def overlay_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build two deterministic holding periods for overlay-backtest tests."""
    dates = pd.bdate_range("2023-01-02", periods=6)
    returns = pd.DataFrame(
        {
            "ENERGY_A": [0.01, 0.00, -0.01, 0.02, 0.00, 0.01],
            "TECH_A": [0.00, 0.02, 0.01, -0.01, 0.01, 0.00],
            "BTC-USD": [0.03, -0.01, 0.02, 0.00, -0.02, 0.01],
        },
        index=dates,
    )
    rows = []
    specifications = [
        ("2022-12-30", "2023-01-02", [0.40, 0.30, 0.30]),
        ("2023-01-04", "2023-01-05", [0.30, 0.40, 0.30]),
    ]
    for decision_date, effective_date, targets in specifications:
        for ticker, target in zip(returns.columns, targets, strict=True):
            rows.append(
                {
                    "decision_date": decision_date,
                    "effective_start_date": effective_date,
                    "fund_id": "combined_synthetic",
                    "universe": "Combined",
                    "method": "equal_weight",
                    "ticker": ticker,
                    "target_weight": target,
                }
            )
    weights = pd.DataFrame(rows)
    sectors = pd.DataFrame(
        {
            "ticker": ["ENERGY_A", "TECH_A"],
            "sector": ["Energy", "Tech"],
        }
    )
    sentiment_rows = []
    for decision_date, source_date, energy, tech in [
        ("2022-12-30", "2022-12-29", -0.20, 0.40),
        ("2023-01-04", "2023-01-03", 0.30, -0.10),
    ]:
        for sector, score in [("Energy", energy), ("Tech", tech)]:
            sentiment_rows.append(
                {
                    "date": decision_date,
                    "sector": sector,
                    "signal_source_date": source_date,
                    "lagged_sentiment_compound": score,
                    "lagged_coverage_rate": 1.0,
                    "signal_available": True,
                }
            )
    sentiment = pd.DataFrame(sentiment_rows)
    return returns, weights, sentiment, sectors


def test_zero_strength_backtest_reproduces_base_buy_and_hold_returns():
    """Zero strength reconciles daily returns, not only formation weights."""
    returns, weights, sentiment, sectors = overlay_inputs()
    daily, adjusted, _audit = fusion._backtest_sentiment_overlay(
        returns,
        weights,
        sentiment,
        sectors,
        tilt_strength=0.0,
    )

    expected_blocks = []
    for effective_start, holding in [
        (pd.Timestamp("2023-01-02"), returns.iloc[:3]),
        (pd.Timestamp("2023-01-05"), returns.iloc[3:]),
    ]:
        target = (
            weights.loc[
                pd.to_datetime(weights["effective_start_date"]).eq(
                    effective_start
                )
            ]
            .set_index("ticker")
            .loc[returns.columns, "target_weight"]
            .to_numpy()
        )
        block_returns, _ending_weights = portfolios.buy_and_hold_block_returns(
            holding,
            target,
        )
        expected_blocks.append(block_returns)
    expected = pd.concat(expected_blocks)

    np.testing.assert_allclose(daily["return"], expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        adjusted["target_weight"],
        adjusted["base_target_weight"],
        atol=1e-12,
        rtol=0.0,
    )


def test_overlay_backtest_uses_drifted_weights_for_turnover():
    """Later turnover compares the new target with live pre-trade holdings."""
    returns, weights, sentiment, sectors = overlay_inputs()
    _daily, adjusted, audit = fusion._backtest_sentiment_overlay(
        returns,
        weights,
        sentiment,
        sectors,
    )
    first_target = (
        adjusted.loc[
            adjusted["effective_start_date"].eq(pd.Timestamp("2023-01-02"))
        ]
        .set_index("ticker")
        .loc[returns.columns, "target_weight"]
        .to_numpy()
    )
    _block_returns, live_weights = portfolios.buy_and_hold_block_returns(
        returns.iloc[:3],
        first_target,
    )
    second_target = (
        adjusted.loc[
            adjusted["effective_start_date"].eq(pd.Timestamp("2023-01-05"))
        ]
        .set_index("ticker")
        .loc[returns.columns, "target_weight"]
        .to_numpy()
    )
    expected_turnover = portfolios.one_way_turnover(
        live_weights,
        second_target,
    )

    assert pd.isna(audit.iloc[0]["one_way_turnover"])
    assert audit.iloc[1]["one_way_turnover"] == pytest.approx(
        expected_turnover
    )


def test_future_sentiment_cannot_change_earlier_fusion_weights():
    """Changing a later signal leaves the already formed target unchanged."""
    returns, weights, sentiment, sectors = overlay_inputs()
    altered = sentiment.copy()
    altered.loc[altered["date"].eq("2023-01-04"), "lagged_sentiment_compound"] *= -1

    _daily, original, _audit = fusion._backtest_sentiment_overlay(
        returns,
        weights,
        sentiment,
        sectors,
    )
    _changed_daily, changed, _changed_audit = (
        fusion._backtest_sentiment_overlay(
            returns,
            weights,
            altered,
            sectors,
        )
    )
    first_date = pd.Timestamp("2023-01-02")
    original_first = original.loc[
        original["effective_start_date"].eq(first_date), "target_weight"
    ]
    changed_first = changed.loc[
        changed["effective_start_date"].eq(first_date), "target_weight"
    ]

    np.testing.assert_allclose(
        original_first,
        changed_first,
        atol=1e-12,
        rtol=0.0,
    )


def test_holdout_costs_include_the_first_2023_rebalance():
    """Period costs charge the live turnover entering the locked holdout."""
    metadata = {
        "base_fund_id": "combined_synthetic",
        "universe": "Combined",
        "method": "equal_weight",
    }
    return_rows = []
    audit_rows = []
    variants = [
        ("combined_synthetic", "base", 0.0, "Base Fund", 0.10),
        (
            "combined_synthetic_coverage_rank_sentiment",
            fusion.PRIMARY_FUSION_RULE,
            fusion.PRIMARY_TILT_STRENGTH,
            "Coverage-Aware Rank Sentiment",
            0.20,
        ),
    ]
    for fund_id, rule, strength, label, holdout_turnover in variants:
        variant_metadata = {
            **metadata,
            "fund_id": fund_id,
            "fusion_rule": rule,
            "tilt_strength": strength,
            "variant_label": label,
        }
        return_rows.extend(
            [
                {**variant_metadata, "date": "2022-12-29", "return": 0.00},
                {**variant_metadata, "date": "2022-12-30", "return": 0.01},
                {**variant_metadata, "date": "2023-01-03", "return": 0.02},
                {**variant_metadata, "date": "2023-01-04", "return": 0.00},
            ]
        )
        audit_rows.extend(
            [
                {
                    **variant_metadata,
                    "effective_start_date": "2022-12-29",
                    "one_way_turnover": np.nan,
                },
                {
                    **variant_metadata,
                    "effective_start_date": "2023-01-03",
                    "one_way_turnover": holdout_turnover,
                },
            ]
        )

    robustness = run_part_b._fusion_cost_robustness(
        pd.DataFrame(return_rows),
        pd.DataFrame(audit_rows),
    )
    holdout = robustness.loc[
        robustness["evaluation_period"].eq("locked_holdout_2023")
        & robustness["fusion_rule"].eq(fusion.PRIMARY_FUSION_RULE)
        & robustness["cost_bps"].eq(50.0)
    ].iloc[0]
    expected_return = (1.0 + 0.02) * (1.0 - 0.20 * 0.005) - 1.0

    assert len(robustness) == 18
    assert holdout["charged_rebalance_count"] == 1
    assert holdout["total_return"] == pytest.approx(expected_return)
