"""Tests for the Project B OOS portfolio engine."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import portfolios


def synthetic_returns(
    start: str = "2020-01-01",
    end: str = "2021-06-30",
) -> pd.DataFrame:
    """Create a deterministic business-day return panel with three assets."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(5545)
    common = rng.normal(0.0002, 0.0060, len(dates))
    data = {
        "Asset_A": common + rng.normal(0.0002, 0.0050, len(dates)),
        "Asset_B": 0.5 * common + rng.normal(0.0001, 0.0035, len(dates)),
        "Asset_C": -0.2 * common + rng.normal(0.0003, 0.0080, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


def synthetic_combined_returns(
    start: str = "2020-01-01",
    end: str = "2021-06-30",
) -> pd.DataFrame:
    """Create two lower-risk equity and two higher-risk crypto assets."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(2026)
    common = rng.normal(0.0002, 0.0040, len(dates))
    data = {
        "EQ_A": common + rng.normal(0.0002, 0.0040, len(dates)),
        "EQ_B": 0.6 * common + rng.normal(0.0001, 0.0050, len(dates)),
        "CR_A-USD": 0.4 * common + rng.normal(0.0005, 0.0180, len(dates)),
        "CR_B-USD": -0.2 * common + rng.normal(0.0004, 0.0220, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


def combined_sleeves() -> dict[str, list[str]]:
    """Return the synthetic asset-class sleeve definitions."""
    return {
        "Equity": ["EQ_A", "EQ_B"],
        "Crypto": ["CR_A-USD", "CR_B-USD"],
    }


@pytest.mark.parametrize("method", portfolios.METHODS)
def test_all_weight_methods_are_long_only_and_fully_invested(method):
    """Every baseline method produces finite long-only weights summing to one."""
    returns = synthetic_returns(end="2020-12-31")
    weights, _solver = portfolios.estimate_weights(
        returns,
        method,
        periods_per_year=252,
    )

    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0)


def test_schedule_separates_decision_and_holding_dates():
    """The first live month is January 2021 and decisions use prior data."""
    returns = synthetic_returns()
    schedule = portfolios.build_monthly_schedule(returns)

    assert schedule.iloc[0]["effective_start_date"] == pd.Timestamp("2021-01-01")
    assert schedule.iloc[0]["decision_date"] == pd.Timestamp("2020-12-31")
    assert (schedule["decision_date"] < schedule["effective_start_date"]).all()
    assert schedule.iloc[0]["window_end_date"] == schedule.iloc[0]["decision_date"]


def test_future_returns_do_not_change_past_weights():
    """Changing future observations cannot alter already formed target weights."""
    returns = synthetic_returns()
    altered = returns.copy()
    altered.loc[altered.index >= "2021-06-01", "Asset_A"] += 0.25

    original = portfolios.oos_backtest(
        returns,
        method="minimum_variance",
        universe="Synthetic",
    )
    changed = portfolios.oos_backtest(
        altered,
        method="minimum_variance",
        universe="Synthetic",
    )

    original_weights = original.target_weights.loc[
        original.target_weights["effective_start_date"] < pd.Timestamp("2021-06-01")
    ]
    changed_weights = changed.target_weights.loc[
        changed.target_weights["effective_start_date"] < pd.Timestamp("2021-06-01")
    ]

    np.testing.assert_allclose(
        original_weights["target_weight"],
        changed_weights["target_weight"],
        atol=1e-12,
    )


def test_hierarchical_sleeve_weights_equalise_group_risk():
    """The innovation is long-only and equalises ex-ante sleeve risk."""
    returns = synthetic_combined_returns(end="2020-12-31")
    target, sleeve_targets = (
        portfolios.hierarchical_sleeve_risk_parity_weights(
            returns,
            combined_sleeves(),
        )
    )

    assert np.isfinite(target).all()
    assert (target >= 0).all()
    assert target.sum() == pytest.approx(1.0)
    assert sum(sleeve_targets.values()) == pytest.approx(1.0)

    equity_weight = sleeve_targets["Equity"]
    crypto_weight = sleeve_targets["Crypto"]
    equity_returns = (
        returns[["EQ_A", "EQ_B"]]
        @ (target[:2] / equity_weight)
    )
    crypto_returns = (
        returns[["CR_A-USD", "CR_B-USD"]]
        @ (target[2:] / crypto_weight)
    )
    sleeve_returns = pd.concat(
        [equity_returns, crypto_returns],
        axis=1,
    )
    covariance = sleeve_returns.cov().to_numpy() * 252
    sleeve_vector = np.array([equity_weight, crypto_weight])
    contributions = (
        sleeve_vector
        * (covariance @ sleeve_vector)
        / (sleeve_vector @ covariance @ sleeve_vector)
    )

    np.testing.assert_allclose(contributions, [0.5, 0.5], atol=1e-4)


def test_hierarchical_backtest_is_look_ahead_safe():
    """Future crypto shocks cannot change earlier hierarchical weights."""
    returns = synthetic_combined_returns()
    altered = returns.copy()
    altered.loc[altered.index >= "2021-06-01", "CR_A-USD"] += 0.25

    original = portfolios.oos_backtest(
        returns,
        method=portfolios.HIERARCHICAL_METHOD,
        universe="Combined",
        sleeve_assets=combined_sleeves(),
    )
    changed = portfolios.oos_backtest(
        altered,
        method=portfolios.HIERARCHICAL_METHOD,
        universe="Combined",
        sleeve_assets=combined_sleeves(),
    )

    original_weights = original.target_weights.loc[
        original.target_weights["effective_start_date"] < pd.Timestamp("2021-06-01")
    ]
    changed_weights = changed.target_weights.loc[
        changed.target_weights["effective_start_date"] < pd.Timestamp("2021-06-01")
    ]

    np.testing.assert_allclose(
        original_weights["target_weight"],
        changed_weights["target_weight"],
        atol=1e-12,
    )
    sleeve_sums = (
        original.rebalance_audit["equity_sleeve_target_weight"]
        + original.rebalance_audit["crypto_sleeve_target_weight"]
    )
    assert np.allclose(sleeve_sums, 1.0)


def test_buy_and_hold_weights_drift_with_asset_returns():
    """Month-block returns use live weights after each day's asset movement."""
    dates = pd.bdate_range("2021-01-04", periods=2)
    holding = pd.DataFrame(
        {
            "Asset_A": [0.10, 0.00],
            "Asset_B": [0.00, 0.10],
        },
        index=dates,
    )

    returns, ending_weights = portfolios.buy_and_hold_block_returns(
        holding,
        np.array([0.5, 0.5]),
    )

    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.10 * (0.5 / 1.05))
    assert ending_weights.sum() == pytest.approx(1.0)
    assert ending_weights[0] == pytest.approx(ending_weights[1])


def test_turnover_growth_drawdown_and_metrics():
    """Portfolio diagnostics match transparent hand calculations."""
    turnover = portfolios.one_way_turnover(
        np.array([0.60, 0.40]),
        np.array([0.50, 0.50]),
    )
    assert turnover == pytest.approx(0.10)

    returns = pd.Series([0.10, -0.20, 0.05])
    metrics = portfolios.performance_metrics(returns, periods_per_year=3)
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0

    assert metrics["final_growth_of_one"] == pytest.approx(wealth.iloc[-1])
    assert metrics["total_return"] == pytest.approx(wealth.iloc[-1] - 1.0)
    assert metrics["annualized_return"] == pytest.approx(returns.mean() * 3)
    assert metrics["max_drawdown"] == pytest.approx(drawdown.min())


def test_transaction_costs_apply_only_on_later_rebalance_dates():
    """Cost-adjusted returns match a transparent multiplicative calculation."""
    dates = pd.bdate_range("2021-01-04", periods=3)
    gross = pd.Series([0.01, 0.02, -0.01], index=dates)
    turnover = pd.Series(
        [np.nan, 0.25],
        index=[dates[0], dates[1]],
    )

    net = portfolios.apply_rebalance_transaction_costs(
        gross,
        turnover,
        cost_bps=100,
    )

    assert net.iloc[0] == pytest.approx(gross.iloc[0])
    assert net.iloc[1] == pytest.approx((1.0 + 0.02) * (1.0 - 0.0025) - 1.0)
    assert net.iloc[2] == pytest.approx(gross.iloc[2])

    zero_cost = portfolios.apply_rebalance_transaction_costs(
        gross,
        turnover,
        cost_bps=0,
    )
    pd.testing.assert_series_equal(
        zero_cost,
        gross.rename("return"),
        check_freq=False,
    )


def test_oos_backtest_output_contract():
    """One fund returns complete daily, weight, audit, and metric outputs."""
    result = portfolios.oos_backtest(
        synthetic_returns(),
        method="risk_parity",
        universe="Synthetic",
        fund_id="synthetic_risk_parity",
    )

    assert not result.daily_returns.empty
    assert not result.target_weights.empty
    assert not result.rebalance_audit.empty
    assert result.daily_returns["date"].min() == pd.Timestamp("2021-01-01")
    assert result.daily_returns["growth_of_one"].notna().all()
    assert result.daily_returns["drawdown"].le(0).all()

    weight_sums = result.target_weights.groupby("effective_start_date")[
        "target_weight"
    ].sum()
    assert np.allclose(weight_sums, 1.0)
    assert result.metrics["rebalance_count"] == 6
    assert result.metrics["periods_per_year"] == 252
