"""Tests for AssetFund's investor allocation and fee engine."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import allocation, app_data


def _return_frame() -> pd.DataFrame:
    """Create two fund calendars with weekend crypto observations."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-04",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                ]
            ),
            "fund_id": [
                "equity",
                "equity",
                "crypto",
                "crypto",
                "crypto",
                "crypto",
            ],
            "return": [0.10, 0.10, 0.00, 0.10, 0.00, 0.00],
        }
    )


def test_allocation_history_preserves_crypto_weekends():
    """The blend uses the union calendar and carries a closed fund unchanged."""
    history = allocation.build_allocation_history(
        _return_frame(),
        {"equity": 0.5, "crypto": 0.5},
        annual_management_fee=0.0,
    )

    assert history["date"].tolist() == list(pd.date_range("2021-01-01", periods=4))
    assert history.loc[1, "gross_growth_of_one"] == pytest.approx(1.10)
    assert history.loc[3, "gross_growth_of_one"] == pytest.approx(1.155)


def test_allocation_is_buy_and_hold_across_funds():
    """Initial product weights drift instead of being silently reset each day."""
    history = allocation.build_allocation_history(
        _return_frame(),
        {"equity": 0.5, "crypto": 0.5},
        annual_management_fee=0.0,
    )

    assert history.loc[3, "gross_growth_of_one"] == pytest.approx(
        0.5 * 1.21 + 0.5 * 1.10
    )


def test_fee_uses_elapsed_calendar_days():
    """One effective annual fee is charged independent of observation count."""
    flat = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01", "2022-01-01", "2021-01-01", "2022-01-01"]
            ),
            "fund_id": ["fund_a", "fund_a", "fund_b", "fund_b"],
            "return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    history = allocation.build_allocation_history(
        flat,
        {"fund_a": 0.5, "fund_b": 0.5},
        annual_management_fee=0.005,
    )

    expected = 0.995 ** (365 / allocation.CALENDAR_DAYS_PER_YEAR)
    assert history["net_growth_of_one"].iloc[-1] == pytest.approx(expected)
    assert history["cumulative_fee_drag"].iloc[-1] > 0.0


def test_zero_fee_makes_gross_and_net_identical():
    """The zero-fee scenario is a transparent gross-return baseline."""
    history = allocation.build_allocation_history(
        _return_frame(),
        {"equity": 0.4, "crypto": 0.6},
        annual_management_fee=0.0,
    )

    np.testing.assert_allclose(
        history["gross_growth_of_one"], history["net_growth_of_one"]
    )
    np.testing.assert_allclose(history["cumulative_fee_drag"], 0.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"equity": 0.5, "crypto": 0.4}, "sum to 100%"),
        ({"equity": 1.0, "crypto": 0.0}, "at least two funds"),
        ({"equity": 1.1, "crypto": -0.1}, "cannot be negative"),
    ],
)
def test_invalid_allocations_fail_clearly(weights, message):
    """Incomplete or economically invalid orders cannot reach the chart."""
    with pytest.raises(ValueError, match=message):
        allocation.build_allocation_history(_return_frame(), weights)


def test_unknown_funds_fail_clearly():
    """Stale URL or widget state cannot silently substitute another fund."""
    with pytest.raises(KeyError, match="Unknown fund IDs"):
        allocation.build_allocation_history(
            _return_frame(),
            {"equity": 0.5, "missing": 0.5},
        )


def test_latest_lookthrough_aggregates_overlapping_holdings():
    """Shared securities are combined and retain an overlap count."""
    weights = pd.DataFrame(
        {
            "effective_start_date": pd.to_datetime(
                ["2021-01-01", "2021-01-01", "2021-02-01", "2021-02-01"]
            ),
            "fund_id": ["fund_a", "fund_a", "fund_b", "fund_b"],
            "ticker": ["AAA", "BBB", "AAA", "BTC-USD"],
            "target_weight": [0.5, 0.5, 0.25, 0.75],
        }
    )
    as_of, lookthrough = allocation.latest_lookthrough_holdings(
        weights,
        {"fund_a": 0.4, "fund_b": 0.6},
    )

    indexed = lookthrough.set_index("ticker")
    assert as_of == pd.Timestamp("2021-01-01")
    assert indexed.loc["AAA", "lookthrough_weight"] == pytest.approx(0.35)
    assert indexed.loc["AAA", "fund_count"] == 2
    assert indexed.loc["BTC-USD", "asset_class"] == "Crypto"
    assert lookthrough["lookthrough_weight"].sum() == pytest.approx(1.0)


def test_allocation_metrics_report_net_investor_outcomes():
    """Summary fields reconcile to the transparent history columns."""
    history = allocation.build_allocation_history(
        _return_frame(),
        {"equity": 0.5, "crypto": 0.5},
        annual_management_fee=0.005,
        initial_investment=25_000.0,
    )
    metrics = allocation.allocation_metrics(history)

    assert metrics["ending_net_value"] == pytest.approx(
        history["net_value"].iloc[-1]
    )
    assert metrics["cumulative_fee_drag"] == pytest.approx(
        history["cumulative_fee_drag"].iloc[-1]
    )
    assert metrics["max_drawdown"] <= 0.0


def test_engine_runs_on_all_committed_funds():
    """The deployable artifacts support a real cross-universe allocation."""
    artifacts = app_data.load_app_artifacts()
    weights = {
        "equity_minimum_variance": 0.40,
        "crypto_equal_weight": 0.20,
        "combined_hierarchical_risk_parity": 0.40,
    }

    history = allocation.build_allocation_history(
        artifacts.fund_returns,
        weights,
    )
    as_of, lookthrough = allocation.latest_lookthrough_holdings(
        artifacts.fund_weights,
        weights,
    )

    assert not history.empty
    assert history["date"].is_monotonic_increasing
    assert history["net_growth_of_one"].le(
        history["gross_growth_of_one"] + 1e-12
    ).all()
    assert as_of == pd.Timestamp("2023-12-01")
    assert set(lookthrough["asset_class"]) == {"Equity", "Crypto"}
