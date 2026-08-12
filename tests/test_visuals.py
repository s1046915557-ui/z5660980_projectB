"""Synthetic tests for the AssetFund Project B figure pipeline."""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import visuals


def test_combined_weights_are_aggregated_into_valid_sleeves():
    """Ticker weights become complete equity-versus-crypto allocations."""
    weights = pd.DataFrame(
        {
            "effective_start_date": pd.to_datetime(
                ["2023-01-03"] * 4 + ["2023-02-01"] * 4
            ),
            "fund_id": ["combined_equal_weight"] * 8,
            "universe": ["Combined"] * 8,
            "method": ["equal_weight"] * 8,
            "ticker": [
                "AAPL",
                "MSFT",
                "BTC-USD",
                "ETH-USD",
                "AAPL",
                "MSFT",
                "BTC-USD",
                "ETH-USD",
            ],
            "target_weight": [0.30, 0.20, 0.35, 0.15, 0.25, 0.25, 0.30, 0.20],
        }
    )

    sleeves = visuals._combined_sleeve_weights(weights)

    assert len(sleeves) == 2
    np.testing.assert_allclose(sleeves["Equity"], [0.50, 0.50])
    np.testing.assert_allclose(sleeves["Crypto"], [0.50, 0.50])
    np.testing.assert_allclose(sleeves["Equity"] + sleeves["Crypto"], 1.0)


def test_invalid_combined_weight_total_is_rejected():
    """The figure must not conceal an incomplete portfolio target."""
    weights = pd.DataFrame(
        {
            "effective_start_date": ["2023-01-03", "2023-01-03"],
            "fund_id": ["combined_equal_weight"] * 2,
            "universe": ["Combined"] * 2,
            "method": ["equal_weight"] * 2,
            "ticker": ["AAPL", "BTC-USD"],
            "target_weight": [0.40, 0.40],
        }
    )

    with pytest.raises(ValueError, match="do not sum to one"):
        visuals._combined_sleeve_weights(weights)


def test_holdout_growth_uses_only_2023_and_rebases_each_variant():
    """The frozen comparison excludes development-period wealth."""
    fund_returns = pd.DataFrame(
        {
            "date": ["2022-12-30", "2023-01-03", "2023-01-04"],
            "fund_id": [visuals.PRIMARY_FUND_ID] * 3,
            "return": [0.50, 0.10, -0.05],
        }
    )
    fusion_rows = []
    for variant in visuals.VARIANT_ORDER[1:]:
        fusion_rows.extend(
            [
                {
                    "date": "2023-01-03",
                    "base_fund_id": visuals.PRIMARY_FUND_ID,
                    "variant_label": variant,
                    "return": 0.08,
                },
                {
                    "date": "2023-01-04",
                    "base_fund_id": visuals.PRIMARY_FUND_ID,
                    "variant_label": variant,
                    "return": -0.02,
                },
            ]
        )
    fusion_returns = pd.DataFrame(fusion_rows)

    growth = visuals._primary_holdout_growth(fund_returns, fusion_returns)

    assert growth["date"].min() == pd.Timestamp("2023-01-03")
    assert len(growth) == 6
    base = growth.loc[growth["variant_label"].eq("Base Fund")]
    assert base["holdout_growth_of_one"].iloc[0] == pytest.approx(1.10)
    assert base["holdout_growth_of_one"].iloc[-1] == pytest.approx(1.045)


def test_cost_gate_selects_exact_locked_primary_scenarios():
    """Only the frozen primary fund and 2023 holdout enter the cost panel."""
    rows = []
    for variant in visuals.VARIANT_ORDER[1:]:
        for cost, delta in [(0.0, 0.02), (10.0, 0.01), (50.0, -0.01)]:
            rows.append(
                {
                    "base_fund_id": visuals.PRIMARY_FUND_ID,
                    "variant_label": variant,
                    "evaluation_period": "locked_holdout_2023",
                    "cost_bps": cost,
                    "delta_sharpe_ratio": delta,
                }
            )
    rows.append(
        {
            "base_fund_id": "combined_equal_weight",
            "variant_label": "Naive Sentiment",
            "evaluation_period": "locked_holdout_2023",
            "cost_bps": 0.0,
            "delta_sharpe_ratio": 99.0,
        }
    )

    selected = visuals._primary_holdout_cost_deltas(pd.DataFrame(rows))

    assert len(selected) == 6
    assert set(selected["cost_bps"]) == {0.0, 10.0, 50.0}
    assert selected["delta_sharpe_ratio"].max() == pytest.approx(0.02)


def test_growth_figure_renders_non_blank_png(tmp_path):
    """The report figure renders successfully from a small valid panel."""
    dates = pd.bdate_range("2021-01-04", periods=5)
    rows = []
    for universe in ("Equity", "Crypto", "Combined"):
        for method, slope in [("equal_weight", 0.02), ("risk_parity", 0.01)]:
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "fund_id": f"{universe.lower()}_{method}",
                        "universe": universe,
                        "method": method,
                        "growth_of_one": 1.0 + slope * index,
                    }
                )

    output = visuals.plot_fund_growth_comparison(pd.DataFrame(rows), tmp_path)
    image = plt.imread(output)

    assert output.is_file()
    assert output.stat().st_size > 10_000
    assert image.shape[0] > image.shape[1]
    assert float(np.std(image)) > 0.01


def test_standardised_sentiment_figure_renders_non_blank_png(tmp_path):
    """The sentiment exhibit requires a finite zero-centred sector series."""
    dates = pd.bdate_range("2021-01-04", periods=30)
    rows = []
    for sector_index, sector in enumerate(visuals.SECTOR_ORDER):
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "sentiment_expanding_zscore": (
                        np.nan
                        if date_index < 21
                        else np.sin(date_index / 4.0 + sector_index / 3.0)
                    ),
                }
            )

    output = visuals.plot_sector_sentiment_index(pd.DataFrame(rows), tmp_path)
    image = plt.imread(output)

    assert output.is_file()
    assert output.stat().st_size > 10_000
    assert image.shape[0] > image.shape[1]
    assert float(np.std(image)) > 0.01
