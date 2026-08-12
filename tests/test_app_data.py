"""Tests for the precomputed AssetFund app data contract."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import app_data


@pytest.fixture(scope="module")
def artifacts():
    """Load the committed app artifacts once for this test module."""
    return app_data.load_app_artifacts()


def test_app_artifacts_cover_the_complete_product(artifacts):
    """The app exposes all built funds, sectors, and fusion variants."""
    assert artifacts.performance_metrics["fund_id"].nunique() == 13
    assert artifacts.fund_returns["fund_id"].nunique() == 13
    assert artifacts.fund_weights["fund_id"].nunique() == 13
    assert artifacts.sector_sentiment["sector"].nunique() == 10
    assert artifacts.sentiment_development_metrics["model_name"].nunique() == 2
    assert artifacts.sentiment_holdout_metrics["model_name"].nunique() == 2
    assert len(artifacts.sentiment_model_diagnostics) == 1
    assert set(artifacts.fusion_returns["fusion_rule"]) == {
        "coverage_rank",
        "naive",
    }


def test_catalog_has_one_display_name_per_fund(artifacts):
    """Fund comparison receives stable, client-facing identifiers."""
    catalog = app_data.fund_catalog(artifacts)

    assert len(catalog) == 13
    assert catalog["fund_id"].is_unique
    assert catalog["display_name"].is_unique
    assert catalog["display_name"].str.contains(" · ", regex=False).all()
    assert "Hierarchical Sleeve Risk Parity" not in set(catalog["method_label"])
    assert "Two-stage Sleeve Risk Parity" in set(catalog["method_label"])


def test_every_fund_builds_a_complete_fact_sheet(artifacts):
    """Every shelf item has OOS history, metrics, and current holdings."""
    for fund_id in artifacts.performance_metrics["fund_id"]:
        sheet = app_data.fund_fact_sheet(artifacts, fund_id)

        assert sheet.fund_id == fund_id
        assert not sheet.daily_returns.empty
        assert sheet.daily_returns["date"].is_monotonic_increasing
        assert not sheet.current_holdings.empty
        assert sheet.current_holdings_date == sheet.current_holdings_date.normalize()
        assert sheet.current_holdings["target_weight"].sum() == pytest.approx(1.0)
        assert set(sheet.current_holdings["asset_class"]).issubset(
            {"Equity", "Crypto"}
        )
        assert np.isfinite(sheet.metrics["sharpe_ratio"])


def test_standardised_sentiment_is_available_for_every_sector(artifacts):
    """The app uses the new past-only zero-centred reporting signal."""
    sentiment = app_data.sector_sentiment_view(artifacts)
    counts = sentiment.groupby("sector")["sentiment_expanding_zscore"].count()

    assert counts.index.nunique() == 10
    assert counts.gt(0).all()
    assert sentiment["standardization_rule"].nunique() == 1
    assert sentiment["standardization_min_periods"].eq(21).all()


def test_primary_fusion_evidence_contains_base_and_both_rules(artifacts):
    """The app can disclose the frozen holdout and all cost scenarios."""
    comparison, costs = app_data.fusion_evidence(
        artifacts,
        "combined_hierarchical_risk_parity",
    )

    assert set(comparison["fusion_rule"]) == {
        "base",
        "coverage_rank",
        "naive",
    }
    assert set(costs["cost_bps"]) == {0.0, 10.0, 50.0}
    assert set(costs["fusion_rule"]) == {
        "base",
        "coverage_rank",
        "naive",
    }


def test_sentiment_validation_summary_keeps_the_locked_split(artifacts):
    """The app compares identical models without leaking holdout hashes."""
    summary = app_data.sentiment_validation_summary(artifacts)

    assert len(summary) == 4
    assert set(summary["evaluation_split"]) == {
        "Development",
        "Locked holdout",
    }
    assert summary.groupby("evaluation_split")["model_name"].nunique().eq(2).all()
    assert "holdout_file_sha256" not in summary
    diagnostic = artifacts.sentiment_model_diagnostics.iloc[0]
    assert diagnostic["scored_headline_rows"] == 146_836
    assert diagnostic["outside_equity_calendar_rows"] == 6


def test_unknown_product_requests_fail_clearly(artifacts):
    """Invalid URL or selector state cannot silently show another product."""
    with pytest.raises(KeyError, match="Unknown fund_id"):
        app_data.fund_fact_sheet(artifacts, "not_a_fund")
    with pytest.raises(KeyError, match="Unknown sectors"):
        app_data.sector_sentiment_view(artifacts, ["not_a_sector"])


def test_missing_artifacts_fail_before_streamlit_renders(tmp_path):
    """A clean deployment reports a missing build instead of a partial app."""
    with pytest.raises(FileNotFoundError, match="Required app artifact"):
        app_data.load_app_artifacts(tmp_path)
