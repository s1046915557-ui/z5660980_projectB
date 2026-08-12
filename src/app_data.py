"""Validated, precomputed data contracts for the AssetFund Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_PATHS = {
    "fund_returns": Path("results/data/fund_returns.csv"),
    "fund_weights": Path("results/data/fund_weights.csv"),
    "sector_sentiment": Path("results/data/sector_sentiment_index.csv"),
    "sentiment_development_metrics": Path(
        "results/tables/sentiment_development_metrics.csv"
    ),
    "sentiment_holdout_metrics": Path(
        "results/tables/sentiment_holdout_metrics.csv"
    ),
    "sentiment_model_diagnostics": Path(
        "results/tables/sentiment_model_diagnostics.csv"
    ),
    "fusion_returns": Path("results/data/fusion_returns.csv"),
    "performance_metrics": Path("results/tables/performance_metrics.csv"),
    "fusion_comparison": Path("results/tables/fusion_period_comparison.csv"),
    "fusion_costs": Path(
        "results/tables/fusion_transaction_cost_robustness.csv"
    ),
}

REQUIRED_COLUMNS = {
    "fund_returns": {
        "date",
        "decision_date",
        "effective_start_date",
        "fund_id",
        "universe",
        "method",
        "return",
        "growth_of_one",
        "drawdown",
    },
    "fund_weights": {
        "decision_date",
        "effective_start_date",
        "fund_id",
        "universe",
        "method",
        "ticker",
        "target_weight",
    },
    "sector_sentiment": {
        "date",
        "sector",
        "sentiment_compound",
        "coverage_rate",
        "sentiment_expanding_zscore",
        "standardization_rule",
        "standardization_min_periods",
    },
    "sentiment_development_metrics": {
        "model_name",
        "observations",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    },
    "sentiment_holdout_metrics": {
        "evaluation_split",
        "model_name",
        "observations",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "model_rule_sha256",
        "holdout_file_sha256",
    },
    "sentiment_model_diagnostics": {
        "model_name",
        "benchmark_model",
        "nltk_version",
        "scored_headline_rows",
        "outside_equity_calendar_rows",
        "finance_lexicon_terms",
        "changed_headline_classifications",
    },
    "fusion_returns": {
        "date",
        "base_fund_id",
        "fund_id",
        "fusion_rule",
        "tilt_strength",
        "return",
        "growth_of_one",
        "drawdown",
        "variant_label",
    },
    "performance_metrics": {
        "fund_id",
        "universe",
        "method",
        "method_label",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_one_way_turnover",
        "latest_effective_number_of_holdings",
    },
    "fusion_comparison": {
        "base_fund_id",
        "fund_id",
        "fusion_rule",
        "variant_label",
        "evaluation_period",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "delta_total_return",
        "delta_sharpe_ratio",
    },
    "fusion_costs": {
        "base_fund_id",
        "fund_id",
        "fusion_rule",
        "variant_label",
        "evaluation_period",
        "cost_bps",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "delta_total_return",
        "delta_sharpe_ratio",
    },
}

DATE_COLUMNS = {
    "fund_returns": ("date", "decision_date", "effective_start_date"),
    "fund_weights": ("decision_date", "effective_start_date"),
    "sector_sentiment": ("date",),
    "fusion_returns": ("date",),
    "fusion_comparison": ("period_start", "period_end"),
    "fusion_costs": ("period_start", "period_end"),
}

NUMERIC_COLUMNS = {
    "fund_returns": ("return", "growth_of_one", "drawdown"),
    "fund_weights": ("target_weight",),
    "sector_sentiment": (
        "sentiment_compound",
        "coverage_rate",
        "sentiment_expanding_zscore",
        "standardization_min_periods",
    ),
    "sentiment_development_metrics": (
        "observations",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ),
    "sentiment_holdout_metrics": (
        "observations",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ),
    "sentiment_model_diagnostics": (
        "scored_headline_rows",
        "outside_equity_calendar_rows",
        "finance_lexicon_terms",
        "changed_headline_classifications",
    ),
    "fusion_returns": (
        "tilt_strength",
        "return",
        "growth_of_one",
        "drawdown",
    ),
    "performance_metrics": (
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_one_way_turnover",
        "latest_effective_number_of_holdings",
    ),
    "fusion_comparison": (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "delta_total_return",
        "delta_sharpe_ratio",
    ),
    "fusion_costs": (
        "cost_bps",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "delta_total_return",
        "delta_sharpe_ratio",
    ),
}


@dataclass(frozen=True)
class AppArtifacts:
    """All precomputed tables needed by the AssetFund investor journey."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    sector_sentiment: pd.DataFrame
    sentiment_development_metrics: pd.DataFrame
    sentiment_holdout_metrics: pd.DataFrame
    sentiment_model_diagnostics: pd.DataFrame
    fusion_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    fusion_comparison: pd.DataFrame
    fusion_costs: pd.DataFrame


@dataclass(frozen=True)
class FundFactSheet:
    """One investable fund's metrics, OOS history, and current holdings."""

    fund_id: str
    display_name: str
    universe: str
    method: str
    metrics: pd.Series
    daily_returns: pd.DataFrame
    current_holdings: pd.DataFrame
    current_holdings_date: pd.Timestamp


def _read_artifact(project_root: Path, name: str) -> pd.DataFrame:
    """Read one required CSV and normalise its typed fields."""
    path = project_root / ARTIFACT_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"Required app artifact not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"App artifact is empty: {path}")

    missing = REQUIRED_COLUMNS[name].difference(frame.columns)
    if missing:
        raise ValueError(
            f"{ARTIFACT_PATHS[name]} is missing columns: {sorted(missing)}"
        )

    for column in DATE_COLUMNS.get(name, ()):
        if column not in frame:
            continue
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(
                f"{ARTIFACT_PATHS[name]} contains invalid {column} values."
            )

    for column in NUMERIC_COLUMNS.get(name, ()):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        available = frame[column].dropna()
        if available.empty or not np.isfinite(available).all():
            raise ValueError(
                f"{ARTIFACT_PATHS[name]} contains invalid {column} values."
            )

    return frame


def _validate_fund_contracts(artifacts: AppArtifacts) -> None:
    """Check cross-table fund identity, return, and weight contracts."""
    returns = artifacts.fund_returns
    weights = artifacts.fund_weights
    metrics = artifacts.performance_metrics

    fund_sets = {
        "returns": set(returns["fund_id"]),
        "weights": set(weights["fund_id"]),
        "metrics": set(metrics["fund_id"]),
    }
    if len({frozenset(values) for values in fund_sets.values()}) != 1:
        raise ValueError(f"Fund IDs differ across app artifacts: {fund_sets}")
    if metrics["fund_id"].duplicated().any():
        raise ValueError("Performance metrics must contain one row per fund.")
    if returns.duplicated(["fund_id", "date"]).any():
        raise ValueError("Fund returns contain duplicate fund-date rows.")
    if weights.duplicated(
        ["fund_id", "effective_start_date", "ticker"]
    ).any():
        raise ValueError("Fund weights contain duplicate rebalance holdings.")
    if (returns["return"] <= -1.0).any() or (returns["growth_of_one"] <= 0).any():
        raise ValueError("Fund returns contain an invalid wealth path.")
    if (returns["drawdown"] > 1e-12).any():
        raise ValueError("Fund drawdown cannot be positive.")
    if (weights["target_weight"] < -1e-12).any():
        raise ValueError("Fund target weights cannot be negative.")

    weight_sums = weights.groupby(
        ["fund_id", "effective_start_date"], observed=True
    )["target_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-8):
        raise ValueError("At least one fund target does not sum to one.")


def _validate_sentiment_contracts(artifacts: AppArtifacts) -> None:
    """Check that the standalone sector view is complete and interpretable."""
    sentiment = artifacts.sector_sentiment
    if sentiment.duplicated(["sector", "date"]).any():
        raise ValueError("Sector sentiment contains duplicate sector-date rows.")
    if not sentiment["sentiment_compound"].between(-1.0, 1.0).all():
        raise ValueError("Sector compound sentiment must lie in [-1, 1].")
    if not sentiment["coverage_rate"].between(0.0, 1.0).all():
        raise ValueError("Sector coverage must lie in [0, 1].")

    standardised = sentiment["sentiment_expanding_zscore"].dropna()
    if standardised.empty or not np.isfinite(standardised).all():
        raise ValueError("Standardised sector sentiment is unavailable.")
    available_by_sector = sentiment.groupby("sector", observed=True)[
        "sentiment_expanding_zscore"
    ].count()
    if available_by_sector.eq(0).any():
        raise ValueError("A sector has no standardised sentiment observations.")


def _validate_sentiment_evaluation_contracts(
    artifacts: AppArtifacts,
) -> None:
    """Check that development and locked-holdout model evidence reconcile."""
    development = artifacts.sentiment_development_metrics
    holdout = artifacts.sentiment_holdout_metrics
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1"]

    for label, frame in {
        "development": development,
        "locked holdout": holdout,
    }.items():
        if frame["model_name"].duplicated().any():
            raise ValueError(f"Sentiment {label} metrics duplicate a model.")
        if frame["model_name"].nunique() != 2:
            raise ValueError(f"Sentiment {label} must compare exactly two models.")
        if (frame["observations"] <= 0).any():
            raise ValueError(f"Sentiment {label} observations must be positive.")
        if not frame[metric_columns].apply(
            lambda column: column.between(0.0, 1.0).all()
        ).all():
            raise ValueError(f"Sentiment {label} metrics must lie in [0, 1].")

    if set(development["model_name"]) != set(holdout["model_name"]):
        raise ValueError("Sentiment evaluation model names do not reconcile.")
    if not holdout["evaluation_split"].eq("locked_holdout").all():
        raise ValueError("Sentiment holdout metrics are not marked as locked.")
    if holdout["model_rule_sha256"].nunique() != 1:
        raise ValueError("Sentiment holdout model rule is not frozen.")
    if holdout["holdout_file_sha256"].nunique() != 1:
        raise ValueError("Sentiment holdout labels are not frozen.")

    diagnostics = artifacts.sentiment_model_diagnostics
    if len(diagnostics) != 1:
        raise ValueError("Sentiment model diagnostics must contain one row.")
    diagnostic = diagnostics.iloc[0]
    indexed_headlines = int(artifacts.sector_sentiment["headline_count"].sum())
    scored_headlines = int(diagnostic["scored_headline_rows"])
    outside_calendar = int(diagnostic["outside_equity_calendar_rows"])
    if scored_headlines - outside_calendar != indexed_headlines:
        raise ValueError(
            "Scored and indexed sentiment headline counts do not reconcile."
        )


def _validate_fusion_contracts(artifacts: AppArtifacts) -> None:
    """Check that every fusion result points to an available base fund."""
    fund_ids = set(artifacts.performance_metrics["fund_id"])
    for name, frame in {
        "fusion_returns": artifacts.fusion_returns,
        "fusion_comparison": artifacts.fusion_comparison,
        "fusion_costs": artifacts.fusion_costs,
    }.items():
        unknown = set(frame["base_fund_id"]).difference(fund_ids)
        if unknown:
            raise ValueError(f"{name} refers to unknown base funds: {unknown}")
    if artifacts.fusion_returns.duplicated(["fund_id", "date"]).any():
        raise ValueError("Fusion returns contain duplicate fund-date rows.")


def load_app_artifacts(
    project_root: str | Path = PROJECT_ROOT,
) -> AppArtifacts:
    """Load and validate every precomputed table used by the app."""
    root = Path(project_root).resolve()
    frames = {
        name: _read_artifact(root, name)
        for name in ARTIFACT_PATHS
    }
    artifacts = AppArtifacts(**frames)
    _validate_fund_contracts(artifacts)
    _validate_sentiment_contracts(artifacts)
    _validate_sentiment_evaluation_contracts(artifacts)
    _validate_fusion_contracts(artifacts)
    return artifacts


def fund_catalog(artifacts: AppArtifacts) -> pd.DataFrame:
    """Return one client-facing, sortable row per investable fund."""
    catalog = artifacts.performance_metrics.copy()
    display_method = catalog["method_label"].replace(
        {
            "Hierarchical Sleeve Risk Parity": (
                "Two-stage Sleeve Risk Parity"
            )
        }
    )
    catalog["method_label"] = display_method
    catalog["display_name"] = (
        catalog["universe"].astype(str)
        + " · "
        + catalog["method_label"].astype(str)
    )
    universe_order = pd.CategoricalDtype(
        ["Equity", "Crypto", "Combined"],
        ordered=True,
    )
    catalog["universe"] = catalog["universe"].astype(universe_order)
    return catalog.sort_values(
        ["universe", "sharpe_ratio"],
        ascending=[True, False],
    ).reset_index(drop=True)


def latest_holdings(
    artifacts: AppArtifacts,
    fund_id: str,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Return a fund's latest fully invested target vector."""
    weights = artifacts.fund_weights.loc[
        artifacts.fund_weights["fund_id"].eq(fund_id)
    ].copy()
    if weights.empty:
        raise KeyError(f"Unknown fund_id: {fund_id}")

    latest_date = pd.Timestamp(weights["effective_start_date"].max())
    current = weights.loc[
        weights["effective_start_date"].eq(latest_date),
        ["ticker", "target_weight"],
    ].copy()
    current["asset_class"] = np.where(
        current["ticker"].str.endswith("-USD"),
        "Crypto",
        "Equity",
    )
    current["weight_percent"] = current["target_weight"] * 100.0
    current = current.sort_values(
        ["target_weight", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)

    if not np.isclose(current["target_weight"].sum(), 1.0, atol=1e-8):
        raise ValueError(f"Latest holdings for {fund_id} do not sum to one.")
    return latest_date, current


def fund_fact_sheet(
    artifacts: AppArtifacts,
    fund_id: str,
) -> FundFactSheet:
    """Assemble the complete factsheet payload for one fund."""
    catalog = fund_catalog(artifacts)
    match = catalog.loc[catalog["fund_id"].eq(fund_id)]
    if match.empty:
        raise KeyError(f"Unknown fund_id: {fund_id}")

    metric_row = match.iloc[0].copy()
    daily = artifacts.fund_returns.loc[
        artifacts.fund_returns["fund_id"].eq(fund_id)
    ].sort_values("date").reset_index(drop=True)
    holdings_date, holdings = latest_holdings(artifacts, fund_id)
    return FundFactSheet(
        fund_id=fund_id,
        display_name=str(metric_row["display_name"]),
        universe=str(metric_row["universe"]),
        method=str(metric_row["method"]),
        metrics=metric_row,
        daily_returns=daily,
        current_holdings=holdings,
        current_holdings_date=holdings_date,
    )


def sector_sentiment_view(
    artifacts: AppArtifacts,
    sectors: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return a sorted, optional sector subset for the sentiment page."""
    data = artifacts.sector_sentiment.copy()
    if sectors is not None:
        requested = set(sectors)
        available = set(data["sector"])
        unknown = requested.difference(available)
        if unknown:
            raise KeyError(f"Unknown sectors: {sorted(unknown)}")
        data = data.loc[data["sector"].isin(requested)]
    return data.sort_values(["date", "sector"]).reset_index(drop=True)


def sentiment_validation_summary(artifacts: AppArtifacts) -> pd.DataFrame:
    """Return comparable development and locked-holdout model metrics."""
    columns = [
        "model_name",
        "observations",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    development = artifacts.sentiment_development_metrics[columns].copy()
    development.insert(0, "evaluation_split", "Development")
    holdout = artifacts.sentiment_holdout_metrics[columns].copy()
    holdout.insert(0, "evaluation_split", "Locked holdout")
    return pd.concat([development, holdout], ignore_index=True)


def fusion_evidence(
    artifacts: AppArtifacts,
    base_fund_id: str,
    evaluation_period: str = "locked_holdout_2023",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return before-after and cost evidence for one base fund and period."""
    filters = (
        artifacts.fusion_comparison["base_fund_id"].eq(base_fund_id)
        & artifacts.fusion_comparison["evaluation_period"].eq(
            evaluation_period
        )
    )
    comparison = artifacts.fusion_comparison.loc[filters].copy()
    cost_filters = (
        artifacts.fusion_costs["base_fund_id"].eq(base_fund_id)
        & artifacts.fusion_costs["evaluation_period"].eq(evaluation_period)
    )
    costs = artifacts.fusion_costs.loc[cost_filters].copy()
    if comparison.empty or costs.empty:
        raise KeyError(
            f"No fusion evidence for {base_fund_id} in {evaluation_period}."
        )
    return (
        comparison.sort_values(["fusion_rule", "variant_label"])
        .reset_index(drop=True),
        costs.sort_values(["cost_bps", "fusion_rule", "variant_label"])
        .reset_index(drop=True),
    )
