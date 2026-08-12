"""Reproduce the Project B results from the project root.

Run with::

    python scripts/run_part_b.py
"""

import hashlib
import pathlib
import sys

import nltk
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402, I001
    etl,
    features,
    fusion,
    portfolios,
    sentiment,
    visuals,
)


INITIAL_ESTIMATION_END = "2020-12-31"
TRANSACTION_COST_BPS = (0.0, 10.0, 50.0)
SENTIMENT_CLASSES = ("negative", "neutral", "positive")
SENTIMENT_ERROR_TYPES = {
    "ambiguous",
    "context_error",
    "correct",
    "finance_lexicon_gap",
    "target_mismatch",
}
DEVELOPMENT_LABELS_PATH = (
    PROJECT_ROOT / "results" / "data" / "sentiment_development_labels.csv"
)
HOLDOUT_LABELS_PATH = (
    PROJECT_ROOT / "results" / "data" / "sentiment_holdout_labels.csv"
)
HOLDOUT_CANDIDATES_PATH = (
    PROJECT_ROOT / "results" / "data" / "sentiment_holdout_candidates.csv"
)
FROZEN_SENTIMENT_RULE_SHA256 = (
    "24A8151D790268C8FA715DC96EDE8ED53D300B945004E492166239A2981F6041"
)
LOCKED_HOLDOUT_SHA256 = (
    "0AC5C5972DF398881F40A462A1170677509DA07FE0B3682796382C0425EFAC26"
)
UNIVERSE_PERIODS = {
    "Equity": 252,
    "Crypto": 365,
    "Combined": 252,
}
PRIMARY_FUSION_BASE_FUND = "combined_hierarchical_risk_parity"
FUSION_VARIANTS = (
    (fusion.NAIVE_FUSION_RULE, fusion.NAIVE_TILT_STRENGTH, "Naive Sentiment"),
    (
        fusion.PRIMARY_FUSION_RULE,
        fusion.PRIMARY_TILT_STRENGTH,
        "Coverage-Aware Rank Sentiment",
    ),
)
FUSION_PERIODS = (
    ("full_oos", None, None),
    ("development_2021_2022", None, pd.Timestamp(fusion.FUSION_HOLDOUT_START)),
    ("locked_holdout_2023", pd.Timestamp(fusion.FUSION_HOLDOUT_START), None),
)


def _wide_return_panel(return_rows: pd.DataFrame) -> pd.DataFrame:
    """Convert long ticker returns into a complete, date-indexed panel."""
    panel = (
        return_rows.pivot(index="date", columns="ticker", values="return")
        .sort_index()
        .dropna(how="any")
    )
    panel.columns.name = None
    return panel


def build_return_panels() -> dict[str, pd.DataFrame]:
    """Load clean prices and construct the three fund universes."""
    equities = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()

    equity_returns = features.daily_returns(equities)
    crypto_returns = features.daily_returns(crypto)

    equity_panel = _wide_return_panel(equity_returns)
    crypto_panel = _wide_return_panel(crypto_returns)
    combined_panel = (
        features.build_combined_returns_panel(equity_returns, crypto_returns)
        .set_index("date")
        .sort_index()
        .dropna(how="any")
    )

    return {
        "Equity": equity_panel,
        "Crypto": crypto_panel,
        "Combined": combined_panel,
    }


def build_portfolio_artifacts(
    return_panels: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the twelve baseline funds plus the combined innovation fund."""
    daily_frames = []
    weight_frames = []
    audit_frames = []
    metric_rows = []

    for universe, periods_per_year in UNIVERSE_PERIODS.items():
        panel = return_panels[universe]
        for method in portfolios.METHODS:
            fund_id = f"{universe.lower()}_{method}"
            print(f"Running {fund_id} ...")
            result = portfolios.oos_backtest(
                panel,
                method,
                universe=universe,
                fund_id=fund_id,
                initial_estimation_end=INITIAL_ESTIMATION_END,
                periods_per_year=periods_per_year,
                risk_free_rate=0.0,
            )
            daily_frames.append(result.daily_returns)
            weight_frames.append(result.target_weights)
            audit_frames.append(result.rebalance_audit)
            metric_rows.append(result.metrics)

    combined_panel = return_panels["Combined"]
    sleeve_assets = {
        "Equity": [
            asset
            for asset in combined_panel.columns
            if not asset.endswith("-USD")
        ],
        "Crypto": [
            asset
            for asset in combined_panel.columns
            if asset.endswith("-USD")
        ],
    }
    innovation_fund_id = "combined_hierarchical_risk_parity"
    print(f"Running {innovation_fund_id} ...")
    innovation = portfolios.oos_backtest(
        combined_panel,
        method=portfolios.HIERARCHICAL_METHOD,
        universe="Combined",
        fund_id=innovation_fund_id,
        initial_estimation_end=INITIAL_ESTIMATION_END,
        periods_per_year=UNIVERSE_PERIODS["Combined"],
        risk_free_rate=0.0,
        sleeve_assets=sleeve_assets,
    )
    daily_frames.append(innovation.daily_returns)
    weight_frames.append(innovation.target_weights)
    audit_frames.append(innovation.rebalance_audit)
    metric_rows.append(innovation.metrics)

    fund_returns = pd.concat(daily_frames, ignore_index=True)
    fund_weights = pd.concat(weight_frames, ignore_index=True)
    rebalance_audit = pd.concat(audit_frames, ignore_index=True)
    performance_metrics = pd.DataFrame(metric_rows)

    fund_returns = fund_returns.sort_values(["fund_id", "date"]).reset_index(
        drop=True
    )
    fund_weights = fund_weights.sort_values(
        ["fund_id", "effective_start_date", "ticker"]
    ).reset_index(drop=True)
    rebalance_audit = rebalance_audit.sort_values(
        ["fund_id", "effective_start_date"]
    ).reset_index(drop=True)
    performance_metrics = performance_metrics.sort_values(
        ["universe", "method"]
    ).reset_index(drop=True)

    return fund_returns, fund_weights, rebalance_audit, performance_metrics


def validate_portfolio_artifacts(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    rebalance_audit: pd.DataFrame,
    performance_metrics: pd.DataFrame,
) -> None:
    """Fail before export if the fund shelf or time ordering is incomplete."""
    expected_funds = {
        f"{universe.lower()}_{method}"
        for universe in UNIVERSE_PERIODS
        for method in portfolios.METHODS
    }
    expected_funds.add("combined_hierarchical_risk_parity")
    artifact_funds = {
        "returns": set(fund_returns["fund_id"]),
        "weights": set(fund_weights["fund_id"]),
        "audit": set(rebalance_audit["fund_id"]),
        "metrics": set(performance_metrics["fund_id"]),
    }
    for artifact_name, observed_funds in artifact_funds.items():
        if observed_funds != expected_funds:
            raise ValueError(
                f"{artifact_name} fund shelf differs from the expected 13 funds."
            )

    weight_sums = fund_weights.groupby(
        ["fund_id", "effective_start_date"]
    )["target_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-8):
        raise ValueError("At least one target portfolio is not fully invested.")

    weights = fund_weights["target_weight"].to_numpy()
    if not np.isfinite(weights).all() or (weights < -1e-10).any():
        raise ValueError("Fund weights contain a non-finite or negative value.")

    if not np.isfinite(fund_returns["return"].to_numpy()).all():
        raise ValueError("Fund returns contain a non-finite value.")

    return_dates = pd.to_datetime(fund_returns["date"])
    effective_dates = pd.to_datetime(fund_returns["effective_start_date"])
    decision_dates = pd.to_datetime(fund_returns["decision_date"])
    if (return_dates < effective_dates).any() or (
        decision_dates >= effective_dates
    ).any():
        raise ValueError("A fund return violates the decision/effective-date order.")

    audit_window_end = pd.to_datetime(rebalance_audit["window_end_date"])
    audit_decision = pd.to_datetime(rebalance_audit["decision_date"])
    audit_effective = pd.to_datetime(rebalance_audit["effective_start_date"])
    if (audit_window_end > audit_decision).any() or (
        audit_decision >= audit_effective
    ).any():
        raise ValueError("The rebalance audit contains look-ahead timing.")


def build_transaction_cost_robustness(
    fund_returns: pd.DataFrame,
    rebalance_audit: pd.DataFrame,
    performance_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every combined fund under three one-way cost assumptions."""
    combined_metrics = performance_metrics.loc[
        performance_metrics["universe"] == "Combined"
    ].set_index("fund_id")
    rows = []

    for fund_id, baseline in combined_metrics.iterrows():
        daily = fund_returns.loc[
            fund_returns["fund_id"] == fund_id,
            ["date", "return"],
        ].copy()
        daily["date"] = pd.to_datetime(daily["date"])
        gross_returns = daily.set_index("date")["return"]

        fund_audit = rebalance_audit.loc[
            rebalance_audit["fund_id"] == fund_id,
            ["effective_start_date", "one_way_turnover"],
        ].copy()
        fund_audit["effective_start_date"] = pd.to_datetime(
            fund_audit["effective_start_date"]
        )
        turnover = fund_audit.set_index("effective_start_date")[
            "one_way_turnover"
        ]

        for cost_bps in TRANSACTION_COST_BPS:
            net_returns = portfolios.apply_rebalance_transaction_costs(
                gross_returns,
                turnover,
                cost_bps,
            )
            scenario_metrics = portfolios.performance_metrics(
                net_returns,
                periods_per_year=int(baseline["periods_per_year"]),
                risk_free_rate=float(baseline["risk_free_rate"]),
            )
            scenario_metrics.update(
                {
                    "fund_id": fund_id,
                    "universe": baseline["universe"],
                    "method": baseline["method"],
                    "method_label": baseline["method_label"],
                    "cost_bps": cost_bps,
                    "total_one_way_turnover": float(turnover.dropna().sum()),
                    "charged_rebalance_count": int(turnover.notna().sum()),
                    "initial_formation_cost_included": False,
                    "cost_model": "one_way_turnover_times_cost_bps",
                }
            )
            rows.append(scenario_metrics)

    return pd.DataFrame(rows).sort_values(
        ["fund_id", "cost_bps"]
    ).reset_index(drop=True)


def validate_transaction_cost_robustness(
    robustness: pd.DataFrame,
    performance_metrics: pd.DataFrame,
) -> None:
    """Check scenario completeness and zero-cost reconciliation."""
    combined = performance_metrics.loc[
        performance_metrics["universe"] == "Combined"
    ]
    expected_rows = len(combined) * len(TRANSACTION_COST_BPS)
    if len(robustness) != expected_rows:
        raise ValueError("Transaction-cost robustness table is incomplete.")

    observed_costs = set(robustness["cost_bps"])
    if observed_costs != set(TRANSACTION_COST_BPS):
        raise ValueError("Transaction-cost scenarios differ from the specification.")

    zero_cost = robustness.loc[
        robustness["cost_bps"] == 0,
        ["fund_id", "total_return", "sharpe_ratio", "max_drawdown"],
    ]
    baseline = combined[
        ["fund_id", "total_return", "sharpe_ratio", "max_drawdown"]
    ]
    reconciled = zero_cost.merge(
        baseline,
        on="fund_id",
        suffixes=("_scenario", "_baseline"),
        validate="one_to_one",
    )
    for metric in ["total_return", "sharpe_ratio", "max_drawdown"]:
        if not np.allclose(
            reconciled[f"{metric}_scenario"],
            reconciled[f"{metric}_baseline"],
            atol=1e-12,
        ):
            raise ValueError(f"Zero-cost {metric} does not match the baseline.")

    total_returns = robustness.pivot(
        index="fund_id",
        columns="cost_bps",
        values="total_return",
    )
    if not (
        (total_returns[0.0] >= total_returns[10.0])
        & (total_returns[10.0] >= total_returns[50.0])
    ).all():
        raise ValueError("Higher transaction costs did not reduce total returns.")


def save_portfolio_artifacts(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    rebalance_audit: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    transaction_cost_robustness: pd.DataFrame,
) -> None:
    """Write the precomputed portfolio artifacts used by the app."""
    data_directory = PROJECT_ROOT / "results" / "data"
    table_directory = PROJECT_ROOT / "results" / "tables"
    data_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)

    fund_returns.to_csv(data_directory / "fund_returns.csv", index=False)
    fund_weights.to_csv(data_directory / "fund_weights.csv", index=False)
    rebalance_audit.to_csv(
        data_directory / "fund_rebalance_audit.csv",
        index=False,
    )
    performance_metrics.to_csv(
        table_directory / "performance_metrics.csv",
        index=False,
    )
    transaction_cost_robustness.to_csv(
        table_directory / "transaction_cost_robustness.csv",
        index=False,
    )


def _sentiment_validation_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    """Select two unique headlines per sector for each diagnostic stratum."""
    unique = scored.drop_duplicates("text_raw").copy()
    strata = {
        "negative": unique["vader_class"].eq("negative"),
        "positive": unique["vader_class"].eq("positive"),
        "zero_score": np.isclose(unique["vader_compound"], 0.0),
    }
    samples = []

    for stratum, mask in strata.items():
        eligible = unique.loc[mask].copy()
        counts = eligible.groupby("sector").size()
        if len(counts) != 10 or counts.min() < 2:
            raise ValueError(
                f"Insufficient sector coverage for the {stratum} validation stratum."
            )
        sample = (
            eligible.groupby("sector", group_keys=False)
            .sample(n=2, random_state=5545)
            .copy()
        )
        sample["sample_stratum"] = stratum
        samples.append(sample)

    candidates = pd.concat(samples, ignore_index=True).sort_values(
        ["sample_stratum", "sector", "trading_date", "ticker"]
    ).reset_index(drop=True)
    candidates.insert(
        0,
        "sample_id",
        [f"SV{number:03d}" for number in range(1, len(candidates) + 1)],
    )
    columns = [
        "sample_id",
        "sample_stratum",
        "date",
        "trading_date",
        "ticker",
        "sector",
        "text_raw",
        "vader_compound",
        "vader_class",
        "vader_neg",
        "vader_neu",
        "vader_pos",
    ]
    return candidates[columns]


def _sentiment_holdout_candidates(
    scored: pd.DataFrame,
    development_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Lock one unseen headline per sector and diagnostic stratum."""
    unique = scored.drop_duplicates("text_raw").copy()
    unique = unique.loc[
        ~unique["text_raw"].isin(development_candidates["text_raw"])
    ]
    strata = {
        "negative": unique["vader_class"].eq("negative"),
        "positive": unique["vader_class"].eq("positive"),
        "zero_score": np.isclose(unique["vader_compound"], 0.0),
    }
    samples = []

    for offset, (stratum, mask) in enumerate(strata.items()):
        eligible = unique.loc[mask].copy()
        counts = eligible.groupby("sector").size()
        if len(counts) != 10 or counts.min() < 1:
            raise ValueError(
                f"Insufficient sector coverage for the {stratum} holdout stratum."
            )
        sample = (
            eligible.groupby("sector", group_keys=False)
            .sample(n=1, random_state=260809 + offset)
            .copy()
        )
        sample["sample_stratum"] = stratum
        samples.append(sample)

    holdout = pd.concat(samples, ignore_index=True).sort_values(
        ["sample_stratum", "sector", "trading_date", "ticker"]
    ).reset_index(drop=True)
    holdout.insert(
        0,
        "sample_id",
        [f"SH{number:03d}" for number in range(1, len(holdout) + 1)],
    )
    columns = [
        "sample_id",
        "sample_stratum",
        "date",
        "trading_date",
        "ticker",
        "sector",
        "text_raw",
        "vader_compound",
        "vader_class",
        "vader_neg",
        "vader_neu",
        "vader_pos",
    ]
    return holdout[columns]


def _merge_sentiment_development_labels(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and attach the approved development labels one-to-one."""
    required = {"sample_id", "approved_label", "error_type", "review_reason"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(
            f"Development labels are missing columns: {sorted(missing)}"
        )

    clean = labels[list(required)].copy()
    for column in required:
        clean[column] = clean[column].astype(str).str.strip()
    clean["approved_label"] = clean["approved_label"].str.lower()
    clean["error_type"] = clean["error_type"].str.lower()

    if clean["sample_id"].duplicated().any():
        raise ValueError("Development labels contain duplicate sample IDs.")
    if not clean["approved_label"].isin(SENTIMENT_CLASSES).all():
        raise ValueError("Development labels contain an invalid sentiment class.")
    if not clean["error_type"].isin(SENTIMENT_ERROR_TYPES).all():
        raise ValueError("Development labels contain an invalid error type.")
    if clean["review_reason"].eq("").any():
        raise ValueError("Development labels contain a blank review reason.")

    candidate_ids = set(candidates["sample_id"].astype(str))
    label_ids = set(clean["sample_id"])
    if candidate_ids != label_ids:
        missing_ids = sorted(candidate_ids.difference(label_ids))
        extra_ids = sorted(label_ids.difference(candidate_ids))
        raise ValueError(
            "Development label IDs do not match the locked candidates: "
            f"missing={missing_ids}, extra={extra_ids}."
        )

    return candidates.merge(
        clean,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )


def _merge_sentiment_holdout_labels(
    holdout: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and attach the approved blind holdout labels one-to-one."""
    required = {"sample_id", "approved_label", "review_reason"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Holdout labels are missing columns: {sorted(missing)}")

    clean = labels[["sample_id", "approved_label", "review_reason"]].copy()
    for column in required:
        clean[column] = clean[column].astype(str).str.strip()
    clean["approved_label"] = clean["approved_label"].str.lower()

    if clean["sample_id"].duplicated().any():
        raise ValueError("Holdout labels contain duplicate sample IDs.")
    if not clean["approved_label"].isin(SENTIMENT_CLASSES).all():
        raise ValueError("Holdout labels contain an invalid sentiment class.")
    if clean["review_reason"].eq("").any():
        raise ValueError("Holdout labels contain a blank review reason.")

    holdout_ids = set(holdout["sample_id"].astype(str))
    label_ids = set(clean["sample_id"])
    if holdout_ids != label_ids:
        missing_ids = sorted(holdout_ids.difference(label_ids))
        extra_ids = sorted(label_ids.difference(holdout_ids))
        raise ValueError(
            "Holdout label IDs do not match the locked candidates: "
            f"missing={missing_ids}, extra={extra_ids}."
        )

    return holdout.merge(
        clean,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )


def _file_sha256(path: pathlib.Path) -> str:
    """Return an uppercase SHA-256 digest for a locked evaluation file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_locked_holdout_inputs() -> None:
    """Refuse evaluation if the frozen model or holdout file has changed."""
    model_path = PROJECT_ROOT / "src" / "sentiment.py"
    for path in (model_path, HOLDOUT_CANDIDATES_PATH, HOLDOUT_LABELS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Locked holdout input not found: {path}")
    if _file_sha256(model_path) != FROZEN_SENTIMENT_RULE_SHA256:
        raise ValueError("Frozen sentiment model hash does not match.")
    if _file_sha256(HOLDOUT_CANDIDATES_PATH) != LOCKED_HOLDOUT_SHA256:
        raise ValueError("Locked holdout candidate hash does not match.")


def _sentiment_classification_metrics(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transparent multiclass review metrics."""
    rows = []
    actual = evaluation["approved_label"]
    for model_name, prediction_column in (
        ("Unmodified NLTK VADER", "baseline_class"),
        (
            "NLTK VADER + conservative finance extension",
            "finance_class",
        ),
    ):
        predicted = evaluation[prediction_column]
        row = {
            "model_name": model_name,
            "observations": len(evaluation),
            "accuracy": predicted.eq(actual).mean(),
        }
        recalls = []
        f1_scores = []
        for sentiment_class in SENTIMENT_CLASSES:
            true_positive = (
                actual.eq(sentiment_class) & predicted.eq(sentiment_class)
            ).sum()
            actual_count = actual.eq(sentiment_class).sum()
            predicted_count = predicted.eq(sentiment_class).sum()
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / actual_count if actual_count else 0.0
            f1_score = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            row[f"{sentiment_class}_precision"] = precision
            row[f"{sentiment_class}_recall"] = recall
            row[f"{sentiment_class}_f1"] = f1_score
            recalls.append(recall)
            f1_scores.append(f1_score)
        row["balanced_accuracy"] = float(np.mean(recalls))
        row["macro_f1"] = float(np.mean(f1_scores))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_sentiment_development_evaluation(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the frozen baseline and finance model on approved labels."""
    if not DEVELOPMENT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Approved development labels not found: {DEVELOPMENT_LABELS_PATH}"
        )
    labels = pd.read_csv(DEVELOPMENT_LABELS_PATH)
    evaluation = _merge_sentiment_development_labels(candidates, labels)
    finance_scored = sentiment.score_headlines(
        candidates[
            ["date", "trading_date", "ticker", "sector", "text_raw"]
        ],
        finance_aware=True,
    )
    if not finance_scored["text_raw"].equals(candidates["text_raw"]):
        raise ValueError("Finance scoring changed development sample order.")

    evaluation = evaluation.rename(
        columns={
            "vader_compound": "baseline_compound",
            "vader_class": "baseline_class",
        }
    )
    evaluation["finance_compound"] = finance_scored["vader_compound"]
    evaluation["finance_class"] = finance_scored["vader_class"]
    evaluation["baseline_correct"] = evaluation["baseline_class"].eq(
        evaluation["approved_label"]
    )
    evaluation["finance_correct"] = evaluation["finance_class"].eq(
        evaluation["approved_label"]
    )
    evaluation["classification_changed"] = ~evaluation["baseline_class"].eq(
        evaluation["finance_class"]
    )
    columns = [
        "sample_id",
        "sample_stratum",
        "date",
        "trading_date",
        "ticker",
        "sector",
        "text_raw",
        "approved_label",
        "error_type",
        "review_reason",
        "baseline_compound",
        "baseline_class",
        "finance_compound",
        "finance_class",
        "baseline_correct",
        "finance_correct",
        "classification_changed",
    ]
    evaluation = evaluation[columns]
    metrics = _sentiment_classification_metrics(evaluation)
    return evaluation, metrics


def _build_sentiment_holdout_evaluation(
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both frozen models once on approved blind holdout labels."""
    _validate_locked_holdout_inputs()
    labels = pd.read_csv(HOLDOUT_LABELS_PATH)
    evaluation = _merge_sentiment_holdout_labels(holdout, labels)
    finance_scored = sentiment.score_headlines(
        holdout[["date", "trading_date", "ticker", "sector", "text_raw"]],
        finance_aware=True,
    )
    if not finance_scored["text_raw"].equals(holdout["text_raw"]):
        raise ValueError("Finance scoring changed holdout sample order.")

    evaluation = evaluation.rename(
        columns={
            "vader_compound": "baseline_compound",
            "vader_class": "baseline_class",
        }
    )
    evaluation["finance_compound"] = finance_scored["vader_compound"]
    evaluation["finance_class"] = finance_scored["vader_class"]
    evaluation["baseline_correct"] = evaluation["baseline_class"].eq(
        evaluation["approved_label"]
    )
    evaluation["finance_correct"] = evaluation["finance_class"].eq(
        evaluation["approved_label"]
    )
    evaluation["classification_changed"] = ~evaluation["baseline_class"].eq(
        evaluation["finance_class"]
    )
    columns = [
        "sample_id",
        "sample_stratum",
        "date",
        "trading_date",
        "ticker",
        "sector",
        "text_raw",
        "approved_label",
        "review_reason",
        "baseline_compound",
        "baseline_class",
        "finance_compound",
        "finance_class",
        "baseline_correct",
        "finance_correct",
        "classification_changed",
    ]
    evaluation = evaluation[columns]
    metrics = _sentiment_classification_metrics(evaluation)
    metrics.insert(0, "evaluation_split", "locked_holdout")
    metrics["model_rule_sha256"] = FROZEN_SENTIMENT_RULE_SHA256
    metrics["holdout_file_sha256"] = LOCKED_HOLDOUT_SHA256
    return evaluation, metrics


def _build_sentiment_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Score real headlines and build the standalone sentiment artifacts."""
    equities = etl.load_clean_equities()
    headlines = etl.load_clean_headlines()
    equity_calendar = equities["date"].drop_duplicates().sort_values()
    sector_universe = (
        equities[["ticker", "sector"]]
        .drop_duplicates()
        .sort_values(["sector", "ticker"])
        .reset_index(drop=True)
    )
    aligned = features.assemble_headline_panel(
        headlines,
        equity_calendar,
    )

    print(
        f"Scoring {len(aligned):,} headlines "
        f"({aligned['text_raw'].nunique():,} unique strings) with baseline and "
        "finance-aware NLTK VADER ..."
    )
    baseline_scored = sentiment.score_headlines(
        aligned,
        finance_aware=False,
    )
    scored = sentiment.score_headlines(
        aligned,
        finance_aware=True,
    )
    ticker_signal = sentiment.ticker_sentiment_signal(
        scored,
        trading_calendar=equity_calendar,
        sector_universe=sector_universe,
    )
    sector_index = sentiment.sector_sentiment_index(
        scored,
        trading_calendar=equity_calendar,
        sector_universe=sector_universe,
    )
    sector_index["sentiment_expanding_zscore"] = sector_index.groupby(
        "sector", sort=False
    )["sentiment_compound"].transform(features.past_only_expanding_zscore)
    sector_index["standardization_rule"] = (
        features.SENTIMENT_STANDARDIZATION_RULE
    )
    sector_index["standardization_min_periods"] = (
        features.SENTIMENT_STANDARDIZATION_MIN_PERIODS
    )
    # Keep both audit samples locked to the original VADER strata. This makes
    # the reviewed development labels reproducible and prevents the extension
    # from selecting its own evaluation observations.
    candidates = _sentiment_validation_candidates(baseline_scored)
    holdout = _sentiment_holdout_candidates(baseline_scored, candidates)
    development_evaluation, development_metrics = (
        _build_sentiment_development_evaluation(candidates)
    )
    holdout_evaluation, holdout_metrics = _build_sentiment_holdout_evaluation(
        holdout
    )

    outside_calendar = ~scored["trading_date"].isin(equity_calendar)
    changed_classification = ~scored["vader_class"].eq(
        baseline_scored["vader_class"]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "model_name": "NLTK VADER + conservative finance extension",
                "benchmark_model": "Unmodified NLTK VADER",
                "nltk_version": nltk.__version__,
                "score_text_column": "text_raw",
                "vader_class_threshold": sentiment.VADER_THRESHOLD,
                "no_news_rule": sentiment.NO_NEWS_RULE,
                "lag_trading_days": 1,
                "sentiment_standardization_rule": (
                    features.SENTIMENT_STANDARDIZATION_RULE
                ),
                "sentiment_standardization_min_periods": (
                    features.SENTIMENT_STANDARDIZATION_MIN_PERIODS
                ),
                "scored_headline_rows": len(scored),
                "unique_scoring_strings": scored["text_raw"].nunique(),
                "equity_trading_dates": equity_calendar.nunique(),
                "equity_tickers": sector_universe["ticker"].nunique(),
                "sectors": sector_universe["sector"].nunique(),
                "outside_equity_calendar_rows": int(outside_calendar.sum()),
                "finance_lexicon_terms": len(sentiment.FINANCE_VADER_LEXICON),
                "finance_idioms": len(sentiment.FINANCE_VADER_IDIOMS),
                "changed_headline_classifications": int(
                    changed_classification.sum()
                ),
                "changed_headline_classification_share": (
                    changed_classification.mean()
                ),
                "baseline_positive_share": baseline_scored["vader_class"]
                .eq("positive")
                .mean(),
                "baseline_neutral_share": baseline_scored["vader_class"]
                .eq("neutral")
                .mean(),
                "baseline_negative_share": baseline_scored["vader_class"]
                .eq("negative")
                .mean(),
                "baseline_exact_zero_share": np.isclose(
                    baseline_scored["vader_compound"],
                    0.0,
                ).mean(),
                "baseline_mean_compound": baseline_scored[
                    "vader_compound"
                ].mean(),
                "positive_headlines": int(scored["vader_class"].eq("positive").sum()),
                "neutral_headlines": int(scored["vader_class"].eq("neutral").sum()),
                "negative_headlines": int(scored["vader_class"].eq("negative").sum()),
                "positive_share": scored["vader_class"].eq("positive").mean(),
                "neutral_share": scored["vader_class"].eq("neutral").mean(),
                "negative_share": scored["vader_class"].eq("negative").mean(),
                "exact_zero_share": np.isclose(scored["vader_compound"], 0.0).mean(),
                "mean_compound": scored["vader_compound"].mean(),
                "median_compound": scored["vader_compound"].median(),
                "minimum_compound": scored["vader_compound"].min(),
                "maximum_compound": scored["vader_compound"].max(),
            }
        ]
    )

    coverage = (
        sector_index.groupby("sector", as_index=False, observed=True)
        .agg(
            trading_days=("date", "size"),
            total_headlines=("headline_count", "sum"),
            mean_daily_headlines=("headline_count", "mean"),
            mean_ticker_coverage=("coverage_rate", "mean"),
            median_ticker_coverage=("coverage_rate", "median"),
            zero_coverage_days=("coverage_rate", lambda values: values.eq(0).sum()),
            full_coverage_days=("coverage_rate", lambda values: values.eq(1).sum()),
            mean_sentiment_compound=("sentiment_compound", "mean"),
            sentiment_volatility=("sentiment_compound", "std"),
        )
        .sort_values("sector")
        .reset_index(drop=True)
    )

    return (
        ticker_signal,
        sector_index,
        candidates,
        holdout,
        diagnostics,
        coverage,
        development_evaluation,
        development_metrics,
        holdout_evaluation,
        holdout_metrics,
    )


def _validate_sentiment_artifacts(
    ticker_signal: pd.DataFrame,
    sector_index: pd.DataFrame,
    candidates: pd.DataFrame,
    holdout: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: pd.DataFrame,
    development_evaluation: pd.DataFrame,
    development_metrics: pd.DataFrame,
    holdout_evaluation: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
) -> None:
    """Check schemas, complete grids, lag timing, and audit-sample balance."""
    if len(diagnostics) != 1:
        raise ValueError("Sentiment diagnostics must contain exactly one model row.")
    metadata = diagnostics.iloc[0]
    expected_ticker_rows = int(
        metadata["equity_trading_dates"] * metadata["equity_tickers"]
    )
    expected_sector_rows = int(
        metadata["equity_trading_dates"] * metadata["sectors"]
    )
    if len(ticker_signal) != expected_ticker_rows:
        raise ValueError("Ticker sentiment grid is incomplete.")
    if len(sector_index) != expected_sector_rows:
        raise ValueError("Sector sentiment grid is incomplete.")
    required_standardization = {
        "sentiment_expanding_zscore",
        "standardization_rule",
        "standardization_min_periods",
    }
    if not required_standardization.issubset(sector_index.columns):
        raise ValueError("Sector sentiment standardization fields are missing.")
    finite_standardized = sector_index["sentiment_expanding_zscore"].dropna()
    if finite_standardized.empty or not np.isfinite(finite_standardized).all():
        raise ValueError("Standardised sector sentiment must contain finite values.")
    if ticker_signal.duplicated(["date", "ticker"]).any():
        raise ValueError("Ticker sentiment contains duplicate ticker-dates.")
    if sector_index.duplicated(["date", "sector"]).any():
        raise ValueError("Sector sentiment contains duplicate sector-dates.")
    if not sector_index["sector_ticker_count"].eq(5).all():
        raise ValueError("Every sector-day must equal-weight five tickers.")

    available_ticker = ticker_signal.loc[ticker_signal["signal_available"]]
    available_sector = sector_index.loc[sector_index["signal_available"]]
    if not (available_ticker["signal_source_date"] < available_ticker["date"]).all():
        raise ValueError("Ticker sentiment lag contains look-ahead timing.")
    if not (available_sector["signal_source_date"] < available_sector["date"]).all():
        raise ValueError("Sector sentiment lag contains look-ahead timing.")

    first_date = ticker_signal["date"].min()
    if ticker_signal.loc[ticker_signal["date"] == first_date, "signal_available"].any():
        raise ValueError("The first ticker signal date cannot have lagged information.")
    if sector_index.loc[sector_index["date"] == first_date, "signal_available"].any():
        raise ValueError("The first sector signal date cannot have lagged information.")

    if len(candidates) != 60:
        raise ValueError("Sentiment validation sample must contain 60 headlines.")
    if not candidates.groupby("sample_stratum").size().eq(20).all():
        raise ValueError("Each sentiment validation stratum must contain 20 rows.")
    if not candidates.groupby("sector").size().eq(6).all():
        raise ValueError("Each sector must contribute six validation headlines.")
    if not candidates.groupby(["sample_stratum", "sector"]).size().eq(2).all():
        raise ValueError("Each validation sector-stratum must contain two rows.")
    if len(holdout) != 30:
        raise ValueError("Locked sentiment holdout must contain 30 headlines.")
    if not holdout.groupby("sample_stratum").size().eq(10).all():
        raise ValueError("Each locked holdout stratum must contain ten rows.")
    if not holdout.groupby("sector").size().eq(3).all():
        raise ValueError("Each sector must contribute three locked holdout headlines.")
    if not holdout.groupby(["sample_stratum", "sector"]).size().eq(1).all():
        raise ValueError("Each locked holdout sector-stratum must contain one row.")
    if candidates["text_raw"].isin(holdout["text_raw"]).any():
        raise ValueError("Development and locked holdout headlines must not overlap.")
    if len(coverage) != int(metadata["sectors"]):
        raise ValueError("Sector coverage summary is incomplete.")
    if len(development_evaluation) != len(candidates):
        raise ValueError("Development evaluation must cover every candidate.")
    if development_evaluation["sample_id"].duplicated().any():
        raise ValueError("Development evaluation contains duplicate sample IDs.")
    if not development_evaluation["approved_label"].isin(SENTIMENT_CLASSES).all():
        raise ValueError("Development evaluation contains an invalid label.")
    if len(development_metrics) != 2:
        raise ValueError("Development metrics must compare exactly two models.")
    if not development_metrics["observations"].eq(len(candidates)).all():
        raise ValueError("Development metrics use an incorrect sample size.")
    if len(holdout_evaluation) != len(holdout):
        raise ValueError("Holdout evaluation must cover every locked candidate.")
    if holdout_evaluation["sample_id"].duplicated().any():
        raise ValueError("Holdout evaluation contains duplicate sample IDs.")
    if not holdout_evaluation["approved_label"].isin(SENTIMENT_CLASSES).all():
        raise ValueError("Holdout evaluation contains an invalid label.")
    if len(holdout_metrics) != 2:
        raise ValueError("Holdout metrics must compare exactly two models.")
    if not holdout_metrics["observations"].eq(len(holdout)).all():
        raise ValueError("Holdout metrics use an incorrect sample size.")
    if not holdout_metrics["model_rule_sha256"].eq(
        FROZEN_SENTIMENT_RULE_SHA256
    ).all():
        raise ValueError("Holdout metrics contain an incorrect model hash.")
    if not holdout_metrics["holdout_file_sha256"].eq(
        LOCKED_HOLDOUT_SHA256
    ).all():
        raise ValueError("Holdout metrics contain an incorrect sample hash.")


def _save_sentiment_artifacts(
    ticker_signal: pd.DataFrame,
    sector_index: pd.DataFrame,
    candidates: pd.DataFrame,
    holdout: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: pd.DataFrame,
    development_evaluation: pd.DataFrame,
    development_metrics: pd.DataFrame,
    holdout_evaluation: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
) -> None:
    """Write precomputed sentiment data and diagnostic tables."""
    data_directory = PROJECT_ROOT / "results" / "data"
    table_directory = PROJECT_ROOT / "results" / "tables"
    data_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)

    sector_index.to_csv(
        data_directory / "sector_sentiment_index.csv",
        index=False,
    )
    ticker_signal.to_csv(
        data_directory / "ticker_sentiment_signal.csv",
        index=False,
    )
    candidates.to_csv(
        data_directory / "sentiment_validation_candidates.csv",
        index=False,
    )
    holdout.to_csv(
        data_directory / "sentiment_holdout_candidates.csv",
        index=False,
    )
    development_evaluation.to_csv(
        data_directory / "sentiment_development_evaluation.csv",
        index=False,
    )
    holdout_evaluation.to_csv(
        data_directory / "sentiment_holdout_evaluation.csv",
        index=False,
    )
    diagnostics.to_csv(
        table_directory / "sentiment_model_diagnostics.csv",
        index=False,
    )
    coverage.to_csv(
        table_directory / "sector_sentiment_coverage.csv",
        index=False,
    )
    development_metrics.to_csv(
        table_directory / "sentiment_development_metrics.csv",
        index=False,
    )
    holdout_metrics.to_csv(
        table_directory / "sentiment_holdout_metrics.csv",
        index=False,
    )


def _fusion_period_metrics(
    variant_returns: pd.DataFrame,
    variant_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate full, development, and locked-holdout fusion metrics."""
    rows = []
    group_columns = [
        "base_fund_id",
        "fund_id",
        "universe",
        "method",
        "fusion_rule",
        "tilt_strength",
        "variant_label",
    ]
    returns = variant_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    audit = variant_audit.copy()
    audit["effective_start_date"] = pd.to_datetime(
        audit["effective_start_date"]
    )

    for period_name, start, end in FUSION_PERIODS:
        period_returns = returns.copy()
        period_audit = audit.copy()
        if start is not None:
            period_returns = period_returns.loc[period_returns["date"] >= start]
            period_audit = period_audit.loc[
                period_audit["effective_start_date"] >= start
            ]
        if end is not None:
            period_returns = period_returns.loc[period_returns["date"] < end]
            period_audit = period_audit.loc[
                period_audit["effective_start_date"] < end
            ]

        for keys, daily in period_returns.groupby(
            group_columns,
            observed=True,
            sort=False,
        ):
            metadata = dict(zip(group_columns, keys, strict=True))
            fund_audit = period_audit.loc[
                period_audit["fund_id"].eq(metadata["fund_id"])
            ]
            metrics = portfolios.performance_metrics(
                daily.set_index("date")["return"],
                periods_per_year=UNIVERSE_PERIODS[metadata["universe"]],
                risk_free_rate=0.0,
            )
            metrics.update(metadata)
            metrics.update(
                {
                    "evaluation_period": period_name,
                    "period_start": daily["date"].min(),
                    "period_end": daily["date"].max(),
                    "rebalance_count": len(fund_audit),
                    "average_one_way_turnover": float(
                        fund_audit["one_way_turnover"].dropna().mean()
                    ),
                    "average_one_way_active_shift_from_base": float(
                        fund_audit["one_way_active_shift_from_base"].mean()
                    ),
                    "latest_effective_number_of_holdings": float(
                        fund_audit["effective_number_of_holdings"].iloc[-1]
                    ),
                    "primary_base_fund": metadata["base_fund_id"]
                    == PRIMARY_FUSION_BASE_FUND,
                    "primary_fusion_rule": metadata["fusion_rule"]
                    == fusion.PRIMARY_FUSION_RULE,
                }
            )
            rows.append(metrics)

    comparison = pd.DataFrame(rows)
    baseline_columns = [
        "evaluation_period",
        "base_fund_id",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "total_return",
        "average_one_way_turnover",
    ]
    baseline = comparison.loc[
        comparison["fusion_rule"].eq("base"),
        baseline_columns,
    ].rename(
        columns={
            column: f"base_{column}"
            for column in baseline_columns
            if column not in {"evaluation_period", "base_fund_id"}
        }
    )
    comparison = comparison.merge(
        baseline,
        on=["evaluation_period", "base_fund_id"],
        how="left",
        validate="many_to_one",
    )
    for metric in [
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "total_return",
        "average_one_way_turnover",
    ]:
        comparison[f"delta_{metric}"] = (
            comparison[metric] - comparison[f"base_{metric}"]
        )
    comparison["method_label"] = comparison["method"].map(
        portfolios.METHOD_LABELS
    )
    return comparison.sort_values(
        ["evaluation_period", "base_fund_id", "fusion_rule"]
    ).reset_index(drop=True)


def _fusion_cost_robustness(
    variant_returns: pd.DataFrame,
    variant_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the pre-specified one-way costs to every fusion comparison."""
    rows = []
    group_columns = [
        "base_fund_id",
        "fund_id",
        "universe",
        "method",
        "fusion_rule",
        "tilt_strength",
        "variant_label",
    ]
    audit = variant_audit.copy()
    audit["effective_start_date"] = pd.to_datetime(
        audit["effective_start_date"]
    )

    for keys, daily in variant_returns.groupby(
        group_columns,
        observed=True,
        sort=False,
    ):
        metadata = dict(zip(group_columns, keys, strict=True))
        dated_returns = daily[["date", "return"]].copy()
        dated_returns["date"] = pd.to_datetime(dated_returns["date"])
        fund_audit = audit.loc[
            audit["fund_id"].eq(metadata["fund_id"]),
            ["effective_start_date", "one_way_turnover"],
        ]

        for period_name, start, end in FUSION_PERIODS:
            period_returns = dated_returns.copy()
            period_audit = fund_audit.copy()
            if start is not None:
                period_returns = period_returns.loc[
                    period_returns["date"] >= start
                ]
                period_audit = period_audit.loc[
                    period_audit["effective_start_date"] >= start
                ]
            if end is not None:
                period_returns = period_returns.loc[period_returns["date"] < end]
                period_audit = period_audit.loc[
                    period_audit["effective_start_date"] < end
                ]
            gross_returns = period_returns.set_index("date")[
                "return"
            ].sort_index()
            turnover = period_audit.set_index("effective_start_date")[
                "one_way_turnover"
            ]

            for cost_bps in TRANSACTION_COST_BPS:
                net_returns = portfolios.apply_rebalance_transaction_costs(
                    gross_returns,
                    turnover,
                    cost_bps,
                )
                metrics = portfolios.performance_metrics(
                    net_returns,
                    periods_per_year=UNIVERSE_PERIODS[metadata["universe"]],
                    risk_free_rate=0.0,
                )
                metrics.update(metadata)
                metrics.update(
                    {
                        "evaluation_period": period_name,
                        "period_start": period_returns["date"].min(),
                        "period_end": period_returns["date"].max(),
                        "cost_bps": cost_bps,
                        "total_one_way_turnover": float(turnover.dropna().sum()),
                        "average_one_way_turnover": float(
                            turnover.dropna().mean()
                        ),
                        "charged_rebalance_count": int(turnover.notna().sum()),
                        "initial_formation_cost_included": False,
                        "cost_model": "one_way_turnover_times_cost_bps",
                    }
                )
                rows.append(metrics)

    robustness = pd.DataFrame(rows)
    baseline = robustness.loc[
        robustness["fusion_rule"].eq("base"),
        [
            "evaluation_period",
            "base_fund_id",
            "cost_bps",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
        ],
    ].rename(
        columns={
            "total_return": "base_total_return",
            "sharpe_ratio": "base_sharpe_ratio",
            "max_drawdown": "base_max_drawdown",
        }
    )
    robustness = robustness.merge(
        baseline,
        on=["evaluation_period", "base_fund_id", "cost_bps"],
        how="left",
        validate="many_to_one",
    )
    for metric in ["total_return", "sharpe_ratio", "max_drawdown"]:
        robustness[f"delta_{metric}"] = (
            robustness[metric] - robustness[f"base_{metric}"]
        )
    robustness["method_label"] = robustness["method"].map(
        portfolios.METHOD_LABELS
    )
    return robustness.sort_values(
        ["evaluation_period", "base_fund_id", "cost_bps", "fusion_rule"]
    ).reset_index(drop=True)


def _build_fusion_artifacts(
    return_panels: dict[str, pd.DataFrame],
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    rebalance_audit: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    ticker_signal: pd.DataFrame,
    sector_index: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the frozen benchmark and coverage-aware sentiment overlays."""
    eligible_metrics = performance_metrics.loc[
        performance_metrics["universe"].isin(["Equity", "Combined"])
    ].copy()
    base_fund_ids = set(eligible_metrics["fund_id"])
    ticker_sectors = ticker_signal[["ticker", "sector"]].drop_duplicates()
    fusion_daily_frames = []
    fusion_weight_frames = []
    fusion_audit_frames = []

    for base_fund_id in sorted(base_fund_ids):
        metadata = eligible_metrics.loc[
            eligible_metrics["fund_id"].eq(base_fund_id)
        ].iloc[0]
        universe = str(metadata["universe"])
        panel = return_panels[universe]
        base_target_rows = fund_weights.loc[
            fund_weights["fund_id"].eq(base_fund_id)
        ].copy()

        zero_daily, _zero_weights, _zero_audit = (
            fusion._backtest_sentiment_overlay(
                panel,
                base_target_rows,
                sector_index,
                ticker_sectors,
                rule=fusion.PRIMARY_FUSION_RULE,
                tilt_strength=0.0,
            )
        )
        baseline_daily = fund_returns.loc[
            fund_returns["fund_id"].eq(base_fund_id),
            ["date", "return"],
        ].sort_values("date")
        if not pd.to_datetime(zero_daily["date"]).reset_index(drop=True).equals(
            pd.to_datetime(baseline_daily["date"]).reset_index(drop=True)
        ):
            raise ValueError(
                f"Zero-strength dates do not reconcile for {base_fund_id}."
            )
        if not np.allclose(
            zero_daily["return"],
            baseline_daily["return"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"Zero-strength returns do not reconcile for {base_fund_id}."
            )

        for rule, strength, variant_label in FUSION_VARIANTS:
            print(f"Running {base_fund_id} + {rule} sentiment ...")
            daily, weights, audit = fusion._backtest_sentiment_overlay(
                panel,
                base_target_rows,
                sector_index,
                ticker_sectors,
                rule=rule,
                tilt_strength=strength,
            )
            daily["variant_label"] = variant_label
            weights["variant_label"] = variant_label
            audit["variant_label"] = variant_label
            fusion_daily_frames.append(daily)
            fusion_weight_frames.append(weights)
            fusion_audit_frames.append(audit)

    fusion_returns = pd.concat(fusion_daily_frames, ignore_index=True).sort_values(
        ["fund_id", "date"]
    ).reset_index(drop=True)
    fusion_weights = pd.concat(
        fusion_weight_frames,
        ignore_index=True,
    ).sort_values(["fund_id", "effective_start_date", "ticker"]).reset_index(
        drop=True
    )
    fusion_audit = pd.concat(fusion_audit_frames, ignore_index=True).sort_values(
        ["fund_id", "effective_start_date"]
    ).reset_index(drop=True)

    signal_columns = [
        "decision_date",
        "effective_start_date",
        "fusion_rule",
        "tilt_strength",
        "variant_label",
        "sector",
        "sentiment_date",
        "signal_source_date",
        "lagged_sentiment_compound",
        "lagged_coverage_rate",
        "sector_signal",
        "sentiment_multiplier",
    ]
    signal_audit = (
        fusion_weights.loc[
            fusion_weights["asset_class"].eq("Equity"),
            signal_columns,
        ]
        .drop_duplicates()
        .sort_values(["decision_date", "fusion_rule", "sector"])
        .reset_index(drop=True)
    )

    base_returns = fund_returns.loc[
        fund_returns["fund_id"].isin(base_fund_ids)
    ].copy()
    base_returns["base_fund_id"] = base_returns["fund_id"]
    base_returns["fusion_rule"] = "base"
    base_returns["tilt_strength"] = 0.0
    base_returns["variant_label"] = "Base Fund"
    variant_returns = pd.concat(
        [base_returns, fusion_returns],
        ignore_index=True,
        sort=False,
    )

    base_audit = rebalance_audit.loc[
        rebalance_audit["fund_id"].isin(base_fund_ids)
    ].copy()
    base_audit["base_fund_id"] = base_audit["fund_id"]
    base_audit["fusion_rule"] = "base"
    base_audit["tilt_strength"] = 0.0
    base_audit["variant_label"] = "Base Fund"
    base_audit["one_way_active_shift_from_base"] = 0.0
    variant_audit = pd.concat(
        [base_audit, fusion_audit],
        ignore_index=True,
        sort=False,
    )

    period_comparison = _fusion_period_metrics(
        variant_returns,
        variant_audit,
    )
    performance_comparison = period_comparison.loc[
        period_comparison["evaluation_period"].eq("full_oos")
    ].reset_index(drop=True)
    cost_robustness = _fusion_cost_robustness(
        variant_returns,
        variant_audit,
    )
    return (
        fusion_returns,
        fusion_weights,
        fusion_audit,
        signal_audit,
        performance_comparison,
        period_comparison,
        cost_robustness,
    )


def _validate_fusion_artifacts(
    fusion_returns: pd.DataFrame,
    fusion_weights: pd.DataFrame,
    fusion_audit: pd.DataFrame,
    signal_audit: pd.DataFrame,
    performance_comparison: pd.DataFrame,
    period_comparison: pd.DataFrame,
    cost_robustness: pd.DataFrame,
) -> None:
    """Check shelf completeness, timing, sleeve preservation, and costs."""
    base_funds = set(performance_comparison["base_fund_id"])
    if len(base_funds) != 9 or PRIMARY_FUSION_BASE_FUND not in base_funds:
        raise ValueError("Fusion must cover four equity and five combined funds.")
    expected_augmented = {
        f"{base_fund}_{rule}_sentiment"
        for base_fund in base_funds
        for rule, _strength, _label in FUSION_VARIANTS
    }
    for name, frame in {
        "returns": fusion_returns,
        "weights": fusion_weights,
        "audit": fusion_audit,
    }.items():
        if set(frame["fund_id"]) != expected_augmented:
            raise ValueError(f"Fusion {name} shelf is incomplete.")

    weight_sums = fusion_weights.groupby(
        ["fund_id", "effective_start_date"]
    )["target_weight"].sum()
    if not np.allclose(weight_sums, 1.0, atol=1e-10):
        raise ValueError("A fusion target is not fully invested.")
    weights = fusion_weights["target_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < -1e-12).any():
        raise ValueError("Fusion targets contain invalid weights.")

    equity = fusion_weights["asset_class"].eq("Equity")
    crypto = fusion_weights["asset_class"].eq("Crypto")
    if not (
        pd.to_datetime(fusion_weights.loc[equity, "signal_source_date"])
        < pd.to_datetime(fusion_weights.loc[equity, "decision_date"])
    ).all():
        raise ValueError("Fusion weights contain look-ahead sentiment.")
    if not (
        pd.to_datetime(fusion_weights["decision_date"])
        < pd.to_datetime(fusion_weights["effective_start_date"])
    ).all():
        raise ValueError("Fusion decisions do not precede holding periods.")
    if not np.allclose(
        fusion_weights.loc[crypto, "target_weight"],
        fusion_weights.loc[crypto, "base_target_weight"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("A fusion rule changed crypto target weights.")

    sleeve_totals = fusion_weights.groupby(
        ["fund_id", "effective_start_date", "asset_class"],
        observed=True,
    )[["target_weight", "base_target_weight"]].sum()
    if not np.allclose(
        sleeve_totals["target_weight"],
        sleeve_totals["base_target_weight"],
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("A fusion rule changed an asset-class sleeve weight.")

    decision_count = fusion_weights["decision_date"].nunique()
    sector_count = fusion_weights.loc[equity, "sector"].nunique()
    expected_signal_rows = decision_count * sector_count * len(FUSION_VARIANTS)
    if len(signal_audit) != expected_signal_rows:
        raise ValueError("Fusion signal audit is incomplete.")
    if signal_audit.duplicated(
        ["decision_date", "fusion_rule", "sector"]
    ).any():
        raise ValueError("Fusion signal audit contains duplicate sector signals.")

    expected_comparison_rows = len(base_funds) * (1 + len(FUSION_VARIANTS))
    if len(performance_comparison) != expected_comparison_rows:
        raise ValueError("Full-OOS fusion comparison is incomplete.")
    if len(period_comparison) != expected_comparison_rows * len(FUSION_PERIODS):
        raise ValueError("Fusion development/holdout comparison is incomplete.")
    if set(period_comparison["evaluation_period"]) != {
        period[0] for period in FUSION_PERIODS
    }:
        raise ValueError("Fusion evaluation periods differ from the specification.")

    expected_cost_rows = (
        expected_comparison_rows
        * len(FUSION_PERIODS)
        * len(TRANSACTION_COST_BPS)
    )
    if len(cost_robustness) != expected_cost_rows:
        raise ValueError("Fusion transaction-cost scenarios are incomplete.")
    if set(cost_robustness["cost_bps"]) != set(TRANSACTION_COST_BPS):
        raise ValueError("Fusion transaction costs differ from the specification.")
    if set(cost_robustness["evaluation_period"]) != {
        period[0] for period in FUSION_PERIODS
    }:
        raise ValueError("Fusion cost periods differ from the specification.")
    zero_cost = cost_robustness.loc[
        cost_robustness["cost_bps"].eq(0),
        [
            "evaluation_period",
            "fund_id",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
        ],
    ]
    gross = period_comparison[
        [
            "evaluation_period",
            "fund_id",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
        ]
    ]
    reconciled = zero_cost.merge(
        gross,
        on=["evaluation_period", "fund_id"],
        suffixes=("_cost", "_gross"),
        validate="one_to_one",
    )
    for metric in ["total_return", "sharpe_ratio", "max_drawdown"]:
        if not np.allclose(
            reconciled[f"{metric}_cost"],
            reconciled[f"{metric}_gross"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"Zero-cost fusion {metric} does not reconcile.")

    total_returns = cost_robustness.pivot(
        index=["evaluation_period", "fund_id"],
        columns="cost_bps",
        values="total_return",
    )
    if not (
        (total_returns[0.0] >= total_returns[10.0])
        & (total_returns[10.0] >= total_returns[50.0])
    ).all():
        raise ValueError("Higher fusion transaction costs did not reduce returns.")


def _save_fusion_artifacts(
    fusion_returns: pd.DataFrame,
    fusion_weights: pd.DataFrame,
    fusion_audit: pd.DataFrame,
    signal_audit: pd.DataFrame,
    performance_comparison: pd.DataFrame,
    period_comparison: pd.DataFrame,
    cost_robustness: pd.DataFrame,
) -> None:
    """Write reproducible fusion data and comparison tables."""
    data_directory = PROJECT_ROOT / "results" / "data"
    table_directory = PROJECT_ROOT / "results" / "tables"
    data_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)
    fusion_returns.to_csv(data_directory / "fusion_returns.csv", index=False)
    fusion_weights.to_csv(data_directory / "fusion_weights.csv", index=False)
    fusion_audit.to_csv(
        data_directory / "fusion_rebalance_audit.csv",
        index=False,
    )
    signal_audit.to_csv(
        data_directory / "fusion_signal_audit.csv",
        index=False,
    )
    performance_comparison.to_csv(
        table_directory / "fusion_performance_comparison.csv",
        index=False,
    )
    period_comparison.to_csv(
        table_directory / "fusion_period_comparison.csv",
        index=False,
    )
    cost_robustness.to_csv(
        table_directory / "fusion_transaction_cost_robustness.csv",
        index=False,
    )


def main() -> None:
    """Build, validate, and save the fund and sentiment artifacts."""
    return_panels = build_return_panels()
    for universe, panel in return_panels.items():
        print(
            f"{universe}: {panel.shape[0]} dates, {panel.shape[1]} assets, "
            f"{panel.index.min().date()} to {panel.index.max().date()}"
        )

    artifacts = build_portfolio_artifacts(return_panels)
    validate_portfolio_artifacts(*artifacts)

    fund_returns, fund_weights, rebalance_audit, performance_metrics = artifacts
    transaction_cost_robustness = build_transaction_cost_robustness(
        fund_returns,
        rebalance_audit,
        performance_metrics,
    )
    validate_transaction_cost_robustness(
        transaction_cost_robustness,
        performance_metrics,
    )
    save_portfolio_artifacts(*artifacts, transaction_cost_robustness)

    print(
        "Saved portfolio artifacts:",
        f"{len(fund_returns):,} fund-return rows,",
        f"{len(fund_weights):,} target-weight rows,",
        f"{len(rebalance_audit):,} rebalance audit rows,",
        f"{len(performance_metrics)} fund metrics rows,",
        f"{len(transaction_cost_robustness)} cost-robustness rows.",
    )

    sentiment_artifacts = _build_sentiment_artifacts()
    _validate_sentiment_artifacts(*sentiment_artifacts)
    _save_sentiment_artifacts(*sentiment_artifacts)
    (
        ticker_signal,
        sector_index,
        candidates,
        holdout,
        diagnostics,
        coverage,
        development_evaluation,
        development_metrics,
        holdout_evaluation,
        holdout_metrics,
    ) = sentiment_artifacts
    print(
        "Saved sentiment artifacts:",
        f"{len(ticker_signal):,} ticker-day rows,",
        f"{len(sector_index):,} sector-day rows,",
        f"{len(candidates)} validation candidates,",
        f"{len(holdout)} locked holdout candidates,",
        f"{len(development_evaluation)} development evaluations,",
        f"{len(development_metrics)} development metric rows,",
        f"{len(holdout_evaluation)} holdout evaluations,",
        f"{len(holdout_metrics)} holdout metric rows,",
        f"{len(diagnostics)} model diagnostic row,",
        f"{len(coverage)} sector coverage rows.",
    )

    fusion_artifacts = _build_fusion_artifacts(
        return_panels,
        fund_returns,
        fund_weights,
        rebalance_audit,
        performance_metrics,
        ticker_signal,
        sector_index,
    )
    _validate_fusion_artifacts(*fusion_artifacts)
    _save_fusion_artifacts(*fusion_artifacts)
    (
        fusion_returns,
        fusion_weights,
        fusion_audit,
        signal_audit,
        fusion_comparison,
        fusion_periods,
        fusion_costs,
    ) = fusion_artifacts
    print(
        "Saved fusion artifacts:",
        f"{len(fusion_returns):,} augmented return rows,",
        f"{len(fusion_weights):,} augmented weight rows,",
        f"{len(fusion_audit):,} augmented rebalance rows,",
        f"{len(signal_audit):,} locked signal rows,",
        f"{len(fusion_comparison)} full-OOS comparison rows,",
        f"{len(fusion_periods)} period comparison rows,",
        f"{len(fusion_costs)} cost-robustness rows.",
    )

    figure_manifest = visuals.build_all_figures(PROJECT_ROOT)
    print(
        "Saved figure artifacts:",
        f"{len(figure_manifest)} report-ready PNGs,",
        "1 figure manifest.",
    )


if __name__ == "__main__":
    main()
