"""Long-only portfolio methods and walk-forward out-of-sample backtests."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHODS = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
)
HIERARCHICAL_METHOD = "hierarchical_risk_parity"

METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    HIERARCHICAL_METHOD: "Hierarchical Sleeve Risk Parity",
}


@dataclass
class BacktestResult:
    """Outputs from one universe-method fund backtest."""

    daily_returns: pd.DataFrame
    target_weights: pd.DataFrame
    rebalance_audit: pd.DataFrame
    metrics: dict


def _validate_return_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, finite, complete daily return matrix."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must use a pandas DatetimeIndex.")
    if returns.empty:
        raise ValueError("Returns cannot be empty.")
    if returns.shape[1] < 2:
        raise ValueError("A portfolio return panel must contain at least two assets.")
    if returns.columns.duplicated().any():
        raise ValueError("Return-panel asset names must be unique.")

    panel = returns.copy().sort_index()
    panel.index = pd.DatetimeIndex(panel.index).normalize()

    if panel.index.duplicated().any():
        raise ValueError("Return-panel dates must be unique.")
    if panel.isna().any().any():
        raise ValueError("Return panel contains missing values.")
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError("Return panel contains non-finite values.")
    if (panel <= -1.0).any().any():
        raise ValueError("Asset returns must be greater than -100%.")

    return panel.astype(float)


def validate_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate long-only, fully invested weights and normalise tiny drift."""
    vector = np.asarray(weights, dtype=float)

    if vector.shape != (n_assets,):
        raise ValueError(
            f"Expected {n_assets} weights, received shape {vector.shape}."
        )
    if not np.isfinite(vector).all():
        raise ValueError("Portfolio weights contain non-finite values.")
    if vector.min() < -1e-8:
        raise ValueError("Long-only portfolio contains a negative weight.")

    vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())

    if total <= 0:
        raise ValueError("Portfolio weights do not have a positive total.")

    vector = vector / total

    if not np.isclose(vector.sum(), 1.0, atol=1e-10):
        raise ValueError("Portfolio weights do not sum to one.")

    return vector


def equal_weight_weights(n_assets: int) -> np.ndarray:
    """Return the long-only 1/N benchmark."""
    if n_assets < 1:
        raise ValueError("Equal weight requires at least one asset.")
    return np.repeat(1.0 / n_assets, n_assets)


def _optimisation_inputs(
    training_returns: pd.DataFrame,
    periods_per_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build annualised mean and covariance inputs for the optimisers."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    mean_returns = training_returns.mean().to_numpy(dtype=float) * periods_per_year
    covariance = training_returns.cov().to_numpy(dtype=float) * periods_per_year
    covariance = 0.5 * (covariance + covariance.T)

    diagonal_scale = float(np.nanmean(np.diag(covariance)))
    if not np.isfinite(diagonal_scale) or diagonal_scale <= 0:
        raise ValueError("Training covariance does not have positive variance.")

    covariance = covariance + np.eye(covariance.shape[0]) * diagonal_scale * 1e-10
    return mean_returns, covariance


def _optimisation_constraints(n_assets: int) -> tuple[list, list]:
    """Return SLSQP long-only bounds and a fully invested constraint."""
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        }
    ]
    return bounds, constraints


def minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """Solve the long-only, fully invested minimum-variance portfolio."""
    matrix = np.asarray(covariance, dtype=float)
    n_assets = matrix.shape[0]
    initial = equal_weight_weights(n_assets)
    bounds, constraints = _optimisation_constraints(n_assets)

    scale = float(np.nanmean(np.diag(matrix)))
    scaled_matrix = matrix / scale
    result = minimize(
        fun=lambda weights: float(weights @ scaled_matrix @ weights),
        x0=initial,
        jac=lambda weights: 2.0 * scaled_matrix @ weights,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1_000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"Minimum-variance optimisation failed: {result.message}"
        )

    return validate_weights(result.x, n_assets)


def maximum_sharpe_weights(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Solve the long-only portfolio with the highest ex-ante Sharpe ratio."""
    mean_vector = np.asarray(mean_returns, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    n_assets = len(mean_vector)
    bounds, constraints = _optimisation_constraints(n_assets)

    inverse_vol = 1.0 / np.sqrt(np.diag(matrix))
    inverse_vol = validate_weights(inverse_vol, n_assets)
    starts = [equal_weight_weights(n_assets), inverse_vol]

    def objective(weights: np.ndarray) -> float:
        portfolio_volatility = float(np.sqrt(weights @ matrix @ weights))
        if portfolio_volatility <= 0:
            return 1e12
        excess_return = float(weights @ mean_vector - risk_free_rate)
        return -excess_return / portfolio_volatility

    successful = []
    for initial in starts:
        result = minimize(
            fun=objective,
            x0=initial,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1_000, "ftol": 1e-12},
        )
        if result.success and np.isfinite(result.fun):
            weights = validate_weights(result.x, n_assets)
            successful.append((objective(weights), weights))

    if not successful:
        raise RuntimeError("Maximum-Sharpe optimisation failed for every start.")

    return min(successful, key=lambda item: item[0])[1]


def risk_parity_weights(covariance: np.ndarray) -> np.ndarray:
    """Solve a long-only equal-risk-contribution portfolio."""
    matrix = np.asarray(covariance, dtype=float)
    n_assets = matrix.shape[0]
    inverse_vol = 1.0 / np.sqrt(np.diag(matrix))
    initial = validate_weights(inverse_vol, n_assets)
    bounds = [(1e-10, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    target = np.repeat(1.0 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ matrix @ weights)
        if portfolio_variance <= 0:
            return 1e12
        contributions = weights * (matrix @ weights) / portfolio_variance
        return float(np.sum((contributions - target) ** 2))

    result = minimize(
        fun=objective,
        x0=initial,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 2_000, "ftol": 1e-14},
    )

    if not result.success:
        raise RuntimeError(f"Risk-parity optimisation failed: {result.message}")

    weights = validate_weights(result.x, n_assets)
    if objective(weights) > 1e-8:
        raise RuntimeError("Risk-parity weights do not equalise risk contributions.")

    return weights


def hierarchical_sleeve_risk_parity_weights(
    training_returns: pd.DataFrame,
    sleeve_assets: dict[str, list[str]],
    periods_per_year: int = 252,
) -> tuple[np.ndarray, dict[str, float]]:
    """Allocate equal risk within sleeves and then across the sleeves.

    This two-stage construction prevents the number of securities in each
    asset class from mechanically determining that class's risk budget.
    """
    panel = _validate_return_panel(training_returns)

    if len(sleeve_assets) < 2:
        raise ValueError("Hierarchical risk parity requires at least two sleeves.")

    grouped_assets = [
        asset
        for assets in sleeve_assets.values()
        for asset in assets
    ]
    if len(grouped_assets) != len(set(grouped_assets)):
        raise ValueError("An asset cannot appear in more than one sleeve.")
    if set(grouped_assets) != set(panel.columns):
        raise ValueError("Sleeve assets must cover the return panel exactly.")

    within_sleeve_weights = {}
    sleeve_return_series = {}

    for sleeve_name, assets in sleeve_assets.items():
        if len(assets) < 2:
            raise ValueError("Each sleeve must contain at least two assets.")
        sleeve_panel = panel.loc[:, assets]
        _means, covariance = _optimisation_inputs(
            sleeve_panel,
            periods_per_year,
        )
        weights = risk_parity_weights(covariance)
        within_sleeve_weights[sleeve_name] = weights
        sleeve_return_series[sleeve_name] = sleeve_panel @ weights

    sleeve_returns = pd.DataFrame(sleeve_return_series, index=panel.index)
    _means, sleeve_covariance = _optimisation_inputs(
        sleeve_returns,
        periods_per_year,
    )
    sleeve_weights = risk_parity_weights(sleeve_covariance)
    sleeve_targets = dict(
        zip(sleeve_assets, sleeve_weights, strict=True)
    )

    target = np.zeros(panel.shape[1], dtype=float)
    asset_locations = {asset: index for index, asset in enumerate(panel.columns)}
    for sleeve_name, assets in sleeve_assets.items():
        sleeve_target = sleeve_targets[sleeve_name]
        within_weights = within_sleeve_weights[sleeve_name]
        for asset, within_weight in zip(assets, within_weights, strict=True):
            target[asset_locations[asset]] = sleeve_target * within_weight

    return validate_weights(target, panel.shape[1]), sleeve_targets


def estimate_weights(
    training_returns: pd.DataFrame,
    method: str,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> tuple[np.ndarray, str]:
    """Estimate one target-weight vector and return its solver label."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")

    panel = _validate_return_panel(training_returns)
    mean_returns, covariance = _optimisation_inputs(panel, periods_per_year)
    n_assets = panel.shape[1]

    if method == "equal_weight":
        return equal_weight_weights(n_assets), "direct"
    if method == "minimum_variance":
        return minimum_variance_weights(covariance), "SLSQP"
    if method == "maximum_sharpe":
        weights = maximum_sharpe_weights(
            mean_returns,
            covariance,
            risk_free_rate,
        )
        return weights, "SLSQP_multistart"

    return risk_parity_weights(covariance), "SLSQP_risk_contribution"


def build_monthly_schedule(
    returns: pd.DataFrame,
    initial_estimation_end: str | pd.Timestamp = "2020-12-31",
) -> pd.DataFrame:
    """Build monthly OOS blocks using only data before each holding month."""
    panel = _validate_return_panel(returns)
    cutoff = pd.Timestamp(initial_estimation_end).normalize()
    live = panel.loc[panel.index > cutoff]

    if live.empty:
        raise ValueError("No OOS returns exist after the initial estimation period.")
    if panel.loc[panel.index <= cutoff].shape[0] < 2:
        raise ValueError("Initial estimation period contains too few observations.")

    rows = []
    for _month, holding_returns in live.groupby(live.index.to_period("M")):
        effective_start = pd.Timestamp(holding_returns.index.min())
        effective_end = pd.Timestamp(holding_returns.index.max())
        training_returns = panel.loc[panel.index < effective_start]

        rows.append(
            {
                "decision_date": pd.Timestamp(training_returns.index.max()),
                "effective_start_date": effective_start,
                "effective_end_date": effective_end,
                "window_start_date": pd.Timestamp(training_returns.index.min()),
                "window_end_date": pd.Timestamp(training_returns.index.max()),
                "window_observations": len(training_returns),
            }
        )

    schedule = pd.DataFrame(rows)
    if not (schedule["decision_date"] < schedule["effective_start_date"]).all():
        raise ValueError("A decision date is not earlier than its holding period.")

    return schedule


def buy_and_hold_block_returns(
    holding_returns: pd.DataFrame,
    target_weights: np.ndarray,
) -> tuple[pd.Series, np.ndarray]:
    """Apply target weights and let holdings drift until the next rebalance."""
    panel = _validate_return_panel(holding_returns)
    current_weights = validate_weights(target_weights, panel.shape[1])
    portfolio_returns = []

    for date, asset_returns in panel.iterrows():
        return_vector = asset_returns.to_numpy(dtype=float)
        portfolio_return = float(current_weights @ return_vector)

        if portfolio_return <= -1.0:
            raise ValueError(f"Portfolio was wiped out on {date.date()}.")

        portfolio_returns.append(portfolio_return)
        current_weights = current_weights * (1.0 + return_vector)
        current_weights = validate_weights(
            current_weights / (1.0 + portfolio_return),
            panel.shape[1],
        )

    series = pd.Series(portfolio_returns, index=panel.index, name="return")
    return series, current_weights


def one_way_turnover(
    previous_live_weights: np.ndarray,
    new_target_weights: np.ndarray,
) -> float:
    """Calculate one-way turnover between live and new target weights."""
    previous = np.asarray(previous_live_weights, dtype=float)
    target = np.asarray(new_target_weights, dtype=float)

    if previous.shape != target.shape:
        raise ValueError("Turnover weight vectors must have the same shape.")

    return float(0.5 * np.abs(target - previous).sum())


def apply_rebalance_transaction_costs(
    gross_returns: pd.Series,
    turnover_by_date: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Deduct one-way trading costs on subsequent rebalance dates.

    The first portfolio formation is excluded because its turnover is undefined.
    On a charged date, wealth first pays ``turnover * cost_bps`` and then earns
    that day's gross portfolio return.
    """
    returns = pd.Series(gross_returns, dtype=float).copy().sort_index()
    turnover = pd.Series(turnover_by_date, dtype=float).copy().dropna()

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Gross returns must use a pandas DatetimeIndex.")
    if not isinstance(turnover.index, pd.DatetimeIndex):
        raise TypeError("Turnover must use a pandas DatetimeIndex.")
    if returns.index.duplicated().any() or turnover.index.duplicated().any():
        raise ValueError("Return and turnover dates must be unique.")
    if returns.empty or not np.isfinite(returns).all():
        raise ValueError("Gross returns must be finite and non-empty.")
    if (returns <= -1.0).any():
        raise ValueError("Gross returns must be greater than -100%.")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction cost must be a finite non-negative value.")
    if not np.isfinite(turnover).all() or (turnover < 0).any():
        raise ValueError("Turnover must contain finite non-negative values.")

    returns.index = returns.index.normalize()
    turnover.index = turnover.index.normalize()
    if not turnover.index.isin(returns.index).all():
        raise ValueError("Every charged rebalance date must have a fund return.")
    if cost_bps == 0 or turnover.empty:
        return returns.rename("return")

    cost_fraction = turnover * (cost_bps / 10_000.0)
    if (cost_fraction >= 1.0).any():
        raise ValueError("Transaction costs would exhaust the portfolio.")

    net_returns = returns.copy()
    charged_dates = turnover.index
    net_returns.loc[charged_dates] = (
        (1.0 + returns.loc[charged_dates])
        * (1.0 - cost_fraction)
        - 1.0
    )
    return net_returns.rename("return")


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Calculate required and investor-facing OOS performance metrics."""
    returns = pd.Series(daily_returns, dtype=float).dropna()

    if returns.empty:
        raise ValueError("Performance metrics require at least one return.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if (returns <= -1.0).any():
        raise ValueError("Daily returns must be greater than -100%.")

    annualized_return = float(returns.mean() * periods_per_year)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    growth_of_one = float(wealth.iloc[-1])
    years = len(returns) / periods_per_year
    cagr = growth_of_one ** (1.0 / years) - 1.0 if years > 0 else np.nan

    return {
        "observations": len(returns),
        "total_return": growth_of_one - 1.0,
        "annualized_return": annualized_return,
        "cagr": float(cagr),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.min()),
        "final_growth_of_one": growth_of_one,
    }


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "minimum_variance",
    *,
    universe: str = "Combined",
    fund_id: str | None = None,
    initial_estimation_end: str | pd.Timestamp = "2020-12-31",
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    sleeve_assets: dict[str, list[str]] | None = None,
) -> BacktestResult:
    """Run one monthly, expanding-window, long-only OOS fund backtest."""
    panel = _validate_return_panel(returns)
    schedule = build_monthly_schedule(panel, initial_estimation_end)
    identifier = fund_id or f"{universe.lower()}_{method}"

    daily_blocks = []
    target_weight_rows = []
    audit_rows = []
    previous_live_weights = None

    for block in schedule.itertuples(index=False):
        training = panel.loc[panel.index <= block.decision_date]
        holding = panel.loc[
            (panel.index >= block.effective_start_date)
            & (panel.index <= block.effective_end_date)
        ]

        if method == HIERARCHICAL_METHOD:
            if sleeve_assets is None:
                raise ValueError(
                    "Hierarchical risk parity requires sleeve asset definitions."
                )
            target_weights, sleeve_targets = (
                hierarchical_sleeve_risk_parity_weights(
                    training,
                    sleeve_assets,
                    periods_per_year,
                )
            )
            solver = "two_stage_SLSQP_risk_contribution"
        else:
            target_weights, solver = estimate_weights(
                training,
                method,
                periods_per_year,
                risk_free_rate,
            )
            sleeve_targets = {}

        turnover = (
            np.nan
            if previous_live_weights is None
            else one_way_turnover(previous_live_weights, target_weights)
        )
        block_returns, previous_live_weights = buy_and_hold_block_returns(
            holding,
            target_weights,
        )

        daily_blocks.append(
            pd.DataFrame(
                {
                    "date": block_returns.index,
                    "decision_date": block.decision_date,
                    "effective_start_date": block.effective_start_date,
                    "fund_id": identifier,
                    "universe": universe,
                    "method": method,
                    "return": block_returns.to_numpy(),
                }
            )
        )

        for ticker, weight in zip(panel.columns, target_weights, strict=True):
            target_weight_rows.append(
                {
                    "decision_date": block.decision_date,
                    "effective_start_date": block.effective_start_date,
                    "fund_id": identifier,
                    "universe": universe,
                    "method": method,
                    "ticker": ticker,
                    "target_weight": float(weight),
                }
            )

        effective_holdings = float(1.0 / np.square(target_weights).sum())
        audit_row = {
            "decision_date": block.decision_date,
            "effective_start_date": block.effective_start_date,
            "effective_end_date": block.effective_end_date,
            "window_start_date": block.window_start_date,
            "window_end_date": block.window_end_date,
            "window_observations": block.window_observations,
            "fund_id": identifier,
            "universe": universe,
            "method": method,
            "solver": solver,
            "one_way_turnover": turnover,
            "largest_target_weight": float(target_weights.max()),
            "effective_number_of_holdings": effective_holdings,
        }
        audit_row.update(
            {
                f"{sleeve_name.lower()}_sleeve_target_weight": sleeve_weight
                for sleeve_name, sleeve_weight in sleeve_targets.items()
            }
        )
        audit_rows.append(audit_row)

    daily = pd.concat(daily_blocks, ignore_index=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["growth_of_one"] = (1.0 + daily["return"]).cumprod()
    daily["drawdown"] = daily["growth_of_one"] / daily["growth_of_one"].cummax() - 1.0

    weights = pd.DataFrame(target_weight_rows)
    audit = pd.DataFrame(audit_rows)
    metrics = performance_metrics(
        daily.set_index("date")["return"],
        periods_per_year,
        risk_free_rate,
    )
    metrics.update(
        {
            "fund_id": identifier,
            "universe": universe,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "periods_per_year": periods_per_year,
            "risk_free_rate": risk_free_rate,
            "rebalance_count": len(audit),
            "average_one_way_turnover": float(
                audit["one_way_turnover"].dropna().mean()
            ),
            "latest_effective_number_of_holdings": float(
                audit["effective_number_of_holdings"].iloc[-1]
            ),
        }
    )

    return BacktestResult(
        daily_returns=daily,
        target_weights=weights,
        rebalance_audit=audit,
        metrics=metrics,
    )
