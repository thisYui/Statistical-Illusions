import numpy as np
import pandas as pd
from typing import Dict


# ==========================================================
# 1. Basic Distribution Metrics
# ==========================================================

def compute_mean_median_gap(series: pd.Series) -> Dict[str, float]:
    """
    Compute mean, median, and their difference.
    Useful for skewed distribution analysis.
    """
    mean_val = series.mean()
    median_val = series.median()

    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "gap": float(mean_val - median_val),
    }


def compute_quantiles(series: pd.Series, quantiles=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    """
    Compute selected quantiles.
    """
    q_vals = series.quantile(quantiles)
    return {f"q_{int(q*100)}": float(v) for q, v in zip(quantiles, q_vals)}


# ==========================================================
# 2. Correlation & Regression Metrics
# ==========================================================

def compute_correlation(df: pd.DataFrame, x: str, y: str) -> float:
    """
    Compute Pearson correlation between two columns.
    """
    return float(df[[x, y]].corr().iloc[0, 1])


def compute_groupwise_correlation(df: pd.DataFrame, group_col: str, x: str, y: str) -> Dict[str, float]:
    """
    Compute correlation within each group.
    """
    correlations = {}
    for g, subdf in df.groupby(group_col):
        correlations[g] = float(subdf[[x, y]].corr().iloc[0, 1])
    return correlations


def compute_linear_regression_slope(df: pd.DataFrame, x: str, y: str) -> float:
    """
    Compute simple OLS slope (closed form).
    """
    X = df[x].values
    Y = df[y].values

    cov = np.cov(X, Y, ddof=1)[0, 1]
    var = np.var(X, ddof=1)

    if var == 0:
        return np.nan

    return float(cov / var)


# ==========================================================
# 3. Survivorship Metrics
# ==========================================================

def compute_survival_rate(df: pd.DataFrame, survival_col: str = "survived") -> float:
    """
    Compute proportion of surviving entities.
    """
    return float(df[survival_col].mean())


def compute_survivor_vs_full_mean(
    df: pd.DataFrame,
    value_col: str = "final_wealth",
    survival_col: str = "survived"
) -> Dict[str, float]:
    """
    Compare mean wealth of full sample vs survivors only.
    """
    full_mean = df[value_col].mean()
    survivor_mean = df[df[survival_col]][value_col].mean()

    return {
        "full_sample_mean": float(full_mean),
        "survivor_mean": float(survivor_mean),
        "difference": float(survivor_mean - full_mean),
    }


# ==========================================================
# 4. Contribution Decomposition (Optional Advanced)
# ==========================================================

def compute_expectation_contributions(
    values: pd.Series,
    probabilities: pd.Series
) -> pd.DataFrame:
    """
    Compute contribution of each state to total expectation.
    Contribution = probability * value
    """
    contributions = probabilities * values

    df = pd.DataFrame({
        "value": values,
        "probability": probabilities,
        "contribution": contributions
    })

    df["contribution_share"] = df["contribution"] / df["contribution"].sum()

    return df.sort_values("contribution_share", ascending=False).reset_index(drop=True)