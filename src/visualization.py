import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Optional global style
sns.set(style="whitegrid")


# ==========================================================
# 1. Distribution Plots
# ==========================================================

def plot_distribution(series: pd.Series, bins: int = 50, title: str = ""):
    """
    Plot histogram with KDE overlay.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(series, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(series.name if series.name else "Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_mean_median(series: pd.Series, bins: int = 50, title: str = ""):
    """
    Plot histogram with mean and median lines.
    """
    mean_val = series.mean()
    median_val = series.median()

    plt.figure(figsize=(8, 5))
    sns.histplot(series, bins=bins, kde=True)

    plt.axvline(mean_val, linestyle="--", label=f"Mean = {mean_val:.2f}")
    plt.axvline(median_val, linestyle="-", label=f"Median = {median_val:.2f}")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 2. Simpson's Paradox Visualization
# ==========================================================

def plot_simpsons_scatter(df: pd.DataFrame, x: str, y: str, group_col: str):
    """
    Scatter plot colored by group.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=group_col, alpha=0.7)
    plt.title("Simpson's Paradox: Grouped Scatter")
    plt.tight_layout()
    plt.show()


def plot_simpsons_regression(df: pd.DataFrame, x: str, y: str, group_col: str):
    """
    Plot regression lines per group and overall.
    """
    plt.figure(figsize=(8, 6))

    # Overall regression
    sns.regplot(data=df, x=x, y=y, scatter=False, label="Overall")

    # Group regressions
    for group, subdf in df.groupby(group_col):
        sns.regplot(
            data=subdf,
            x=x,
            y=y,
            scatter=False,
            label=f"Group {group}"
        )

    plt.legend()
    plt.title("Regression: Group vs Aggregate")
    plt.tight_layout()
    plt.show()


# ==========================================================
# 3. Survivorship Bias Visualization
# ==========================================================

def plot_survival_distribution(
    df: pd.DataFrame,
    value_col: str = "final_wealth",
    survival_col: str = "survived"
):
    """
    Compare wealth distribution of full sample vs survivors.
    """
    plt.figure(figsize=(8, 5))

    sns.kdeplot(df[value_col], label="Full Sample", fill=True)
    sns.kdeplot(
        df[df[survival_col]][value_col],
        label="Survivors Only",
        fill=True
    )

    plt.title("Survivorship Bias: Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 4. Spurious Correlation Visualization
# ==========================================================

def plot_spurious_correlation(df: pd.DataFrame, x: str, y: str):
    """
    Scatter plot to illustrate spurious correlation.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    sns.regplot(data=df, x=x, y=y, scatter=False)
    plt.title("Spurious Correlation")
    plt.tight_layout()
    plt.show()


# ==========================================================
# 5. Contribution Visualization (Advanced)
# ==========================================================

def plot_contribution_bar(df: pd.DataFrame, top_n: int = 10):
    """
    Plot top states contributing most to expectation.
    Expects dataframe with 'value', 'probability', 'contribution_share'.
    """
    df_top = df.head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=df_top["contribution_share"],
        y=np.arange(len(df_top))
    )

    plt.title("Top Contribution to Expectation")
    plt.xlabel("Contribution Share")
    plt.ylabel("State Rank")
    plt.tight_layout()
    plt.show()


def plot_sample_wealth_paths(
    df: pd.DataFrame,
    n_paths: int = 50,
    log_scale: bool = True,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot sample wealth trajectories.

    Parameters:
    - df: DataFrame from simulate_wealth_paths
    - n_paths: number of random paths to display
    - log_scale: whether to use log-scale on y-axis
    """

    sample_paths = df["path"].unique()[:n_paths]

    plt.figure(figsize=figsize)

    for p in sample_paths:
        path_df = df[df["path"] == p]
        plt.plot(path_df["time"], path_df["wealth"], alpha=0.3)

    if log_scale:
        plt.yscale("log")

    plt.title("Sample Wealth Paths")
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.show()


def plot_final_wealth_distribution(
    df: pd.DataFrame,
    bins: int = 50,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot histogram of final wealth distribution.
    """

    final_time = df["time"].max()
    final_df = df[df["time"] == final_time]

    plt.figure(figsize=figsize)
    sns.histplot(final_df["wealth"], bins=bins)
    plt.title("Final Wealth Distribution")
    plt.xlabel("Wealth")
    plt.ylabel("Frequency")
    plt.show()


def plot_spurious_scatter(
    df: pd.DataFrame,
    x_col: str = "X",
    y_col: str = "Y",
    alpha: float = 0.4,
    figsize: tuple = (6, 5),
    show_corr: bool = True,
) -> float:
    """
    Plot scatter of two variables and optionally display correlation.

    Returns:
        Pearson correlation coefficient.
    """

    corr = df[[x_col, y_col]].corr().iloc[0, 1]

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=alpha)

    title = f"Scatter Plot of {x_col} vs {y_col}"
    if show_corr:
        title += f"\nCorrelation = {corr:.4f}"

    plt.title(title)
    plt.show()

    return corr