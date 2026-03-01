# ==========================================================
# Data Generating Processes
# ==========================================================

from .dgp import (
    generate_simpsons_data,
    generate_lognormal_data,
    generate_survivorship_data,
    generate_latent_factor_data,
)

# ==========================================================
# Simulations
# ==========================================================

from .simulations import (
    simulate_coin_flips,
    simulate_sample_mean_distribution,
    simulate_survival_paths,
    simulate_spurious_correlation_instability,
    simulate_wealth_paths,
    simulate_correlation_instability,
    simulate_mean_instability,
)

# ==========================================================
# Metrics
# ==========================================================

from .metrics import (
    compute_mean_median_gap,
    compute_quantiles,
    compute_correlation,
    compute_groupwise_correlation,
    compute_linear_regression_slope,
    compute_survival_rate,
    compute_survivor_vs_full_mean,
    compute_expectation_contributions,
    summarize_survivorship,
    summarize_sampling_distribution,
)

# ==========================================================
# Visualization
# ==========================================================

from .visualization import (
    plot_distribution,
    plot_mean_median,
    plot_simpsons_scatter,
    plot_simpsons_regression,
    plot_survival_distribution,
    plot_spurious_correlation,
    plot_contribution_bar,
    plot_sample_wealth_paths,
    plot_final_wealth_distribution,
    plot_spurious_scatter,
    plot_sampling_distribution,
)

# ==========================================================
# Public API
# ==========================================================

__all__ = [
    # DGP
    "generate_simpsons_data",
    "generate_lognormal_data",
    "generate_survivorship_data",
    "generate_latent_factor_data",

    # Simulations
    "simulate_coin_flips",
    "simulate_sample_mean_distribution",
    "simulate_survival_paths",
    "simulate_spurious_correlation_instability",
    "simulate_wealth_paths",
    "simulate_correlation_instability",
    "simulate_mean_instability",

    # Metrics
    "compute_mean_median_gap",
    "compute_quantiles",
    "compute_correlation",
    "compute_groupwise_correlation",
    "compute_linear_regression_slope",
    "compute_survival_rate",
    "compute_survivor_vs_full_mean",
    "compute_expectation_contributions",
    "summarize_survivorship",
    "summarize_sampling_distribution",

    # Visualization
    "plot_distribution",
    "plot_mean_median",
    "plot_simpsons_scatter",
    "plot_simpsons_regression",
    "plot_survival_distribution",
    "plot_spurious_correlation",
    "plot_contribution_bar",
    "plot_sample_wealth_paths",
    "plot_final_wealth_distribution",
    "plot_spurious_scatter",
    "plot_sampling_distribution",
]