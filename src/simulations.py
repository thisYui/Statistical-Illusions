import numpy as np
import pandas as pd


# ==========================================================
# 1. Law of Small Numbers Simulation
# ==========================================================

def simulate_coin_flips(
    n_flips: int,
    n_simulations: int = 1000,
    p: float = 0.5,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Simulate multiple coin flip experiments.

    Returns a DataFrame with proportion of heads per simulation.
    """
    rng = np.random.default_rng(seed)

    results = []

    for i in range(n_simulations):
        flips = rng.binomial(1, p, size=n_flips)
        results.append({
            "simulation": i,
            "proportion_heads": flips.mean()
        })

    return pd.DataFrame(results)


# ==========================================================
# 2. Monte Carlo for Mean Instability
# ==========================================================

def simulate_sample_mean_distribution(
    distribution: str = "lognormal",
    n: int = 100,
    n_simulations: int = 1000,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Compare sampling variability of mean under different distributions.
    """
    rng = np.random.default_rng(seed)

    means = []

    for i in range(n_simulations):

        if distribution == "normal":
            sample = rng.normal(mu, sigma, size=n)

        elif distribution == "lognormal":
            sample = rng.lognormal(mu, sigma, size=n)

        else:
            raise ValueError("Unsupported distribution")

        means.append(sample.mean())

    return pd.DataFrame({
        "simulation": np.arange(n_simulations),
        "sample_mean": means
    })


# ==========================================================
# 3. Survivorship Over Time (Path Simulation)
# ==========================================================

def simulate_survival_paths(
    n_entities: int = 1000,
    survival_prob: float = 0.95,
    growth_mean: float = 0.05,
    growth_std: float = 0.2,
    periods: int = 20,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Simulate time evolution of wealth with survival filtering.
    Returns long-format DataFrame for path visualization.
    """
    rng = np.random.default_rng(seed)

    records = []

    for entity in range(n_entities):
        wealth = 1.0
        alive = True

        for t in range(periods):

            if not alive:
                break

            # Survival check
            if rng.random() > survival_prob:
                alive = False
                break

            # Growth
            growth = rng.normal(growth_mean, growth_std)
            wealth *= (1 + growth)

            records.append({
                "entity": entity,
                "time": t,
                "wealth": wealth
            })

    return pd.DataFrame(records)


# ==========================================================
# 4. Correlation Instability Simulation
# ==========================================================

def simulate_spurious_correlation_instability(
    n: int = 100,
    n_simulations: int = 1000,
    noise_std: float = 1.0,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Simulate repeated draws of spurious correlation
    from hidden-variable structure.
    """
    rng = np.random.default_rng(seed)

    correlations = []

    for i in range(n_simulations):
        Z = rng.normal(0, 1, n)
        X = Z + rng.normal(0, noise_std, n)
        Y = Z + rng.normal(0, noise_std, n)

        corr = np.corrcoef(X, Y)[0, 1]

        correlations.append(corr)

    return pd.DataFrame({
        "simulation": np.arange(n_simulations),
        "correlation": correlations
    })

def simulate_wealth_paths(
    n_paths: int = 1000,
    periods: int = 50,
    mu: float = 0.03,
    sigma: float = 0.4,
    initial_wealth: float = 1.0,
    bankruptcy_threshold: float = 0.3,
    crash_prob: float = 0.05,
    crash_severity: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate multiplicative wealth dynamics with realistic extinction.

    Wealth evolves as:
        W_{t+1} = W_t * exp(μ - 0.5σ² + σ Z_t)

    Additional crash risk:
        With probability crash_prob,
        wealth is multiplied by (1 - crash_severity)

    Bankruptcy occurs if wealth < bankruptcy_threshold.
    """

    rng = np.random.default_rng(seed)

    records = []

    for path in range(n_paths):

        wealth = initial_wealth
        alive = True

        for t in range(periods):

            if alive:

                # Lognormal return
                z = rng.normal()
                growth_factor = np.exp(mu - 0.5 * sigma**2 + sigma * z)
                wealth *= growth_factor

                # Occasional crash
                if rng.random() < crash_prob:
                    wealth *= (1 - crash_severity)

                # Bankruptcy check
                if wealth < bankruptcy_threshold:
                    wealth = 0.0
                    alive = False

            # Record even after death (wealth stays 0)
            records.append({
                "path": path,
                "time": t,
                "wealth": wealth,
                "alive": alive
            })

    return pd.DataFrame(records)