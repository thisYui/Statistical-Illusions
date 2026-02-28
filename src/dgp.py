import numpy as np
import pandas as pd


# ==========================================================
# 1. Simpson's Paradox
# ==========================================================

def generate_simpsons_data(
    n_per_group: int = 500,
    beta: float = 1.0,
    group_shift: float = 5.0,
    noise_std: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate data exhibiting Simpson's Paradox.

    Within each group: positive correlation between X and Y.
    After aggregation: overall slope may reverse depending on parameters.
    """
    rng = np.random.default_rng(seed)

    groups = []
    for g, shift in zip(["A", "B"], [0, group_shift]):
        X = rng.normal(loc=shift, scale=1.0, size=n_per_group)
        Y = beta * X - shift + rng.normal(0, noise_std, size=n_per_group)

        df_group = pd.DataFrame({
            "group": g,
            "X": X,
            "Y": Y
        })
        groups.append(df_group)

    return pd.concat(groups, ignore_index=True)


# ==========================================================
# 2. Lognormal Distribution (Mean vs Median Illusion)
# ==========================================================

def generate_lognormal_data(
    n: int = 1000,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate lognormal distributed values.
    Used to illustrate mean > median under skewed distributions.
    """
    rng = np.random.default_rng(seed)
    values = rng.lognormal(mean=mu, sigma=sigma, size=n)

    return pd.DataFrame({"value": values})


# ==========================================================
# 3. Survivorship Bias
# ==========================================================

def generate_survivorship_data(
    n_entities: int = 1000,
    survival_prob: float = 0.9,
    growth_mean: float = 0.05,
    growth_std: float = 0.2,
    periods: int = 10,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate entities over time with survival probability.
    Survivors accumulate multiplicative growth.
    """
    rng = np.random.default_rng(seed)

    data = []

    for i in range(n_entities):
        alive = True
        wealth = 1.0

        for t in range(periods):
            if not alive:
                break

            # Survival check
            if rng.random() > survival_prob:
                alive = False
                break

            # Multiplicative growth
            growth = rng.normal(growth_mean, growth_std)
            wealth *= (1 + growth)

        data.append({
            "entity": i,
            "final_wealth": wealth if alive else 0.0,
            "survived": alive
        })

    return pd.DataFrame(data)


# ==========================================================
# 4. Spurious Correlation (Hidden Variable)
# ==========================================================

def generate_spurious_correlation_data(
    n: int = 1000,
    noise_std: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate two variables X and Y that appear correlated
    due to a shared hidden variable Z.
    """
    rng = np.random.default_rng(seed)

    Z = rng.normal(0, 1, n)
    X = Z + rng.normal(0, noise_std, n)
    Y = Z + rng.normal(0, noise_std, n)

    return pd.DataFrame({
        "X": X,
        "Y": Y,
        "Z": Z
    })