"""Monte Carlo VaR.

Simulate many scenarios of asset returns from a fitted multivariate
distribution, aggregate to portfolio P&L, and take the empirical quantile
of simulated losses.

Baseline model: multivariate normal with sample mean vector and sample
covariance matrix. Extension (optional): multivariate Student-t to capture
joint fat tails.

Pipeline
--------
1. Estimate mu (N,) and Sigma (N, N) from historical asset returns.
2. Draw ``n_sims`` scenarios: r ~ N(mu, Sigma)  -> (n_sims, N).
3. Portfolio P&L per scenario: p = r @ w.
4. VaR_alpha = -quantile(p, 1 - alpha).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> float:
    """Monte Carlo VaR for a portfolio of assets.

    Parameters
    ----------
    returns : asset return history (T x N), used to fit mu and Sigma.
    weights : portfolio weights (N,).
    n_sims  : number of simulated scenarios.
    seed    : RNG seed for reproducibility (important for tests!).

    Returns
    -------
    Positive VaR estimate.
    """
    # TODO: implement using np.random.default_rng(seed).multivariate_normal
    raise NotImplementedError
