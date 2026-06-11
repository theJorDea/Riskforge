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
    dist: str = "normal",
    df: float | None = None,
) -> float:
    """Monte Carlo VaR for a portfolio of assets.

    Parameters
    ----------
    returns : asset return history (T x N), used to fit mu and Sigma.
    weights : portfolio weights (N,).
    n_sims  : number of simulated scenarios.
    seed    : RNG seed for reproducibility (important for tests!).
    dist    : "normal" or "t" (multivariate Student-t with joint fat tails).
    df      : degrees of freedom for dist="t". If None, estimated by fitting
              a univariate Student-t to portfolio returns (simple, robust proxy).

    Returns
    -------
    Positive VaR estimate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (returns.shape[1],):
        raise ValueError("weights length must match the number of assets")

    mu = returns.mean().to_numpy()
    sigma = returns.cov().to_numpy()

    rng = np.random.default_rng(seed)
    if dist == "normal":
        scenarios = rng.multivariate_normal(mu, sigma, size=n_sims)  # (n_sims, N)
    elif dist == "t":
        if df is None:
            from scipy import stats

            df = float(stats.t.fit(returns.to_numpy() @ weights)[0])
        if df <= 2:
            raise ValueError(f"df must be > 2 for finite covariance, got {df:.2f}")
        # Multivariate t as a normal mixture: r = mu + Z / sqrt(W / df),
        # Z ~ N(0, Sigma_z), W ~ chi2(df). Scale Sigma_z so that the
        # resulting covariance equals the sample Sigma: Cov = Sigma_z * df/(df-2).
        sigma_z = sigma * (df - 2.0) / df
        z = rng.multivariate_normal(np.zeros_like(mu), sigma_z, size=n_sims)
        w_mix = rng.chisquare(df, size=n_sims) / df
        scenarios = mu + z / np.sqrt(w_mix)[:, None]
    else:
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    pnl = scenarios @ weights
    return float(-np.quantile(pnl, 1.0 - alpha))
