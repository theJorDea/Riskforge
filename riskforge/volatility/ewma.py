"""EWMA volatility (RiskMetrics, JPMorgan 1996).

Recursion
---------
sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

with lambda = 0.94 for daily data (the classic RiskMetrics choice).
This is GARCH(1,1) with omega=0, alpha=1-lambda, beta=lambda — i.e. an
IGARCH process without mean reversion.

Implement BY HAND on NumPy (no statsmodels/arch) — the point of this module
is to demonstrate understanding of the recursion, not library usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Conditional volatility sigma_t via the EWMA recursion.

    Parameters
    ----------
    returns : return series r_t.
    lam     : decay factor lambda in (0, 1).

    Returns
    -------
    Series of sigma_t (same index as ``returns``). Initialize sigma_0^2
    with the sample variance of the first ~30 observations (document the
    choice — initialization matters for the first few dozen points).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 observations")

    sigma2 = np.empty(n)
    # Initialize with the sample variance of the first ~30 points (or fewer).
    init_window = min(30, n)
    sigma2[0] = np.var(r[:init_window], ddof=1)
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(sigma2), index=returns.index, name="ewma_vol")
