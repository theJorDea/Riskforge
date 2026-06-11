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
    # TODO: implement the recursion with a simple loop or np cumulative trick
    raise NotImplementedError
