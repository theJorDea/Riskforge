"""Conditional VaR: a Student-t quantile on top of a GARCH volatility forecast.

Why this module exists
----------------------
Notebook 03 ends on an honest verdict: a *static* historical/parametric VaR
reacts to regime changes with a lag (Christoffersen rejects — breaches
cluster), while a GARCH-VaR with a *Gaussian* quantile tracks the regime but
still under-covers the tail (Kupiec rejects — too many breaches, because
daily standardized residuals are fat-tailed). The fix is to combine the two
strengths:

    dynamic sigma_t (GARCH)  +  fat-tailed quantile (Student-t).

Construction
------------
1. Fit GARCH(1,1) by MLE and forecast next-day volatility sigma_{t+1}.
2. Standardize the in-sample residuals z_t = (r_t - mu) / sigma_t and fit a
   Student-t to them with location fixed at 0:  z ~ scale * t_nu.
   The fitted nu captures the fat tails the Gaussian GARCH misses; the fitted
   scale (~ sqrt((nu-2)/nu) for unit-variance residuals) keeps the units right.
3. The one-day-ahead VaR is the lower tail quantile of mu + sigma_{t+1} * z:

       VaR_alpha = -( mu + sigma_{t+1} * scale * t^{-1}_{1-alpha, nu} )

Because t^{-1}_{1-alpha, nu} is negative for alpha > 0.5, VaR comes out
positive, consistent with the rest of the package.

This estimator has the signature ``returns -> positive float`` and so drops
straight into ``rolling_var_backtest`` for an out-of-sample Kupiec/
Christoffersen check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskforge.volatility.garch import (
    GarchParams,
    _sigma2_recursion,
    garch_fit,
    garch_forecast,
)


def garch_t_var(
    returns: pd.Series,
    alpha: float = 0.99,
    params: GarchParams | None = None,
) -> float:
    """One-day-ahead conditional VaR: Student-t quantile scaled by GARCH sigma.

    Parameters
    ----------
    returns : return history used to fit GARCH and the residual t-distribution.
    alpha   : VaR confidence level (e.g. 0.99).
    params  : optional pre-fitted ``GarchParams`` (skip the MLE if you already
              have them; handy when sweeping alpha on the same window).

    Returns
    -------
    Positive VaR estimate for the next day.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    from scipy import stats

    mu = float(returns.mean())
    if params is None:
        params = garch_fit(returns)

    # In-sample conditional variances -> standardized residuals.
    r = returns.to_numpy(dtype=float) - mu
    sigma2 = _sigma2_recursion(
        r, params.omega, params.alpha, params.beta, float(np.var(r, ddof=1))
    )
    z = r / np.sqrt(sigma2)

    # Fit Student-t to the standardized residuals with location pinned at 0.
    nu, _, scale = stats.t.fit(z, floc=0.0)

    # Next-day volatility forecast (sqrt of the 1-step variance forecast).
    sigma_next = float(np.sqrt(garch_forecast(returns, params, horizon=1)))

    quantile = scale * stats.t.ppf(1.0 - alpha, nu)  # negative (lower tail)
    return float(-(mu + sigma_next * quantile))
