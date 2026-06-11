"""Parametric (variance-covariance) VaR.

Assume returns follow a known distribution; estimate its parameters and
read VaR off the analytical quantile.

Normal:    VaR_alpha = -(mu + sigma * z_{1-alpha}),  z = Phi^{-1}
Student-t: VaR_alpha = -(mu + sigma * sqrt((nu-2)/nu) * t_{1-alpha,nu})
           (the sqrt((nu-2)/nu) factor rescales the standard t so that the
           distribution has variance sigma^2; nu estimated by MLE)

Why Student-t: daily returns have fat tails (kurtosis >> 3); the normal
model systematically underestimates tail risk. Show this in notebook 02.
"""

from __future__ import annotations

import pandas as pd


def parametric_var(
    returns: pd.Series,
    alpha: float = 0.99,
    dist: str = "normal",
) -> float:
    """Parametric VaR under ``dist`` in {"normal", "t"}.

    For dist="t", fit nu (degrees of freedom) by MLE via
    ``scipy.stats.t.fit(returns, floc=mu)`` or full ``t.fit``.

    Returns positive VaR.
    """
    # TODO: implement
    #  normal: mu, sigma = returns.mean(), returns.std(ddof=1)
    #          var = -(mu + sigma * norm.ppf(1 - alpha))
    #  t:      nu, loc, scale = t.fit(returns)
    #          var = -(loc + scale * t.ppf(1 - alpha, nu))
    raise NotImplementedError
