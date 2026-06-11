"""Historical (non-parametric) VaR and ES.

Idea: use the empirical distribution of past returns as the forecast
distribution. No distributional assumptions, but assumes the recent past
is representative of the near future.

Formulas
--------
VaR_alpha = -Quantile_{1-alpha}(r)          (positive number)
ES_alpha  = -E[ r | r <= Quantile_{1-alpha}(r) ]
"""

from __future__ import annotations

import pandas as pd


def historical_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """Historical VaR at confidence level ``alpha``.

    Example: alpha=0.99 -> the 1% empirical quantile of returns, negated.
    """
    # TODO: implement (validate 0 < alpha < 1; use returns.quantile(1 - alpha))
    raise NotImplementedError


def historical_es(returns: pd.Series, alpha: float = 0.99) -> float:
    """Historical Expected Shortfall: mean loss beyond the VaR threshold."""
    # TODO: implement (mean of returns below the (1-alpha) quantile, negated)
    raise NotImplementedError
