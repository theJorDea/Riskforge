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
    _validate_alpha(alpha)
    return float(-returns.quantile(1.0 - alpha))


def historical_es(returns: pd.Series, alpha: float = 0.99) -> float:
    """Historical Expected Shortfall: mean loss beyond the VaR threshold."""
    _validate_alpha(alpha)
    threshold = returns.quantile(1.0 - alpha)
    tail = returns[returns <= threshold]
    if tail.empty:
        raise ValueError("no observations in the tail; need more data or lower alpha")
    return float(-tail.mean())


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
