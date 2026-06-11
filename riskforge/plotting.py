"""Standard diagnostic plots (matplotlib).

Keep plots boring and readable: one idea per figure, labeled axes, title
with the key number (e.g. breach count vs expected).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_returns_with_var(backtest_df: pd.DataFrame, alpha: float = 0.99) -> plt.Figure:
    """Realized returns vs -VaR boundary; breaches highlighted in red.

    Expects the output of ``rolling_var_backtest`` (columns: return, var, breach).
    """
    # TODO: implement
    raise NotImplementedError


def plot_qq(returns: pd.Series) -> plt.Figure:
    """QQ-plot of returns against the normal distribution (shows fat tails)."""
    # TODO: implement with scipy.stats.probplot
    raise NotImplementedError


def plot_volatility(returns: pd.Series, vol_series: dict[str, pd.Series]) -> plt.Figure:
    """Overlay |returns| with one or more conditional volatility estimates.

    ``vol_series`` maps label -> sigma_t series, e.g. {"EWMA": ..., "GARCH": ...}.
    """
    # TODO: implement
    raise NotImplementedError
