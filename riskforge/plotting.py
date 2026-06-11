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
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(backtest_df.index, backtest_df["return"], lw=0.6, color="0.4", label="return")
    ax.plot(backtest_df.index, -backtest_df["var"], lw=1.2, color="C0", label=f"-VaR {alpha:.0%}")
    br = backtest_df[backtest_df["breach"]]
    ax.scatter(br.index, br["return"], color="red", s=18, zorder=3, label="breach")
    expected = (1 - alpha) * len(backtest_df)
    ax.set_title(f"VaR backtest: {len(br)} breaches vs {expected:.1f} expected")
    ax.set_xlabel("date")
    ax.set_ylabel("daily log return")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_qq(returns: pd.Series) -> plt.Figure:
    """QQ-plot of returns against the normal distribution (shows fat tails)."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title("QQ-plot vs Normal (deviations in the tails = fat tails)")
    fig.tight_layout()
    return fig


def plot_volatility(returns: pd.Series, vol_series: dict[str, pd.Series]) -> plt.Figure:
    """Overlay |returns| with one or more conditional volatility estimates.

    ``vol_series`` maps label -> sigma_t series, e.g. {"EWMA": ..., "GARCH": ...}.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(returns.index, returns.abs(), lw=0.5, color="0.7", label="|return|")
    for label, vol in vol_series.items():
        ax.plot(vol.index, vol, lw=1.2, label=label)
    ax.set_title("Conditional volatility estimates")
    ax.set_xlabel("date")
    ax.set_ylabel("daily volatility")
    ax.legend()
    fig.tight_layout()
    return fig
