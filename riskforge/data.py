"""Data loading and return computation.

Responsibilities
----------------
1. Download adjusted close prices via yfinance.
2. Cache raw prices to ``data/*.parquet`` so notebooks are reproducible offline.
3. Compute log returns: r_t = ln(P_t / P_{t-1}).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data"


def load_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load adjusted close prices for ``tickers``.

    Parameters
    ----------
    tickers : list of ticker symbols, e.g. ``["SPY", "AAPL", "GLD"]``.
    start, end : ISO date strings passed to yfinance.
    use_cache : if True, read/write ``data/prices_<hash>.parquet``.

    Returns
    -------
    DataFrame indexed by date, one column per ticker, NaNs dropped.
    """
    key = "_".join(sorted(t.upper() for t in tickers)) + f"_{start}_{end or 'latest'}"
    cache_file = CACHE_DIR / f"prices_{key}.parquet"

    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):  # single ticker -> Series
        raw = raw.to_frame(tickers[0])
    prices = raw.dropna(how="any").sort_index()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_file)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns r_t = ln(P_t) - ln(P_{t-1}).

    First row (NaN) is dropped. Why log returns: they are time-additive
    and approximately equal to simple returns for small moves.
    """
    return np.log(prices).diff().dropna(how="any")
