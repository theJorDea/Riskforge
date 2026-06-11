"""Portfolio construction utilities.

A portfolio is defined by a weight vector w (sums to 1).
Portfolio return: r_p,t = w' r_t.
Portfolio variance (given covariance matrix S): sigma_p^2 = w' S w.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    """Compute portfolio return series r_p,t = w' r_t.

    Parameters
    ----------
    returns : DataFrame of asset log returns (T x N).
    weights : array of length N. Defaults to equal weights 1/N.

    Returns
    -------
    Series of portfolio returns indexed like ``returns``.

    Notes
    -----
    Strictly speaking, log returns do not aggregate linearly across assets,
    but for daily horizons the approximation error is negligible. Document
    this assumption in the README.
    """
    # TODO: implement (validate weights length, default to equal weights,
    #       check weights sum ~ 1.0, return returns @ weights as a Series)
    raise NotImplementedError


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix of asset returns (pandas ``.cov()``)."""
    # TODO: implement
    raise NotImplementedError
