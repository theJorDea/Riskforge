"""Rolling-window VaR backtest runner.

For each day t in the out-of-sample period:
1. Take the trailing ``window`` returns [t-window, t).
2. Estimate VaR_t with the supplied estimator.
3. Record a breach if the realized return r_t < -VaR_t.

Output feeds directly into kupiec_test / christoffersen_test and the
breach plot in plotting.py.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def rolling_var_backtest(
    returns: pd.Series,
    var_estimator: Callable[[pd.Series], float],
    window: int = 250,
) -> pd.DataFrame:
    """Run a rolling out-of-sample VaR backtest.

    Parameters
    ----------
    returns       : full return series.
    var_estimator : function mapping a return window -> positive VaR,
                    e.g. ``lambda r: historical_var(r, alpha=0.99)``.
    window        : estimation window length (250 ~ one trading year).

    Returns
    -------
    DataFrame indexed by date with columns:
        ``return`` (realized), ``var`` (forecast, positive), ``breach`` (bool).

    Note: this is a plain loop over t — keep it simple and readable first,
    optimize later only if needed.
    """
    # TODO: implement
    raise NotImplementedError
