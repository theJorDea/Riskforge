"""Kupiec (1995) Proportion-of-Failures (POF) test.

Question: is the observed number of VaR breaches consistent with the
promised coverage p = 1 - alpha?

Setup
-----
T observations, x breaches (days when loss > VaR), p = 1 - alpha.
H0: true breach probability equals p.

Likelihood ratio statistic:
LR_pof = -2 * ln[ (1-p)^(T-x) * p^x ] + 2 * ln[ (1-x/T)^(T-x) * (x/T)^x ]

Under H0, LR_pof ~ chi2(1). Reject if p-value < significance level.
Use log-space arithmetic throughout to avoid underflow for large T.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class KupiecResult:
    n_obs: int
    n_breaches: int
    expected_breaches: float
    lr_stat: float
    p_value: float
    reject_h0: bool  # at 5% significance


def kupiec_test(breaches: pd.Series | list[bool], alpha: float = 0.99) -> KupiecResult:
    """Run the Kupiec POF test on a boolean breach series.

    Parameters
    ----------
    breaches : boolean sequence, True on days when loss exceeded VaR.
    alpha    : VaR confidence level (so expected breach rate is 1 - alpha).

    Edge cases to handle: x = 0 and x = T (the LR formula has 0*ln(0) terms —
    treat them as 0).
    """
    import numpy as np
    from scipy import stats

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    b = np.asarray(breaches, dtype=bool)
    t = len(b)
    if t == 0:
        raise ValueError("empty breach series")
    x = int(b.sum())
    p = 1.0 - alpha

    def _xlogy(k: float, q: float) -> float:
        """k * ln(q) with the convention 0 * ln(0) = 0."""
        return 0.0 if k == 0 else k * np.log(q)

    loglik_h0 = _xlogy(t - x, 1.0 - p) + _xlogy(x, p)
    pi_hat = x / t
    loglik_h1 = _xlogy(t - x, 1.0 - pi_hat) + _xlogy(x, pi_hat)
    lr = -2.0 * (loglik_h0 - loglik_h1)
    p_value = float(stats.chi2.sf(lr, df=1))

    return KupiecResult(
        n_obs=t,
        n_breaches=x,
        expected_breaches=t * p,
        lr_stat=float(lr),
        p_value=p_value,
        reject_h0=p_value < 0.05,
    )
