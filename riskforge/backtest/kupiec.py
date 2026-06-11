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
    # TODO: implement with scipy.stats.chi2.sf(lr_stat, df=1)
    raise NotImplementedError
