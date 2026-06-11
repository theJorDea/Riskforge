"""Christoffersen (1998) independence test (bonus module).

Kupiec only checks the NUMBER of breaches. Christoffersen checks whether
breaches CLUSTER in time (a good model has independent breaches; clustered
breaches mean the model reacts too slowly to volatility regimes).

Setup: model breaches as a first-order Markov chain with transition
probabilities pi_01 = P(breach today | no breach yesterday) and
pi_11 = P(breach today | breach yesterday).

H0: pi_01 = pi_11 (no clustering).
LR_ind ~ chi2(1) under H0.

Combined test: LR_cc = LR_pof + LR_ind ~ chi2(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ChristoffersenResult:
    lr_ind: float
    p_value: float
    reject_h0: bool


def christoffersen_test(breaches: pd.Series | list[bool]) -> ChristoffersenResult:
    """Independence test on the breach indicator sequence.

    Count transitions n00, n01, n10, n11; estimate pi_01, pi_11; build the
    likelihood ratio. Handle degenerate cases (no breaches, no transitions).
    """
    import numpy as np
    from scipy import stats

    b = np.asarray(breaches, dtype=int)
    if len(b) < 2:
        raise ValueError("need at least 2 observations")

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    def _xlogy(k: float, q: float) -> float:
        return 0.0 if k == 0 or q <= 0.0 else k * np.log(q)

    # Degenerate case: no breaches at all -> nothing to test, do not reject.
    if n01 + n11 == 0:
        return ChristoffersenResult(lr_ind=0.0, p_value=1.0, reject_h0=False)

    pi = (n01 + n11) / (n00 + n01 + n10 + n11)  # pooled breach prob (H0)
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    loglik_h0 = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    loglik_h1 = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
        + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    lr = -2.0 * (loglik_h0 - loglik_h1)
    p_value = float(stats.chi2.sf(lr, df=1))

    return ChristoffersenResult(lr_ind=float(lr), p_value=p_value, reject_h0=p_value < 0.05)
