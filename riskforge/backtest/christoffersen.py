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
    # TODO: implement
    raise NotImplementedError
