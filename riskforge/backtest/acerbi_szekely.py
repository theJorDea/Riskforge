"""Acerbi-Szekely (2014) backtests for Expected Shortfall.

Why ES needs its own backtest
-----------------------------
Kupiec and Christoffersen test VaR: they only look at *whether* the loss
crossed the threshold (a 0/1 breach), never at *how far* past it the loss
went. ES is precisely the expected size of those tail losses, so a breach
indicator cannot validate it. ES is also not "elicitable", which is why a
simple score-comparison test does not exist; Acerbi & Szekely's answer is a
pair of statistics with E[Z] = 0 under a correct model, calibrated by Monte
Carlo simulation.

Conventions (consistent with the rest of the package)
-----------------------------------------------------
* ``returns`` X_t are realised P&L (negative = loss).
* ``var`` and ``es`` are POSITIVE forecasts at confidence ``alpha``; tail
  probability is ``p = 1 - alpha`` (e.g. 0.01).
* Breach on day t:  I_t = 1{ X_t < -VaR_t }  (the loss exceeded VaR).

The two statistics
------------------
Both use the per-day tail contribution  X_t * I_t / ES_t  (negative on breach
days). Let N = sum I_t be the number of breaches.

Z1 (magnitude, conditional on the breaches):
    Z1 = (1/N) * sum_t [ X_t * I_t / ES_t ] + 1
    Tests only whether the *average* breach size matches ES. Undefined if
    there are no breaches.

Z2 (frequency AND magnitude, the recommended one):
    Z2 = (1 / (T * p)) * sum_t [ X_t * I_t / ES_t ] + 1
    Jointly penalises too many breaches and breaches that are too deep.

Under H0 (the forecast distribution is correct) E[Z1] = E[Z2] = 0. A
*negative* value means realised tail losses are worse than predicted ES, i.e.
the model under-estimates risk -> the test is one-sided to the left.

Significance (Monte Carlo, as prescribed in the paper)
------------------------------------------------------
There is no closed form, so we simulate. Under H0 each day's P&L is
sigma_t * g, where g is a standardized innovation with distribution G and
sigma_t is the day's scale. We recover sigma_t from the forecasts themselves:
since ES_t = sigma_t * ES_G (ES_G = standardized ES of G at level p),
sigma_t = ES_t / ES_G. We then draw many synthetic paths, recompute Z2 with
the *same* VaR_t/ES_t, and report the left-tail p-value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AcerbiSzekelyResult:
    z1: float          # magnitude statistic (nan if no breaches)
    z2: float          # frequency+magnitude statistic (the recommended one)
    p_value: float     # one-sided MC p-value for Z2 (left tail)
    reject_h0: bool    # at 5% significance
    n_obs: int
    n_breaches: int


def _as_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def _tail_contrib(returns: np.ndarray, var: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Per-day term X_t * I_t / ES_t, zero on non-breach days."""
    breach = returns < -var
    contrib = np.zeros_like(returns)
    contrib[breach] = returns[breach] / es[breach]
    return contrib


def _standardized_innovations(
    p: float, innovation: str, df: float | None, rng: np.random.Generator
) -> tuple[Callable[[int], np.ndarray], float]:
    """Return ``(sampler, es_g)`` for the H0 innovation distribution G.

    ``sampler(size)`` draws standardized innovations; ``es_g`` is the
    standardized, positive ES of G at tail probability ``p`` (used to back out
    each day's scale sigma_t = ES_t / es_g). ``df`` is required for "t".
    """
    from scipy import stats

    if innovation == "normal":
        z_p = stats.norm.ppf(p)
        es_g = float(stats.norm.pdf(z_p) / p)  # standard-normal ES, positive
        sampler = lambda size: rng.standard_normal(size)  # noqa: E731
        return sampler, es_g
    if innovation == "t":
        if df is None or df <= 2:
            raise ValueError("innovation='t' needs df > 2 for finite variance")
        # Standardized (unit-variance) Student-t: T * sqrt((df-2)/df).
        c = np.sqrt((df - 2.0) / df)
        t_p = stats.t.ppf(p, df)
        # ES of a standard t_nu (positive): pdf(t_p)/p * (nu + t_p^2)/(nu - 1).
        es_std_t = stats.t.pdf(t_p, df) / p * (df + t_p**2) / (df - 1.0)
        es_g = float(c * es_std_t)
        sampler = lambda size: c * rng.standard_t(df, size=size)  # noqa: E731
        return sampler, es_g
    raise ValueError(f"innovation must be 'normal' or 't', got {innovation!r}")


def acerbi_szekely_test(
    returns: pd.Series | np.ndarray,
    var: pd.Series | np.ndarray,
    es: pd.Series | np.ndarray,
    alpha: float = 0.99,
    *,
    innovation: str = "normal",
    df: float | None = None,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> AcerbiSzekelyResult:
    """Acerbi-Szekely ES backtest with a Monte Carlo p-value for Z2.

    Parameters
    ----------
    returns      : realised P&L series (negative = loss).
    var, es      : positive VaR and ES forecasts at confidence ``alpha``.
    alpha        : confidence level; tail probability p = 1 - alpha.
    innovation   : H0 innovation distribution, "normal" or "t".
    df           : degrees of freedom when innovation="t".
    n_sims, seed : Monte Carlo controls (seed pinned for reproducible tests).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    x = _as_array(returns)
    v = _as_array(var)
    e = _as_array(es)
    t = len(x)
    if not (len(v) == len(e) == t):
        raise ValueError("returns, var and es must have the same length")
    if t == 0:
        raise ValueError("empty series")
    if np.any(e <= 0):
        raise ValueError("ES forecasts must be strictly positive")
    p = 1.0 - alpha

    contrib = _tail_contrib(x, v, e)
    n_breaches = int(np.count_nonzero(x < -v))
    sum_contrib = float(contrib.sum())

    z2 = sum_contrib / (t * p) + 1.0
    z1 = (sum_contrib / n_breaches + 1.0) if n_breaches > 0 else float("nan")

    # ----- Monte Carlo p-value for Z2 under H0 -----
    rng = np.random.default_rng(seed)
    sampler, es_g = _standardized_innovations(p, innovation, df, rng)
    sigma_t = e / es_g  # back out each day's scale from its ES forecast

    denom = t * p
    z2_sim = np.empty(n_sims)
    for s in range(n_sims):
        x_sim = sigma_t * sampler(t)
        contrib_sim = _tail_contrib(x_sim, v, e)
        z2_sim[s] = contrib_sim.sum() / denom + 1.0

    # One-sided: reject when realised Z2 sits in the left tail (risk under-est.).
    p_value = float((np.count_nonzero(z2_sim <= z2) + 1) / (n_sims + 1))

    return AcerbiSzekelyResult(
        z1=z1,
        z2=z2,
        p_value=p_value,
        reject_h0=p_value < 0.05,
        n_obs=t,
        n_breaches=n_breaches,
    )
