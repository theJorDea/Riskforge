"""GJR-GARCH(1,1,1): GARCH with a leverage (asymmetry) effect.

Motivation
----------
Plain GARCH treats a +5% day and a -5% day identically: only ``r_{t-1}^2``
enters the variance recursion. But equity volatility reacts *more* to drops
than to rallies (the "leverage effect"). GJR-GARCH (Glosten-Jagannathan-
Runkle, 1993) adds an extra term that switches on only after negative
returns.

Model
-----
r_t = sigma_t * eps_t,   eps_t ~ N(0, 1) i.i.d.
sigma_t^2 = omega + (alpha + gamma * I[r_{t-1} < 0]) * r_{t-1}^2 + beta * sigma_{t-1}^2

``gamma > 0`` means bad news raises tomorrow's variance by an extra
``gamma * r_{t-1}^2`` on top of the symmetric ``alpha`` term.

Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + gamma >= 0.
Stationarity (for symmetric innovations, E[I] = 1/2):
    alpha + beta + gamma/2 < 1.
Unconditional variance: sigma^2 = omega / (1 - alpha - beta - gamma/2).

Estimation: maximize the Gaussian log-likelihood with
``scipy.optimize.minimize`` (L-BFGS-B with bounds), exactly like ``garch.py``.
The test suite cross-checks the hand-written estimates against ``arch``
(``arch_model(..., p=1, o=1, q=1)``), the same proof strategy used for GARCH.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class GjrGarchParams:
    omega: float
    alpha: float
    gamma: float  # leverage term: extra ARCH weight after negative returns
    beta: float
    loglik: float

    @property
    def persistence(self) -> float:
        # E[I[r<0]] = 1/2 for symmetric innovations, so the gamma term
        # contributes gamma/2 to the mean reversion speed.
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)


def gjr_garch_fit(returns: pd.Series) -> GjrGarchParams:
    """Fit GJR-GARCH(1,1,1) by maximum likelihood (own implementation).

    Returns are demeaned first (for daily data the mean is ~0 anyway). The
    conditional-variance recursion is seeded with the sample variance.
    """
    r = returns.to_numpy(dtype=float)
    r = r - r.mean()
    sample_var = float(np.var(r, ddof=1))
    neg = r < 0.0  # leverage indicator I[r_{t-1} < 0], evaluated on r_{t-1}

    def neg_loglik(params: np.ndarray) -> float:
        omega, alpha, gamma, beta = params
        # Validity: positive variance everywhere + stationarity.
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + gamma < 0:
            return np.inf
        if alpha + beta + 0.5 * gamma >= 0.9999:
            return np.inf
        sigma2 = _sigma2_recursion(r, neg, omega, alpha, gamma, beta, sample_var)
        return 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + r**2 / sigma2)

    # Start from a typical equity fit: most persistence in beta, a little in
    # alpha, a small positive leverage term.
    x0 = np.array([0.05 * sample_var, 0.02, 0.06, 0.90])
    res = optimize.minimize(
        neg_loglik,
        x0,
        method="L-BFGS-B",
        bounds=[(1e-12, None), (0.0, 0.9999), (-0.5, 0.9999), (0.0, 0.9999)],
    )
    omega, alpha, gamma, beta = res.x
    return GjrGarchParams(
        omega=float(omega),
        alpha=float(alpha),
        gamma=float(gamma),
        beta=float(beta),
        loglik=float(-res.fun),
    )


def _sigma2_recursion(
    r: np.ndarray,
    neg: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    init_var: float,
) -> np.ndarray:
    """Conditional variance recursion with the asymmetric leverage term."""
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = init_var
    for t in range(1, n):
        shock = (alpha + gamma * neg[t - 1]) * r[t - 1] ** 2
        sigma2[t] = omega + shock + beta * sigma2[t - 1]
    return sigma2


def gjr_garch_forecast(
    returns: pd.Series, params: GjrGarchParams, horizon: int = 1
) -> float:
    """h-step-ahead variance forecast.

    The one-step update uses the realised last return (so the leverage term
    is known). For h > 1 we mean-revert with the *expected* persistence
    ``alpha + beta + gamma/2`` (the indicator's expectation under symmetry):

        sigma_{t+h}^2 = sigma^2_uncond
                        + persistence^{h-1} * (sigma_{t+1}^2 - sigma^2_uncond)
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r = returns.to_numpy(dtype=float)
    r = r - r.mean()
    neg = r < 0.0
    sigma2 = _sigma2_recursion(
        r, neg, params.omega, params.alpha, params.gamma, params.beta,
        float(np.var(r, ddof=1)),
    )
    shock = (params.alpha + params.gamma * (r[-1] < 0.0)) * r[-1] ** 2
    sigma2_next = params.omega + shock + params.beta * sigma2[-1]
    uncond = params.unconditional_variance
    forecast = uncond + params.persistence ** (horizon - 1) * (sigma2_next - uncond)
    return float(forecast)
