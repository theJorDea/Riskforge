"""GARCH(1,1) with hand-written MLE, cross-checked against the `arch` package.

Model
-----
r_t = sigma_t * eps_t,   eps_t ~ N(0, 1) i.i.d.
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (stationarity).
Unconditional variance: sigma^2 = omega / (1 - alpha - beta).

Log-likelihood (Gaussian):
L = -0.5 * sum( ln(2*pi) + ln(sigma_t^2) + r_t^2 / sigma_t^2 )

Estimation: maximize L over (omega, alpha, beta) with
``scipy.optimize.minimize`` (L-BFGS-B with bounds, minimize -L).
The test suite checks the hand-written estimates against ``arch``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GarchParams:
    omega: float
    alpha: float
    beta: float
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)


def garch_fit(returns: pd.Series) -> GarchParams:
    """Fit GARCH(1,1) by maximum likelihood (own implementation on scipy).

    Steps
    -----
    1. Demean returns (or assume zero mean for daily data — document choice).
    2. Define negative log-likelihood: run the sigma_t^2 recursion,
       initialize sigma_0^2 with sample variance.
    3. Minimize with L-BFGS-B, bounds: omega>1e-12, 0<=alpha<1, 0<=beta<1,
       and reject alpha+beta>=1 inside the objective (return +inf).
    4. Return fitted GarchParams.
    """
    # TODO: implement
    raise NotImplementedError


def garch_forecast(returns: pd.Series, params: GarchParams, horizon: int = 1) -> float:
    """h-step-ahead variance forecast.

    sigma_{t+h}^2 = sigma^2_uncond + (alpha+beta)^h * (sigma_t^2 - sigma^2_uncond)
    """
    # TODO: implement
    raise NotImplementedError
