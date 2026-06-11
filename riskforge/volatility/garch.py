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

import numpy as np
import pandas as pd
from scipy import optimize


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
    r = returns.to_numpy(dtype=float)
    r = r - r.mean()  # demean; for daily data the mean is ~0 anyway
    sample_var = float(np.var(r, ddof=1))

    def neg_loglik(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return np.inf
        sigma2 = _sigma2_recursion(r, omega, alpha, beta, sample_var)
        return 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + r**2 / sigma2)

    x0 = np.array([0.1 * sample_var, 0.05, 0.90])
    res = optimize.minimize(
        neg_loglik,
        x0,
        method="L-BFGS-B",
        bounds=[(1e-12, None), (0.0, 0.9999), (0.0, 0.9999)],
    )
    omega, alpha, beta = res.x
    return GarchParams(omega=float(omega), alpha=float(alpha), beta=float(beta),
                       loglik=float(-res.fun))


def _sigma2_recursion(
    r: np.ndarray, omega: float, alpha: float, beta: float, init_var: float
) -> np.ndarray:
    """Run the conditional variance recursion sigma_t^2."""
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = init_var
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def garch_forecast(returns: pd.Series, params: GarchParams, horizon: int = 1) -> float:
    """h-step-ahead variance forecast.

    sigma_{t+h}^2 = sigma^2_uncond + (alpha+beta)^h * (sigma_t^2 - sigma^2_uncond)
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r = returns.to_numpy(dtype=float)
    r = r - r.mean()
    sigma2 = _sigma2_recursion(r, params.omega, params.alpha, params.beta,
                               float(np.var(r, ddof=1)))
    # sigma_{t+1}^2 from the last observation, then mean-revert:
    sigma2_next = params.omega + params.alpha * r[-1] ** 2 + params.beta * sigma2[-1]
    uncond = params.unconditional_variance
    forecast = uncond + params.persistence ** (horizon - 1) * (sigma2_next - uncond)
    return float(forecast)
