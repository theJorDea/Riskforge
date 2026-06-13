"""Tests for VaR estimators.

Strategy: on synthetic data with a KNOWN distribution the estimators must
converge to the analytical answer. This validates the math, not just the code.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

ALPHA = 0.99
SIGMA = 0.02
ANALYTICAL_VAR = -(SIGMA * norm.ppf(1 - ALPHA))  # mu = 0


@pytest.fixture
def normal_returns() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, SIGMA, size=100_000))


def test_historical_var_matches_analytical(normal_returns):
    from riskforge.var import historical_var

    var = historical_var(normal_returns, alpha=ALPHA)
    assert var == pytest.approx(ANALYTICAL_VAR, rel=0.05)


def test_es_not_less_than_var(normal_returns):
    from riskforge.var import historical_es, historical_var

    assert historical_es(normal_returns, ALPHA) >= historical_var(normal_returns, ALPHA)


def test_parametric_normal_var(normal_returns):
    from riskforge.var import parametric_var

    var = parametric_var(normal_returns, alpha=ALPHA, dist="normal")
    assert var == pytest.approx(ANALYTICAL_VAR, rel=0.05)


def test_monte_carlo_single_asset(normal_returns):
    from riskforge.var import monte_carlo_var

    df = normal_returns.to_frame("X")
    var = monte_carlo_var(df, weights=np.array([1.0]), alpha=ALPHA, seed=1)
    assert var == pytest.approx(ANALYTICAL_VAR, rel=0.05)


def test_monte_carlo_student_t(normal_returns):
    """t-MC with huge df ~ normal MC; with small df the tail is fatter."""
    from riskforge.var import monte_carlo_var

    df_ = normal_returns.to_frame("X")
    w = np.array([1.0])
    var_norm = monte_carlo_var(df_, w, alpha=ALPHA, seed=1, dist="normal")
    var_t_inf = monte_carlo_var(df_, w, alpha=ALPHA, seed=1, dist="t", df=1000)
    var_t_fat = monte_carlo_var(df_, w, alpha=ALPHA, seed=1, dist="t", df=4)
    assert var_t_inf == pytest.approx(var_norm, rel=0.05)
    assert var_t_fat > var_norm * 1.08


def test_package_imports():
    """Smoke test: package is importable (keeps CI green from day one)."""
    import riskforge

    assert riskforge.__version__


def _simulate_garch(n, innovation, rng, df=None):
    """GARCH(1,1) path with either normal or standardized-t innovations."""
    omega, alpha, beta = 1e-6, 0.08, 0.90
    sigma2 = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        if innovation == "normal":
            eps = rng.standard_normal()
        else:  # unit-variance standardized t
            eps = rng.standard_t(df) * np.sqrt((df - 2.0) / df)
        r[t] = np.sqrt(sigma2) * eps
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
    return pd.Series(r)


def test_garch_t_var_positive():
    """garch_t_var returns a positive next-day VaR."""
    from riskforge.var import garch_t_var

    r = _simulate_garch(3000, "normal", np.random.default_rng(0))
    assert garch_t_var(r, alpha=0.99) > 0


def test_garch_t_var_fat_tail_exceeds_gaussian_quantile():
    """With fat-tailed residuals the t-quantile VaR must exceed the Gaussian
    quantile applied to the *same* GARCH sigma forecast."""
    from scipy.stats import norm

    from riskforge.var import garch_t_var
    from riskforge.volatility import garch_fit, garch_forecast

    r = _simulate_garch(4000, "t", np.random.default_rng(1), df=4)
    params = garch_fit(r)
    sigma_next = np.sqrt(garch_forecast(r, params))
    var_gauss = -(r.mean() + sigma_next * norm.ppf(0.01))
    var_t = garch_t_var(r, alpha=0.99, params=params)
    assert var_t > var_gauss


def test_garch_t_var_matches_gaussian_on_normal_residuals():
    """When residuals are Gaussian, the fitted t has large nu, so the t-quantile
    VaR collapses back onto the Gaussian-quantile VaR."""
    from scipy.stats import norm

    from riskforge.var import garch_t_var
    from riskforge.volatility import garch_fit, garch_forecast

    r = _simulate_garch(6000, "normal", np.random.default_rng(2))
    params = garch_fit(r)
    sigma_next = np.sqrt(garch_forecast(r, params))
    var_gauss = -(r.mean() + sigma_next * norm.ppf(0.01))
    var_t = garch_t_var(r, alpha=0.99, params=params)
    assert var_t == pytest.approx(var_gauss, rel=0.15)
