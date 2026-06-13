"""Tests for volatility models."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def garch_returns() -> pd.Series:
    """Simulate GARCH(1,1) data with known parameters."""
    rng = np.random.default_rng(7)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    n = 5000
    sigma2 = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(sigma2) * rng.standard_normal()
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
    return pd.Series(r)


def test_ewma_constant_series():
    """On constant-variance i.i.d. data EWMA must hover near the true sigma."""
    from riskforge.volatility import ewma_volatility

    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, size=10_000))
    vol = ewma_volatility(r, lam=0.94)
    assert vol.iloc[-500:].mean() == pytest.approx(0.01, rel=0.1)


def test_garch_recovers_true_params(garch_returns):
    """Own MLE should recover simulated parameters (loose tolerance)."""
    from riskforge.volatility import garch_fit

    params = garch_fit(garch_returns)
    assert params.alpha == pytest.approx(0.08, abs=0.04)
    assert params.beta == pytest.approx(0.90, abs=0.05)
    assert params.persistence < 1.0


def test_garch_matches_arch_package(garch_returns):
    """Cross-check own MLE against the reference `arch` implementation."""
    from arch import arch_model

    from riskforge.volatility import garch_fit

    own = garch_fit(garch_returns)
    ref = arch_model(garch_returns * 100, vol="GARCH", p=1, q=1, mean="Zero").fit(disp="off")
    assert own.alpha == pytest.approx(ref.params["alpha[1]"], abs=0.03)
    assert own.beta == pytest.approx(ref.params["beta[1]"], abs=0.03)


@pytest.fixture
def gjr_returns() -> pd.Series:
    """Simulate GJR-GARCH(1,1,1) data with a known positive leverage effect."""
    rng = np.random.default_rng(7)
    omega, alpha, gamma, beta = 1e-6, 0.03, 0.08, 0.90
    n = 6000
    sigma2 = omega / (1 - alpha - beta - gamma / 2)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(sigma2) * rng.standard_normal()
        sigma2 = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * sigma2
    return pd.Series(r)


def test_gjr_recovers_true_params(gjr_returns):
    """Own MLE should recover the simulated leverage parameters (loose tol)."""
    from riskforge.volatility import gjr_garch_fit

    params = gjr_garch_fit(gjr_returns)
    assert params.gamma == pytest.approx(0.08, abs=0.04)
    assert params.gamma > 0  # leverage effect detected
    assert params.beta == pytest.approx(0.90, abs=0.05)
    assert params.persistence < 1.0


def test_gjr_matches_arch_package(gjr_returns):
    """Cross-check own GJR MLE against arch (vol='GARCH', o=1)."""
    from arch import arch_model

    from riskforge.volatility import gjr_garch_fit

    own = gjr_garch_fit(gjr_returns)
    ref = arch_model(
        gjr_returns * 100, vol="GARCH", p=1, o=1, q=1, mean="Zero"
    ).fit(disp="off")
    assert own.alpha == pytest.approx(ref.params["alpha[1]"], abs=0.03)
    assert own.gamma == pytest.approx(ref.params["gamma[1]"], abs=0.03)
    assert own.beta == pytest.approx(ref.params["beta[1]"], abs=0.03)


def test_gjr_nests_garch_in_likelihood(gjr_returns):
    """GJR nests GARCH (gamma=0), so its max log-likelihood cannot be lower."""
    from riskforge.volatility import garch_fit, gjr_garch_fit

    garch = garch_fit(gjr_returns)
    gjr = gjr_garch_fit(gjr_returns)
    assert gjr.loglik >= garch.loglik - 1e-6
