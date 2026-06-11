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
