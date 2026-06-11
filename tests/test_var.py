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


def test_package_imports():
    """Smoke test: package is importable (keeps CI green from day one)."""
    import riskforge

    assert riskforge.__version__
