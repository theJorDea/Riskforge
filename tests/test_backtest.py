"""Tests for VaR backtesting (Kupiec / Christoffersen / runner)."""

import numpy as np


def test_kupiec_accepts_correct_coverage():
    """Breaches generated with the true rate p=1% must NOT be rejected."""
    from riskforge.backtest import kupiec_test

    rng = np.random.default_rng(3)
    breaches = rng.random(1000) < 0.01
    res = kupiec_test(breaches, alpha=0.99)
    assert not res.reject_h0


def test_kupiec_rejects_bad_coverage():
    """Breach rate 5% with promised 1% must be strongly rejected."""
    from riskforge.backtest import kupiec_test

    rng = np.random.default_rng(4)
    breaches = rng.random(1000) < 0.05
    res = kupiec_test(breaches, alpha=0.99)
    assert res.reject_h0
    assert res.p_value < 0.01


def test_kupiec_zero_breaches_edge_case():
    from riskforge.backtest import kupiec_test

    res = kupiec_test([False] * 500, alpha=0.99)
    assert np.isfinite(res.lr_stat)


def test_christoffersen_detects_clustering():
    """A block of consecutive breaches must be flagged as dependent."""
    from riskforge.backtest import christoffersen_test

    breaches = [False] * 480 + [True] * 20
    res = christoffersen_test(breaches)
    assert res.reject_h0


def test_rolling_backtest_shape_and_coverage():
    """End-to-end: historical VaR on i.i.d. normal data ~ 1% breach rate."""
    import pandas as pd

    from riskforge.backtest import kupiec_test, rolling_var_backtest
    from riskforge.var import historical_var

    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0, 0.01, size=2000))
    bt = rolling_var_backtest(returns, lambda r: historical_var(r, alpha=0.99), window=250)
    assert len(bt) == 2000 - 250
    assert set(bt.columns) == {"return", "var", "breach"}
    res = kupiec_test(bt["breach"], alpha=0.99)
    assert not res.reject_h0


def _normal_var_es(sigma, alpha):
    """Analytical positive VaR and ES for N(0, sigma) at confidence alpha."""
    from scipy import stats

    p = 1.0 - alpha
    z_p = stats.norm.ppf(p)
    var = -sigma * z_p
    es = sigma * stats.norm.pdf(z_p) / p
    return var, es


def test_acerbi_szekely_accepts_correct_es():
    """ES forecast equal to the true ES of the data must NOT be rejected."""
    import numpy as np

    from riskforge.backtest import acerbi_szekely_test

    rng = np.random.default_rng(11)
    t, sigma, alpha = 1500, 0.01, 0.99
    x = rng.normal(0.0, sigma, size=t)
    var, es = _normal_var_es(sigma, alpha)
    res = acerbi_szekely_test(
        x, np.full(t, var), np.full(t, es), alpha, n_sims=3000, seed=1
    )
    assert not res.reject_h0
    assert abs(res.z2) < 0.6  # close to the E[Z2] = 0 ideal


def test_acerbi_szekely_rejects_underestimated_es():
    """Halving the VaR/ES forecasts (risk under-estimation) is strongly rejected."""
    import numpy as np

    from riskforge.backtest import acerbi_szekely_test

    rng = np.random.default_rng(11)
    t, sigma, alpha = 1500, 0.01, 0.99
    x = rng.normal(0.0, sigma, size=t)
    var, es = _normal_var_es(sigma, alpha)
    res = acerbi_szekely_test(
        x, np.full(t, var * 0.5), np.full(t, es * 0.5), alpha, n_sims=3000, seed=1
    )
    assert res.reject_h0
    assert res.z2 < 0  # realised tail losses far exceed predicted ES


def test_acerbi_szekely_one_sided_ignores_conservative_model():
    """An over-conservative model (ES too large) is not flagged by the left-tail test."""
    import numpy as np

    from riskforge.backtest import acerbi_szekely_test

    rng = np.random.default_rng(11)
    t, sigma, alpha = 1500, 0.01, 0.99
    x = rng.normal(0.0, sigma, size=t)
    var, es = _normal_var_es(sigma, alpha)
    res = acerbi_szekely_test(
        x, np.full(t, var * 2), np.full(t, es * 2), alpha, n_sims=2000, seed=1
    )
    assert not res.reject_h0


def test_acerbi_szekely_no_breaches_edge_case():
    """No breaches -> Z1 undefined (nan), Z2 finite, nothing rejected."""
    import numpy as np

    from riskforge.backtest import acerbi_szekely_test

    rng = np.random.default_rng(11)
    t, sigma, alpha = 800, 0.01, 0.99
    x = rng.normal(0.0, sigma, size=t)
    var, es = _normal_var_es(sigma, alpha)
    res = acerbi_szekely_test(
        x, np.full(t, var * 5), np.full(t, es * 5), alpha, n_sims=1000, seed=1
    )
    assert res.n_breaches == 0
    assert np.isnan(res.z1)
    assert np.isfinite(res.z2)
