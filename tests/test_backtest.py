"""Tests for VaR backtesting (Kupiec / Christoffersen / runner)."""

import numpy as np
import pytest


@pytest.mark.skip(reason="TODO: enable after implementing kupiec_test")
def test_kupiec_accepts_correct_coverage():
    """Breaches generated with the true rate p=1% must NOT be rejected."""
    from riskforge.backtest import kupiec_test

    rng = np.random.default_rng(3)
    breaches = rng.random(1000) < 0.01
    res = kupiec_test(breaches, alpha=0.99)
    assert not res.reject_h0


@pytest.mark.skip(reason="TODO: enable after implementing kupiec_test")
def test_kupiec_rejects_bad_coverage():
    """Breach rate 5% with promised 1% must be strongly rejected."""
    from riskforge.backtest import kupiec_test

    rng = np.random.default_rng(4)
    breaches = rng.random(1000) < 0.05
    res = kupiec_test(breaches, alpha=0.99)
    assert res.reject_h0
    assert res.p_value < 0.01


@pytest.mark.skip(reason="TODO: enable after implementing kupiec_test")
def test_kupiec_zero_breaches_edge_case():
    from riskforge.backtest import kupiec_test

    res = kupiec_test([False] * 500, alpha=0.99)
    assert np.isfinite(res.lr_stat)


@pytest.mark.skip(reason="TODO: enable after implementing christoffersen_test")
def test_christoffersen_detects_clustering():
    """A block of consecutive breaches must be flagged as dependent."""
    from riskforge.backtest import christoffersen_test

    breaches = [False] * 480 + [True] * 20
    res = christoffersen_test(breaches)
    assert res.reject_h0
