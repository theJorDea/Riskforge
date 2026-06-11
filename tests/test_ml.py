"""Tests for the optional LSTM volatility forecaster (skipped without torch)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _garch_series(n=1500, omega=2e-6, alpha=0.1, beta=0.85, seed=7):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(1, n):
        s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        r[t] = np.sqrt(s2) * rng.standard_normal()
    return pd.Series(r)


def test_lstm_vol_trains_and_predicts():
    from riskforge.ml import lstm_vol_fit, lstm_vol_predict

    r = _garch_series()
    res = lstm_vol_fit(r, lookback=22, epochs=60, seed=0)
    # training loss should clearly decrease
    assert res.train_loss[-1] < res.train_loss[0] - 0.01
    vol = lstm_vol_predict(res, r)
    assert len(vol) == len(r) - 22
    assert (vol > 0).all()
    # forecasts should correlate with true conditional vol regime: check
    # against a 22d realized vol proxy
    realized = r.rolling(22).std().shift(-1).dropna()
    joined = pd.concat([vol, realized], axis=1).dropna()
    corr = joined.corr().iloc[0, 1]
    assert corr > 0.5
