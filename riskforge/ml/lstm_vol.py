"""LSTM volatility forecaster (PyTorch) — ML baseline vs GARCH.

Task: predict next-day variance sigma_{t+1}^2 from a window of the last
``lookback`` squared returns. The model is trained on the QLIKE loss —
the same metric used to compare volatility forecasts, so training and
evaluation are consistent:

    QLIKE(sigma2_hat, r2) = ln(sigma2_hat) + r2 / sigma2_hat

(minimized in expectation by the true conditional variance; robust to the
noise of the r^2 proxy, unlike MSE).

Architecture is deliberately small (single-layer LSTM + linear head):
the dataset is ~2500 points, anything bigger memorizes noise. The network
outputs ln(sigma2) so positivity is guaranteed by construction.

PyTorch is an optional dependency: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@dataclass
class LSTMVolResult:
    """Fitted forecaster + per-epoch training loss for diagnostics."""

    model: nn.Module
    train_loss: list[float]
    lookback: int
    scale: float  # std of r used to normalize inputs


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -e '.[ml]'")


class _LSTMVol(nn.Module if HAS_TORCH else object):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)            # (B, T, H)
        log_sigma2 = self.head(out[:, -1, :])  # last step -> (B, 1)
        return log_sigma2.squeeze(-1)


def _make_windows(r2: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """X[i] = r2[i : i+lookback], y[i] = r2[i+lookback] (next-day proxy)."""
    n = len(r2) - lookback
    x = np.lib.stride_tricks.sliding_window_view(r2, lookback)[:n]
    y = r2[lookback:]
    return x, y


def lstm_vol_fit(
    returns: pd.Series,
    lookback: int = 22,
    hidden: int = 16,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
) -> LSTMVolResult:
    """Train the LSTM variance forecaster on (demeaned) returns.

    Inputs are normalized by the sample std (r/s)^2 so the network works on
    O(1) numbers; the predicted log-variance is shifted back by ln(s^2).
    """
    _require_torch()
    torch.manual_seed(seed)

    r = returns.to_numpy(dtype=float)
    r = r - r.mean()
    scale = float(r.std(ddof=1))
    r2 = (r / scale) ** 2

    x_np, y_np = _make_windows(r2, lookback)
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(-1)  # (B, T, 1)
    y = torch.tensor(y_np, dtype=torch.float32)

    model = _LSTMVol(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    eps = 1e-8
    for _ in range(epochs):
        opt.zero_grad()
        log_s2 = model(x)
        # QLIKE in log-parameterization: log_s2 + y * exp(-log_s2)
        loss = (log_s2 + (y + eps) * torch.exp(-log_s2)).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return LSTMVolResult(model=model, train_loss=losses, lookback=lookback, scale=scale)


def lstm_vol_predict(result: LSTMVolResult, returns: pd.Series) -> pd.Series:
    """One-step-ahead sigma_t forecasts for every day a full window exists.

    Output index is aligned with the target day t (forecast made from the
    window ending at t-1), so it can be compared directly with GARCH/EWMA
    series and with realized r_t^2.
    """
    _require_torch()
    r = returns.to_numpy(dtype=float)
    r = r - r.mean()
    r2 = (r / result.scale) ** 2
    x_np, _ = _make_windows(r2, result.lookback)
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        log_s2 = result.model(x).numpy()
    sigma = np.sqrt(np.exp(log_s2)) * result.scale
    return pd.Series(sigma, index=returns.index[result.lookback :], name="lstm_vol")
