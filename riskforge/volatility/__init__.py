"""Volatility models: EWMA (RiskMetrics) and GARCH(1,1)."""

from riskforge.volatility.ewma import ewma_volatility
from riskforge.volatility.garch import garch_fit, garch_forecast

__all__ = ["ewma_volatility", "garch_fit", "garch_forecast"]
