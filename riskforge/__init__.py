"""riskforge — market risk toolkit.

Modules
-------
data        : price loading (yfinance) with local parquet cache, log returns
portfolio   : portfolio returns, weights, covariance estimation
var         : Value-at-Risk / Expected Shortfall (historical, parametric, Monte Carlo)
volatility  : EWMA (RiskMetrics) and GARCH(1,1) volatility models
backtest    : VaR backtesting (Kupiec POF, Christoffersen independence, rolling runner)
plotting    : standard diagnostic plots
"""

__version__ = "0.1.0"
