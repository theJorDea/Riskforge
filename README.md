# Riskforge

Market risk toolkit built from first principles: **Value-at-Risk / Expected Shortfall estimation, volatility modeling, and statistical backtesting** — implemented by hand on the Python data stack (NumPy / pandas / SciPy) and validated against reference implementations.

The goal of this project is not to wrap libraries, but to demonstrate the mathematics behind market risk models: where they come from, what assumptions they make, and how to verify that they actually work.

> Status: 🚧 work in progress. Module skeletons, tests, and CI are in place; implementations are being filled in module by module (see roadmap below).

## What's inside

| Area | Methods |
|---|---|
| **VaR / ES** | Historical (empirical quantile) · Parametric (Normal & Student-t, MLE) · Monte Carlo (multivariate normal scenarios) |
| **Volatility** | EWMA / RiskMetrics (λ = 0.94, hand-written recursion) · GARCH(1,1) with own MLE on `scipy.optimize`, cross-checked against the `arch` package |
| **Backtesting** | Rolling out-of-sample VaR backtest · Kupiec POF test · Christoffersen independence test |
| **Research notebooks** | EDA of stylized facts (fat tails, volatility clustering) · model comparison with honest conclusions |

## Project structure

```
riskforge/
├── riskforge/                  # the package itself
│   ├── data.py                 # price loading (yfinance) + parquet cache, log returns
│   ├── portfolio.py            # portfolio returns w'r, covariance estimation
│   ├── var/
│   │   ├── historical.py       # VaR/ES from the empirical quantile of past returns
│   │   ├── parametric.py       # Normal and Student-t VaR (parameters via MLE)
│   │   └── monte_carlo.py      # scenario simulation from fitted N(mu, Sigma)
│   ├── volatility/
│   │   ├── ewma.py             # RiskMetrics recursion, implemented by hand on NumPy
│   │   └── garch.py            # GARCH(1,1): own log-likelihood + scipy MLE,
│   │                           #   verified against the `arch` package in tests
│   ├── backtest/
│   │   ├── kupiec.py           # POF test: is the breach COUNT consistent with 1-alpha?
│   │   ├── christoffersen.py   # independence test: do breaches CLUSTER in time?
│   │   └── runner.py           # rolling-window out-of-sample backtest engine
│   └── plotting.py             # returns vs VaR boundary, QQ-plots, volatility overlays
├── notebooks/
│   ├── 01_eda.ipynb            # stylized facts: fat tails, clustering, QQ-plots
│   ├── 02_var_comparison.ipynb # 3 VaR methods on one portfolio + backtest verdicts
│   └── 03_vol_forecast.ipynb   # EWMA vs GARCH forecasting comparison
├── tests/                      # math validated on synthetic data with known answers
├── data/                       # local price cache (gitignored)
└── .github/workflows/ci.yml    # ruff + pytest on every push (Python 3.10 / 3.12)
```

A detailed module-by-module description with formulas and conventions lives in [`docs/STRUCTURE.md`](docs/STRUCTURE.md).

## Conventions

* `alpha` is the confidence level (e.g. `0.99`); expected breach rate is `1 - alpha`.
* **VaR and ES are reported as positive numbers** (loss magnitudes). `ES >= VaR` always.
* Returns are daily log returns `r_t = ln(P_t / P_{t-1})`.
* Every estimator is a pure function: same inputs → same outputs (Monte Carlo takes an explicit `seed`).

## Testing philosophy

The math is validated on synthetic data where the exact answer is known:

* On `N(0, σ²)` samples, historical/parametric/MC VaR must converge to the analytical quantile `-σ·z₁₋α`.
* The hand-written GARCH MLE must recover the parameters of a simulated GARCH(1,1) process **and** agree with the `arch` package on the same data.
* Kupiec must *not* reject a breach series generated with the true rate, and must strongly reject a miscalibrated one.

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Package skeleton, module contracts, test suite design, CI
- [ ] `data` + `portfolio` + EDA notebook (stylized facts)
- [ ] Historical & parametric VaR/ES + tests
- [ ] Monte Carlo VaR + tests
- [ ] Rolling backtest + Kupiec/Christoffersen + comparison notebook
- [ ] EWMA + GARCH(1,1) own MLE + `arch` cross-check + forecast notebook
- [ ] Model limitations write-up (when and why each method fails)
- [ ] (Optional) Student-t Monte Carlo; LSTM volatility forecast vs GARCH baseline (PyTorch)

## Known model limitations (to be expanded with results)

* Historical VaR is blind to anything not in the estimation window and reacts slowly to regime changes.
* Normal parametric VaR underestimates tail risk on fat-tailed daily returns.
* All methods here assume the portfolio is static and liquid; no intraday, funding, or liquidity risk.
