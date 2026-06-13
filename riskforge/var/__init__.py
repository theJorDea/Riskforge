"""Value-at-Risk and Expected Shortfall estimators.

Conventions (used everywhere in this package)
---------------------------------------------
* ``alpha`` is the confidence level, e.g. 0.99.
* VaR is reported as a POSITIVE number: the loss threshold such that
  P(loss > VaR) = 1 - alpha.
* ES (a.k.a. CVaR) is the expected loss GIVEN the loss exceeds VaR;
  also positive, and ES >= VaR by construction.
"""

from riskforge.var.garch_t import garch_t_var
from riskforge.var.historical import historical_es, historical_var
from riskforge.var.monte_carlo import monte_carlo_var
from riskforge.var.parametric import parametric_var

__all__ = [
    "historical_var",
    "historical_es",
    "parametric_var",
    "monte_carlo_var",
    "garch_t_var",
]
