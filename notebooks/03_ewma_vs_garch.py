# %% [markdown]
# # 03 — EWMA vs GARCH(1,1): оценка и прогноз волатильности
#
# SPY, 2015–2025. Сравниваем две модели условной волатильности:
# фиксированную EWMA (λ=0.94) и GARCH(1,1) с параметрами из собственной MLE.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskforge.backtest import christoffersen_test, kupiec_test, rolling_var_backtest
from riskforge.data import load_prices, log_returns
from riskforge.plotting import plot_volatility
from riskforge.volatility import ewma_volatility, garch_fit
from riskforge.volatility.garch import _sigma2_recursion

prices = load_prices(["SPY", "QQQ", "TLT", "GLD"], start="2015-01-01", end="2025-06-01")
r = log_returns(prices)["SPY"]

# %% [markdown]
# ## Подгонка GARCH(1,1) собственной MLE

# %%
params = garch_fit(r)
print(f"omega = {params.omega:.3e}")
print(f"alpha = {params.alpha:.4f}  (реакция на шок)")
print(f"beta  = {params.beta:.4f}  (инерция)")
print(f"alpha+beta = {params.persistence:.4f}  (персистентность)")
print(f"безусловная vol: {np.sqrt(params.unconditional_variance * 252):.1%} годовых")
print(f"выборочная vol:  {r.std() * np.sqrt(252):.1%} годовых")

# %% [markdown]
# Типичная картина для дневного индекса: α ≈ 0.1–0.2, β ≈ 0.8–0.9,
# α+β близко к 1 — шоки волатильности затухают неделями.
#
# ## Траектории σ_t

# %%
ewma_vol = ewma_volatility(r, lam=0.94)
rc = (r - r.mean()).to_numpy()
garch_sigma2 = _sigma2_recursion(rc, params.omega, params.alpha, params.beta, float(np.var(rc, ddof=1)))
garch_vol = pd.Series(np.sqrt(garch_sigma2), index=r.index, name="garch_vol")

fig = plot_volatility(r, {"EWMA (λ=0.94)": ewma_vol, "GARCH(1,1)": garch_vol})

# %% [markdown]
# Траектории близки (EWMA — частный случай GARCH), различия:
# GARCH быстрее «отпускает» шок назад к среднему уровню, EWMA после COVID-пика
# дольше держит завышенную оценку.
#
# ## Качество прогноза на 1 день: QLIKE и MSE
#
# Прокси истинной дисперсии — r²_{t+1} (шумный, но несмещённый). QLIKE
# `ln σ̂² + r²/σ̂²` устойчивее MSE к выбросам прокси и является стандартом
# в литературе по сравнению vol-моделей (Patton, 2011).

# %%
sigma2_f = pd.DataFrame({"EWMA": ewma_vol**2, "GARCH": garch_vol**2}).shift(0)
target = (r**2).rename("r2")
scores = {}
for name in sigma2_f:
    s2 = sigma2_f[name].iloc[1:]
    y = target.iloc[1:]
    scores[name] = {
        "QLIKE": float(np.mean(np.log(s2) + y / s2)),
        "MSE ×1e8": float(np.mean((s2 - y) ** 2) * 1e8),
    }
print(pd.DataFrame(scores).T.round(4).to_string())

# %% [markdown]
# ## GARCH-VaR против исторического: лечится ли кластеризация пробитий
#
# Динамический VaR: `VaR_t = σ_{t+1|t} · z_{0.99}` (среднее ≈ 0).
# Оценка параметров — на скользящем окне 500 дней, без look-ahead.

# %%
from scipy import stats

from riskforge.var import historical_var

Z99 = -stats.norm.ppf(0.01)


def garch_var(window: pd.Series) -> float:
    p = garch_fit(window)
    rc = (window - window.mean()).to_numpy()
    s2 = _sigma2_recursion(rc, p.omega, p.alpha, p.beta, float(np.var(rc, ddof=1)))
    s2_next = p.omega + p.alpha * rc[-1] ** 2 + p.beta * s2[-1]
    return float(np.sqrt(s2_next) * Z99)


bt_garch = rolling_var_backtest(r, garch_var, window=500)
bt_hist = rolling_var_backtest(r, lambda w: historical_var(w, 0.99), window=500)

rows = []
for name, bt in [("GARCH-VaR", bt_garch), ("Historical VaR", bt_hist)]:
    kup = kupiec_test(bt["breach"], alpha=0.99)
    ch = christoffersen_test(bt["breach"])
    rows.append(
        {
            "модель": name,
            "пробитий": kup.n_breaches,
            "ожидалось": round(kup.expected_breaches, 1),
            "Kupiec p": round(kup.p_value, 4),
            "Kupiec": "REJECT" if kup.reject_h0 else "pass",
            "Christ. p": round(ch.p_value, 4),
            "Christ.": "REJECT" if ch.reject_h0 else "pass",
        }
    )
print(pd.DataFrame(rows).set_index("модель").to_string())

# %% [markdown]
# ## Выводы (по фактическим числам выше)
#
# 1. Собственная MLE сходится к адекватным параметрам (β ≫ α, заметная
#    персистентность); на этом периоде α+β ≈ 0.90 — ниже хрестоматийных 0.95+,
#    оценка на 10 годах с несколькими режимами это допускает.
# 2. По QLIKE на 1 день вперёд EWMA здесь даже чуть лучше GARCH: для
#    однодневного горизонта возврат к среднему почти не успевает сыграть,
#    а λ=0.94 близка к оптимуму. GARCH выигрывает на многодневных горизонтах.
# 3. Бэктест честно показывает: динамическая σ_t заметно снижает кластеризацию
#    (p-value Кристофферсена 0.024 против 0.0003 у исторического), но
#    гауссов квантиль занижает хвост — пробитий больше ожидаемого, Купик
#    отвергает. Правильное направление — GARCH-σ_t + t-квантиль
#    (комбинация динамики и жирных хвостов); это пункт roadmap.
# 4. Главный урок: ни «правильное распределение со статичной σ», ни
#    «динамическая σ с тонкими хвостами» по отдельности тест не проходят —
#    нужны обе компоненты.
