"""Midterm Problem 2 -- VaR and ES for a single stock.

    python problem2.py

Model: r_XYZ = alpha + beta * r_SPY + eps, with r_SPY and eps normal.

The exam text does not say whether to keep the fitted mean, so every number is
printed for both variants, side by side:
  "with mean"  : alpha and the SPY sample mean are kept
  "zero mean"  : alpha = 0 and the SPY mean = 0
Each variant is computed in closed form (a normal) and by simulation.
Output is also saved to output/problem2.txt.
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import MIDTERM_DIR, Report, rl
import settings

rep = Report()
LEVELS = (0.95, 0.99)   # confidence levels; alpha = 1 - level
SHARES = 100

prices = pd.read_csv(MIDTERM_DIR / "problem2.csv")
ret = rl.return_calculate(prices, method="DISCRETE", date_column="Date")   # arithmetic returns
spy, xyz = ret["SPY"].to_numpy(), ret["XYZ"].to_numpy()
T = len(ret)

# ---------------------------------------------------------------------------
# Regression of XYZ on SPY (OLS, normal equations)
# ---------------------------------------------------------------------------
rep.section("Regression: r_XYZ = alpha + beta * r_SPY + eps")
X = np.column_stack([np.ones(T), spy])
alpha, beta = np.linalg.solve(X.T @ X, X.T @ xyz)
resid = xyz - X @ np.array([alpha, beta])
sd_eps = float(np.sqrt(resid @ resid / (T - 2)))     # divides by T - 2 (observations minus coefficients)
mu_spy, sd_spy = float(spy.mean()), float(spy.std(ddof=1))

rep.say(f"{T} arithmetic returns")
rep.say(f"alpha = {alpha:.6f}   beta = {beta:.6f}   residual SD = {sd_eps:.6f}")
rep.say(f"SPY return: mean = {mu_spy:.6f}   SD = {sd_spy:.6f}")

# ---------------------------------------------------------------------------
# Distribution of r_XYZ under each variant (normal SPY + normal eps, independent)
# ---------------------------------------------------------------------------
variants = {
    "with mean": (alpha + beta * mu_spy, alpha, mu_spy),
    "zero mean": (0.0, 0.0, 0.0),
}
sd_xyz = float(np.sqrt(beta ** 2 * sd_spy ** 2 + sd_eps ** 2))

rng = np.random.default_rng(settings.SEED)
spy_draw = rng.standard_normal(settings.N_SIM_Q2)
eps_draw = rng.standard_normal(settings.N_SIM_Q2)

position_value = SHARES * float(prices["XYZ"].iloc[-1])
results = []

for label, (mean_xyz, a, m_spy) in variants.items():
    dist = stats.norm(mean_xyz, sd_xyz)
    sim = a + beta * (m_spy + sd_spy * spy_draw) + sd_eps * eps_draw
    for level in LEVELS:
        alpha_tail = 1 - level
        results.append({
            "variant": label, "level": f"{level:.0%}", "method": "closed form",
            "VaR (%)": 100 * rl.value_at_risk(dist, alpha_tail),
            "ES (%)": 100 * rl.expected_shortfall(dist, alpha_tail)})
        results.append({
            "variant": label, "level": f"{level:.0%}", "method": f"simulation ({settings.N_SIM_Q2:,})",
            "VaR (%)": 100 * rl.value_at_risk(sim, alpha_tail),
            "ES (%)": 100 * rl.expected_shortfall(sim, alpha_tail)})

# ---------------------------------------------------------------------------
# Problem 2(a): VaR at 95% and 99%, in percent
# ---------------------------------------------------------------------------
table = pd.DataFrame(results)
rep.section("Problem 2(a): VaR in percent of position value")
rep.say(f"Position: {SHARES} shares at the last price {prices['XYZ'].iloc[-1]:.4f} = ${position_value:,.2f}")
rep.say(f"r_XYZ standard deviation = sqrt(beta^2 * var(SPY) + var(eps)) = {sd_xyz:.6f}\n")
rep.say(table[["variant", "level", "method", "VaR (%)"]].to_string(index=False, float_format=lambda v: f"{v: .4f}"))

# ---------------------------------------------------------------------------
# Problem 2(b): ES at 95% and 99%, in percent
# ---------------------------------------------------------------------------
rep.section("Problem 2(b): ES in percent of position value")
rep.say(table[["variant", "level", "method", "ES (%)"]].to_string(index=False, float_format=lambda v: f"{v: .4f}"))

rep.save("problem2.txt")
