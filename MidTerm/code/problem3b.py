"""Midterm Problem 3(b) -- Gaussian copula vs Student's t copula.

    python problem3b.py

Uses the per-stock distributions you set in settings.py (DISTRIBUTION_CHOICE).
Prints the log likelihood, nu, AICc and BIC of each copula and the correlation
matrix. Output is also saved to output/problem3b.txt.
"""

import pandas as pd

from common import Report, fit_margin, load_q3_returns, require_choices, rl

choices = require_choices()
rep = Report()
returns = load_q3_returns()
stocks = list(returns.columns)
n = len(returns)

# ---------------------------------------------------------------------------
# Problem 3(b): uniforms from the chosen margins, then both copulas
# ---------------------------------------------------------------------------
rep.section("Problem 3(b): margins used")
rep.say(pd.Series(choices).rename("distribution").to_string())

fits = {s: fit_margin(returns[s].to_numpy(), choices[s]) for s in stocks}
U = pd.DataFrame({s: fits[s].u for s in stocks}).to_numpy()   # each fitted margin's CDF applied to its data

R, ll_gaussian = rl.fit_gaussian_copula(U)         # R from Kendall's tau (rho = sin(pi * tau / 2))
_, nu, ll_t = rl.fit_t_copula(U)

# k counts the one parameter that differs between the copulas, nu; R is held fixed
# from tau in both (the same convention as the course's test 13.7).
k_gaussian, k_t = 0, 1
table = pd.DataFrame({
    "log likelihood": [ll_gaussian, ll_t],
    "nu-hat": [float("nan"), nu],
    "k": [k_gaussian, k_t],
    "AICc": [rl.copula_aicc(ll_gaussian, k_gaussian, n), rl.copula_aicc(ll_t, k_t, n)],
    "BIC": [rl.copula_bic(ll_gaussian, k_gaussian, n), rl.copula_bic(ll_t, k_t, n)],
}, index=["Gaussian", "t"])

rep.section(f"Problem 3(b): copula fit ({n} days)")
rep.say(table.to_string(float_format=lambda v: f"{v: .4f}", na_rep="--"))
rep.say(f"\nt minus Gaussian:  log likelihood {ll_t - ll_gaussian:+.4f}   "
        f"AICc {table.loc['t', 'AICc'] - table.loc['Gaussian', 'AICc']:+.4f}   "
        f"BIC {table.loc['t', 'BIC'] - table.loc['Gaussian', 'BIC']:+.4f}   (negative = t lower)")

rep.section("Problem 3(b): correlation matrix (Kendall's tau)")
rep.say(pd.DataFrame(R, index=stocks, columns=stocks).to_string(float_format=lambda v: f"{v: .4f}"))

rep.save("problem3b.txt")
