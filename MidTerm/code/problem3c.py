"""Midterm Problem 3(c) -- copula simulation, VaR and ES at 95% in dollars.

    python problem3c.py

Uses the margins (DISTRIBUTION_CHOICE) and copula (COPULA_CHOICE) set in
settings.py, N_SIM simulated days, and each position valued at the last price.
Output is also saved to output/problem3c.txt and output/problem3c_table.csv.
"""

import numpy as np
import pandas as pd

import settings
from common import (OUTPUT_DIR, Report, fit_margin, load_q3_positions, load_q3_returns,
                    margin_quantile, require_choices, rl)

choices = require_choices(need_copula=True)
rep = Report()
ALPHA = 0.05   # 95% level

returns = load_q3_returns()
stocks = list(returns.columns)
positions = load_q3_positions().set_index("Stock").loc[stocks]

# ---------------------------------------------------------------------------
# Problem 3(c): fit margins and the chosen copula, then simulate
# ---------------------------------------------------------------------------
fits = {s: fit_margin(returns[s].to_numpy(), choices[s]) for s in stocks}
U = pd.DataFrame({s: fits[s].u for s in stocks}).to_numpy()

if settings.COPULA_CHOICE == "gaussian":
    R, _ = rl.fit_gaussian_copula(U)
    u_sim = rl.simulate_gaussian_copula(R, settings.N_SIM, seed=settings.SEED)
    copula_label = "Gaussian copula"
else:
    R, nu, _ = rl.fit_t_copula(U)
    u_sim = rl.simulate_t_copula(R, nu, settings.N_SIM, seed=settings.SEED)
    copula_label = f"t copula (nu = {nu:.4f})"

# Simulated arithmetic returns of each stock through its fitted margin.
sim_returns = np.column_stack([margin_quantile(fits[s], choices[s], u_sim[:, j])
                               for j, s in enumerate(stocks)])

# P&L of each position = position value x simulated return.
value = positions["Value"].to_numpy()
pnl = sim_returns * value                                   # (draws x stocks)
book = positions["Book"].to_numpy()

rep.section("Problem 3(c): setup")
rep.say(f"{copula_label}; {settings.N_SIM:,} simulated days; seed {settings.SEED}")
rep.say(pd.DataFrame({"distribution": pd.Series(choices)}).T.to_string())
rep.say("\nPositions (valued at the last price in the file):")
rep.say(positions[["Book", "Holding", "Price", "Value"]].to_string(float_format=lambda v: f"{v:,.4f}"))

# VaR and ES are positive numbers for a loss, in dollars.
rows = []
for j, s in enumerate(stocks):
    rows.append({"level": "stock", "name": s, "value": value[j],
                 "VaR 95%": rl.value_at_risk(pnl[:, j], ALPHA), "ES 95%": rl.expected_shortfall(pnl[:, j], ALPHA)})
for b in sorted(set(book)):
    p = pnl[:, book == b].sum(axis=1)                       # sum across the book's stocks within each draw
    rows.append({"level": "book", "name": f"Book {b}", "value": value[book == b].sum(),
                 "VaR 95%": rl.value_at_risk(p, ALPHA), "ES 95%": rl.expected_shortfall(p, ALPHA)})
total = pnl.sum(axis=1)
rows.append({"level": "total", "name": "Total", "value": value.sum(),
             "VaR 95%": rl.value_at_risk(total, ALPHA), "ES 95%": rl.expected_shortfall(total, ALPHA)})

table = pd.DataFrame(rows)
rep.section("Problem 3(c): VaR and ES at 95%, in dollars")
rep.say(table.to_string(index=False, formatters={"value": "{:>14,.2f}".format,
                                                  "VaR 95%": "{:>12,.2f}".format,
                                                  "ES 95%": "{:>12,.2f}".format}))

table.to_csv(OUTPUT_DIR / "problem3c_table.csv", index=False)
rep.save("problem3c.txt")
