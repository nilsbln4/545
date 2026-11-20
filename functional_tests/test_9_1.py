"""Test 9.1 -- VaR / ES on two levels from a Gaussian copula simulation.

Input:     testfiles/data/test9_1_returns.csv, testfiles/data/test9_1_portfolio.csv
Reference: testfiles/data/testout9_1.csv
Output:    functional_tests/outputs/testout9_1.csv
Run:       python functional_tests/test_9_1.py

Margins: normal for A, Student's t for B (as the portfolio file says). Spearman correlation
of the fitted uniforms, 100,000 Gaussian copula draws (seed 91). Monte Carlo, so checked
within sampling noise. Position value = Holding x Starting Price.
"""

import numpy as np
import pandas as pd

from _common import MONTE_CARLO_RISK, load, run, scenario_frame

import risk545 as rl


def main() -> bool:
    returns = load("test9_1_returns.csv")
    port = load("test9_1_portfolio.csv")
    port.columns = [c.strip().lstrip("\ufeff") for c in port.columns]
    a = rl.fit_normal(returns["A"].to_numpy())
    b = rl.fit_general_t(returns["B"].to_numpy())
    spearman = pd.DataFrame(np.column_stack([a.u, b.u])).corr(method="spearman").to_numpy()

    u = rl.simulate_gaussian_copula(spearman, 100_000, seed=91)
    sim = np.column_stack([a.eval(u[:, 0]), b.eval(u[:, 1])])
    values = (port["Holding"] * port["Starting Price"]).to_numpy(dtype=float)

    out = rl.agg_risk(scenario_frame(port["Stock"], values, sim), ["Stock"])
    result = out[["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]]
    return run("9.1", "Gaussian copula VaR / ES, two levels", result, "testout9_1.csv", tol=MONTE_CARLO_RISK)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
