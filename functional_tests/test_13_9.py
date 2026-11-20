"""Test 13.9 -- Portfolio VaR and ES under the t copula.

Input:     testfiles/data/test13_returns.csv, test13_portfolio.csv
Reference: testfiles/data/testout13_9.csv
Output:    functional_tests/outputs/testout13_9.csv
Run:       python functional_tests/test_13_9.py

100,000 t copula draws (seed 139) with generalized t margins. Monte Carlo, so checked within
sampling noise.
"""

import numpy as np

from _common import MONTE_CARLO_RISK, load, run, scenario_frame

import risk545 as rl


def main() -> bool:
    X = load("test13_returns.csv").to_numpy(dtype=float)
    port = load("test13_portfolio.csv")
    fits = [rl.fit_general_t(X[:, j]) for j in range(X.shape[1])]
    R, nu, _ = rl.fit_t_copula(np.column_stack([f.u for f in fits]))

    u = rl.simulate_t_copula(R, nu, 100_000, seed=139)
    sim = np.column_stack([fits[j].eval(u[:, j]) for j in range(len(fits))])

    out = rl.agg_risk(scenario_frame(port["Stock"], port["currentValue"].to_numpy(dtype=float), sim), ["Stock"])
    result = out[["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]]
    return run("13.9", "t copula portfolio VaR / ES", result, "testout13_9.csv", tol=MONTE_CARLO_RISK)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
