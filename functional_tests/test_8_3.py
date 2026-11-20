"""Test 8.3 -- VaR from simulation, compared to the 8.2 values.

Input:     testfiles/data/test7_2.csv
Reference: testfiles/data/testout8_2.csv (Tests.xlsx: compare to 8.2)
Output:    functional_tests/outputs/testout8_3.csv
Run:       python functional_tests/test_8_3.py

Simulates 200,000 draws from the fitted t (seed 8). The reference testout8_3.csv is one
draw of Julia's RNG, so the result is checked against the analytic 8.2 values instead,
within Monte Carlo noise.
"""

import numpy as np
import pandas as pd
from scipy import stats

from _common import MONTE_CARLO, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    sim = fit.eval(np.random.default_rng(8).random(200_000))
    result = pd.DataFrame({"VaR Absolute": [rl.value_at_risk(sim)],
                           "VaR Diff from Mean": [rl.value_at_risk(sim - sim.mean())]})
    return run("8.3", "VaR from simulation (vs 8.2)", result, "testout8_3.csv", tol=MONTE_CARLO,
               reference=load("testout8_2.csv"))


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
