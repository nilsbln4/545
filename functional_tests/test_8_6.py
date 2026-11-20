"""Test 8.6 -- ES from simulation, compared to the 8.5 values.

Input:     testfiles/data/test7_2.csv
Reference: testfiles/data/testout8_5.csv (Tests.xlsx: compare to 8.5)
Output:    functional_tests/outputs/testout8_6.csv
Run:       python functional_tests/test_8_6.py

Same approach as 8.3: 200,000 draws (seed 86), checked against the analytic 8.5 values.
"""

import numpy as np
import pandas as pd
from scipy import stats

from _common import MONTE_CARLO, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    sim = fit.eval(np.random.default_rng(86).random(200_000))
    result = pd.DataFrame({"ES Absolute": [rl.expected_shortfall(sim)],
                           "ES Diff from Mean": [rl.expected_shortfall(sim - sim.mean())]})
    return run("8.6", "ES from simulation (vs 8.5)", result, "testout8_6.csv", tol=MONTE_CARLO,
               reference=load("testout8_5.csv"))


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
