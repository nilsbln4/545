"""Test 5.1 -- Normal simulation, PD input, 100,000 draws.

Input:     testfiles/data/test5_1.csv
Reference: testfiles/data/testout_5.1.csv
Output:    functional_tests/outputs/testout_5.1.csv
Run:       python functional_tests/test_5_1.py

The simulated covariance is checked against the input covariance it was drawn from (per Tests.xlsx),
not against the reference file: that file is one particular Julia-RNG realization, so an exact match
is not the right bar for a Monte Carlo result.
"""

import numpy as np

from _common import MONTE_CARLO, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    cov = matrix("test5_1.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5001)
    out = np.cov(sim, rowvar=False)
    result = matrix_df(out)
    return run(
        "5.1",
        "Normal simulation, PD input, 100,000 draws",
        result,
        "testout_5.1.csv",
        tol=MONTE_CARLO,
        reference=cov,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
