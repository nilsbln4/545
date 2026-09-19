"""Test 5.5 -- PCA simulation, 99% variance explained, 100,000 draws.

Input:     testfiles/data/test5_2.csv
Reference: testfiles/data/testout_5.5.csv
Output:    functional_tests/outputs/testout_5.5.csv
Run:       python functional_tests/test_5_5.py

The simulated covariance is checked against the input covariance it was drawn from (per Tests.xlsx),
not against the reference file: that file is one particular Julia-RNG realization, so an exact match
is not the right bar for a Monte Carlo result.
"""

import numpy as np

from _common import MONTE_CARLO, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    cov = matrix("test5_2.csv")
    sim = rl.simulate_pca(cov, 100_000, pct_exp=0.99, seed=5005)
    out = np.cov(sim, rowvar=False)
    result = matrix_df(out)
    return run(
        "5.5",
        "PCA simulation, 99% variance explained, 100,000 draws",
        result,
        "testout_5.5.csv",
        tol=MONTE_CARLO,
        reference=cov,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
