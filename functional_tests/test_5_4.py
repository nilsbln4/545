"""Test 5.4 -- Normal simulation, non-PSD input, Higham fix.

Input:     testfiles/data/test5_3.csv
Reference: testfiles/data/testout_5.4.csv
Output:    functional_tests/outputs/testout_5.4.csv
Run:       python functional_tests/test_5_4.py

Checked against the repaired (Higham) matrix, since the raw input is not a valid covariance matrix.
"""

import numpy as np

from _common import MONTE_CARLO, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    cov = matrix("test5_3.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5004, fix_method=rl.higham_nearest_psd)
    out = np.cov(sim, rowvar=False)
    # The input is deliberately non-PSD, so no valid simulation can reproduce it;
    # the correct target is the repaired matrix.
    target = rl.higham_nearest_psd(cov)
    result = matrix_df(out)
    return run(
        "5.4",
        "Normal simulation, non-PSD input, Higham fix",
        result,
        "testout_5.4.csv",
        tol=MONTE_CARLO,
        reference=target,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
