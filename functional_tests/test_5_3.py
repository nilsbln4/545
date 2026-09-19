"""Test 5.3 -- Normal simulation, non-PSD input, near_psd fix.

Input:     testfiles/data/test5_3.csv
Reference: testfiles/data/testout_5.3.csv
Output:    functional_tests/outputs/testout_5.3.csv
Run:       python functional_tests/test_5_3.py

Checked against the repaired (near_psd) matrix, since the raw input is not a valid covariance matrix.
"""

import numpy as np

from _common import MONTE_CARLO, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    cov = matrix("test5_3.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5003, fix_method=rl.near_psd)
    out = np.cov(sim, rowvar=False)
    # The input is deliberately non-PSD, so no valid simulation can reproduce it;
    # the correct target is the repaired matrix.
    target = rl.near_psd(cov)
    result = matrix_df(out)
    return run(
        "5.3",
        "Normal simulation, non-PSD input, near_psd fix",
        result,
        "testout_5.3.csv",
        tol=MONTE_CARLO,
        reference=target,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
