"""Test 2.3 -- Covariance with EW variance (0.97) and EW correlation (0.94).

Input:     testfiles/data/test2.csv
Reference: testfiles/data/testout_2.3.csv
Output:    functional_tests/outputs/testout_2.3.csv
Run:       python functional_tests/test_2_3.py
"""

import numpy as np

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    x = matrix("test2.csv")
    # Variances from lambda=0.97, correlation structure from lambda=0.94.
    sd_var = np.sqrt(np.diag(rl.ew_cov(x, 0.97)))
    cov94 = rl.ew_cov(x, 0.94)
    sd_corr = 1.0 / np.sqrt(np.diag(cov94))
    out = np.diag(sd_var) @ np.diag(sd_corr) @ cov94 @ np.diag(sd_corr) @ np.diag(sd_var)
    result = matrix_df(out)
    return run(
        "2.3",
        "Covariance with EW variance (0.97) and EW correlation (0.94)",
        result,
        "testout_2.3.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
