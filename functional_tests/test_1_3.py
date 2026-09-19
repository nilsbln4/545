"""Test 1.3 -- Covariance, missing data, pairwise.

Input:     testfiles/data/test1.csv
Reference: testfiles/data/testout_1.3.csv
Output:    functional_tests/outputs/testout_1.3.csv
Run:       python functional_tests/test_1_3.py
"""

from _common import TIGHT, load, matrix_df, run

import risk545 as rl


def main() -> bool:
    x = load("test1.csv")
    out = rl.missing_cov(x, skip_miss=False, method="cov")
    result = matrix_df(out)
    return run(
        "1.3",
        "Covariance, missing data, pairwise",
        result,
        "testout_1.3.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
