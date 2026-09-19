"""Test 2.1 -- EW covariance, lambda=0.97.

Input:     testfiles/data/test2.csv
Reference: testfiles/data/testout_2.1.csv
Output:    functional_tests/outputs/testout_2.1.csv
Run:       python functional_tests/test_2_1.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    x = matrix("test2.csv")
    out = rl.ew_cov(x, 0.97)
    result = matrix_df(out)
    return run(
        "2.1",
        "EW covariance, lambda=0.97",
        result,
        "testout_2.1.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
