"""Test 2.2 -- EW correlation, lambda=0.94.

Input:     testfiles/data/test2.csv
Reference: testfiles/data/testout_2.2.csv
Output:    functional_tests/outputs/testout_2.2.csv
Run:       python functional_tests/test_2_2.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    x = matrix("test2.csv")
    out = rl.ew_corr(x, 0.94)
    result = matrix_df(out)
    return run(
        "2.2",
        "EW correlation, lambda=0.94",
        result,
        "testout_2.2.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
