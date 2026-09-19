"""Test 3.1 -- near_psd, covariance.

Input:     testfiles/data/testout_1.3.csv
Reference: testfiles/data/testout_3.1.csv
Output:    functional_tests/outputs/testout_3.1.csv
Run:       python functional_tests/test_3_1.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    out = rl.near_psd(matrix("testout_1.3.csv"))
    result = matrix_df(out)
    return run(
        "3.1",
        "near_psd, covariance",
        result,
        "testout_3.1.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
