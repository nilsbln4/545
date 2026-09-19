"""Test 3.3 -- Higham nearest PSD, covariance.

Input:     testfiles/data/testout_1.3.csv
Reference: testfiles/data/testout_3.3.csv
Output:    functional_tests/outputs/testout_3.3.csv
Run:       python functional_tests/test_3_3.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    out = rl.higham_nearest_psd(matrix("testout_1.3.csv"))
    result = matrix_df(out)
    return run(
        "3.3",
        "Higham nearest PSD, covariance",
        result,
        "testout_3.3.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
