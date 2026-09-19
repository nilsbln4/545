"""Test 3.4 -- Higham nearest PSD, correlation.

Input:     testfiles/data/testout_1.4.csv
Reference: testfiles/data/testout_3.4.csv
Output:    functional_tests/outputs/testout_3.4.csv
Run:       python functional_tests/test_3_4.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    out = rl.higham_nearest_psd(matrix("testout_1.4.csv"))
    result = matrix_df(out)
    return run(
        "3.4",
        "Higham nearest PSD, correlation",
        result,
        "testout_3.4.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
