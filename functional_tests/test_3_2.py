"""Test 3.2 -- near_psd, correlation.

Input:     testfiles/data/testout_1.4.csv
Reference: testfiles/data/testout_3.2.csv
Output:    functional_tests/outputs/testout_3.2.csv
Run:       python functional_tests/test_3_2.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    out = rl.near_psd(matrix("testout_1.4.csv"))
    result = matrix_df(out)
    return run(
        "3.2",
        "near_psd, correlation",
        result,
        "testout_3.2.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
