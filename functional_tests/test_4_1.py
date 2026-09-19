"""Test 4.1 -- Cholesky of a PSD matrix (chol_psd).

Input:     testfiles/data/testout_3.1.csv
Reference: testfiles/data/testout_4.1.csv
Output:    functional_tests/outputs/testout_4.1.csv
Run:       python functional_tests/test_4_1.py
"""

from _common import TIGHT, matrix, matrix_df, run

import risk545 as rl


def main() -> bool:
    out = rl.chol_psd(matrix("testout_3.1.csv"))
    result = matrix_df(out)
    return run(
        "4.1",
        "Cholesky of a PSD matrix (chol_psd)",
        result,
        "testout_4.1.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
