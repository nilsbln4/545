"""Test 13.1 -- Correlation from Kendall's tau, repaired to positive definite.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_1.csv
Output:    functional_tests/outputs/testout13_1.csv
Run:       python functional_tests/test_13_1.py

rho = sin(pi tau / 2).
"""

import numpy as np
import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    X = load("test13_returns.csv").to_numpy(dtype=float)
    R = rl.kendall_correlation(X)
    result = pd.DataFrame(R, columns=load("testout13_1.csv").columns)
    return run("13.1", "Kendall correlation, repaired", result, "testout13_1.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
