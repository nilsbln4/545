"""Test 10.4 -- Max Sharpe ratio, 0.1 <= w <= 0.5.

Input:     testfiles/data/test5_3.csv, testfiles/data/test10_3_means.csv, rf = 0.04
Reference: testfiles/data/testout10_4.csv
Output:    functional_tests/outputs/testout10_4.csv
Run:       python functional_tests/test_10_4.py

The reference (testfiles/test_setup.jl) reads test5_3.csv, although Tests.xlsx lists
test5_2.csv. The long-only optimum puts ~1e-8 on three assets, so an absolute tolerance.
"""

import numpy as np
import pandas as pd

from _common import WEIGHTS, load, matrix, run

import risk545 as rl


def main() -> bool:
    cov = matrix("test5_3.csv")
    means = load("test10_3_means.csv")["Mean"].to_numpy()
    w = rl.max_sharpe(cov, means, 0.04, bounds=np.tile([0.1, 0.5], (5, 1)))
    return run("10.4", "Max Sharpe ratio, 0.1 <= w <= 0.5", pd.DataFrame({"W": w}), "testout10_4.csv", tol=WEIGHTS)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
