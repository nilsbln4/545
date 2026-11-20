"""Test 10.3 -- Max Sharpe ratio, long only.

Input:     testfiles/data/test5_3.csv, testfiles/data/test10_3_means.csv, rf = 0.04
Reference: testfiles/data/testout10_3.csv
Output:    functional_tests/outputs/testout10_3.csv
Run:       python functional_tests/test_10_3.py

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
    w = rl.max_sharpe(cov, means, 0.04, bounds=np.tile([0.0, 1.0], (5, 1)))
    return run("10.3", "Max Sharpe ratio, long only", pd.DataFrame({"W": w}), "testout10_3.csv", tol=WEIGHTS)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
