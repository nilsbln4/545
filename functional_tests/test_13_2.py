"""Test 13.2 -- Multivariate t fit, the mean vector.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_2.csv
Output:    functional_tests/outputs/testout13_2.csv
Run:       python functional_tests/test_13_2.py
"""

import numpy as np
import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    mu, _, _, _ = rl.fit_multivariate_t(load("test13_returns.csv").to_numpy(dtype=float))
    return run("13.2", "Multivariate t, mean vector", pd.DataFrame({"mu": mu}), "testout13_2.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
