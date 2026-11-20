"""Test 13.4 -- Multivariate t fit, nu by profile and the log likelihood.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_4.csv
Output:    functional_tests/outputs/testout13_4.csv
Run:       python functional_tests/test_13_4.py
"""

import numpy as np
import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    _, _, nu, ll = rl.fit_multivariate_t(load("test13_returns.csv").to_numpy(dtype=float))
    return run("13.4", "Multivariate t, nu and log likelihood", pd.DataFrame({"nu": [nu], "ll": [ll]}),
               "testout13_4.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
