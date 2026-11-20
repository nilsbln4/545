"""Test 13.3 -- Multivariate t fit, the scale matrix S.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_3.csv
Output:    functional_tests/outputs/testout13_3.csv
Run:       python functional_tests/test_13_3.py

S is the scale matrix; the covariance is nu / (nu - 2) x S.
"""

import numpy as np
import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    _, S, _, _ = rl.fit_multivariate_t(load("test13_returns.csv").to_numpy(dtype=float))
    result = pd.DataFrame(S, columns=load("testout13_3.csv").columns)
    return run("13.3", "Multivariate t, scale matrix", result, "testout13_3.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
