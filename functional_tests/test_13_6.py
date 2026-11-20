"""Test 13.6 -- t copula, nu by profile and the log likelihood.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_6.csv
Output:    functional_tests/outputs/testout13_6.csv
Run:       python functional_tests/test_13_6.py
"""

import numpy as np
import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def uniforms() -> np.ndarray:
    """Generalized t margins fitted to each column, mapped to uniforms by their CDFs."""
    X = load("test13_returns.csv").to_numpy(dtype=float)
    return np.column_stack([rl.fit_general_t(X[:, j]).u for j in range(X.shape[1])])


def main() -> bool:
    _, nu, ll = rl.fit_t_copula(uniforms())
    return run("13.6", "t copula, nu and log likelihood", pd.DataFrame({"nu": [nu], "ll": [ll]}),
               "testout13_6.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
