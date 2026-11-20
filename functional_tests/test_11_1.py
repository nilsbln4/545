"""Test 11.1 -- Ex-post return and volatility attribution by asset.

Input:     testfiles/data/test11_1_returns.csv, testfiles/data/test11_1_weights.csv
Reference: testfiles/data/testout11_1.csv
Output:    functional_tests/outputs/testout11_1.csv
Run:       python functional_tests/test_11_1.py

As in the reference: the asset returns are passed as the 'factors' with an identity beta
matrix, and the residual (Alpha) column is dropped.
"""

import numpy as np

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    returns = load("test11_1_returns.csv")
    w = load("test11_1_weights.csv")["W"].to_numpy()
    result = rl.expost_factor(w, returns, returns, np.eye(returns.shape[1])).drop(columns="Alpha")
    return run("11.1", "Ex-post attribution by asset", result, "testout11_1.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
