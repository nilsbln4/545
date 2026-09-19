"""Test 7.5 -- Fit a NIG by the method of moments.

Input:     testfiles/data/test7_5.csv
Reference: testfiles/data/testout7_5.csv
Output:    functional_tests/outputs/testout7_5.csv
Run:       python functional_tests/test_7_5.py

Closed form: matches the sample's mean, variance, skewness and excess kurtosis exactly.
"""

import pandas as pd

from _common import MLE, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_5.csv")["x1"].to_numpy()
    fit = rl.fit_nig_moments(x)
    p = fit.params
    result = pd.DataFrame({"mu": [p["mu"]], "alpha": [p["alpha"]], "beta": [p["beta"]], "delta": [p["delta"]]})
    return run(
        "7.5",
        "Fit a NIG by the method of moments",
        result,
        "testout7_5.csv",
        tol=MLE,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
