"""Test 7.3 -- Regression with t-distributed errors.

Input:     testfiles/data/test7_3.csv
Reference: testfiles/data/testout7_3.csv
Output:    functional_tests/outputs/testout7_3.csv
Run:       python functional_tests/test_7_3.py
"""

import pandas as pd

from _common import MLE, load, run

import risk545 as rl


def main() -> bool:
    df = load("test7_3.csv")
    fit = rl.fit_regression_t(df["y"].to_numpy(), df[["x1", "x2", "x3"]].to_numpy())
    p = fit.params
    # mu is fixed at 0 in the regression error model; Alpha carries the level.
    result = pd.DataFrame({"mu": [0.0], "sigma": [p["sigma"]], "nu": [p["nu"]], "Alpha": [p["Alpha"]], "B1": [p["B1"]], "B2": [p["B2"]], "B3": [p["B3"]]})
    return run(
        "7.3",
        "Regression with t-distributed errors",
        result,
        "testout7_3.csv",
        tol=MLE,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
