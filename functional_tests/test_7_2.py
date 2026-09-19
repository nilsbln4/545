"""Test 7.2 -- Fit a Student's t distribution.

Input:     testfiles/data/test7_2.csv
Reference: testfiles/data/testout7_2.csv
Output:    functional_tests/outputs/testout7_2.csv
Run:       python functional_tests/test_7_2.py
"""

import pandas as pd

from _common import MLE, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    p = fit.params
    result = pd.DataFrame({"mu": [p["mu"]], "sigma": [p["sigma"]], "nu": [p["nu"]]})
    return run(
        "7.2",
        "Fit a Student's t distribution",
        result,
        "testout7_2.csv",
        tol=MLE,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
