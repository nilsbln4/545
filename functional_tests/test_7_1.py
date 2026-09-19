"""Test 7.1 -- Fit a Normal distribution.

Input:     testfiles/data/test7_1.csv
Reference: testfiles/data/testout7_1.csv
Output:    functional_tests/outputs/testout7_1.csv
Run:       python functional_tests/test_7_1.py
"""

import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_1.csv")["x1"].to_numpy()
    fit = rl.fit_normal(x)
    p = fit.params
    result = pd.DataFrame({"mu": [p["mu"]], "sigma": [p["sigma"]]})
    return run(
        "7.1",
        "Fit a Normal distribution",
        result,
        "testout7_1.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
