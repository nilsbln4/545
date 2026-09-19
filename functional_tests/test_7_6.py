"""Test 7.6 -- Fit the same NIG by maximum likelihood.

Input:     testfiles/data/test7_5.csv
Reference: testfiles/data/testout7_6.csv
Output:    functional_tests/outputs/testout7_6.csv
Run:       python functional_tests/test_7_6.py

Numerical MLE (scipy.stats.norminvgauss.fit); maximizes the likelihood rather than matching moments.
"""

import pandas as pd

from _common import MLE, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_5.csv")["x1"].to_numpy()
    fit = rl.fit_nig_mle(x)
    p = fit.params
    result = pd.DataFrame({"mu": [p["mu"]], "alpha": [p["alpha"]], "beta": [p["beta"]], "delta": [p["delta"]]})
    return run(
        "7.6",
        "Fit the same NIG by maximum likelihood",
        result,
        "testout7_6.csv",
        tol=MLE,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
