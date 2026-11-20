"""Test 8.1 -- VaR from a fitted normal.

Input:     testfiles/data/test7_1.csv
Reference: testfiles/data/testout8_1.csv
Output:    functional_tests/outputs/testout8_1.csv
Run:       python functional_tests/test_8_1.py

VaR Absolute is the loss quantile of the fitted distribution; VaR Diff from Mean is
the same quantile measured from the mean (a zero-mean copy of the fit), alpha = 5%.
"""

import numpy as np
import pandas as pd
from scipy import stats

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_1.csv")["x1"].to_numpy()
    fit = rl.fit_normal(x)
    result = pd.DataFrame({"VaR Absolute": [rl.value_at_risk(fit.error_model)],
                           "VaR Diff from Mean": [rl.value_at_risk(stats.norm(0, fit.params["sigma"]))]})
    return run("8.1", "VaR from a fitted normal", result, "testout8_1.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
