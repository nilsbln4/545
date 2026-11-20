"""Test 8.2 -- VaR from a fitted Student's t.

Input:     testfiles/data/test7_2.csv
Reference: testfiles/data/testout8_2.csv
Output:    functional_tests/outputs/testout8_2.csv
Run:       python functional_tests/test_8_2.py

The t fit is a numerical MLE (scipy here, Ipopt in the reference), so the MLE tolerance.
"""

import numpy as np
import pandas as pd
from scipy import stats

from _common import MLE, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    p = fit.params
    result = pd.DataFrame({"VaR Absolute": [rl.value_at_risk(fit.error_model)],
                           "VaR Diff from Mean": [rl.value_at_risk(stats.t(p["nu"], 0, p["sigma"]))]})
    return run("8.2", "VaR from a fitted Student's t", result, "testout8_2.csv", tol=MLE)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
