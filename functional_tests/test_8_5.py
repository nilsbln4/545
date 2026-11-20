"""Test 8.5 -- ES from a fitted Student's t.

Input:     testfiles/data/test7_2.csv
Reference: testfiles/data/testout8_5.csv
Output:    functional_tests/outputs/testout8_5.csv
Run:       python functional_tests/test_8_5.py
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
    result = pd.DataFrame({"ES Absolute": [rl.expected_shortfall(fit.error_model)],
                           "ES Diff from Mean": [rl.expected_shortfall(stats.t(p["nu"], 0, p["sigma"]))]})
    return run("8.5", "ES from a fitted Student's t", result, "testout8_5.csv", tol=MLE)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
