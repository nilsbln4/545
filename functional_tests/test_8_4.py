"""Test 8.4 -- ES from a fitted normal.

Input:     testfiles/data/test7_1.csv
Reference: testfiles/data/testout8_4.csv
Output:    functional_tests/outputs/testout8_4.csv
Run:       python functional_tests/test_8_4.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_1.csv")["x1"].to_numpy()
    fit = rl.fit_normal(x)
    result = pd.DataFrame({"ES Absolute": [rl.expected_shortfall(fit.error_model)],
                           "ES Diff from Mean": [rl.expected_shortfall(stats.norm(0, fit.params["sigma"]))]})
    return run("8.4", "ES from a fitted normal", result, "testout8_4.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
