"""Test 7.4 -- AICc of the fitted t (test 7.2's fit).

Input:     testfiles/data/test7_2.csv
Reference: testfiles/data/testout7_4.csv
Output:    functional_tests/outputs/testout7_4.csv
Run:       python functional_tests/test_7_4.py
"""

import pandas as pd

from _common import MLE, load, run

import risk545 as rl


def main() -> bool:
    x = load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    result = pd.DataFrame({"AICC": [rl.aicc(fit, x)]})
    return run(
        "7.4",
        "AICc of the fitted t (test 7.2's fit)",
        result,
        "testout7_4.csv",
        tol=MLE,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
