"""Test 10.1 -- Risk parity, normal assumption.

Input:     testfiles/data/test5_2.csv
Reference: testfiles/data/testout10_1.csv
Output:    functional_tests/outputs/testout10_1.csv
Run:       python functional_tests/test_10_1.py

Solved as the convex problem min 1/2 w'Sw - sum b_i log w_i (Spinu 2013), whose optimum has
component SDs proportional to the budget, the target of riskParity in library/optimizers.jl.
"""

import pandas as pd

from _common import MLE, matrix, run

import risk545 as rl


def main() -> bool:
    cov = matrix("test5_2.csv")
    w = rl.risk_parity(cov, risk_budget=None)
    return run("10.1", "Risk parity, normal assumption", pd.DataFrame({"W": w}), "testout10_1.csv", tol=MLE)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
