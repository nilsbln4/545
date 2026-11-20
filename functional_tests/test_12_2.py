"""Test 12.2 -- American options, continuous dividends, with Greeks.

Input:     testfiles/data/test12_1.csv
Reference: testfiles/data/testout12_2.csv
Output:    functional_tests/outputs/testout12_2.csv
Run:       python functional_tests/test_12_2.py

Binomial tree with N = 500. Greeks by finite differences on the tree, as the reference
computes them: central differences (FiniteDiff.jl's default step) for delta, vega, rho
(b held fixed) and theta (= dV/dT, the reference's sign), and a 1.5 bump for gamma.
"""

import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    opts = load("test12_1.csv").dropna(subset=["ID"])
    rows = []
    for _, o in opts.iterrows():
        g = rl.bt_american_greeks(o["Option Type"] == "Call", o.Underlying, o.Strike,
                                  o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate,
                                  o.RiskFreeRate - o.DividendRate, o.ImpliedVol, 500)
        rows.append({"ID": int(o.ID), "Value": g["value"], "Delta": g["delta"], "Gamma": g["gamma"],
                     "Vega": g["vega"], "Rho": g["rho"], "Theta": g["theta"]})
    return run("12.2", "American options with Greeks (tree)", pd.DataFrame(rows), "testout12_2.csv",
               tol=dict(rtol=1e-6, atol=1e-8))


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
