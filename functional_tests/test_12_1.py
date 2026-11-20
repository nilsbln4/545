"""Test 12.1 -- European options, GBSM with Greeks.

Input:     testfiles/data/test12_1.csv
Reference: testfiles/data/testout12_1.csv
Output:    functional_tests/outputs/testout12_1.csv
Run:       python functional_tests/test_12_1.py

T = DaysToMaturity / DayPerYear, b = RiskFreeRate - DividendRate. Rho is gbsm.jl's rho
(rf moved with b fixed) and theta its dV/dt.
"""

import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    opts = load("test12_1.csv").dropna(subset=["ID"])
    rows = []
    for _, o in opts.iterrows():
        g = rl.gbsm(o["Option Type"] == "Call", o.Underlying, o.Strike, o.DaysToMaturity / o.DayPerYear,
                    o.RiskFreeRate, o.RiskFreeRate - o.DividendRate, o.ImpliedVol, include_greeks=True)
        rows.append({"ID": int(o.ID), "Value": g.value, "Delta": g.delta, "Gamma": g.gamma, "Vega": g.vega,
                     "Rho": g.rho, "Theta": g.theta})
    return run("12.1", "European options, GBSM with Greeks", pd.DataFrame(rows), "testout12_1.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
