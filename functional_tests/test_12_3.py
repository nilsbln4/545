"""Test 12.3 -- American options with discrete dividends.

Input:     testfiles/data/test12_3.csv
Reference: testfiles/data/testout12_3.csv
Output:    functional_tests/outputs/testout12_3.csv
Run:       python functional_tests/test_12_3.py

Non-recombining tree. As in the reference, N = 2 x DaysToMaturity and the dividend dates
(in days) are doubled to become step indices.
"""

import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    opts = load("test12_3.csv").dropna(subset=["ID"])
    rows = []
    for _, o in opts.iterrows():
        steps = [2 * int(s) for s in str(o.DividendDates).split(",")]
        amts = [float(a) for a in str(o.DividendAmts).split(",")]
        v = rl.bt_american_div(o["Option Type"] == "Call", o.Underlying, o.Strike,
                               o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate, amts, steps,
                               o.ImpliedVol, 2 * int(o.DaysToMaturity))
        rows.append({"ID": int(o.ID), "Value": v})
    return run("12.3", "American options, discrete dividends", pd.DataFrame(rows), "testout12_3.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
