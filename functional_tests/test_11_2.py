"""Test 11.2 -- Ex-post return and volatility attribution to factors.

Input:     testfiles/data/test11_2_factor_returns.csv, test11_2_stock_returns.csv, test11_2_beta.csv, test11_2_weights.csv
Reference: testfiles/data/testout11_2.csv
Output:    functional_tests/outputs/testout11_2.csv
Run:       python functional_tests/test_11_2.py
"""

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    factors = load("test11_2_factor_returns.csv")
    stocks = load("test11_2_stock_returns.csv")
    beta = load("test11_2_beta.csv").drop(columns="Stock").to_numpy(dtype=float)
    w = load("test11_2_weights.csv")["W"].to_numpy()
    result = rl.expost_factor(w, stocks, factors, beta)
    return run("11.2", "Ex-post attribution to factors", result, "testout11_2.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
