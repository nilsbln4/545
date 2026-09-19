"""Test 6.1 -- Arithmetic returns.

Input:     testfiles/data/test6.csv
Reference: testfiles/data/testout6_1.csv
Output:    functional_tests/outputs/testout6_1.csv
Run:       python functional_tests/test_6_1.py
"""

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    prices = load("test6.csv")
    out = rl.return_calculate(prices, method="DISCRETE", date_column="Date")
    result = out
    return run(
        "6.1",
        "Arithmetic returns",
        result,
        "testout6_1.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
