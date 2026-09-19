"""Test 6.2 -- Log returns.

Input:     testfiles/data/test6.csv
Reference: testfiles/data/testout6_2.csv
Output:    functional_tests/outputs/testout6_2.csv
Run:       python functional_tests/test_6_2.py
"""

from _common import TIGHT, load, run

import risk545 as rl


def main() -> bool:
    prices = load("test6.csv")
    out = rl.return_calculate(prices, method="LOG", date_column="Date")
    result = out
    return run(
        "6.2",
        "Log returns",
        result,
        "testout6_2.csv",
        tol=TIGHT,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
