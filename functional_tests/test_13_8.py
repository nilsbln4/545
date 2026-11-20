"""Test 13.8 -- Lower tail dependence of each pair under the fitted t copula.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_8.csv
Output:    functional_tests/outputs/testout13_8.csv
Run:       python functional_tests/test_13_8.py
"""

import numpy as np
import pandas as pd

from _common import TIGHT, load, run

import risk545 as rl


def uniforms() -> np.ndarray:
    """Generalized t margins fitted to each column, mapped to uniforms by their CDFs."""
    X = load("test13_returns.csv").to_numpy(dtype=float)
    return np.column_stack([rl.fit_general_t(X[:, j]).u for j in range(X.shape[1])])


def main() -> bool:
    R, nu, _ = rl.fit_t_copula(uniforms())
    n = R.shape[0]
    rows = [{"I": i + 1, "J": j + 1, "Rho": R[i, j], "Lambda": rl.tail_dependence_t(R[i, j], nu)}
            for i in range(n) for j in range(i + 1, n)]
    return run("13.8", "t copula lower tail dependence", pd.DataFrame(rows), "testout13_8.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
