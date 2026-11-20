"""Test 13.7 -- Choose between the copulas by AICc and BIC.

Input:     testfiles/data/test13_returns.csv
Reference: testfiles/data/testout13_7.csv
Output:    functional_tests/outputs/testout13_7.csv
Run:       python functional_tests/test_13_7.py

K counts the copula's own parameters beyond the correlation matrix: 0 for the Gaussian,
1 (nu) for the t, as the reference does.
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
    U = uniforms()
    m = U.shape[0]
    _, ll_g = rl.fit_gaussian_copula(U)
    _, _, ll_t = rl.fit_t_copula(U)
    result = pd.DataFrame({"Copula": ["Gaussian", "T"], "LL": [ll_g, ll_t], "K": [0, 1],
                           "AICC": [rl.copula_aicc(ll_g, 0, m), rl.copula_aicc(ll_t, 1, m)],
                           "BIC": [rl.copula_bic(ll_g, 0, m), rl.copula_bic(ll_t, 1, m)]})
    return run("13.7", "Copula choice by AICc and BIC", result, "testout13_7.csv", tol=TIGHT)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
