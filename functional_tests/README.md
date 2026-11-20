# Functional tests 1.1 - 13.9

One script per test, named for the test (`test_1_1.py` ... `test_13_9.py`), each
using the `risk545` package at the repo root. All 50 tests in
`testfiles/Tests.xlsx` are covered.

## Run

From the repo root:

```
python functional_tests/run_all.py          # all 50, one PASS/FAIL line each (about 40 s)
python functional_tests/test_12_2.py        # or any single test
```

Needs `numpy`, `scipy` and `pandas`; no `pip install` of the package itself is
required (the scripts add the repo root to the path).

## What each test produces

Every script writes its computed result to `outputs/` under the **same file
name as the reference**, e.g. `outputs/testout12_2.csv`, and compares it to
`testfiles/data/testout12_2.csv`. The two files can be opened side by side or
diffed. The committed `outputs/` folder is the result of the last run.

Each script's docstring lists its input file, its reference file, its output
file, and any convention the test depends on.

## Where the code lives

| Tests | Topic | `risk545` module |
|:--|:--|:--|
| 1 - 4 | Covariance with missing data, EW covariance, PSD repairs, Cholesky | `covariance.py`, `psd.py` |
| 5 | Normal and PCA simulation | `simulate.py` |
| 6 | Returns | `returns.py` |
| 7 | Normal, t, t regression, AICc, NIG fits | `fitted_model.py` |
| 8 | VaR and ES (parametric and simulated) | `risk_stats.py` |
| 9.1, 13 | Multivariate t, Gaussian and t copulas, portfolio risk | `multivariate_t.py`, `copula.py`, `risk_stats.py` |
| 10 | Risk parity, max Sharpe | `optimizers.py` |
| 11 | Ex-post return and volatility attribution | `attribution.py` |
| 12 | GBSM and Greeks, binomial trees, discrete dividends | `options.py` |

## How "pass" is decided

| Tests | Comparison |
|:--|:--|
| 1-4, 6, 7.1, 7.5, 8.1, 8.4, 11, 12, 13.1-13.8 | Match the reference to `rtol=1e-6` (deterministic). In practice 11 and 12 agree to ~1e-9. |
| 7.2 - 7.4, 7.6, 8.2, 8.5, 10.1, 10.2 | Match to `rtol=1e-4`. These rest on optimizers (scipy here, Ipopt in the Julia reference); in practice they agree to 6+ significant figures. |
| 10.3, 10.4 | `rtol=1e-4` plus `atol=1e-6`: where the optimal weight is 0, Ipopt reports ~1e-8. |
| 5.1 - 5.5 | Monte Carlo (100,000 draws), so the simulated covariance is compared to the covariance it was drawn from, within sampling noise. It is not compared to `testout_5.*.csv`, which is one particular Julia-RNG realization. |
| 8.3, 8.6 | Simulated VaR / ES compared to the analytic 8.2 / 8.5 values, as `Tests.xlsx` says ("compare to 8.2 values"), within Monte Carlo noise (200,000 draws). |
| 9.1, 13.9 | Simulated portfolio VaR / ES from 100,000 copula draws, compared to the reference within 5% (sampling noise; a different RNG from Julia's). |

For 5.3 and 5.4 the input matrix is deliberately not positive semi-definite, so
no valid simulation can reproduce it. The target there is the *repaired* matrix
(`near_psd` for 5.3, `higham_nearest_psd` for 5.4).

## Notes on specific tests

- **10.1 / 10.2 (risk parity).** The reference (`riskParity` in
  `library/optimizers.jl`) minimizes the spread of component SD / budget with
  Ipopt. `risk_parity` reaches the same point through the convex problem
  min ½ w'Σw − Σ bᵢ log wᵢ (Spinu 2013), whose optimum has component SDs exactly
  proportional to the budget; it matches the reference to ~1e-9.
- **10.3 / 10.4 (max Sharpe).** `Tests.xlsx` lists `test5_2.csv` as the input, but
  the reference (`testfiles/test_setup.jl`) reads `test5_3.csv`; the scripts follow
  the reference.
- **11 (attribution).** Port of `library/expost_factor.jl`: weights drift with
  returns, Carino linking for multi-period return attribution, and regression-based
  volatility attribution. 11.1 passes the asset returns as the factors with an
  identity beta matrix and drops the Alpha column, as the reference does.
- **12.2 (American Greeks).** Finite differences on the N = 500 tree, as in the
  reference: central differences with FiniteDiff.jl's default step for delta,
  vega, rho and theta, and a 1.5 bump for gamma. Rho moves rf with the cost of
  carry b held fixed, and theta is dV/dT (positive), which is the reference's sign;
  `gbsm`'s theta in 12.1 is dV/dt (negative).
- **12.3.** N = 2 × days to maturity, with the dividend days doubled into step
  indices, as in the reference.

The same cases are also checked by pytest: `tests/test_functional.py` (1-7),
`tests/test_risk_and_copula.py` (8, 9.1, 13) and `tests/test_options.py`
(10, 11, 12).
