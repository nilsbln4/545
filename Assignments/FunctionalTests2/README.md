# Functional Tests 2

The course package `risk545/` (at the repo root) now passes every functional test
in `testfiles/Tests.xlsx`, 1.1 through 13.9: 50 of 50. Functional Tests 1 covered
1.1 - 7.6; this submission adds tests 8 - 13.

## Run

From the repo root (needs numpy, scipy, pandas; the repo's `.venv` has them, or
`pip install -e ".[dev]"`):

```
python functional_tests/run_all.py     # one PASS/FAIL line per test, 50/50
pytest tests -v                        # the same cases (and extra checks) as pytest
```

Each test is also a standalone script, e.g. `python functional_tests/test_11_2.py`.
Every script writes its result to `functional_tests/outputs/` under the reference's
file name, so it can be diffed against `testfiles/data/`.

## What was added for tests 8 - 13

| Tests | Topic | Module | Function(s) |
|:--|:--|:--|:--|
| 8.1 - 8.6 | VaR and ES from a fitted normal / t and from simulation | `risk545/risk_stats.py` | `value_at_risk`, `expected_shortfall` |
| 9.1 | Portfolio VaR / ES on two levels, Gaussian copula | `risk545/copula.py`, `risk_stats.py` | `simulate_gaussian_copula`, `agg_risk` |
| 10.1 - 10.2 | Risk parity, with and without a risk budget | `risk545/optimizers.py` | `risk_parity`, `component_sd` |
| 10.3 - 10.4 | Max Sharpe ratio, long only and bounded | `risk545/optimizers.py` | `max_sharpe` |
| 11.1 - 11.2 | Ex-post return and volatility attribution (by asset, to factors) | `risk545/attribution.py` | `expost_factor` |
| 12.1 | European options: GBSM and Greeks | `risk545/options.py` | `gbsm` |
| 12.2 | American options, continuous dividends, finite-difference Greeks | `risk545/options.py` | `bt_american`, `bt_american_greeks` |
| 12.3 | American options with discrete dividends (non-recombining tree) | `risk545/options.py` | `bt_american_div` |
| 13.1 - 13.4 | Kendall correlation; multivariate t fit | `risk545/multivariate_t.py` | `kendall_correlation`, `fit_multivariate_t` |
| 13.5 - 13.9 | Gaussian and t copulas: fits, AICc / BIC, tail dependence, portfolio risk | `risk545/copula.py` | `fit_gaussian_copula`, `fit_t_copula`, `copula_aicc`, `copula_bic`, `tail_dependence_t`, `simulate_t_copula` |

Each function is a port of the instructor's Julia code in `library/`
(`RiskStats.jl`, `copula.jl`, `multivariate_t.jl`, `optimizers.jl`,
`expost_factor.jl`, `gbsm.jl`, `bt_american.jl`) and documents any place where the
Python route differs from it.

## Tolerances and conventions

Summarized here; the full table is in `functional_tests/README.md`.

- **Deterministic tests** (8.1, 8.4, 11, 12, 13.1 - 13.8) match the reference to
  `rtol = 1e-6`; 11 and 12 agree to about 1e-9.
- **Optimizer-based tests** (8.2, 8.5, 10) match to `rtol = 1e-4` (scipy here,
  Ipopt in the reference); in practice to 6+ significant figures. 10.3 / 10.4 add
  `atol = 1e-6` because Ipopt reports ~1e-8 where the optimal weight is 0.
- **Monte Carlo tests** (8.3, 8.6, 9.1, 13.9) cannot reproduce Julia's random draws.
  8.3 and 8.6 are compared with the analytic 8.2 / 8.5 values, as `Tests.xlsx`
  specifies; 9.1 and 13.9 are compared with the reference within sampling noise
  (100,000 draws, 5%).
- **Risk parity** is solved as min ½ w'Σw − Σ bᵢ log wᵢ (Spinu 2013), a convex
  problem whose optimum has component SDs proportional to the budget, the same
  target as the reference's `riskParity`.
- **12.2 Greeks** follow the reference's finite differences: FiniteDiff.jl's default
  central step for delta, vega, rho (b held fixed) and theta (dV/dT, positive), and
  a 1.5 bump for gamma, because the tree value is piecewise linear in S.
- **10.3 / 10.4 input:** `Tests.xlsx` lists `test5_2.csv`, but the reference reads
  `test5_3.csv`; the scripts follow the reference.
