# Assignment 2 -- Covariance, VaR, and Copulas

Responses: `RESPONSES_NILS.pdf`. All code is in `code/`.

## Setup

From the repo root, in a Python environment with numpy, scipy, pandas and
matplotlib (the repo's `.venv` has them; or `pip install -e ".[dev]"`):

```
cd Assignments/Assignment2/code
python run_all.py
```

`run_all.py` runs every script below in order and reproduces every number and
figure in the write-up. Each script can also be run on its own. The scripts use
the course package `risk545/` at the repo root (they add it to the path
themselves, so no install is needed). The data is `problem1.csv` ... `problem5.csv`
in the folder above `code/`.

## Scripts, by problem

| Problem | Script | What it produces |
|:--|:--|:--|
| 1 | `problem1_counts.py` | Joint-observation table and complete-row count (Predict) |
| 1 | `problem1_fit.py` | Parts d-i: both correlation matrices, eigenvalues, Cholesky, tracking variance, both repairs |
| 2 | `problem2_predict.py --recent-days 30` | Returns plot, moments (c), n_eff / half life and weight shares (b) |
| 2 | `problem2_fit.py --recent-days 30` | Five VaRs and fitted t (d), regime SDs (f), standard errors and gap (g) |
| 3 | `problem3_predict.py` | P&L plot, moments (a), loss counts (b) |
| 3 | `problem3_fit.py` | Historical and Normal VaR / ES at 5% and 1% (c-e), subadditivity (f) |
| 4 | `problem4_predict.py` | Moments and sensitivity (a), rank-transform plots and joint tail counts (b) |
| 4 | `problem4_fit.py` | Margins by AICc (c), copulas (d), portfolio risk (e), implied tail counts (f), daily likelihood gaps (i), tail dependence (j) |
| 5 | `problem5_fit.py` | Regressions (b), simulated VaR with and without the residual covariance (c), delta normal (d), comparison (e, f) |

Console output of the `*_fit.py` scripts (and `problem4_predict.py`) is also saved
to `output/`. Figures are saved to `figures/`.

## Conventions

Where a calculation has more than one defensible convention, the scripts follow
the instructor's Julia code in `library/` and the weekly homework files.

- **Standard deviation and variance** divide by n - 1 (Julia's `std` / `var`).
- **Skewness and excess kurtosis** are the population-style estimators of
  StatsBase (scipy's `bias=True`).
- **Exponentially weighted volatility** uses `ew_cov`, a port of `ewCovar`, which
  removes the weighted mean. Problem 2 also prints the zero-mean variant as a
  footnote.
- **Regression residual variance (Problem 5)** divides by T - 2 (observations minus
  fitted coefficients), as in the Week 9 code. Alphas are reported but left out of
  the simulation, since expected returns are assumed zero.
- **Pseudo-observations for the Problem 4 plots** are rank / (n + 1). The copula fits
  use the uniforms from the fitted margins, with R from Kendall's tau
  (rho = sin(pi * tau / 2)).
- **Copula criteria:** AICc and BIC count one parameter (nu) for the t copula and
  none for the Gaussian, as in the course's test 13.7.
- **Monte Carlo:** all simulations use fixed seeds, so results are reproducible. The
  Gaussian and t copula simulations share their underlying normals. Simulation
  noise is reported alongside the results (10 seeds).
- **VaR and ES** are reported as positive numbers for a loss. A negative VaR
  (Problem 3, single bonds at 5%) means the 5% quantile is itself a gain.

## Package

The new functions this assignment needed live in `risk545/`: `risk_stats.py`
(VaR, ES), `multivariate_t.py` (Kendall correlation, profile for nu) and
`copula.py` (Gaussian and t copulas). They are checked against the course's
reference test outputs by `tests/test_risk_and_copula.py`.
