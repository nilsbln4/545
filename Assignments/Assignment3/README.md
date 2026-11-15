# Assignment 3 -- Options, Option Risk, and Portfolio Construction

Responses: `RESPONSES_Nils_Viehweger_A3.pdf`. All code is in `code/`.

## Setup

From the repo root, in a Python environment with numpy, scipy, pandas, matplotlib
and statsmodels (the repo's `.venv` has them; or `pip install -e ".[dev]"`):

```
cd Assignments/Assignment3/code
python run_all.py
```

`run_all.py` runs every script below in order and reproduces every number and
figure in the write-up (about 5 minutes, almost all in `problem2_fit.py`). Each
script can also be run on its own. The scripts use the course package `risk545/`
at the repo root (they add it to the path themselves, so no install is needed).
The data files are in the folder above `code/`.

Every script prints labelled blocks and saves its console output to `output/`
under the script's name. Figures are saved to `figures/`.

## Scripts, by problem

Each problem follows the assignment's three steps: `*_predict.py` computes only
what the Predict questions ask for (no fitting), `*_fit.py` answers the Fit
questions, and `*_reconcile.py` computes the extra numbers the Reconcile answers use.

| Problem | Script | What it produces |
|:--|:--|:--|
| 1 | `problem1_predict.py` | Put-call parity gap vs band at every strike (a); spreads and spread / mid (c) |
| 1 | `problem1_fit.py` | Implied vol from bid, mid and ask, quotes with no IV and why (d); IV smile chart; vega and IV change per cent (e) |
| 1 | `problem1_reconcile.py` | 175 parity gap in vol points (f); call wing gaps in dollars vs vol points (g) |
| 2 | `problem2_predict.py` | Early exercise condition per dividend (a); PV of dividends |
| 2 | `problem2_fit.py` | All four values with and without dividends, GBSM on S - PV(div) (c); convergence chart and table (d); delta and gamma by bump size (e); dividend sensitivities (f) |
| 2 | `problem2_reconcile.py` | Early exercise premiums; call exercise boundary before dividend 2 (g) |
| 3 | `problem3_predict.py` | Book delta, gamma, theta (a); size of the delta, gamma and theta terms at a typical move |
| 3 | `problem3_fit.py` | Moments, normal and t fits, AICc (c); delta-normal VaR / ES (d); full revaluation simulation (e); P&L vs price chart (f) |
| 3 | `problem3_reconcile.py` | Delta, delta-gamma and delta-gamma-theta approximations vs full revaluation (g) |
| 4 | `problem4_predict.py` | Moments, Sharpe ratios, effect of the most extreme month (a) |
| 4 | `problem4_fit.py` | Max Sharpe portfolios three ways, frontier and CML chart (c); VaR / ES historical and normal (d); CAPM and downside beta (e) |
| 4 | `problem4_reconcile.py` | Standard errors of the means; bootstrap of the tangency weights (h) |
| 5 | `problem5_predict.py` | Excess return, vol, Sharpe, CAPM (a); residual correlations with SMB, HML, Mom (b) |
| 5 | `problem5_fit.py` | CAPM, FF3, Carhart with SEs, t, R^2, AICc (c); expected returns from the last 10 years (d); worst momentum day on $10M (e) |
| 5 | `problem5_reconcile.py` | Exact split of the change in FUND1's and FUND2's alpha into loading x factor premium (f) |

## Conventions

Where a calculation has more than one defensible convention, the scripts follow
the instructor's Julia code in `library/`, the weekly files and old exams.

- **Option pricing** is our own code in `risk545/options.py`: GBSM with Greeks
  (port of `library/gbsm.jl`, cost of carry b = r - q), an implied vol solver
  (Brent's method, the instructor uses `find_zero`), and binomial trees (port of
  `library/bt_american.jl`): CRR, recombining with a continuous yield, and
  non-recombining for discrete cash dividends. Theta is per year, as in `gbsm.jl`.
- **Problem 1:** T = 35 calendar days / 365, as the problem specifies. A quote has no
  implied vol when its price is at or below the European no-arbitrage lower bound.
- **Problem 2:** the main tree uses N = 500 (dividends on steps 100 and 400). The
  European values with dividends come from the same tree without early exercise.
  Early exercise premiums compare American and European values from the same N, so
  the tree's discretisation error cancels.
- **Problem 3:** T = trading days / 255. Delta-normal VaR follows the instructor's
  `OldFinals/Fall2024/exam.jl`: |dollar delta| x sample sd of returns, here x sqrt(10)
  for the 10 day horizon. The simulation draws 10 daily returns per path from each
  fitted distribution with the mean set to zero, compounds them, and revalues every
  option with GBSM at 20 days left and the same implied vol (100,000 paths, seed 545).
- **Problem 4:** rf = 3% / 12 a month (simple). Max Sharpe follows `maxSR` in
  `library/optimizers.jl` (solved with SLSQP; the unconstrained case uses the closed
  form). Historical VaR / ES use the portfolio's 240 monthly returns at fixed weights.
  The bootstrap resamples whole months (1,000 resamples, seed 545).
- **Problem 5:** daily data annualized with 255 days, as in `Week09/week09.jl`.
  Fama-French percentages are divided by 100. Regression standard errors use the
  residual variance over T - p. AICc uses the Gaussian log-likelihood at SSR / T with
  k = coefficients + 1 (the variance). "Most recent 10 years" is 2014-10-01 to
  2024-09-30, the end of the factor files.
- **Moments:** standard deviation with n - 1; skewness and excess kurtosis
  population-style, as StatsBase.
- **VaR and ES** are positive numbers for a loss.

## Package

The new functions this assignment needed live in `risk545/`: `options.py` (GBSM,
Greeks, implied vol, binomial trees) and `optimizers.py` (max Sharpe, frontier).
They are checked against the course's reference outputs (tests 12.1, 12.2, 12.3,
10.3, 10.4) by `tests/test_options.py`.
