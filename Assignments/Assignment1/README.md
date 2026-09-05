# Assignment 1 -- Univariate and Multivariate Statistics

## Setup

```
cd Assignments/Assignment1
pip install -r requirements.txt
```

## Running

Each problem is its own script in `code/`, and can be run on its own:

```
cd code
python problem1.py
python problem2.py
python problem3.py
python problem4.py
python problem5.py
```

Or run everything at once and capture the combined console output:

```
cd code
python run_all.py
```

This writes every printed number to `output/results.txt` (also printed to
the terminal) and regenerates every figure in `figures/`.

## What each script does

- **problem1.py** -- first four sample moments of `problem1.csv`; fits a
  Normal by matching mean and variance; counts observations below the
  fitted Normal's 1% quantile vs. the number expected. Saves
  `figures/problem1_fit.png` (histogram + fitted Normal, and a QQ-plot).
- **problem2.py** -- fits `y ~ alpha + beta*x` on `problem2.csv` three ways:
  OLS (with standard errors), MLE under a Normal error, and MLE under a
  Student's t error (numerically optimized with `scipy.optimize.minimize`).
  Compares the two MLE fits with AICc, and reports the 95%/99.5% quantiles
  of each fitted error distribution. Saves `figures/problem2_scatter.png`.
- **problem3.py** -- Pearson and Spearman correlation matrices for
  `problem3.csv`'s four series, and the pair with the largest gap between
  the two. Saves `figures/problem3_pairs.png` (full pairwise scatter/hist
  grid).
- **problem4.py** -- sample covariance matrix of `problem4.csv`, the
  partitioned-Normal conditional mean/variance of x2 given x1, a 95% band
  around the conditional mean, and that band's coverage overall and split
  by how many standard deviations x1 sits from its mean. Saves
  `figures/problem4_conditional.png`.
- **problem5.py** -- plots the series, ACF, and PACF of `problem5.csv`;
  fits AR(1)-AR(3) and MA(1)-MA(3) (`statsmodels.tsa.arima.model.ARIMA`);
  compares all six with AICc. Saves `figures/problem5_series_acf_pacf.png`.

## Conventions worth knowing

- **AICc**, not AIC. `code/common.py` implements
  `AICc = AIC + 2k(k+1)/(n-k-1)`, with `k` counting every estimated
  parameter (including the error scale/sigma2 and, for the t error, the
  degrees of freedom). Used in both problem2.py and problem5.py so the
  correction is identical everywhere it's applied.
- **Skewness and excess kurtosis** in problem1.py use `scipy.stats`' bias
  correction (`bias=False`) -- the same small-sample-adjusted estimator
  pandas and Excel use -- rather than the plug-in moment estimator.
- **OLS vs. "MLE, Normal error"** in problem2.py: these have numerically
  identical alpha/beta (least squares *is* the Normal MLE for the mean
  parameters). They differ only in how sigma is estimated -- OLS divides
  by `n-2`, the MLE divides by `n` -- so the AICc comparison table only
  carries one Normal-error row, not two.
- **255 trading days/year** is this course's convention elsewhere in the
  repo; it is not used in this assignment since none of these series are
  annualized.

## Where the numbers for the writeup come from

Run `run_all.py` and pull the printed values (and the four saved figures)
directly into the PDF writeup -- every number named in the assignment
(moments, tail counts, alpha/beta/error params, AICc, correlations,
coverage fractions, AICc-selected AR/MA order) is printed with a label.
