# Functional Tests 1

Python package (`risk545/`, at the repo root) implementing the class's
risk-management library, checked against `testfiles/data/` through test 7.6.

## Layout

| Path | What's there |
|:--|:--|
| `risk545/` | The package: `covariance.py`, `psd.py`, `simulate.py`, `returns.py`, `fitted_model.py`. |
| `tests/test_functional.py` | Automated pass/fail check against every `testout*.csv` in `testfiles/data/`, tests 1 through 7.6. |
| `Assignments/FunctionalTests1/functional_tests_1.ipynb` | Guided walkthrough of the package -- what each function does and why, run against the real test data. |

## Setup

From the repo root:

```
pip install -e .
```

## Running the tests

```
pytest tests/test_functional.py -v
```

All 25 cases pass as of this submission.

## Tolerance policy

- Tests 1-4, 6, 7.1, and 7.5 are deterministic linear algebra / closed-form
  fits -- checked tight (they should reproduce the reference near machine
  precision).
- Test 5 (Monte Carlo simulation) is checked against the covariance matrix
  it was supposed to reproduce, not against `testout5_*.csv` directly --
  that file is itself just one particular 100,000-draw realization from a
  different RNG (Julia's), so an exact match isn't the right bar. For the
  two non-PSD cases (5.3, 5.4), the correct comparison target is the
  *repaired* matrix (`near_psd`/`higham_nearest_psd` applied to the input),
  not the raw non-PSD input -- no valid simulation can reproduce a matrix
  that isn't itself a valid covariance matrix.
- Tests 7.2, 7.3, 7.4, and 7.6 involve numerical optimization (`scipy`'s
  optimizer here vs. the reference's Ipopt/Julia call). In practice these
  agree with the reference to 6+ significant figures, not just "close" --
  see the notebook's output for the actual numbers.

## Notes on the NIG fits (7.5 / 7.6)

Same 1,000 observations, two different fitting criteria:

- `fit_nig_moments`: closed form. Matches the sample's mean, variance,
  skewness, and excess kurtosis exactly.
- `fit_nig_mle`: numerical MLE via `scipy.stats.norminvgauss.fit`. Maximizes
  the likelihood of the observed sample instead.

They land on different parameters from the same data because they optimize
different things -- the notebook works through both fits, compares their
log-likelihoods and how well each recovers the sample's own moments.
