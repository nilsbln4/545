# Functional tests 1.1 - 7.6

One script per test, named for the test (`test_1_1.py` ... `test_7_6.py`), each
using the `risk545` package at the repo root.

## Run

From the repo root:

```
python functional_tests/run_all.py          # all 25, one PASS/FAIL line each
python functional_tests/test_7_5.py         # or any single test
```

Needs `numpy`, `scipy` and `pandas`; no `pip install` of the package itself is
required (the scripts add the repo root to the path).

## What each test produces

Every script writes its computed result to `outputs/` under the **same file
name as the reference**, e.g. `outputs/testout_1.1.csv`, and compares it to
`testfiles/data/testout_1.1.csv`. The two files can be opened side by side or
diffed. The committed `outputs/` folder is the result of the last run.

Each script's docstring lists its input file, its reference file, and its
output file.

## How "pass" is decided

| Tests | Comparison |
|:--|:--|
| 1-4, 6, 7.1 | Match the reference to `rtol=1e-6` (deterministic linear algebra). |
| 7.2 - 7.6 | Match to `rtol=1e-4`. These are optimizer fits (scipy here, Ipopt in the Julia reference); in practice they agree to 6+ significant figures. 7.5 is closed form. |
| 5.1 - 5.5 | Monte Carlo (100,000 draws), so the simulated covariance is compared to the covariance it was drawn from, within sampling noise. It is not compared to `testout_5.*.csv`, which is one particular Julia-RNG realization. |

For 5.3 and 5.4 the input matrix is deliberately not positive semi-definite, so
no valid simulation can reproduce it. The target there is the *repaired* matrix
(`near_psd` for 5.3, `higham_nearest_psd` for 5.4).

The same 25 cases are also checked by `tests/test_functional.py` (pytest).
The package source is in `risk545/`.
