"""Functional tests 1-7.6, checked against testfiles/data/testout*.csv.

Tolerance policy:
- Deterministic linear algebra (tests 1-4, 6, 7.1, 7.5) is checked tight --
  it should reproduce the reference near machine precision.
- Monte Carlo simulation (test 5) is checked against the *input* covariance
  it was supposed to reproduce, not the reference testout file -- that file
  is itself just one particular 100,000-draw realization from a different
  RNG, so matching it exactly isn't the point; recovering the input
  covariance to within Monte Carlo noise is.
- Numerical optimization (tests 7.2, 7.3, 7.4, 7.6) is checked at a looser
  tolerance -- a different optimizer (scipy vs. Julia's Ipopt) converges to
  the same likelihood surface's optimum, not to bit-identical parameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import risk545 as rl

DATA = Path(__file__).resolve().parent.parent / "testfiles" / "data"

TIGHT = dict(rtol=1e-6, atol=1e-8)
# Different optimizer (scipy vs. Julia's Ipopt) than the reference, but both
# converge to the same likelihood surface's optimum -- in practice these
# fits agree with the reference to 6+ significant figures, not just "close".
MLE = dict(rtol=1e-4, atol=1e-6)


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def _matrix(name: str) -> np.ndarray:
    return _load(name).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Test 1 -- missing data covariance/correlation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("skip_miss,method,expected", [
    (True, "cov", "testout_1.1.csv"),
    (True, "cor", "testout_1.2.csv"),
    (False, "cov", "testout_1.3.csv"),
    (False, "cor", "testout_1.4.csv"),
])
def test_missing_cov(skip_miss, method, expected):
    x = _load("test1.csv")
    out = rl.missing_cov(x, skip_miss=skip_miss, method=method)
    np.testing.assert_allclose(out, _matrix(expected), **TIGHT)


# ---------------------------------------------------------------------------
# Test 2 -- exponentially weighted covariance/correlation
# ---------------------------------------------------------------------------

def test_ew_cov_2_1():
    x = _matrix("test2.csv")
    out = rl.ew_cov(x, 0.97)
    np.testing.assert_allclose(out, _matrix("testout_2.1.csv"), **TIGHT)


def test_ew_corr_2_2():
    x = _matrix("test2.csv")
    out = rl.ew_corr(x, 0.94)
    np.testing.assert_allclose(out, _matrix("testout_2.2.csv"), **TIGHT)


def test_ew_mixed_var_corr_2_3():
    x = _matrix("test2.csv")
    cov97 = rl.ew_cov(x, 0.97)
    sd1 = np.sqrt(np.diag(cov97))
    cov94 = rl.ew_cov(x, 0.94)
    sd = 1.0 / np.sqrt(np.diag(cov94))
    out = np.diag(sd1) @ np.diag(sd) @ cov94 @ np.diag(sd) @ np.diag(sd1)
    np.testing.assert_allclose(out, _matrix("testout_2.3.csv"), **TIGHT)


# ---------------------------------------------------------------------------
# Test 3 -- non-PSD matrix repair
# ---------------------------------------------------------------------------

def test_near_psd_covariance_3_1():
    out = rl.near_psd(_matrix("testout_1.3.csv"))
    np.testing.assert_allclose(out, _matrix("testout_3.1.csv"), **TIGHT)


def test_near_psd_correlation_3_2():
    out = rl.near_psd(_matrix("testout_1.4.csv"))
    np.testing.assert_allclose(out, _matrix("testout_3.2.csv"), **TIGHT)


def test_higham_covariance_3_3():
    out = rl.higham_nearest_psd(_matrix("testout_1.3.csv"))
    np.testing.assert_allclose(out, _matrix("testout_3.3.csv"), **TIGHT)


def test_higham_correlation_3_4():
    out = rl.higham_nearest_psd(_matrix("testout_1.4.csv"))
    np.testing.assert_allclose(out, _matrix("testout_3.4.csv"), **TIGHT)


# ---------------------------------------------------------------------------
# Test 4 -- Cholesky factorization
# ---------------------------------------------------------------------------

def test_chol_psd_4_1():
    out = rl.chol_psd(_matrix("testout_3.1.csv"))
    np.testing.assert_allclose(out, _matrix("testout_4.1.csv"), **TIGHT)


# ---------------------------------------------------------------------------
# Test 5 -- simulation. Checked against the *input* covariance (Monte Carlo
# tolerance), per Tests.xlsx's own description of these cases.
# ---------------------------------------------------------------------------

MC = dict(rtol=0.08, atol=0.0015)


def test_simulate_normal_pd_5_1():
    cov = _matrix("test5_1.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5001)
    np.testing.assert_allclose(np.cov(sim, rowvar=False), cov, **MC)


def test_simulate_normal_psd_5_2():
    cov = _matrix("test5_2.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5002)
    np.testing.assert_allclose(np.cov(sim, rowvar=False), cov, **MC)


def test_simulate_normal_nonpsd_near_psd_5_3():
    # test5_3 is deliberately non-PSD (one off-diagonal pair is zeroed out
    # against a background ~0.75 correlation structure), so a valid
    # simulation can only reproduce the *repaired* matrix, not the raw input.
    cov = _matrix("test5_3.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5003, fix_method=rl.near_psd)
    np.testing.assert_allclose(np.cov(sim, rowvar=False), rl.near_psd(cov), **MC)


def test_simulate_normal_nonpsd_higham_5_4():
    cov = _matrix("test5_3.csv")
    sim = rl.simulate_normal(100_000, cov, seed=5004, fix_method=rl.higham_nearest_psd)
    np.testing.assert_allclose(np.cov(sim, rowvar=False), rl.higham_nearest_psd(cov), **MC)


def test_simulate_pca_5_5():
    cov = _matrix("test5_2.csv")
    sim = rl.simulate_pca(cov, 100_000, pct_exp=0.99, seed=5005)
    np.testing.assert_allclose(np.cov(sim, rowvar=False), cov, **MC)


# ---------------------------------------------------------------------------
# Test 6 -- returns
# ---------------------------------------------------------------------------

def test_return_calculate_discrete_6_1():
    prices = _load("test6.csv")
    out = rl.return_calculate(prices, method="DISCRETE", date_column="Date")
    expected = _load("testout6_1.csv")
    np.testing.assert_allclose(out.drop(columns="Date").to_numpy(dtype=float),
                                expected.drop(columns="Date").to_numpy(dtype=float), **TIGHT)


def test_return_calculate_log_6_2():
    prices = _load("test6.csv")
    out = rl.return_calculate(prices, method="LOG", date_column="Date")
    expected = _load("testout6_2.csv")
    np.testing.assert_allclose(out.drop(columns="Date").to_numpy(dtype=float),
                                expected.drop(columns="Date").to_numpy(dtype=float), **TIGHT)


# ---------------------------------------------------------------------------
# Test 7.1-7.4 -- Normal, generalized t, t-regression, AICc
# ---------------------------------------------------------------------------

def test_fit_normal_7_1():
    x = _load("test7_1.csv")["x1"].to_numpy()
    fit = rl.fit_normal(x)
    expected = _load("testout7_1.csv").iloc[0]
    np.testing.assert_allclose([fit.params["mu"], fit.params["sigma"]],
                                [expected["mu"], expected["sigma"]], **TIGHT)


def test_fit_general_t_7_2():
    x = _load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    expected = _load("testout7_2.csv").iloc[0]
    np.testing.assert_allclose([fit.params["mu"], fit.params["sigma"], fit.params["nu"]],
                                [expected["mu"], expected["sigma"], expected["nu"]], **MLE)


def test_fit_regression_t_7_3():
    df = _load("test7_3.csv")
    y = df["y"].to_numpy()
    x = df[["x1", "x2", "x3"]].to_numpy()
    fit = rl.fit_regression_t(y, x)
    expected = _load("testout7_3.csv").iloc[0]
    np.testing.assert_allclose([fit.params["sigma"], fit.params["nu"]],
                                [expected["sigma"], expected["nu"]], **MLE)
    np.testing.assert_allclose(fit.beta, [expected["Alpha"], expected["B1"], expected["B2"], expected["B3"]], **MLE)


def test_aicc_7_4():
    x = _load("test7_2.csv")["x1"].to_numpy()
    fit = rl.fit_general_t(x)
    value = rl.aicc(fit, x)
    expected = _load("testout7_4.csv").iloc[0]["AICC"]
    np.testing.assert_allclose(value, expected, **MLE)


# ---------------------------------------------------------------------------
# Test 7.5-7.6 -- NIG, method of moments vs. MLE
# ---------------------------------------------------------------------------

def _nig_params(fit):
    p = fit.params
    return [p["mu"], p["alpha"], p["beta"], p["delta"]]


def test_fit_nig_moments_7_5():
    x = _load("test7_5.csv")["x1"].to_numpy()
    fit = rl.fit_nig_moments(x)
    expected = _load("testout7_5.csv").iloc[0]
    np.testing.assert_allclose(_nig_params(fit),
                                [expected["mu"], expected["alpha"], expected["beta"], expected["delta"]],
                                rtol=1e-4, atol=1e-6)


def test_fit_nig_mle_7_6():
    x = _load("test7_5.csv")["x1"].to_numpy()
    fit = rl.fit_nig_mle(x)
    expected = _load("testout7_6.csv").iloc[0]
    np.testing.assert_allclose(_nig_params(fit),
                                [expected["mu"], expected["alpha"], expected["beta"], expected["delta"]], **MLE)
