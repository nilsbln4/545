"""Option pricing checks: GBSM against functional test 12.1, and the implied-vol solver."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import risk545 as rl

DATA = Path(__file__).resolve().parents[1] / "testfiles" / "data"


def test_gbsm_greeks_12_1():
    opts = pd.read_csv(DATA / "test12_1.csv").dropna(subset=["ID"])
    exp = pd.read_csv(DATA / "testout12_1.csv")
    rows = []
    for _, o in opts.iterrows():
        g = rl.gbsm(o["Option Type"] == "Call", o.Underlying, o.Strike, o.DaysToMaturity / o.DayPerYear,
                    o.RiskFreeRate, o.RiskFreeRate - o.DividendRate, o.ImpliedVol, include_greeks=True)
        rows.append([g.value, g.delta, g.gamma, g.vega, g.rho, g.theta])
    np.testing.assert_allclose(np.array(rows), exp[["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]].to_numpy(),
                               rtol=1e-9)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
def test_implied_vol_round_trip(call, strike):
    args = (100.0, strike, 0.25, 0.045, 0.045 - 0.01)
    price = rl.gbsm(call, *args, 0.37).value
    assert rl.implied_vol(call, *args, price) == pytest.approx(0.37, abs=1e-9)


def test_implied_vol_nan_outside_bounds():
    args = (100.0, 80.0, 0.25, 0.045, 0.045)
    lower, upper = rl.european_bounds(True, *args)
    assert np.isnan(rl.implied_vol(True, *args, lower - 0.01))
    assert np.isnan(rl.implied_vol(True, *args, upper + 0.01))


def test_bt_american_values_12_2():
    opts = pd.read_csv(DATA / "test12_1.csv").dropna(subset=["ID"])
    exp = pd.read_csv(DATA / "testout12_2.csv")
    got = [rl.bt_american(o["Option Type"] == "Call", o.Underlying, o.Strike, o.DaysToMaturity / o.DayPerYear,
                          o.RiskFreeRate, o.RiskFreeRate - o.DividendRate, o.ImpliedVol, 500)
           for _, o in opts.iterrows()]
    np.testing.assert_allclose(got, exp["Value"].to_numpy(), rtol=1e-9)


def test_bt_american_discrete_dividends_12_3():
    opts = pd.read_csv(DATA / "test12_3.csv").dropna(subset=["ID"])
    exp = pd.read_csv(DATA / "testout12_3.csv")
    got = []
    for _, o in opts.iterrows():
        steps = [2 * int(s) for s in str(o.DividendDates).split(",")]
        amts = [float(a) for a in str(o.DividendAmts).split(",")]
        got.append(rl.bt_american_div(o["Option Type"] == "Call", o.Underlying, o.Strike,
                                      o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate, amts, steps,
                                      o.ImpliedVol, 2 * int(o.DaysToMaturity)))
    np.testing.assert_allclose(got, exp["Value"].to_numpy(), rtol=1e-9)


def test_tree_european_converges_to_gbsm():
    args = (100.0, 105.0, 0.5, 0.045, 0.045)
    for call in (True, False):
        tree = rl.bt_american(call, *args, 0.3, 1000, american=False)
        assert tree == pytest.approx(rl.gbsm(call, *args, 0.3).value, abs=5e-3)


def test_zero_dividend_tree_matches_plain_tree():
    plain = rl.bt_american(False, 48.5, 50, 0.5, 0.045, 0.045, 0.3, 100)
    div = rl.bt_american_div(False, 48.5, 50, 0.5, 0.045, [0.0, 0.0], [20, 80], 0.3, 100)
    assert div == pytest.approx(plain, rel=1e-12)


# ---------------------------------------------------------------------------
# Optimizers (tests 10.3 and 10.4)
# ---------------------------------------------------------------------------

def _msr_inputs():
    cov = pd.read_csv(DATA / "test5_3.csv").to_numpy()
    means = pd.read_csv(DATA / "test10_3_means.csv")["Mean"].to_numpy()
    return cov, means, 0.04


def test_max_sharpe_long_only_10_3():
    cov, means, rf = _msr_inputs()
    w = rl.max_sharpe(cov, means, rf, bounds=np.tile([0.0, 1.0], (len(means), 1)))
    np.testing.assert_allclose(w, pd.read_csv(DATA / "testout10_3.csv")["W"].to_numpy(), atol=1e-5)


def test_max_sharpe_bounded_10_4():
    cov, means, rf = _msr_inputs()
    w = rl.max_sharpe(cov, means, rf, bounds=np.tile([0.1, 0.5], (len(means), 1)))
    np.testing.assert_allclose(w, pd.read_csv(DATA / "testout10_4.csv")["W"].to_numpy(), atol=1e-5)


def test_unconstrained_max_sharpe_beats_long_only():
    # test5_3 is not positive definite, so use a valid covariance here.
    rng = np.random.default_rng(10)
    a = rng.normal(size=(200, 5)) @ np.diag([0.1, 0.15, 0.2, 0.12, 0.08])
    cov, means, rf = np.cov(a, rowvar=False), np.array([0.09, 0.08, 0.07, 0.06, 0.05]), 0.04
    w_u = rl.max_sharpe(cov, means, rf)
    w_l = rl.max_sharpe(cov, means, rf, bounds=np.tile([0.0, 1.0], (len(means), 1)))
    assert w_u.sum() == pytest.approx(1.0)
    assert rl.portfolio_sharpe(w_u, means, cov, rf) >= rl.portfolio_sharpe(w_l, means, cov, rf) - 1e-12
    # the closed form is the optimum: no other fully invested portfolio does better
    for _ in range(200):
        w = rng.normal(size=5)
        w /= w.sum()
        assert rl.portfolio_sharpe(w, means, cov, rf) <= rl.portfolio_sharpe(w_u, means, cov, rf) + 1e-12
