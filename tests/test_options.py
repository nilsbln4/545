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


# ---------------------------------------------------------------------------
# Risk parity (10.1, 10.2), attribution (11.1, 11.2), tree Greeks (12.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("budget,ref", [(None, "testout10_1.csv"), ([1, 1, 1, 1, 0.5], "testout10_2.csv")])
def test_risk_parity_10_1_10_2(budget, ref):
    cov = pd.read_csv(DATA / "test5_2.csv").to_numpy()
    w = rl.risk_parity(cov, budget)
    np.testing.assert_allclose(w, pd.read_csv(DATA / ref)["W"].to_numpy(), rtol=1e-6)
    # the defining property: component SDs proportional to the budget
    b = np.ones(5) if budget is None else np.asarray(budget, dtype=float)
    csd = rl.component_sd(w, cov)
    np.testing.assert_allclose(csd / b / (csd / b).mean(), 1.0, rtol=1e-6)


def test_expost_attribution_11_1():
    r = pd.read_csv(DATA / "test11_1_returns.csv")
    w = pd.read_csv(DATA / "test11_1_weights.csv")["W"].to_numpy()
    got = rl.expost_factor(w, r, r, np.eye(3)).drop(columns="Alpha")
    exp = pd.read_csv(DATA / "testout11_1.csv")
    assert list(got["Value"]) == list(exp["Value"])
    np.testing.assert_allclose(got.iloc[:, 1:].to_numpy(), exp.iloc[:, 1:].to_numpy(), rtol=1e-9)


def test_expost_factor_attribution_11_2():
    f = pd.read_csv(DATA / "test11_2_factor_returns.csv")
    s = pd.read_csv(DATA / "test11_2_stock_returns.csv")
    beta = pd.read_csv(DATA / "test11_2_beta.csv").drop(columns="Stock").to_numpy()
    w = pd.read_csv(DATA / "test11_2_weights.csv")["W"].to_numpy()
    got = rl.expost_factor(w, s, f, beta)
    exp = pd.read_csv(DATA / "testout11_2.csv")
    np.testing.assert_allclose(got.iloc[:, 1:].to_numpy(), exp.iloc[:, 1:].to_numpy(), rtol=1e-9)
    # attributions add up to the portfolio's total return and standard deviation
    row = got.set_index("Value")
    for r in ("Return Attribution", "Vol Attribution"):
        np.testing.assert_allclose(row.loc[r].drop("Portfolio").sum(), row.loc[r, "Portfolio"], rtol=1e-10)


def test_bt_american_greeks_12_2():
    opts = pd.read_csv(DATA / "test12_1.csv").dropna(subset=["ID"])
    exp = pd.read_csv(DATA / "testout12_2.csv")
    got = []
    for _, o in opts.iterrows():
        g = rl.bt_american_greeks(o["Option Type"] == "Call", o.Underlying, o.Strike,
                                  o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate,
                                  o.RiskFreeRate - o.DividendRate, o.ImpliedVol, 500)
        got.append([g[k] for k in ("value", "delta", "gamma", "vega", "rho", "theta")])
    np.testing.assert_allclose(np.array(got), exp[["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]].to_numpy(),
                               rtol=1e-6)
