"""Problem 3 -- Fit step: VaR and ES of four option books, delta-normal and full revaluation.

(c) First four moments of ABC's arithmetic daily returns (mean, std with ddof = 1,
    population skewness and excess kurtosis as StatsBase). Fit a normal and a
    Student's t, compare by AICc, report the parameters.
(d) 10 day delta-normal VaR and ES at 5% and 1%: the book's dollar delta
    (sum of holding x delta x S) times the daily std of returns times sqrt(10),
    valued under Normal(0, .), the instructor's convention (OldFinals/Fall2024/exam.jl).
(e) Simulate 10 daily returns from each fitted distribution with its mean set to
    zero (scale and nu kept), compound them into the price in 10 days, and revalue
    every position in full with GBSM at 10 fewer days to expiry and the same implied
    vol. Mean P&L, VaR and ES at 5% and 1%, and ES / VaR, per book and distribution.
(f) Each book's 10 day P&L against the ABC price in 10 days (full revaluation),
    with the delta-only approximation on the same axes.

Every number is printed in a labelled block and saved to output/problem3_fit.txt.
The chart is saved to figures/problem3_pnl_vs_price.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (DATA_DIR, FIGURES_DIR, P3_DAYS_PER_YEAR, P3_HORIZON, P3_RATE, Report, p3_positions,
                    p3_spot)

import risk545 as rl

N_SIM = 100_000
SEED = 545
ALPHAS = [0.05, 0.01]
BOOKS = ["CoveredCall", "ShortStraddle", "RiskReversal", "HedgedShortCall"]


def position_values(pos: pd.DataFrame, prices: np.ndarray, days_left_shift: int) -> np.ndarray:
    """Value of every position (rows) at every price (columns), with DaysToExpiry
    reduced by days_left_shift. Options are valued with GBSM, b = r, constant IV."""
    prices = np.asarray(prices, dtype=float)
    out = np.empty((len(pos), len(prices)))
    for row, p in enumerate(pos.itertuples()):
        if p.Type == "Stock":
            unit = prices
        else:
            ttm = (p.DaysToExpiry - days_left_shift) / P3_DAYS_PER_YEAR
            unit = rl.gbsm(p.Type == "Call", prices, p.Strike, ttm, P3_RATE, P3_RATE, p.ImpliedVol).value
        out[row] = p.Holding * unit
    return out


def book_pnl(pos: pd.DataFrame, spot: float, future_prices: np.ndarray) -> pd.DataFrame:
    """10 day P&L of each book at each future price (full revaluation)."""
    now = position_values(pos, np.array([spot]), 0)[:, 0]
    later = position_values(pos, future_prices, P3_HORIZON)
    pnl = later - now[:, None]
    return pd.DataFrame({b: pnl[(pos["Portfolio"] == b).to_numpy()].sum(axis=0) for b in BOOKS})


def book_deltas(pos: pd.DataFrame, spot: float) -> pd.Series:
    d = []
    for p in pos.itertuples():
        if p.Type == "Stock":
            d.append(p.Holding)
        else:
            g = rl.gbsm(p.Type == "Call", spot, p.Strike, p.DaysToExpiry / P3_DAYS_PER_YEAR, P3_RATE, P3_RATE,
                        p.ImpliedVol)
            d.append(p.Holding * g.delta)
    return pd.Series(d, index=pos.index).groupby(pos["Portfolio"]).sum().reindex(BOOKS)


def load_returns() -> np.ndarray:
    prices = pd.read_csv(DATA_DIR / "problem3.csv")
    return rl.return_calculate(prices, method="DISCRETE", date_column="Date")["ABC"].to_numpy()


def simulate_daily_returns(fit_n, fit_t) -> dict:
    """N_SIM x 10 daily returns from each fitted distribution, mean set to zero."""
    rng = np.random.default_rng(SEED)
    return {
        "normal": stats.norm(0, fit_n.params["sigma"]).rvs(size=(N_SIM, P3_HORIZON), random_state=rng),
        "t": stats.t(df=fit_t.params["nu"], loc=0, scale=fit_t.params["sigma"]).rvs(
            size=(N_SIM, P3_HORIZON), random_state=rng),
    }


def main() -> None:
    rep = Report()
    spot = p3_spot()
    pos = p3_positions()
    ret = load_returns()

    rep.section("Inputs")
    rep.say(f"S = {spot}   r = {P3_RATE}   {P3_DAYS_PER_YEAR} trading days   horizon {P3_HORIZON} days   "
            f"{len(ret)} daily returns   {N_SIM:,} simulations, seed {SEED}")

    # ---------------------------------------------------------------- (c)
    rep.section("(c) Moments of ABC daily arithmetic returns")
    rep.say(f"mean            = {ret.mean():.8f}")
    rep.say(f"variance        = {ret.var(ddof=1):.8e}   (std = {ret.std(ddof=1):.8f}, ddof = 1)")
    rep.say(f"skewness        = {stats.skew(ret, bias=True):.6f}")
    rep.say(f"excess kurtosis = {stats.kurtosis(ret, fisher=True, bias=True):.6f}")

    fit_n = rl.fit_normal(ret)
    fit_t = rl.fit_general_t(ret)
    a_n, a_t = rl.aicc(fit_n, ret), rl.aicc(fit_t, ret)
    rep.section("(c) Fitted distributions and AICc (lower is better)")
    rep.say(f"normal : mu = {fit_n.params['mu']:.8f}   sigma = {fit_n.params['sigma']:.8f}   AICc = {a_n:.4f}")
    rep.say(f"t      : mu = {fit_t.params['mu']:.8f}   sigma = {fit_t.params['sigma']:.8f}   "
            f"nu = {fit_t.params['nu']:.4f}   AICc = {a_t:.4f}")
    rep.say(f"AICc(normal) - AICc(t) = {a_n - a_t:.4f}  ->  chosen: {'t' if a_t < a_n else 'normal'}")
    t_sd = fit_t.params["sigma"] * np.sqrt(fit_t.params["nu"] / (fit_t.params["nu"] - 2))
    rep.say(f"std implied by the t fit = sigma sqrt(nu / (nu - 2)) = {t_sd:.8f}")

    # ---------------------------------------------------------------- (d)
    sd = ret.std(ddof=1)
    sd10 = sd * np.sqrt(P3_HORIZON)
    deltas = book_deltas(pos, spot)
    rep.section("(d) 10 day delta-normal VaR and ES")
    rep.say(f"P&L ~ Normal(0, |dollar delta| x sd x sqrt(10)),  sd = {sd:.8f},  sd x sqrt(10) = {sd10:.8f}")
    rep.say("dollar delta = book delta x S\n")
    rep.say(f"{'book':<16} {'delta':>9} {'$ delta':>10} {'P&L sd':>9} {'VaR 5%':>9} {'ES 5%':>9} "
            f"{'VaR 1%':>9} {'ES 1%':>9}")
    dn = {}
    for b in BOOKS:
        dollar = deltas[b] * spot
        pnl_sd = abs(dollar) * sd10
        dist = stats.norm(0, pnl_sd)
        v5, e5, v1, e1 = (rl.value_at_risk(dist, 0.05), rl.expected_shortfall(dist, 0.05),
                          rl.value_at_risk(dist, 0.01), rl.expected_shortfall(dist, 0.01))
        dn[b] = (v5, e5, v1, e1)
        rep.say(f"{b:<16} {deltas[b]:9.4f} {dollar:10.2f} {pnl_sd:9.2f} {v5:9.2f} {e5:9.2f} {v1:9.2f} {e1:9.2f}")
    rep.say(f"(ES / VaR under a normal: {stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05 / -stats.norm.ppf(0.05):.4f} "
            f"at 5%, {stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01 / -stats.norm.ppf(0.01):.4f} at 1%)")

    # ---------------------------------------------------------------- (e)
    sims = simulate_daily_returns(fit_n, fit_t)
    rep.section("(e) Full revaluation: 10 simulated daily returns (zero mean), compounded; options at 20 days left")
    sim_prices = {}
    for name, r in sims.items():
        s10 = spot * np.prod(1 + r, axis=1)
        sim_prices[name] = s10
        cum = s10 / spot - 1
        rep.say(f"\n{name} -- simulated 10 day return: mean {cum.mean():.6f}, std {cum.std(ddof=1):.6f}, "
                f"1% quantile {np.quantile(cum, 0.01):.6f}, 99% quantile {np.quantile(cum, 0.99):.6f}")
        pnl = book_pnl(pos, spot, s10)
        rep.say(f"{'book':<16} {'mean P&L':>9} {'VaR 5%':>9} {'ES 5%':>9} {'ES/VaR':>7} "
                f"{'VaR 1%':>9} {'ES 1%':>9} {'ES/VaR':>7}")
        for b in BOOKS:
            x = pnl[b].to_numpy()
            v5, e5 = rl.value_at_risk(x, 0.05), rl.expected_shortfall(x, 0.05)
            v1, e1 = rl.value_at_risk(x, 0.01), rl.expected_shortfall(x, 0.01)
            rep.say(f"{b:<16} {x.mean():9.2f} {v5:9.2f} {e5:9.2f} {e5 / v5:7.3f} {v1:9.2f} {e1:9.2f} {e1 / v1:7.3f}")

    rep.section("(d) vs (e): VaR and ES side by side")
    rep.say(f"{'book':<16} {'measure':<7} {'delta-N':>9} {'full, N':>9} {'full, t':>9}")
    full = {name: book_pnl(pos, spot, sim_prices[name]) for name in sims}
    for b in BOOKS:
        for k, (lab, a) in enumerate((("VaR 5%", 0.05), ("ES 5%", 0.05), ("VaR 1%", 0.01), ("ES 1%", 0.01))):
            f = rl.value_at_risk if lab.startswith("VaR") else rl.expected_shortfall
            rep.say(f"{b:<16} {lab:<7} {dn[b][k]:9.2f} {f(full['normal'][b].to_numpy(), a):9.2f} "
                    f"{f(full['t'][b].to_numpy(), a):9.2f}")

    # ---------------------------------------------------------------- (f)
    lo, hi = np.quantile(sim_prices["t"], [0.0005, 0.9995])
    grid = np.linspace(min(lo, spot * 0.8), max(hi, spot * 1.2), 401)
    pnl_grid = book_pnl(pos, spot, grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, b in zip(axes.ravel(), BOOKS):
        ax.plot(grid, pnl_grid[b], color="#2a78d6", linewidth=2, label="full revaluation")
        ax.plot(grid, deltas[b] * (grid - spot), color="#eb6834", linewidth=2, linestyle="--", label="delta only")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(spot, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(f"{b}  (delta {deltas[b]:.2f})")
        ax.set_xlabel("ABC price in 10 days")
        ax.set_ylabel("10 day P&L ($)")
        ax.grid(alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"10 day P&L vs ABC price (today {spot}; dotted line), options revalued at 20 days left")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "problem3_pnl_vs_price.png"
    fig.savefig(fig_path, dpi=150)
    rep.section("(f) Chart")
    rep.say(f"Saved to figures/{fig_path.name}; price grid {grid[0]:.2f} to {grid[-1]:.2f}")
    rep.say("P&L at an unchanged price (S_10 = S), full revaluation:")
    flat = book_pnl(pos, spot, np.array([spot]))
    for b in BOOKS:
        rep.say(f"  {b:<16} {flat[b].iloc[0]:9.2f}")

    rep.save("problem3_fit.txt")


if __name__ == "__main__":
    main()
