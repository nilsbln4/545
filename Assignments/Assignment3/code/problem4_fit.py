"""Problem 4 -- Fit step: maximum Sharpe portfolios, their tail risk, and PUTW's CAPM.

Inputs to the optimizer: the sample mean (arithmetic, monthly) and the sample
covariance (ddof = 1) of the four assets, rf = 3% / 12 a month.

(c) Maximum Sharpe portfolio three ways: unconstrained (shorting allowed, closed
    form), long only, and long only with PUTW capped at 20%. Plot the long only
    efficient frontier and the capital market line through the long only tangency
    portfolio, with the four assets and three portfolios marked (annualized axes).
(d) For each portfolio: annualized Sharpe ratio, and monthly VaR and ES at 5% and
    1%, historical (the portfolio's 240 monthly returns, fixed weights) and normal
    (a normal with the portfolio's mean and standard deviation). Positive = loss,
    as a fraction of portfolio value.
(e) CAPM regression of PUTW's excess return on EQ's excess return (OLS): alpha, its
    t statistic, beta, R^2. Then the same regression on the months with EQ's excess
    return below -5% only (and, for comparison, on the remaining months).

Every number is printed in a labelled block and saved to output/problem4_fit.txt.
The chart is saved to figures/problem4_frontier.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import FIGURES_DIR, P4_RF_MONTHLY, Report, p4_returns

import risk545 as rl

PUTW_CAP = 0.20
DOWN_MONTH = -0.05


def annual_sharpe(x: np.ndarray) -> float:
    return (x.mean() - P4_RF_MONTHLY) / x.std(ddof=1) * np.sqrt(12)


def ols(y: pd.Series, x: pd.Series):
    """OLS with an intercept; the residual variance divides by T - 2 (statsmodels default)."""
    return sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit()


def main() -> None:
    rep = Report()
    r = p4_returns()
    assets = list(r.columns)
    mean = r.mean().to_numpy()
    cov = r.cov().to_numpy()
    n = len(assets)
    i_putw = assets.index("PUTW")

    rep.section("Inputs")
    rep.say(f"{len(r)} months; rf = {P4_RF_MONTHLY:.4%} a month; sample mean and covariance (ddof = 1)")
    rep.say("monthly means: " + ", ".join(f"{a} {m:.6f}" for a, m in zip(assets, mean)))

    # ---------------------------------------------------------------- (c)
    long_only = np.tile([0.0, 1.0], (n, 1))
    capped = long_only.copy()
    capped[i_putw, 1] = PUTW_CAP
    ports = {
        "Unconstrained": rl.max_sharpe(cov, mean, P4_RF_MONTHLY),
        "Long only": rl.max_sharpe(cov, mean, P4_RF_MONTHLY, long_only),
        f"Long only, PUTW <= {PUTW_CAP:.0%}": rl.max_sharpe(cov, mean, P4_RF_MONTHLY, capped),
    }
    rep.section("(c) Maximum Sharpe ratio portfolios (weights)")
    w_tbl = pd.DataFrame({k: v for k, v in ports.items()}, index=assets).T
    w_tbl["sum"] = w_tbl.sum(axis=1)
    rep.say(w_tbl.to_string(float_format=lambda v: f"{v:10.4f}"))

    rep.say("\nPortfolio mean and volatility (monthly and annualized):")
    for k, w in ports.items():
        m, s = w @ mean, np.sqrt(w @ cov @ w)
        rep.say(f"  {k:<26} mean {m:.6f} ({12 * m:.4%}/yr)   sd {s:.6f} ({s * np.sqrt(12):.4%}/yr)")

    # Frontier (long only) and CML through the long only tangency portfolio.
    targets = np.linspace(mean.min(), mean.max(), 120)
    frontier = []
    for t in targets:
        try:
            w = rl.min_variance_for_return(cov, mean, t, long_only)
            frontier.append((np.sqrt(w @ cov @ w), t))
        except RuntimeError:
            pass
    fr = np.array(frontier)
    # keep the efficient (upper) branch: returns above the minimum-variance point
    k_min = fr[:, 0].argmin()
    fr_eff = fr[k_min:]

    w_tan = ports["Long only"]
    tan_s, tan_m = np.sqrt(w_tan @ cov @ w_tan), w_tan @ mean
    slope = (tan_m - P4_RF_MONTHLY) / tan_s
    ann = lambda s, m: (s * np.sqrt(12), 12 * m)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    s_a, m_a = ann(fr[:, 0], fr[:, 1])
    ax.plot(s_a[:k_min + 1], m_a[:k_min + 1], color="#2a78d6", linewidth=1, linestyle=":", alpha=0.6)
    s_a, m_a = ann(fr_eff[:, 0], fr_eff[:, 1])
    ax.plot(s_a, m_a, color="#2a78d6", linewidth=2, label="long only efficient frontier")
    s_line = np.linspace(0, fr[:, 0].max() * 1.05, 50)
    ax.plot(*ann(s_line, P4_RF_MONTHLY + slope * s_line), color="#eb6834", linewidth=1.5, linestyle="--",
            label=f"capital market line (Sharpe {slope * np.sqrt(12):.3f})")
    for a, m_, s_ in zip(assets, mean, np.sqrt(np.diag(cov))):
        x, y = ann(s_, m_)
        ax.scatter(x, y, color="#333333", s=30, zorder=3)
        ax.annotate(a, (x, y), textcoords="offset points", xytext=(6, -3), fontsize=9)
    markers = {"Long only": ("*", 180), f"Long only, PUTW <= {PUTW_CAP:.0%}": ("D", 50)}
    for k, (mk, sz) in markers.items():
        w = ports[k]
        x, y = ann(np.sqrt(w @ cov @ w), w @ mean)
        ax.scatter(x, y, marker=mk, s=sz, color="#1baf7a", edgecolor="white", zorder=4, label=f"max Sharpe: {k}")
    ax.scatter(0, 12 * P4_RF_MONTHLY, color="#333333", s=20, zorder=3)
    ax.annotate("rf", (0, 12 * P4_RF_MONTHLY), textcoords="offset points", xytext=(5, -10), fontsize=9)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized mean return")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Long only efficient frontier and capital market line (monthly data, annualized)")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "problem4_frontier.png"
    fig.savefig(fig_path, dpi=150)
    w_u = ports["Unconstrained"]
    rep.say(f"\nChart saved to figures/{fig_path.name}. The unconstrained portfolio (sd "
            f"{np.sqrt(w_u @ cov @ w_u) * np.sqrt(12):.2%}/yr) lies on the unconstrained frontier and is not drawn.")

    # ---------------------------------------------------------------- (d)
    rep.section("(d) Sharpe ratio and monthly VaR / ES (fraction of portfolio value, positive = loss)")
    rep.say(f"{'portfolio':<26} {'Sharpe':>7} {'':>5} {'VaR 5%':>8} {'ES 5%':>8} {'VaR 1%':>8} {'ES 1%':>8}")
    port_ret = {}
    for k, w in ports.items():
        x = r.to_numpy() @ w
        port_ret[k] = x
        dist = stats.norm(x.mean(), x.std(ddof=1))
        hist = [rl.value_at_risk(x, 0.05), rl.expected_shortfall(x, 0.05),
                rl.value_at_risk(x, 0.01), rl.expected_shortfall(x, 0.01)]
        norm = [rl.value_at_risk(dist, 0.05), rl.expected_shortfall(dist, 0.05),
                rl.value_at_risk(dist, 0.01), rl.expected_shortfall(dist, 0.01)]
        rep.say(f"{k:<26} {annual_sharpe(x):7.4f} {'hist':>5} " + " ".join(f"{v:8.4%}" for v in hist))
        rep.say(f"{'':<26} {'':>7} {'norm':>5} " + " ".join(f"{v:8.4%}" for v in norm))
        rep.say(f"{'':<26} {'':>7} {'h/n':>5} " + " ".join(f"{h / m:8.3f}" for h, m in zip(hist, norm)))
    rep.say("\nWorst three months of each portfolio:")
    for k, x in port_ret.items():
        worst = pd.Series(x, index=r.index).nsmallest(3)
        rep.say(f"  {k:<26} " + ", ".join(f"{d:%Y-%m} {v:+.4%}" for d, v in worst.items()))

    # ---------------------------------------------------------------- (e)
    ex = r.sub(P4_RF_MONTHLY)
    rep.section("(e) CAPM: PUTW excess return on EQ excess return")
    rep.say("OLS with intercept; standard errors from the residual variance over T - 2.\n")
    rep.say(f"{'sample':<34} {'months':>6} {'alpha':>10} {'t(alpha)':>9} {'beta':>8} {'t(beta)':>8} "
            f"{'se(beta)':>9} {'R^2':>7}")
    down = ex["EQ"] < DOWN_MONTH
    for label, mask in (("all months", np.ones(len(ex), bool)),
                        (f"EQ excess < {DOWN_MONTH:.0%}", down.to_numpy()),
                        (f"EQ excess >= {DOWN_MONTH:.0%} (the rest)", (~down).to_numpy())):
        f = ols(ex["PUTW"][mask], ex["EQ"][mask])
        rep.say(f"{label:<34} {mask.sum():6d} {f.params[0]:10.6f} {f.tvalues[0]:9.4f} {f.params[1]:8.4f} "
                f"{f.tvalues[1]:8.4f} {f.bse[1]:9.4f} {f.rsquared:7.4f}")
    f_all = ols(ex["PUTW"], ex["EQ"])
    rep.say(f"\nAll months: alpha annualized = 12 x alpha = {12 * f_all.params[0]:.4%};  "
            f"se(alpha) = {f_all.bse[0]:.6f}")
    rep.say(f"Months with EQ excess < {DOWN_MONTH:.0%}: " +
            ", ".join(f"{d:%Y-%m}" for d in ex.index[down]))

    rep.save("problem4_fit.txt")


if __name__ == "__main__":
    main()
