"""Problem 5 -- Fit and Reconcile numbers for parts (b)-(f).

Each stock return is modelled as r_i = alpha_i + beta_i * r_MKT + eps_i with a
Normal market and Normal errors. VaR of two portfolios is computed three ways:
  - simulation with the full residual covariance matrix
  - simulation with its off-diagonal set to zero (the "shortcut")
  - delta normal, straight from the sample covariance of the two stock returns

Expected returns are assumed zero throughout, so the fitted alphas are reported
but not used in the simulation or the delta normal VaR.

Every number is printed in a labelled block and saved to output/problem5_fit.txt.
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, Report

import risk545 as rl

POSITION = 1_000_000
ALPHA = 0.05
N_SIM = 100_000
SEED = 2024
STOCKS = ["A", "B"]

# Portfolio weights on (A, B), in dollars.
PORTFOLIOS = {"P1: long A, long B": np.array([POSITION, POSITION]),
              "P2: long A, short B": np.array([POSITION, -POSITION])}


def normal_var(weights: np.ndarray, cov: np.ndarray) -> float:
    """Delta normal / analytic VaR in dollars: zero mean, sigma_p = sqrt(w' S w)."""
    sigma_p = float(np.sqrt(weights @ cov @ weights))
    return rl.value_at_risk(stats.norm(0, sigma_p), ALPHA)


def simulate_stock_returns(betas, var_mkt, resid_cov, seed):
    """Simulate (market, eps_A, eps_B) jointly Normal with the market independent
    of the errors, then build the stock returns through the fitted betas.
    alpha is left out: expected returns are assumed zero."""
    cov3 = np.zeros((3, 3))
    cov3[0, 0] = var_mkt
    cov3[1:, 1:] = resid_cov
    draws = rl.simulate_normal(N_SIM, cov3, seed=seed)
    mkt, eps = draws[:, 0], draws[:, 1:]
    return mkt[:, None] * betas[None, :] + eps


def simulated_var(stock_returns: np.ndarray) -> dict:
    return {name: rl.value_at_risk((stock_returns @ w), ALPHA) for name, w in PORTFOLIOS.items()}


def main() -> None:
    rep = Report()
    prices = pd.read_csv(DATA_DIR / "problem5.csv")
    ret = rl.return_calculate(prices, method="DISCRETE", date_column="Day")
    mkt = ret["MKT"].to_numpy()
    stock = ret[STOCKS].to_numpy()
    T = len(ret)

    # ---------------------------------------------------------------- (b)
    rep.section("(b) Regressions of each stock on the market: r_i = alpha + beta * r_MKT + eps")
    # OLS by the normal equations, as in the instructor's Week 9 code.
    X = np.column_stack([np.ones(T), mkt])
    theta = np.linalg.solve(X.T @ X, X.T @ stock)         # 2 x 2: row 0 = alphas, row 1 = betas
    alphas, betas = theta[0], theta[1]
    resid = stock - X @ theta                             # T x 2

    # Residual (co)variance divides by T - 2 (observations minus fitted coefficients),
    # the instructor's T - m - 1 convention.
    resid_cov = resid.T @ resid / (T - 2)
    resid_sd = np.sqrt(np.diag(resid_cov))
    resid_corr = resid_cov[0, 1] / (resid_sd[0] * resid_sd[1])

    rep.say(f"{T} daily arithmetic returns; residual variance divides by T - 2 = {T - 2}\n")
    rep.say(f"{'':<8}{'alpha':>14}{'beta':>12}{'residual SD':>14}")
    for j, s in enumerate(STOCKS):
        rep.say(f"{s:<8}{alphas[j]:>14.6f}{betas[j]:>12.6f}{resid_sd[j]:>14.6f}")
    rep.say(f"\nResidual covariance matrix:\n{pd.DataFrame(resid_cov, index=STOCKS, columns=STOCKS).to_string(float_format=lambda v: f'{v: .4e}')}")
    rep.say(f"\nCorrelation between the residuals of A and B = {resid_corr:.6f}")
    var_mkt = float(mkt.var(ddof=1))
    rep.say(f"Market return variance = {var_mkt:.4e}   (SD {np.sqrt(var_mkt):.6f})")

    # ---------------------------------------------------------------- (c), (d)
    cov_full = resid_cov
    cov_diag = np.diag(np.diag(resid_cov))          # off-diagonal set to zero

    sim_full = simulated_var(simulate_stock_returns(betas, var_mkt, cov_full, SEED))
    sim_diag = simulated_var(simulate_stock_returns(betas, var_mkt, cov_diag, SEED))

    # Closed-form VaR of the same two models (no simulation noise), for comparison.
    model_cov = lambda e: np.outer(betas, betas) * var_mkt + e
    exact_full = {n: normal_var(w, model_cov(cov_full)) for n, w in PORTFOLIOS.items()}
    exact_diag = {n: normal_var(w, model_cov(cov_diag)) for n, w in PORTFOLIOS.items()}

    sample_cov = np.cov(stock, rowvar=False)        # sample covariance of the two stock returns
    delta = {n: normal_var(w, sample_cov) for n, w in PORTFOLIOS.items()}

    rep.section(f"(c) Simulated VaR at {ALPHA:.0%}, dollars ({N_SIM:,} days)")
    rows = []
    for name in PORTFOLIOS:
        rows.append({"portfolio": name,
                     "full residual cov": sim_full[name],
                     "off-diagonal zero": sim_diag[name],
                     "shortcut effect": sim_diag[name] / sim_full[name] - 1})
    t = pd.DataFrame(rows).set_index("portfolio")
    rep.say(t.to_string(formatters={"full residual cov": "{:>14,.0f}".format,
                                    "off-diagonal zero": "{:>14,.0f}".format,
                                    "shortcut effect": "{:>+14.1%}".format}))

    rep.say("\nSame models in closed form (no simulation noise):")
    rows = [{"portfolio": n, "full residual cov": exact_full[n], "off-diagonal zero": exact_diag[n],
             "shortcut effect": exact_diag[n] / exact_full[n] - 1} for n in PORTFOLIOS]
    rep.say(pd.DataFrame(rows).set_index("portfolio").to_string(
        formatters={"full residual cov": "{:>14,.0f}".format, "off-diagonal zero": "{:>14,.0f}".format,
                    "shortcut effect": "{:>+14.1%}".format}))

    rep.say("\nSimulation noise: SD of the simulated VaR over 10 seeds")
    noise = {"full residual cov": [], "off-diagonal zero": []}
    for s in range(100, 110):
        noise["full residual cov"].append(simulated_var(simulate_stock_returns(betas, var_mkt, cov_full, s)))
        noise["off-diagonal zero"].append(simulated_var(simulate_stock_returns(betas, var_mkt, cov_diag, s)))
    for k, v in noise.items():
        d = pd.DataFrame(v)
        rep.say(f"  {k:<20}" + "   ".join(f"{c}: ${d[c].std():,.0f}" for c in d.columns))

    rep.section(f"(d) Delta normal VaR at {ALPHA:.0%} from the sample covariance of the two stock returns, dollars")
    rep.say("Sample covariance matrix of (A, B) returns:")
    rep.say(pd.DataFrame(sample_cov, index=STOCKS, columns=STOCKS).to_string(float_format=lambda v: f"{v: .4e}"))
    rep.say("")
    for n, v in delta.items():
        rep.say(f"{n:<24}${v:>12,.0f}")

    # ---------------------------------------------------------------- (e), (f)
    rep.section("(e)/(f) Side by side, dollars")
    rows = [{"portfolio": n, "delta normal": delta[n], "full resid (closed form)": exact_full[n],
             "full resid (simulated)": sim_full[n], "off-diag zero (simulated)": sim_diag[n]}
            for n in PORTFOLIOS]
    rep.say(pd.DataFrame(rows).set_index("portfolio").to_string(float_format=lambda v: f"{v:>14,.0f}"))

    rep.say("\nWhy the full-residual and delta normal figures should agree: the model's covariance of (A, B)")
    rep.say("is beta beta' var(MKT) + residual covariance. Compare it with the sample covariance:")
    rep.say(pd.DataFrame(model_cov(cov_full) - sample_cov, index=STOCKS, columns=STOCKS)
            .to_string(float_format=lambda v: f"{v: .3e}"))
    rep.say(f"(largest entry is {np.abs(model_cov(cov_full) - sample_cov).max() / np.abs(sample_cov).max():.3%} "
            "of the largest sample covariance entry)")

    rep.save("problem5_fit.txt")


if __name__ == "__main__":
    main()
