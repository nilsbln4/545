"""Problem 4 -- Fit and Reconcile numbers for parts (c)-(j).

  (c) Normal vs Student's t for each margin, chosen by AICc
  (d) Kendall's-tau correlation matrix; Gaussian and t copulas fitted to the
      uniforms from the chosen margins
  (e) VaR and ES for $1,000,000 in each asset, simulated from each copula and
      historical
  (f) joint tail days each copula implies, next to the data's
  (i) each day's contribution to the log-likelihood gap between the copulas
  (j) tail dependence coefficients

Every number is printed in a labelled block and saved to output/problem4_fit.txt.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, Report

import risk545 as rl

POSITION = 1_000_000      # dollars held in each asset
N_SIM = 100_000
SEED = 2024               # both copulas share their underlying normals (see simulate_t_copula)
TAIL = 0.025
LEVELS = (0.05, 0.01)


def portfolio_risk(pnl: np.ndarray) -> dict:
    return {"VaR 5%": rl.value_at_risk(pnl, 0.05), "ES 5%": rl.expected_shortfall(pnl, 0.05),
            "VaR 1%": rl.value_at_risk(pnl, 0.01), "ES 1%": rl.expected_shortfall(pnl, 0.01)}


def simulate_returns(kind: str, R, nu, margins, seed: int) -> tuple:
    """Simulated returns (n_sim x 3) and the underlying uniforms."""
    if kind == "Gaussian":
        u = rl.simulate_gaussian_copula(R, N_SIM, seed=seed)
    else:
        u = rl.simulate_t_copula(R, nu, N_SIM, seed=seed)
    returns = np.column_stack([m.eval(u[:, j]) for j, m in enumerate(margins)])
    return returns, u


def daily_copula_loglik(U: np.ndarray, R: np.ndarray, nu: float) -> tuple:
    """Per-day log copula density under the Gaussian and the t copula. The
    package's gaussian_copula_ll / t_copula_ll return only the sum; this is the
    same formula before summing (joint log density minus the margins')."""
    z = stats.norm.ppf(U)
    gauss = (stats.multivariate_normal(mean=np.zeros(U.shape[1]), cov=R).logpdf(z)
             - stats.norm.logpdf(z).sum(axis=1))
    t = stats.t.ppf(U, df=nu)
    tcop = (stats.multivariate_t(loc=np.zeros(U.shape[1]), shape=R, df=nu).logpdf(t)
            - stats.t.logpdf(t, df=nu).sum(axis=1))
    return gauss, tcop


def joint_tail_counts(u: pd.DataFrame, per_days: int) -> dict:
    """Days (scaled to `per_days`) on which both series of each pair sit in
    their own worst / best 2.5%, from uniforms."""
    out = {}
    for p, q in itertools.combinations(u.columns, 2):
        out[f"{p}-{q}"] = (per_days * float(((u[p] < TAIL) & (u[q] < TAIL)).mean()),
                           per_days * float(((u[p] > 1 - TAIL) & (u[q] > 1 - TAIL)).mean()))
    return out


def main() -> None:
    rep = Report()
    df = pd.read_csv(DATA_DIR / "problem4.csv")
    cols = list(df.columns)
    n = len(df)

    # ---------------------------------------------------------------- (c)
    rep.section("(c) Normal vs Student's t for each margin, by AICc (lower is better)")
    margins, rows = [], []
    for c in cols:
        x = df[c].to_numpy()
        fit_n, fit_t = rl.fit_normal(x), rl.fit_general_t(x)
        aicc_n, aicc_t = rl.aicc(fit_n, x), rl.aicc(fit_t, x)
        winner = fit_t if aicc_t < aicc_n else fit_n
        margins.append(winner)
        rows.append({"series": c,
                     "Normal mu": fit_n.params["mu"], "Normal sigma": fit_n.params["sigma"], "Normal AICc": aicc_n,
                     "t mu": fit_t.params["mu"], "t sigma": fit_t.params["sigma"], "t nu": fit_t.params["nu"],
                     "t AICc": aicc_t, "chosen": "t" if aicc_t < aicc_n else "Normal",
                     "AICc gap (Normal - t)": aicc_n - aicc_t})
    table = pd.DataFrame(rows).set_index("series")
    # A t whose nu runs off to astronomically large values is the optimizer's way of
    # saying "this is a Normal"; print those in scientific notation.
    rep.say(table.drop(columns="chosen").to_string(
        float_format=lambda v: f"{v: .3e}" if abs(v) > 1e6 else f"{v: .6f}"))
    rep.say("\nChosen margin: " + ", ".join(f"{c} = {r['chosen']}" for c, r in zip(cols, rows)))
    if any(r["t nu"] > 1e6 for r in rows):
        rep.say("(nu above 1e6 means the t fit collapsed to a Normal: same likelihood, one more parameter.)")

    # ---------------------------------------------------------------- (d)
    rep.section("(d) Copulas fitted to the uniforms from the chosen margins")
    U = np.column_stack([m.u for m in margins])

    R = rl.kendall_correlation(U)
    rep.say("Correlation matrix from Kendall's tau (rho = sin(pi*tau/2)), shared by both copulas:")
    rep.say(pd.DataFrame(R, index=cols, columns=cols).to_string(float_format=lambda v: f"{v: .6f}"))

    _, ll_g = rl.fit_gaussian_copula(U)
    _, nu_t, ll_t = rl.fit_t_copula(U)
    # k counts only the parameter that differs: nu (R is held fixed from tau), as in test 13.7.
    k_g, k_t = 0, 1
    copulas = pd.DataFrame({
        "log likelihood": [ll_g, ll_t],
        "nu-hat": [np.nan, nu_t],
        "k": [k_g, k_t],
        "AICc": [rl.copula_aicc(ll_g, k_g, n), rl.copula_aicc(ll_t, k_t, n)],
        "BIC": [rl.copula_bic(ll_g, k_g, n), rl.copula_bic(ll_t, k_t, n)],
    }, index=["Gaussian", "t"])
    rep.say("\n" + copulas.to_string(float_format=lambda v: f"{v: .4f}", na_rep="--"))
    rep.say(f"\nt minus Gaussian:  log likelihood {ll_t - ll_g:+.4f}   AICc {copulas.loc['t', 'AICc'] - copulas.loc['Gaussian', 'AICc']:+.4f}"
            f"   BIC {copulas.loc['t', 'BIC'] - copulas.loc['Gaussian', 'BIC']:+.4f}   (negative = t preferred)")

    # ---------------------------------------------------------------- (e)
    rep.section(f"(e) Portfolio risk, ${POSITION:,} in each of {', '.join(cols)} (dollars, positive = loss)")
    sims = {"Gaussian": simulate_returns("Gaussian", R, nu_t, margins, SEED),
            "t": simulate_returns("t", R, nu_t, margins, SEED)}
    risk = {f"{k} copula ({N_SIM:,} sims)": portfolio_risk(POSITION * r.sum(axis=1)) for k, (r, _) in sims.items()}
    risk["Historical (1,000 days)"] = portfolio_risk(POSITION * df.to_numpy().sum(axis=1))
    rep.say(pd.DataFrame(risk).T.to_string(float_format=lambda v: f"{v:>12,.0f}"))

    rep.say("\nSimulation noise: the same risk numbers over 10 different seeds (mean and SD):")
    noise = {}
    for kind in ("Gaussian", "t"):
        draws = [portfolio_risk(POSITION * simulate_returns(kind, R, nu_t, margins, s)[0].sum(axis=1))
                 for s in range(100, 110)]
        d = pd.DataFrame(draws)
        noise[f"{kind}: mean"], noise[f"{kind}: SD"] = d.mean(), d.std()
    rep.say(pd.DataFrame(noise).T.to_string(float_format=lambda v: f"{v:>12,.0f}"))

    # ---------------------------------------------------------------- (f)
    rep.section("(f) Joint tail days over 1,000 days: each copula's, next to the data's")
    ranks = df.rank(method="average")
    k_days = int(round(TAIL * n))
    data_counts = {f"{p}-{q}": (int(((ranks[p] <= k_days) & (ranks[q] <= k_days)).sum()),
                                int(((ranks[p] > n - k_days) & (ranks[q] > n - k_days)).sum()))
                   for p, q in itertools.combinations(cols, 2)}
    implied = {k: joint_tail_counts(pd.DataFrame(u, columns=cols), 1000) for k, (_, u) in sims.items()}

    rep.say(f"{'pair':<9}{'both worst 2.5%':^38}{'both best 2.5%':^38}")
    rep.say(f"{'':<9}{'data':>10}{'Gaussian':>14}{'t':>14}{'data':>10}{'Gaussian':>14}{'t':>14}")
    for pair in data_counts:
        rep.say(f"{pair:<9}{data_counts[pair][0]:>10}{implied['Gaussian'][pair][0]:>14.2f}{implied['t'][pair][0]:>14.2f}"
                f"{data_counts[pair][1]:>10}{implied['Gaussian'][pair][1]:>14.2f}{implied['t'][pair][1]:>14.2f}")
    rep.say(f"\n(independence would give {1000 * TAIL * TAIL:.3f} days)")

    # ---------------------------------------------------------------- (i)
    rep.section("(i) Each day's contribution to (log likelihood of t copula - log likelihood of Gaussian copula)")
    ll_gauss_day, ll_t_day = daily_copula_loglik(U, R, nu_t)
    diff = ll_t_day - ll_gauss_day
    rep.say(f"check: sum of daily gaps = {diff.sum():.4f}  vs  total gap {ll_t - ll_g:.4f}")

    # "No series in an outer 5% on either side": every uniform lies in [0.05, 0.95].
    central = ((U >= 0.05) & (U <= 0.95)).all(axis=1)
    rep.say(f"\nDays with no series in an outer 5% on either side: {int(central.sum())} of {n}")
    rep.say(f"  their total contribution:           {diff[central].sum():+.4f}   ({diff[central].sum() / diff.sum():.1%} of the total)")
    rep.say(f"Days with at least one series in an outer 5%: {int((~central).sum())} of {n}")
    rep.say(f"  their total contribution:           {diff[~central].sum():+.4f}   ({diff[~central].sum() / diff.sum():.1%} of the total)")

    top = np.argsort(diff)[::-1][:5]
    rep.say("\nFive days with the largest contribution (observation number, contribution, uniforms):")
    for i in top:
        rep.say(f"  obs {i + 1:>4}   {diff[i]:+8.4f}   u = " + "  ".join(f"{v:.4f}" for v in U[i]))
    rep.say(f"\nDays where the t copula is the better fit (contribution > 0): {int((diff > 0).sum())} of {n}")

    # ---------------------------------------------------------------- (j)
    rep.section("(j) Lower tail dependence coefficient")
    rep.say(f"{'pair':<9}{'correlation':>13}{'t copula':>12}{'Gaussian':>12}")
    best_pair, best_rho = None, -np.inf
    for (i, p), (j, q) in itertools.combinations(list(enumerate(cols)), 2):
        lam = rl.tail_dependence_t(R[i, j], nu_t)
        rep.say(f"{p + '-' + q:<9}{R[i, j]:>13.4f}{lam:>12.4f}{0.0:>12.4f}")
        if R[i, j] > best_rho:
            best_pair, best_rho = f"{p}-{q}", R[i, j]
    rep.say(f"\nMost correlated pair: {best_pair} (correlation {best_rho:.4f}), fitted nu = {nu_t:.4f}")

    rep.save("problem4_fit.txt")


if __name__ == "__main__":
    main()
