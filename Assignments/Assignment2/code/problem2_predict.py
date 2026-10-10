"""Problem 2 -- Predict step: the returns plot, the moments, and the weight numbers.

Computes arithmetic returns, removes the sample mean, plots them, prints the
first four moments (part c), and prints effective sample size and half life
for lambda = 0.94 and 0.97 (part b). No VaR is computed here.

Part (b) also needs the share of weight each lambda puts on the "recent days
that look different". Choose that number of days from the plot, then pass it:

    python problem2_predict.py --recent-days 40

Without the flag, the script prints the weight share for a range of day counts.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import DATA_DIR, FIGURES_DIR, ew_weights, problem2_returns, section

LAMBDAS = [0.94, 0.97]


def recent_weight_share(lam: float, n_recent: int, n_obs: int) -> float:
    """Share of the total exponential weight on the most recent n_recent days,
    for a sample of n_obs returns (weights normalized to sum to 1 over the
    sample, newest observation heaviest -- the same scheme ew_cov uses)."""
    return float(ew_weights(lam, n_obs)[-n_recent:].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-days", type=int, default=None,
                        help="number of recent days that look different in the plot")
    args = parser.parse_args()

    r, mean_raw = problem2_returns()  # arithmetic returns, sample mean removed
    n = len(r)

    section("Returns")
    say = print
    say(f"{n} arithmetic returns from {n + 1} prices; sample mean removed = {mean_raw:.6f}")

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, n + 1), r, lw=0.8, color="steelblue")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("observation (1 = first return, 500 = most recent)")
    ax.set_ylabel("return (mean removed)")
    ax.set_title("Problem 2: daily returns")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "problem2_returns.png"
    fig.savefig(out, dpi=150)
    say(f"Saved plot to {out.relative_to(DATA_DIR)}")

    # ---- (c) first four moments --------------------------------------
    section("(c) First four moments of the full sample (mean removed)")
    # Conventions follow the instructor's Julia code: var()/std() divide by n-1,
    # and StatsBase's skewness()/kurtosis() are the population-style (not
    # small-sample-corrected) estimators, i.e. scipy's bias=True.
    say(f"mean            = {r.mean(): .6e}")
    say(f"variance        = {r.var(ddof=1): .6e}")
    say(f"std deviation   = {r.std(ddof=1): .6f}")
    say(f"skewness        = {stats.skew(r, bias=True): .6f}")
    say(f"excess kurtosis = {stats.kurtosis(r, fisher=True, bias=True): .6f}")

    # ---- (b) effective sample size and half life ----------------------
    section("(b) Effective sample size and half life")
    # Week 3: n_eff = (1+lambda)/(1-lambda) (infinite horizon), h = ln(0.5)/ln(lambda).
    say(f"{'lambda':>8} {'n_eff':>10} {'half life (days)':>18}")
    for lam in LAMBDAS:
        n_eff = (1 + lam) / (1 - lam)
        half = np.log(0.5) / np.log(lam)
        say(f"{lam:>8.2f} {n_eff:>10.2f} {half:>18.2f}")

    section("(b) Share of total weight on the most recent k days")
    ks = [args.recent_days] if args.recent_days else [5, 10, 20, 30, 40, 50, 75, 100]
    say(f"{'k':>6} " + " ".join(f"{'lambda=' + str(lam):>14}" for lam in LAMBDAS)
        + f" {'equal weight':>14}")
    for k in ks:
        shares = [recent_weight_share(lam, k, n) for lam in LAMBDAS]
        say(f"{k:>6} " + " ".join(f"{s:>14.4f}" for s in shares) + f" {k / n:>14.4f}")


if __name__ == "__main__":
    main()
