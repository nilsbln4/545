"""Problem 4 -- Predict step: moments, rank plots, and joint tail counts.

Prints the first four moments of each series (part a), plots every pair after
converting each series to ranks scaled into (0,1), and counts the days on which
both series of a pair sit in their own worst (and best) 2.5% (part b). No model
is fitted here.
"""

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, FIGURES_DIR, Report

TAIL = 0.025


def moments(x: np.ndarray) -> dict:
    # Conventions follow the instructor's Julia code: std() divides by n-1, and
    # StatsBase's skewness()/kurtosis() are the population-style estimators
    # (scipy's bias=True).
    return {"mean": x.mean(), "SD": x.std(ddof=1),
            "skewness": stats.skew(x, bias=True),
            "excess kurtosis": stats.kurtosis(x, fisher=True, bias=True),
            "min": x.min(), "max": x.max()}


def main() -> None:
    rep = Report()
    df = pd.read_csv(DATA_DIR / "problem4.csv")
    cols = list(df.columns)
    n = len(df)

    # ---------------------------------------------------------------- (a)
    rep.section("(a) First four moments of each series")
    rep.say(pd.DataFrame({c: moments(df[c].to_numpy()) for c in cols})
            .to_string(float_format=lambda v: f"{v: .6f}"))

    rep.section("(a) Sensitivity to a single observation")
    rep.say("Moments recomputed after dropping the one observation furthest from the mean.")
    rep.say("Use this if one of the numbers above looks out of place.\n")
    rows = []
    for c in cols:
        x = df[c].to_numpy()
        i = int(np.argmax(np.abs(x - x.mean())))
        kept = np.delete(x, i)
        full, drop = moments(x), moments(kept)
        rows.append({"series": c, "dropped value": x[i], "obs #": i + 1,
                     "excess kurt (all)": full["excess kurtosis"],
                     "excess kurt (dropped)": drop["excess kurtosis"],
                     "skew (all)": full["skewness"], "skew (dropped)": drop["skewness"]})
    rep.say(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # ---------------------------------------------------------------- (b)
    # Ranks scaled into (0,1): u = rank / (n+1). Tied values (none expected in
    # continuous returns) would share the average rank.
    ranks = df.rank(method="average")
    u = ranks / (n + 1)

    pairs = list(itertools.combinations(cols, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.6))
    for ax, (p, q) in zip(axes, pairs):
        ax.scatter(u[p], u[q], s=6, alpha=0.5, color="steelblue")
        ax.set_xlabel(f"{p} (rank / (n+1))"); ax.set_ylabel(f"{q} (rank / (n+1))")
        ax.set_title(f"{p} vs {q}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.suptitle("Problem 4: pairs after the rank transform")
    fig.tight_layout()
    out = FIGURES_DIR / "problem4_rank_pairs.png"
    fig.savefig(out, dpi=150)
    rep.say(f"\nSaved plot to {out.relative_to(DATA_DIR)}")

    rep.section("(b) Days both series sit in their own tail")
    # "Own worst 2.5%" = the series' lowest 2.5% of days; with n = 1,000 that is
    # the 25 lowest ranks. Best 2.5% = the 25 highest ranks.
    k = int(round(TAIL * n))
    worst = ranks <= k
    best = ranks > n - k
    expected = n * TAIL * TAIL
    rep.say(f"Tail size: {k} days ({TAIL:.1%} of {n}) per series.")
    rep.say(f"Expected joint-tail days under independence: {n} x {TAIL} x {TAIL} = {expected:.3f}\n")
    rep.say(f"{'pair':<10}{'both worst 2.5%':>18}{'both best 2.5%':>18}")
    for p, q in pairs:
        rep.say(f"{p + '-' + q:<10}{int((worst[p] & worst[q]).sum()):>18}{int((best[p] & best[q]).sum()):>18}")

    rep.save("problem4_predict.txt")


if __name__ == "__main__":
    main()
