"""Problem 3 -- Predict step: P&L plots, moments, and tail counts.

Two choices for $2,000,000:
  concentrated: all of it in bond A
  diversified:  $1,000,000 in A and $1,000,000 in B
Prints the first four moments of each bond (part a) and the counts of scenarios
with a loss over 20% (part b). No VaR or ES is computed here.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, FIGURES_DIR, section

MILLION = 1_000_000
LOSS_THRESHOLD = -0.20


def moments(x: np.ndarray) -> dict:
    # Conventions follow the instructor's Julia code: std() divides by n-1, and
    # StatsBase's skewness()/kurtosis() are the population-style estimators
    # (scipy's bias=True).
    return {"mean": x.mean(), "SD": x.std(ddof=1),
            "skewness": stats.skew(x, bias=True),
            "excess kurtosis": stats.kurtosis(x, fisher=True, bias=True),
            "min": x.min(), "max": x.max()}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem3.csv")
    a, b = df["A"].to_numpy(), df["B"].to_numpy()
    n = len(df)

    # ---- P&L of the two choices --------------------------------------
    pnl_concentrated = 2 * MILLION * a                      # $2,000,000 in A
    pnl_diversified = MILLION * a + MILLION * b             # $1,000,000 in each

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    # Dollar signs are escaped: matplotlib treats a pair of bare $ as math mode.
    for ax, pnl, title in ((axes[0], pnl_concentrated, r"\$2,000,000 in A"),
                           (axes[1], pnl_diversified, r"\$1,000,000 in A + \$1,000,000 in B")):
        ax.hist(pnl, bins=100, color="steelblue")
        ax.set_yscale("log")  # the default losses are rare; a log axis keeps them visible
        ax.set_title(title)
        ax.set_xlabel(r"one-year P&L (\$)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("number of scenarios (log scale)")
    fig.suptitle(f"Problem 3: P&L distributions, {n:,} scenarios")
    fig.tight_layout()
    out = FIGURES_DIR / "problem3_pnl.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out.relative_to(DATA_DIR)}")

    # ---- (a) moments ---------------------------------------------------
    section("(a) First four moments of each bond's return")
    table = pd.DataFrame({"A": moments(a), "B": moments(b)})
    print(table.to_string(float_format=lambda v: f"{v: .6f}"))

    # ---- (b) tail counts -----------------------------------------------
    section(f"(b) Scenarios with a loss of more than {abs(LOSS_THRESHOLD):.0%}")
    loss_a, loss_b = a < LOSS_THRESHOLD, b < LOSS_THRESHOLD
    counts = {
        "A loses more than 20%": loss_a.sum(),
        "B loses more than 20%": loss_b.sum(),
        "at least one of the two": (loss_a | loss_b).sum(),
        "both": (loss_a & loss_b).sum(),
    }
    for label, c in counts.items():
        print(f"{label:<28}{int(c):>6}  ({c / n:.2%} of {n:,})")


if __name__ == "__main__":
    main()
