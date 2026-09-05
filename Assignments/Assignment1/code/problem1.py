"""Problem 1 -- Reading the Shape of a Sample.

Computes the first four moments of problem1.csv, fits a Normal by matching
mean and variance, and compares the fitted Normal's 1% tail against the
sample's actual tail.
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, FIGURES_DIR, section

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem1.csv")
    x = df["x"].to_numpy()
    n = len(x)

    # Sample moments. bias=False applies the standard small-sample
    # correction to skewness and kurtosis (matplotlib/pandas/Excel-style),
    # rather than the plug-in (biased) estimator.
    mean = x.mean()
    var = x.var(ddof=1)
    skew = stats.skew(x, bias=False)
    exkurt = stats.kurtosis(x, fisher=True, bias=False)

    section("Sample moments")
    print(f"n            = {n}")
    print(f"mean         = {mean:.6f}")
    print(f"variance     = {var:.6f}")
    print(f"std dev      = {np.sqrt(var):.6f}")
    print(f"skewness     = {skew:.6f}")
    print(f"excess kurt. = {exkurt:.6f}")

    # Fit a Normal by moment matching: mu = sample mean, sigma = sample std.
    mu, sigma = mean, np.sqrt(var)
    fitted = stats.norm(loc=mu, scale=sigma)

    section("Fitted Normal (moment matching)")
    print(f"mu    = {mu:.6f}")
    print(f"sigma = {sigma:.6f}")

    # How many observations fall below the fitted Normal's 1% quantile?
    q01 = fitted.ppf(0.01)
    n_below = int(np.sum(x < q01))
    n_expected = 0.01 * n

    section("Left-tail check at the 1% quantile")
    print(f"Normal's 1% quantile        = {q01:.6f}")
    print(f"observations below it       = {n_below}")
    print(f"expected under the Normal   = {n_expected:.2f}")

    # Plots to support the "Predict" step: histogram vs fitted Normal, and
    # a QQ-plot against the fitted Normal to show where the tails deviate.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(x, bins=40, density=True, alpha=0.6, color="steelblue", label="sample")
    grid = np.linspace(x.min(), x.max(), 400)
    axes[0].plot(grid, fitted.pdf(grid), color="crimson", lw=2, label="fitted Normal")
    axes[0].axvline(q01, color="black", ls="--", lw=1, label="Normal 1% quantile")
    axes[0].set_title("Histogram vs. fitted Normal")
    axes[0].set_xlabel("x")
    axes[0].legend()

    stats.probplot(x, dist=fitted, plot=axes[1])
    axes[1].set_title("QQ-plot against fitted Normal")

    fig.tight_layout()
    out_path = FIGURES_DIR / "problem1_fit.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
