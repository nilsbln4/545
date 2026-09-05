"""Problem 4 -- Conditional Distributions.

Uses the partitioned-covariance result for the bivariate Normal to get the
conditional mean and variance of x2 given x1, plots a 95% band around the
conditional mean, and checks its coverage overall and by distance from
mean(x1).
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, FIGURES_DIR, section

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Block convention: block 1 = x1 (the conditioning variable), block 2 = x2
# (the variable being predicted). Sigma = [[S11, S12], [S21, S22]].
Z_95 = stats.norm.ppf(0.975)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem4.csv")
    x1, x2 = df["x1"].to_numpy(), df["x2"].to_numpy()
    n = len(x1)

    Sigma = np.cov(x1, x2, ddof=1)
    S11, S12 = Sigma[0, 0], Sigma[0, 1]
    S21, S22 = Sigma[1, 0], Sigma[1, 1]
    mu1, mu2 = x1.mean(), x2.mean()

    section("Sample covariance matrix (block 1 = x1, block 2 = x2)")
    print(f"S11 = Var(x1)  = {S11:.6f}")
    print(f"S22 = Var(x2)  = {S22:.6f}")
    print(f"S12 = S21 = Cov(x1,x2) = {S12:.6f}")

    # Conditional variance: Var(x2|x1) = S22 - S21 * S11^-1 * S12
    cond_var = S22 - S21 * (1.0 / S11) * S12
    factor = cond_var / S22  # = 1 - rho^2, and does not depend on the x1 value observed
    rho = S12 / np.sqrt(S11 * S22)

    section("(a) Conditional variance of x2 given x1")
    print("Formula: Var(x2|x1) = S22 - S21 * S11^-1 * S12")
    print(f"Var(x2|x1)                = {cond_var:.6f}")
    print(f"Var(x2) (unconditional)   = {S22:.6f}")
    print(f"factor = Var(x2|x1)/Var(x2) = 1 - rho^2 = {factor:.6f}  (rho = {rho:.6f})")

    section("(b) Does the factor depend on the observed x1?")
    print("No: S21 * S11^-1 * S12 has no x1 term in it, so the conditional")
    print("variance -- and hence the factor above -- is the same for every x1.")

    # Conditional mean: E[x2|x1] = mu2 + S21 * S11^-1 * (x1 - mu1)
    beta = S21 / S11  # identical to the OLS slope of x2 on x1
    alpha = mu2 - beta * mu1

    section("(c) Conditional mean of x2 given x1")
    print("Formula: E(x2|x1) = mu2 + S21 * S11^-1 * (x1 - mu1)")
    print(f"slope on x1 (= S21/S11, the OLS slope of x2~x1) = {beta:.6f}")
    print(f"intercept implied                                = {alpha:.6f}")

    grid = np.linspace(x1.min(), x1.max(), 200)
    cond_mean_grid = alpha + beta * grid
    half_width = Z_95 * np.sqrt(cond_var)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(x1, x2, s=10, alpha=0.4, color="steelblue", edgecolor="none", label="data")
    ax.plot(grid, cond_mean_grid, color="crimson", lw=2, label="E[x2|x1]")
    ax.fill_between(grid, cond_mean_grid - half_width, cond_mean_grid + half_width,
                     color="crimson", alpha=0.15, label="95% band")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Conditional mean of x2 given x1, with 95% band")
    ax.legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / "problem4_conditional.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path.relative_to(DATA_DIR)}")

    # (d) Coverage: fraction of points inside the band at their own x1.
    cond_mean_at_x1 = alpha + beta * x1
    lower, upper = cond_mean_at_x1 - half_width, cond_mean_at_x1 + half_width
    inside = (x2 >= lower) & (x2 <= upper)
    coverage_all = inside.mean()

    section("(d) Overall coverage of the 95% band")
    print(f"fraction inside band = {coverage_all:.4f}  ({inside.sum()} / {n})")

    # (e) Coverage split by |x1 - mean(x1)| / std(x1) buckets.
    z1 = np.abs(x1 - mu1) / np.sqrt(S11)
    buckets = [
        ("within 1 sd", z1 <= 1),
        ("1 to 2 sd", (z1 > 1) & (z1 <= 2)),
        ("beyond 2 sd", z1 > 2),
    ]

    section("(e) Coverage by distance of x1 from its mean")
    print(f"{'bucket':<14}{'n':>6}{'coverage':>12}")
    for label, mask in buckets:
        cov = inside[mask].mean() if mask.sum() > 0 else float("nan")
        print(f"{label:<14}{mask.sum():>6}{cov:>12.4f}")


if __name__ == "__main__":
    main()
