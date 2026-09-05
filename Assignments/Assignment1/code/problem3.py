"""Problem 3 -- Pearson Against Spearman.

Plots every pair of columns in problem3.csv, computes the Pearson and
Spearman correlation matrices, and reports the pair with the largest gap
between the two.
"""

import itertools

import numpy as np
import pandas as pd

from common import DATA_DIR, FIGURES_DIR, section

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem3.csv")
    cols = list(df.columns)

    fig, axes = plt.subplots(len(cols), len(cols), figsize=(10, 10))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[ci], bins=25, color="steelblue")
            else:
                ax.scatter(df[cj], df[ci], s=6, alpha=0.4, color="steelblue", edgecolor="none")
            if i == len(cols) - 1:
                ax.set_xlabel(cj)
            if j == 0:
                ax.set_ylabel(ci)
    fig.suptitle("problem3.csv: every pair")
    fig.tight_layout()
    out_path = FIGURES_DIR / "problem3_pairs.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved pairwise plot to {out_path.relative_to(DATA_DIR)}")

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    section("Pearson correlation matrix")
    print(pearson.round(4))

    section("Spearman correlation matrix")
    print(spearman.round(4))

    gap = (pearson - spearman).abs()
    section("Absolute gap |Pearson - Spearman|")
    print(gap.round(4))

    best_pair, best_gap = None, -1.0
    for ci, cj in itertools.combinations(cols, 2):
        g = abs(pearson.loc[ci, cj] - spearman.loc[ci, cj])
        if g > best_gap:
            best_gap, best_pair = g, (ci, cj)

    section("Largest gap")
    ci, cj = best_pair
    print(f"pair            = ({ci}, {cj})")
    print(f"Pearson         = {pearson.loc[ci, cj]:.4f}")
    print(f"Spearman        = {spearman.loc[ci, cj]:.4f}")
    print(f"|gap|           = {best_gap:.4f}")


if __name__ == "__main__":
    main()
