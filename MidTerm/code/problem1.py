"""Midterm Problem 1 -- Pairwise correlation with missing data.

    python problem1.py

Each section is labelled with the problem part it answers. Output is also
saved to output/problem1.txt.
"""

import numpy as np
import pandas as pd

from common import MIDTERM_DIR, Report, rl

rep = Report()
df = pd.read_csv(MIDTERM_DIR / "problem1.csv")
names = list(df.columns)


def frame(m: np.ndarray, fmt: str = "{: .6f}") -> str:
    return pd.DataFrame(m, index=names, columns=names).to_string(float_format=fmt.format)


# ---------------------------------------------------------------------------
# Problem 1(a): pairwise correlation matrix
# Each entry uses every day on which both of its two series are observed.
# ---------------------------------------------------------------------------
rep.section("Problem 1(a): pairwise correlation matrix")
corr = rl.missing_cov(df, skip_miss=False, method="cor")
rep.say(frame(corr))

# ---------------------------------------------------------------------------
# Problem 1(b): variance of each series from every observation it has
# (pandas skips the blanks; divides by n - 1).
# ---------------------------------------------------------------------------
rep.section("Problem 1(b): variance of each series (all of its observations)")
var = df.var(ddof=1)
count = df.count()
rep.say(pd.DataFrame({"observations": count, "variance": var, "std dev": np.sqrt(var)})
        .to_string(float_format=lambda v: f"{v: .8f}"))

# ---------------------------------------------------------------------------
# Problem 1(c): is the correlation matrix PSD?
# Eigenvalues, and an attempt at a Cholesky factorization.
# ---------------------------------------------------------------------------
rep.section("Problem 1(c): PSD check on the pairwise correlation matrix")
eig = np.linalg.eigvalsh(corr)  # ascending
rep.say("eigenvalues (ascending): " + "  ".join(f"{v: .6f}" for v in eig))
rep.say(f"smallest eigenvalue = {eig[0]:.6e}")
try:
    np.linalg.cholesky(corr)
    rep.say("Cholesky: succeeded")
except np.linalg.LinAlgError as e:
    rep.say(f"Cholesky: FAILED ({e})")

# ---------------------------------------------------------------------------
# Problem 1(d): Higham's nearest PSD correlation matrix
# ---------------------------------------------------------------------------
rep.section("Problem 1(d): Higham nearest PSD correlation matrix")
fixed = rl.higham_nearest_psd(corr)
rep.say(frame(fixed))
eig_fixed = np.linalg.eigvalsh(fixed)
rep.say("\neigenvalues (ascending): " + "  ".join(f"{v: .6e}" for v in eig_fixed))
rep.say(f"smallest eigenvalue = {eig_fixed[0]:.6e}")

change = fixed - corr
rep.say("\nChange (Higham minus pairwise):")
rep.say(frame(change))

rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        rows.append({"pair": f"{names[i]}-{names[j]}", "pairwise": corr[i, j], "Higham": fixed[i, j],
                     "change": change[i, j], "abs change": abs(change[i, j])})
ranked = pd.DataFrame(rows).sort_values("abs change", ascending=False).drop(columns="abs change")
rep.say("\nPairs ranked by size of change:")
rep.say(ranked.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
rep.say(f"\nFrobenius distance, Higham vs pairwise = {np.linalg.norm(change, 'fro'):.6f}")

# ---------------------------------------------------------------------------
# Problem 1(e): covariance matrix = D R D with D = diag(standard deviations)
# Both correlation matrices are combined with the same variances.
# ---------------------------------------------------------------------------
rep.section("Problem 1(e): covariance matrices")
D = np.diag(np.sqrt(var.to_numpy()))
for label, R in (("pairwise correlation matrix", corr), ("Higham-repaired correlation matrix", fixed)):
    cov = D @ R @ D
    rep.say(f"\nCovariance from the {label}:")
    rep.say(frame(cov, "{: .4e}"))
    rep.say("eigenvalues (ascending): " + "  ".join(f"{v: .4e}" for v in np.linalg.eigvalsh(cov)))

rep.save("problem1.txt")
