"""Problem 1 -- Fit and Reconcile numbers for parts (d)-(i).

Compares the complete-case and pairwise correlation matrices, shows what the
pairwise one does to a tracking portfolio, and repairs it two ways.

Every number the write-up needs is printed in a labelled block and also saved
to output/problem1_fit.txt.
"""

import numpy as np
import pandas as pd

from common import DATA_DIR

import risk545 as rl

ASSETS = ["A", "B", "C", "D", "IDX"]

# IDX is built from the stocks with these weights (plus a little tracking
# error). The tracking portfolio is long 1 unit of IDX and short that basket.
BASKET = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}
TRACKING_WEIGHTS = np.array([-BASKET["A"], -BASKET["B"], -BASKET["C"], -BASKET["D"], 1.0])

_lines: list = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    say(f"\n{title}\n{'=' * len(title)}")


def show_matrix(title: str, m: np.ndarray) -> None:
    say(f"\n{title}")
    say(pd.DataFrame(m, index=ASSETS, columns=ASSETS).to_string(float_format=lambda v: f"{v: .6f}"))


def show_eigenvalues(title: str, m: np.ndarray) -> float:
    # eigvalsh: the matrix is symmetric, and it returns real eigenvalues in
    # ascending order, so the first one is the smallest.
    vals = np.linalg.eigvalsh(m)
    say(f"{title}: " + "  ".join(f"{v: .6f}" for v in vals))
    say(f"  smallest eigenvalue = {vals[0]:.4e}")
    return vals[0]


def try_cholesky(title: str, m: np.ndarray) -> None:
    try:
        np.linalg.cholesky(m)
        say(f"Cholesky of {title}: succeeded")
    except np.linalg.LinAlgError as e:
        say(f"Cholesky of {title}: FAILED ({e})")


def covariance_from(corr: np.ndarray, sd: np.ndarray) -> np.ndarray:
    # Sigma = D R D with D = diag(sd).
    return np.diag(sd) @ corr @ np.diag(sd)


def tracking_variance(cov: np.ndarray) -> float:
    return float(TRACKING_WEIGHTS @ cov @ TRACKING_WEIGHTS)


def frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro"))


def top_changes(change: np.ndarray, joint_days: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Largest absolute changes among the unique off-diagonal pairs, next to
    the number of days that pair is jointly observed."""
    rows = []
    for i in range(len(ASSETS)):
        for j in range(i + 1, len(ASSETS)):
            rows.append({"pair": f"{ASSETS[i]}-{ASSETS[j]}", "change": change[i, j],
                         "abs_change": abs(change[i, j]),
                         "joint_days": int(joint_days.iloc[i, j])})
    return pd.DataFrame(rows).sort_values("abs_change", ascending=False).head(n)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem1.csv")[ASSETS]
    observed = df.notna().astype(int)
    joint_days = observed.T @ observed

    # ---------------------------------------------------------------- (d)
    section("(d) Complete-case vs pairwise correlation matrices")
    # Complete case: drop any day on which any series is missing (81 rows).
    # Pairwise: each entry uses every day both of its series are observed.
    complete = rl.missing_cov(df, skip_miss=True, method="cor")
    pairwise = rl.missing_cov(df, skip_miss=False, method="cor")

    show_matrix("Complete-case correlation matrix (81 days)", complete)
    show_matrix("Pairwise correlation matrix", pairwise)

    say("\nEigenvalues (ascending):")
    show_eigenvalues("complete case", complete)
    show_eigenvalues("pairwise     ", pairwise)

    say()
    try_cholesky("the complete-case matrix", complete)
    try_cholesky("the pairwise matrix", pairwise)

    # ---------------------------------------------------------------- (e)
    section("(e) Covariance from pairwise correlations and full-history standard deviations")
    # "Full history" = every available observation of each series on its own
    # (pandas skips the blanks), not just the days where everything trades.
    sd = df.std(ddof=1).to_numpy()
    say("Full-history standard deviations: " + "  ".join(f"{a}={s:.6f}" for a, s in zip(ASSETS, sd)))

    cov_pairwise = covariance_from(pairwise, sd)
    show_matrix("Covariance matrix (pairwise correlations x full-history SDs)", cov_pairwise)

    var_pairwise = tracking_variance(cov_pairwise)
    say("\nTracking portfolio: long 1 IDX, short 0.40 A, 0.30 B, 0.20 C, 0.10 D")
    say(f"Tracking portfolio variance (pairwise matrix) = {var_pairwise:.6e}")
    if var_pairwise < 0:
        say("  -> NEGATIVE: a variance below zero, which no real portfolio can have.")

    # For (i): the same quantity under the complete-case correlations.
    var_complete = tracking_variance(covariance_from(complete, sd))
    say(f"Tracking portfolio variance (complete-case correlations, same SDs) = {var_complete:.6e}")

    # ---------------------------------------------------------------- (f)
    section("(f) Repairing the pairwise correlation matrix")
    repairs = {
        "Rebonato-Jackel (near_psd)": rl.near_psd(pairwise),
        "Higham": rl.higham_nearest_psd(pairwise),
    }

    summary = []
    for name, fixed in repairs.items():
        say(f"\n--- {name} ---")
        show_matrix("Repaired correlation matrix", fixed)
        smallest = show_eigenvalues("eigenvalues", fixed)
        dist = frobenius(fixed, pairwise)
        var_fixed = tracking_variance(covariance_from(fixed, sd))
        say(f"Frobenius distance from pairwise matrix = {dist:.6f}")
        say(f"Tracking portfolio variance (repaired matrix) = {var_fixed:.6e}")
        summary.append({"method": name, "smallest_eigenvalue": smallest,
                        "frobenius_from_pairwise": dist, "tracking_variance": var_fixed})

    say("\nSummary (f):")
    say(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    # ---------------------------------------------------------------- (h)
    section("(h) Which entries did each repair move?")
    for name, fixed in repairs.items():
        change = fixed - pairwise
        show_matrix(f"{name}: repaired minus pairwise", change)
        say(f"\n{name}: pairs ranked by size of change, with the days each pair is jointly observed")
        say(top_changes(change, joint_days).drop(columns="abs_change")
            .to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # ---------------------------------------------------------------- (i)
    section("(i) Size of the estimator gap vs size of the repairs (Frobenius distances)")
    gap = frobenius(complete, pairwise)
    say(f"complete case vs pairwise        = {gap:.6f}")
    for name, fixed in repairs.items():
        say(f"{name:<28} repair size = {frobenius(fixed, pairwise):.6f}")

    out_path = DATA_DIR / "output" / "problem1_fit.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\nSaved to {out_path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
