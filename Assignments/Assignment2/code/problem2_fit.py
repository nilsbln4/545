"""Problem 2 -- Fit and Reconcile numbers for parts (d), (f) and (g).

Five one-day 5% VaR estimates for a $1,000,000 position, side by side:
  1. Normal, equally weighted SD
  2. Normal, exponentially weighted SD, lambda = 0.97
  3. Normal, exponentially weighted SD, lambda = 0.94
  4. Student's t fitted by maximum likelihood
  5. Historical simulation over the full sample

Part (f) needs the split between the calm and the different recent days that
you chose from the plot in part (b):

    python problem2_fit.py --recent-days 35

Every number is printed in a labelled block and saved to output/problem2_fit.txt.
"""

import argparse

import numpy as np
from scipy import stats

from common import DATA_DIR, ew_weights, problem2_returns

import risk545 as rl

POSITION = 1_000_000
ALPHA = 0.05

_lines: list = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    say(f"\n{title}\n{'=' * len(title)}")


def ew_sd(r: np.ndarray, lam: float) -> float:
    """EW volatility the way the instructor's code computes it (Week 4 homework:
    sqrt(ewCovar(...)[1,1])). ew_cov removes the *weighted* mean before
    squaring, which is the course's covariance convention."""
    return float(np.sqrt(rl.ew_cov(r[:, None], lam)[0, 0]))


def ew_sd_zero_mean(r: np.ndarray, lam: float) -> float:
    """Variant that treats the already-removed sample mean as xbar (Week 3's
    formula read literally): a weighted mean of squared returns. Shown only as
    a footnote."""
    return float(np.sqrt(ew_weights(lam, len(r)) @ r ** 2))


def regime_stats(x: np.ndarray) -> dict:
    return {"n": len(x), "SD": x.std(ddof=1),
            # Population-style moments, as in the instructor's StatsBase code.
            "skewness": stats.skew(x, bias=True),
            "excess kurtosis": stats.kurtosis(x, fisher=True, bias=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-days", type=int, default=None,
                        help="number of recent days that look different (from your plot in part b)")
    args = parser.parse_args()

    r, _ = problem2_returns()
    n = len(r)

    # ---------------------------------------------------------------- (d)
    section(f"(d) One-day {ALPHA:.0%} VaR on ${POSITION:,}, five ways")

    # Equally weighted SD: the ordinary sample SD (divides by n-1), as in the
    # instructor's Week 4 homework (std(meta)).
    sd_equal = float(r.std(ddof=1))
    sd_97 = ew_sd(r, 0.97)
    sd_94 = ew_sd(r, 0.94)

    t_fit = rl.fit_general_t(r)
    t_mu, t_sigma, t_nu = (t_fit.params[k] for k in ("mu", "sigma", "nu"))

    t_implied_sd = t_sigma * np.sqrt(t_nu / (t_nu - 2))

    # VaR is reported as a positive loss, as the assignment specifies.
    # The third entry is the SD shown in the table; for the t it is the SD
    # implied by its fitted parameters (the t's VaR itself uses scale and nu).
    estimates = [
        ("1. Normal, equally weighted SD", f"{sd_equal:.6f}", rl.value_at_risk(stats.norm(0, sd_equal), ALPHA)),
        ("2. Normal, EW SD, lambda = 0.97", f"{sd_97:.6f}", rl.value_at_risk(stats.norm(0, sd_97), ALPHA)),
        ("3. Normal, EW SD, lambda = 0.94", f"{sd_94:.6f}", rl.value_at_risk(stats.norm(0, sd_94), ALPHA)),
        ("4. Student's t (MLE)", f"{t_implied_sd:.6f} *", rl.value_at_risk(t_fit.error_model, ALPHA)),
        ("5. Historical simulation", "--", rl.value_at_risk(r, ALPHA)),
    ]

    say(f"{'estimate':<36}{'SD of the model':>17}{'VaR ($)':>14}{'VaR (% of position)':>22}")
    for name, sd_txt, var in estimates:
        say(f"{name:<36}{sd_txt:>17}{var * POSITION:>14,.0f}{var:>22.4%}")
    say("* implied SD of the fitted t; its VaR is computed from the fitted t's parameters below, not from this SD.")
    say("-- historical simulation assumes no distribution, so there is no model SD.")

    order = sorted(range(5), key=lambda i: estimates[i][2])
    say("\nSmallest to largest: " + " < ".join(str(i + 1) for i in order))

    say("\nFitted Student's t parameters (full sample, mean-removed returns):")
    say(f"  location mu                    = {t_mu:.4e}   (effectively 0)")
    say(f"  scale sigma                    = {t_sigma:.6f}")
    say(f"  degrees of freedom nu          = {t_nu:.4f}")
    say(f"  implied SD = sigma*sqrt(nu/(nu-2)) = {t_implied_sd:.6f}")

    say("\nConvention note: the EW SDs use ew_cov, which removes the *weighted* mean (the instructor's")
    say("ewCovar). Reading Week 3's formula literally, with the sample mean already removed as xbar,")
    say("gives slightly different numbers; for reference:")
    for lam, sd in ((0.97, sd_97), (0.94, sd_94)):
        alt = ew_sd_zero_mean(r, lam)
        alt_var = rl.value_at_risk(stats.norm(0, alt), ALPHA) * POSITION
        say(f"  lambda = {lam}: SD {alt:.6f}  VaR ${alt_var:,.0f}  (vs ${rl.value_at_risk(stats.norm(0, sd), ALPHA) * POSITION:,.0f} above)")

    # ---------------------------------------------------------------- (f)
    section("(f) Volatility inside each regime")
    if args.recent_days is None:
        say("Pass --recent-days K (the number of recent days that look different) to fill this in.")
    else:
        k = args.recent_days
        calm, wild = r[: n - k], r[n - k:]
        say(f"Split: first {n - k} returns (calm) | last {k} returns (recent)\n")
        say(f"{'':<18}{'n':>6}{'SD':>12}{'skewness':>12}{'excess kurt.':>14}")
        for label, x in (("calm", calm), ("recent", wild), ("full sample", r)):
            s = regime_stats(x)
            say(f"{label:<18}{s['n']:>6}{s['SD']:>12.6f}{s['skewness']:>12.4f}{s['excess kurtosis']:>14.4f}")
        say(f"\nratio of SDs (recent / calm) = {wild.std(ddof=1) / calm.std(ddof=1):.3f}")

    # ---------------------------------------------------------------- (g)
    section("(g) Is the gap between the two EW VaRs bigger than the estimation noise?")
    # Week 3: standard error of a volatility estimate ~ sigma / sqrt(2 n_eff),
    # with n_eff = (1+lambda)/(1-lambda).
    say(f"{'lambda':>8}{'n_eff':>9}{'SE of vol (relative)':>23}{'SD':>12}{'SE of SD':>12}")
    for lam, sd in ((0.97, sd_97), (0.94, sd_94)):
        n_eff = (1 + lam) / (1 - lam)
        rel = 1 / np.sqrt(2 * n_eff)
        say(f"{lam:>8.2f}{n_eff:>9.2f}{rel:>23.2%}{sd:>12.6f}{sd * rel:>12.6f}")

    var_97 = rl.value_at_risk(stats.norm(0, sd_97), ALPHA) * POSITION
    var_94 = rl.value_at_risk(stats.norm(0, sd_94), ALPHA) * POSITION
    say(f"\nVaR at lambda = 0.97: ${var_97:,.0f}")
    say(f"VaR at lambda = 0.94: ${var_94:,.0f}")
    say(f"gap = ${var_94 - var_97:,.0f}  ({var_94 / var_97 - 1:.2%} of the lambda = 0.97 VaR)")

    out_path = DATA_DIR / "output" / "problem2_fit.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\nSaved to {out_path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
