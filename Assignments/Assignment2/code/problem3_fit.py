"""Problem 3 -- Fit and Reconcile numbers for parts (c)-(f) and the 1% results for (h).

Historical and Normal VaR / ES for four positions in bonds A and B:
  $1M in A, $1M in B, $2M in A (concentrated), $1M in each (diversified).

Every number is printed in a labelled block and saved to output/problem3_fit.txt.
"""

import pandas as pd
from scipy import stats

from common import DATA_DIR, Report

import risk545 as rl

MILLION = 1_000_000


def risk_table(positions: dict, alpha: float) -> pd.DataFrame:
    """Historical VaR and ES, and Normal VaR from each P&L's own mean and SD,
    in dollars (positive = loss)."""
    rows = []
    for name, pnl in positions.items():
        mean, sd = pnl.mean(), pnl.std(ddof=1)
        rows.append({
            "position": name,
            "hist VaR": rl.value_at_risk(pnl, alpha),
            "hist ES": rl.expected_shortfall(pnl, alpha),
            "Normal VaR": rl.value_at_risk(stats.norm(mean, sd), alpha),
            "P&L mean": mean,
            "P&L SD": sd,
        })
    return pd.DataFrame(rows).set_index("position")


def main() -> None:
    rep = Report()
    df = pd.read_csv(DATA_DIR / "problem3.csv")
    a, b = df["A"].to_numpy(), df["B"].to_numpy()

    positions = {
        "$1M in A": MILLION * a,
        "$1M in B": MILLION * b,
        "$2M in A (concentrated)": 2 * MILLION * a,
        "$1M in each (diversified)": MILLION * a + MILLION * b,
    }

    fmt = {col: (lambda v: f"{v:>14,.0f}") for col in ("hist VaR", "hist ES", "Normal VaR", "P&L mean", "P&L SD")}

    # ---------------------------------------------------------------- (c), (d)
    t5 = risk_table(positions, 0.05)
    rep.section("(c) Historical and (d) Normal VaR / ES at 5%, in dollars")
    rep.say("VaR and ES are positive for a loss; a negative VaR means the 5% mark is itself a gain.")
    rep.say("Normal VaR uses each P&L's own sample mean and SD (shown); hist ES is the average of the worst 5%.\n")
    rep.say(t5.to_string(formatters=fmt))

    # ---------------------------------------------------------------- (e)
    t1 = risk_table(positions, 0.01)
    rep.section("(e) The same at 1%, in dollars")
    rep.say(t1.to_string(formatters=fmt))

    # ---------------------------------------------------------------- (f)
    def subadditivity(table: pd.DataFrame, label: str) -> None:
        rep.say(f"\n{label}")
        rep.say(f"{'':<8}{'$1M A':>14}{'$1M B':>14}{'sum':>14}{'$1M each':>14}{'combined - sum':>17}")
        for col, name in (("hist VaR", "VaR"), ("hist ES", "ES")):
            va, vb = table.loc["$1M in A", col], table.loc["$1M in B", col]
            vc = table.loc["$1M in each (diversified)", col]
            rep.say(f"{name:<8}{va:>14,.0f}{vb:>14,.0f}{va + vb:>14,.0f}{vc:>14,.0f}{vc - (va + vb):>17,.0f}")
            verdict = "holds" if vc <= va + vb else "FAILS (combined risk exceeds the sum)"
            rep.say(f"        subadditivity {verdict}")

    rep.section("(f) Subadditivity check: risk(A + B) <= risk(A) + risk(B)?")
    subadditivity(t5, "At 5% (historical):")
    subadditivity(t1, "At 1% (historical), for part (h):")

    rep.save("problem3_fit.txt")


if __name__ == "__main__":
    main()
