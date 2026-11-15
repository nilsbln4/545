"""Problem 3 -- Reconcile step, part (g): what delta-normal misses, and the delta-gamma fix.

Uses the same simulated 10 day prices as problem3_fit.py (same seed). For each
simulated price change dS, the book's P&L is approximated by

    delta only        : delta dS
    delta + gamma     : delta dS + 1/2 gamma dS^2
    delta+gamma+theta : delta dS + 1/2 gamma dS^2 + theta dt,   dt = 10 / 255 years

with the book Greeks at today's price, and compared with full revaluation.
VaR and ES at 5% and 1% for each, under the normal and the t.

Every number is printed in a labelled block and saved to output/problem3_reconcile.txt.
"""

import numpy as np
import pandas as pd

from common import P3_DAYS_PER_YEAR, P3_HORIZON, P3_RATE, Report, p3_positions, p3_spot
from problem3_fit import BOOKS, book_pnl, load_returns, simulate_daily_returns

import risk545 as rl


def book_greeks(pos: pd.DataFrame, spot: float) -> pd.DataFrame:
    rows = []
    for p in pos.itertuples():
        if p.Type == "Stock":
            rows.append((p.Portfolio, p.Holding, 0.0, 0.0))
        else:
            g = rl.gbsm(p.Type == "Call", spot, p.Strike, p.DaysToExpiry / P3_DAYS_PER_YEAR, P3_RATE, P3_RATE,
                        p.ImpliedVol, include_greeks=True)
            rows.append((p.Portfolio, p.Holding * g.delta, p.Holding * g.gamma, p.Holding * g.theta))
    df = pd.DataFrame(rows, columns=["Portfolio", "delta", "gamma", "theta_yr"])
    return df.groupby("Portfolio").sum().reindex(BOOKS)


def risk(x: np.ndarray) -> tuple:
    return (rl.value_at_risk(x, 0.05), rl.expected_shortfall(x, 0.05),
            rl.value_at_risk(x, 0.01), rl.expected_shortfall(x, 0.01))


def main() -> None:
    rep = Report()
    spot = p3_spot()
    pos = p3_positions()
    ret = load_returns()
    sims = simulate_daily_returns(rl.fit_normal(ret), rl.fit_general_t(ret))
    greeks = book_greeks(pos, spot)
    dt = P3_HORIZON / P3_DAYS_PER_YEAR

    rep.section("(g) Book Greeks at today's price")
    rep.say(f"S = {spot}, dt = {P3_HORIZON}/{P3_DAYS_PER_YEAR} = {dt:.6f} years; theta x dt = 10 day time decay\n")
    g = greeks.copy()
    g["theta x dt"] = g["theta_yr"] * dt
    rep.say(g.to_string(float_format=lambda v: f"{v:12.4f}"))

    for name, r in sims.items():
        ds = spot * np.prod(1 + r, axis=1) - spot
        full = book_pnl(pos, spot, spot + ds)
        rep.section(f"(g) Approximations vs full revaluation -- {name} returns")
        rep.say(f"{'book':<16} {'method':<20} {'mean P&L':>9} {'VaR 5%':>9} {'ES 5%':>9} {'VaR 1%':>9} {'ES 1%':>9}")
        for b in BOOKS:
            d, gm, th = greeks.loc[b, "delta"], greeks.loc[b, "gamma"], greeks.loc[b, "theta_yr"] * dt
            approx = {
                "delta only": d * ds,
                "delta + gamma": d * ds + 0.5 * gm * ds**2,
                "delta + gamma + theta": d * ds + 0.5 * gm * ds**2 + th,
                "full revaluation": full[b].to_numpy(),
            }
            for method, x in approx.items():
                v5, e5, v1, e1 = risk(x)
                rep.say(f"{b:<16} {method:<20} {x.mean():9.2f} {v5:9.2f} {e5:9.2f} {v1:9.2f} {e1:9.2f}")
            rep.say()

    rep.save("problem3_reconcile.txt")


if __name__ == "__main__":
    main()
