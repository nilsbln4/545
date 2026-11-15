"""Problem 3 -- Predict step: each book's delta, gamma and theta.

Part (a) asks for the Greeks before anything is simulated. Each option is valued
with GBSM (b = r, no dividends) at the last ABC price, T = DaysToExpiry / 255.
Book Greeks are holding-weighted sums; the stock has delta 1, gamma 0, theta 0.

For scale only, the size of a typical 10 day price move is taken from the options'
own implied vol (S * IV * sqrt(10/255)), not from the price history, which is
part (c). The three terms of dP ~ delta dS + 1/2 gamma dS^2 + theta dt are then
shown at that move size.

Every number is printed in a labelled block and saved to output/problem3_predict.txt.
"""

import numpy as np
import pandas as pd

from common import P3_DAYS_PER_YEAR, P3_HORIZON, P3_RATE, Report, p3_positions, p3_spot

import risk545 as rl


def main() -> None:
    rep = Report()
    spot = p3_spot()
    pos = p3_positions()

    rep.section("Inputs")
    rep.say(f"S = {spot} (last price)   r = {P3_RATE}   no dividends (b = r)   "
            f"{P3_DAYS_PER_YEAR} trading days a year   horizon {P3_HORIZON} days")

    # ---------------------------------------------------------------- per position
    rows = []
    for _, p in pos.iterrows():
        if p.Type == "Stock":
            value, delta, gamma, theta = spot, 1.0, 0.0, 0.0
        else:
            ttm = p.DaysToExpiry / P3_DAYS_PER_YEAR
            g = rl.gbsm(p.Type == "Call", spot, p.Strike, ttm, P3_RATE, P3_RATE, p.ImpliedVol, include_greeks=True)
            value, delta, gamma, theta = g.value, g.delta, g.gamma, g.theta
        rows.append({"Portfolio": p.Portfolio, "Type": p.Type, "Strike": p.Strike, "IV": p.ImpliedVol,
                     "Holding": p.Holding, "Unit value": value, "Unit delta": delta, "Unit gamma": gamma,
                     "Unit theta/yr": theta,
                     "Value": p.Holding * value, "Delta": p.Holding * delta, "Gamma": p.Holding * gamma,
                     "Theta/day": p.Holding * theta / P3_DAYS_PER_YEAR})
    df = pd.DataFrame(rows)

    rep.section("Per position (unit = one share or one option; position = unit x holding)")
    rep.say("Theta from GBSM is per year; Theta/day = theta / 255 x holding.\n")
    rep.say(df.to_string(index=False, float_format=lambda v: f"{v:10.4f}", na_rep="  -"))

    # ---------------------------------------------------------------- (a) book Greeks
    book = df.groupby("Portfolio", sort=False)[["Value", "Delta", "Gamma", "Theta/day"]].sum()
    book[f"Theta x {P3_HORIZON}d"] = book["Theta/day"] * P3_HORIZON
    rep.section("(a) Book Greeks (delta per $1 of ABC, gamma per $1^2, theta in $ per trading day)")
    rep.say(book.to_string(float_format=lambda v: f"{v:12.4f}"))

    # ---------------------------------------------------------------- scale
    dt = P3_HORIZON / P3_DAYS_PER_YEAR
    iv = pos["ImpliedVol"].dropna()
    move = spot * iv.mean() * np.sqrt(dt)
    rep.section("Scale of a 10 day move (from implied vol, for intuition only)")
    rep.say(f"average IV of the options = {iv.mean():.4f};  dS ~ S x IV x sqrt(10/255) = {move:.4f}\n")
    rep.say("Terms of dP ~ delta dS + 1/2 gamma dS^2 + theta dt at that dS (dollars):")
    terms = pd.DataFrame({"delta x dS": book["Delta"] * move,
                          "1/2 gamma dS^2": 0.5 * book["Gamma"] * move**2,
                          f"theta x {P3_HORIZON}d": book[f"Theta x {P3_HORIZON}d"]})
    rep.say(terms.to_string(float_format=lambda v: f"{v:14.2f}"))

    rep.save("problem3_predict.txt")


if __name__ == "__main__":
    main()
