"""Problem 1 -- Predict step: put-call parity on the mids, and the quotes in the wings.

Part (a): at every strike, the parity gap computed from the mid prices, next to
the band it must exceed to be tradeable (the call's spread plus the put's
spread). Also prints each quote's spread as a share of its mid, to look at the
far wings before answering part (c). No implied volatility is computed here.

Every number is printed in a labelled block and saved to output/problem1_predict.txt.
"""

import numpy as np

from common import P1_DIV_YIELD, P1_EXPIRY, P1_RATE, P1_SPOT, P1_T, P1_TODAY, Report, p1_chain


def main() -> None:
    rep = Report()
    chain = p1_chain()

    rep.section("Inputs")
    rep.say(f"S = {P1_SPOT}   r = {P1_RATE}   q = {P1_DIV_YIELD}")
    rep.say(f"{P1_TODAY} to {P1_EXPIRY}: {(P1_EXPIRY - P1_TODAY).days} calendar days, T = days / 365 = {P1_T:.6f}")

    # ---------------------------------------------------------------- (a)
    # Put-call parity for European options on a stock with a continuous
    # dividend yield q:  C - P = S e^{-qT} - K e^{-rT}.
    # Gap = (C_mid - P_mid) - (S e^{-qT} - K e^{-rT}).
    forward_leg = P1_SPOT * np.exp(-P1_DIV_YIELD * P1_T)
    chain["PV(K)"] = chain.index * np.exp(-P1_RATE * P1_T)
    chain["C-P (mids)"] = chain["C_Mid"] - chain["P_Mid"]
    chain["parity value"] = forward_leg - chain["PV(K)"]
    chain["gap"] = chain["C-P (mids)"] - chain["parity value"]
    chain["band"] = chain["C_Spread"] + chain["P_Spread"]
    chain["flag"] = np.where(chain["gap"].abs() > chain["band"], "FLAG", "")

    rep.section("(a) Put-call parity at each strike, from the mids")
    rep.say(f"S e^(-qT) = {forward_leg:.6f}")
    rep.say("gap = (C_mid - P_mid) - (S e^(-qT) - K e^(-rT));  band = call spread + put spread;"
            "  flagged when |gap| > band\n")
    cols = ["C_Mid", "P_Mid", "C-P (mids)", "parity value", "gap", "band", "flag"]
    rep.say(chain[cols].to_string(float_format=lambda v: f"{v: .4f}"))

    flagged = chain.index[chain["flag"] == "FLAG"].tolist()
    rep.say(f"\nStrikes flagged: {flagged if flagged else 'none'}")

    # ---------------------------------------------------------------- (c)
    rep.section("(c) The quotes: bid, ask, spread, and spread as a share of the mid")
    view = chain[["C_Bid", "C_Ask", "C_Spread", "P_Bid", "P_Ask", "P_Spread"]].copy()
    view["C_Spread/Mid"] = chain["C_Spread"] / chain["C_Mid"]
    view["P_Spread/Mid"] = chain["P_Spread"] / chain["P_Mid"]
    rep.say(view.to_string(formatters={
        **{c: "{:>8.2f}".format for c in ("C_Bid", "C_Ask", "C_Spread", "P_Bid", "P_Ask", "P_Spread")},
        "C_Spread/Mid": "{:>12.1%}".format, "P_Spread/Mid": "{:>12.1%}".format}))

    rep.save("problem1_predict.txt")


if __name__ == "__main__":
    main()
