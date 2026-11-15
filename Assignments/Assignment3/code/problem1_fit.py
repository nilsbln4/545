"""Problem 1 -- Fit step: implied volatilities, vega, and the IV sensitivity to one cent.

(d) The implied vol of every quote from its mid, bid and ask (calls and puts),
    using the GBSM with b = r - q. A quote has no implied vol when its price is at
    or below the European no-arbitrage lower bound (no positive vol reaches it).
    One chart: call and put IV from the mids against strike, with the bid-to-ask
    IV range shaded.
(e) Vega at each strike (at the mid IV), and the change in implied vol caused by
    a one cent change in price: about 0.01 / vega, plus the exact re-solved change.

Every number is printed in a labelled block and saved to output/problem1_fit.txt.
The chart is saved to figures/problem1_iv_smile.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, P1_DIV_YIELD, P1_RATE, P1_SPOT, P1_T, Report, p1_chain

import risk545 as rl

B = P1_RATE - P1_DIV_YIELD           # cost of carry, Merton model
ARGS = (P1_SPOT, None, P1_T, P1_RATE, B)
CALL_COLOR, PUT_COLOR = "#2a78d6", "#eb6834"


def iv(call: bool, strike: float, price: float) -> float:
    return rl.implied_vol(call, P1_SPOT, strike, P1_T, P1_RATE, B, price)


def main() -> None:
    rep = Report()
    chain = p1_chain()
    strikes = chain.index.to_numpy(dtype=float)

    rep.section("Inputs")
    rep.say(f"S = {P1_SPOT}   r = {P1_RATE}   q = {P1_DIV_YIELD}   b = r - q = {B:.4f}   T = 35/365 = {P1_T:.6f}")
    rep.say("Model: GBSM (European), implied vol solved by Brent's method on [1e-6, 5.0].")

    # ---------------------------------------------------------------- (d)
    out = pd.DataFrame(index=chain.index)
    for side, call in (("C", True), ("P", False)):
        for px in ("Bid", "Mid", "Ask"):
            out[f"{side}_IV_{px}"] = [iv(call, k, p) for k, p in zip(strikes, chain[f"{side}_{px}"])]
        out[f"{side}_LowerBound"] = [rl.european_bounds(call, P1_SPOT, k, P1_T, P1_RATE, B)[0] for k in strikes]

    rep.section("(d) Implied volatility from bid, mid and ask (in %)")
    pct = lambda v: "     n/a" if np.isnan(v) else f"{100 * v:8.2f}"
    tbl = out[[c for c in out.columns if "_IV_" in c]]
    rep.say(tbl.to_string(formatters={c: pct for c in tbl.columns}, na_rep="     n/a"))

    rep.section("(d) Quotes with no implied vol")
    rep.say("Lower bound = max(S e^(-qT) - K e^(-rT), 0) for a call, max(K e^(-rT) - S e^(-qT), 0) for a put.")
    rep.say("A price at or below it cannot be reached by any positive vol.\n")
    any_missing = False
    for side, name in (("C", "Call"), ("P", "Put")):
        for px in ("Bid", "Mid", "Ask"):
            for k in out.index[out[f"{side}_IV_{px}"].isna()]:
                any_missing = True
                price, lb = chain.loc[k, f"{side}_{px}"], out.loc[k, f"{side}_LowerBound"]
                rep.say(f"{name} {k:>5} {px:<3}: price {price:.4f}  vs lower bound {lb:.4f}  "
                        f"(short by {lb - price:.4f})")
    if not any_missing:
        rep.say("none")

    # Chart: IV from the mids, with the bid-to-ask IV range shaded.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for side, label, color in (("C", "Call", CALL_COLOR), ("P", "Put", PUT_COLOR)):
        lo, mid, hi = (100 * out[f"{side}_IV_{px}"].to_numpy() for px in ("Bid", "Mid", "Ask"))
        ax.fill_between(strikes, lo, hi, color=color, alpha=0.15, linewidth=0,
                        label=f"{label}: bid-to-ask IV range")
        ax.plot(strikes, mid, color=color, linewidth=2, marker="o", markersize=5, label=f"{label}: IV from mid")
    ax.axvline(P1_SPOT, color="#888888", linewidth=1, linestyle=":")
    ax.text(P1_SPOT, ax.get_ylim()[1], f" spot {P1_SPOT}", va="top", fontsize=9, color="#555555")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Implied volatility by strike, 18 Dec 2026 expiry")
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "problem1_iv_smile.png"
    fig.savefig(fig_path, dpi=150)
    rep.say(f"\nChart saved to figures/{fig_path.name}")

    # ---------------------------------------------------------------- (e)
    rep.section("(e) Vega at the mid IV, and the IV change from a one cent price change")
    rep.say("Vega is per 1.00 of vol (100 vol points), so a price change dP moves IV by about dP / vega.")
    rep.say("dIV approx = 0.01 / vega, in vol points;  dIV exact = IV(mid + 0.01) - IV(mid), re-solved.\n")
    e = pd.DataFrame(index=chain.index)
    for side, call in (("C", True), ("P", False)):
        vegas, approx, exact = [], [], []
        for k in strikes:
            s_mid = out.loc[k, f"{side}_IV_Mid"]
            if np.isnan(s_mid):
                vegas.append(np.nan), approx.append(np.nan), exact.append(np.nan)
                continue
            vg = rl.gbsm(call, P1_SPOT, k, P1_T, P1_RATE, B, s_mid, include_greeks=True).vega
            vegas.append(vg)
            approx.append(100 * 0.01 / vg)
            exact.append(100 * (iv(call, k, chain.loc[k, f"{side}_Mid"] + 0.01) - s_mid))
        e[f"{side}_Vega"], e[f"{side}_dIV_approx"], e[f"{side}_dIV_exact"] = vegas, approx, exact
    rep.say(e.to_string(float_format=lambda v: f"{v:10.4f}"))

    rep.section("(e) Bid-to-ask width in implied vol (ask IV - bid IV, vol points)")
    w = pd.DataFrame({"Call": 100 * (out["C_IV_Ask"] - out["C_IV_Bid"]),
                      "Put": 100 * (out["P_IV_Ask"] - out["P_IV_Bid"])})
    rep.say(w.to_string(float_format=lambda v: f"{v:10.2f}", na_rep="       n/a"))

    rep.save("problem1_fit.txt")


if __name__ == "__main__":
    main()
