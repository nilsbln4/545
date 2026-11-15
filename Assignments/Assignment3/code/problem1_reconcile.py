"""Problem 1 -- Reconcile step, part (f): the flagged strike's parity gap in vol points.

The parity gap is in dollars. Dividing it by vega converts it into the implied vol
change that would be needed to explain it (dIV ~ dPrice / vega). That is compared
with the actual gap between the call's and the put's implied vol at the strike.

Every number is printed in a labelled block and saved to output/problem1_reconcile.txt.
"""

import numpy as np

from common import P1_DIV_YIELD, P1_RATE, P1_SPOT, P1_T, Report, p1_chain

import risk545 as rl

B = P1_RATE - P1_DIV_YIELD
STRIKE = 175.0


def main() -> None:
    rep = Report()
    row = p1_chain().loc[STRIKE]

    parity_value = P1_SPOT * np.exp(-P1_DIV_YIELD * P1_T) - STRIKE * np.exp(-P1_RATE * P1_T)
    gap = (row["C_Mid"] - row["P_Mid"]) - parity_value

    iv_c = rl.implied_vol(True, P1_SPOT, STRIKE, P1_T, P1_RATE, B, row["C_Mid"])
    iv_p = rl.implied_vol(False, P1_SPOT, STRIKE, P1_T, P1_RATE, B, row["P_Mid"])
    vega = lambda call, s: rl.gbsm(call, P1_SPOT, STRIKE, P1_T, P1_RATE, B, s, include_greeks=True).vega
    vega_c, vega_p = vega(True, iv_c), vega(False, iv_p)
    iv_avg = (iv_c + iv_p) / 2
    vega_avg = vega(True, iv_avg)          # call and put vega are equal at the same vol

    rep.section(f"(f) Strike {STRIKE:g}: parity gap expressed in vol points")
    rep.say(f"C_mid = {row['C_Mid']:.4f}   P_mid = {row['P_Mid']:.4f}   C - P = {row['C_Mid'] - row['P_Mid']:.4f}")
    rep.say(f"parity value S e^(-qT) - K e^(-rT) = {parity_value:.4f}")
    rep.say(f"parity gap = {gap:.4f} dollars\n")

    rep.say(f"call IV (mid) = {100 * iv_c:.2f}%   vega = {vega_c:.4f}")
    rep.say(f"put  IV (mid) = {100 * iv_p:.2f}%   vega = {vega_p:.4f}")
    rep.say(f"average IV    = {100 * iv_avg:.2f}%   vega = {vega_avg:.4f}\n")

    rep.say("Gap converted with dIV = gap / vega (vol points):")
    rep.say(f"  using the call's vega    : {100 * gap / vega_c:.2f}")
    rep.say(f"  using the put's vega     : {100 * gap / vega_p:.2f}")
    rep.say(f"  using vega at average IV : {100 * gap / vega_avg:.2f}")
    rep.say(f"Actual call IV - put IV    : {100 * (iv_c - iv_p):.2f}\n")

    # Where the put's IV would sit if its price satisfied parity with the call.
    p_parity = row["C_Mid"] - parity_value
    iv_p_parity = rl.implied_vol(False, P1_SPOT, STRIKE, P1_T, P1_RATE, B, p_parity)
    rep.say(f"Put price implied by parity with the call mid = {p_parity:.4f}  (quoted mid {row['P_Mid']:.4f})")
    rep.say(f"IV of that parity put price = {100 * iv_p_parity:.2f}%  (call IV {100 * iv_c:.2f}%)")

    # ---------------------------------------------------------------- (g)
    rep.section("(g) The call wing above 210: parity gap in dollars vs in vol points")
    rep.say("gap and band in dollars (as in part a); call IV - put IV from the mids;")
    rep.say("gap / vega at the average of the two IVs, in vol points.\n")
    chain = p1_chain()
    rep.say(f"{'Strike':>6} {'C_mid':>7} {'gap $':>8} {'band $':>7} {'C IV%':>7} {'P IV%':>7} "
            f"{'C-P IV':>7} {'avg vega':>9} {'gap/vega':>9}")
    for k in (205.0, 210.0, 215.0, 220.0, 225.0):
        r = chain.loc[k]
        pv = P1_SPOT * np.exp(-P1_DIV_YIELD * P1_T) - k * np.exp(-P1_RATE * P1_T)
        g = (r["C_Mid"] - r["P_Mid"]) - pv
        band = r["C_Spread"] + r["P_Spread"]
        ic = rl.implied_vol(True, P1_SPOT, k, P1_T, P1_RATE, B, r["C_Mid"])
        ip = rl.implied_vol(False, P1_SPOT, k, P1_T, P1_RATE, B, r["P_Mid"])
        va = rl.gbsm(True, P1_SPOT, k, P1_T, P1_RATE, B, (ic + ip) / 2, include_greeks=True).vega
        rep.say(f"{k:6.0f} {r['C_Mid']:7.3f} {g:8.4f} {band:7.2f} {100 * ic:7.2f} {100 * ip:7.2f} "
                f"{100 * (ic - ip):7.2f} {va:9.4f} {100 * g / va:9.2f}")

    rep.save("problem1_reconcile.txt")


if __name__ == "__main__":
    main()
