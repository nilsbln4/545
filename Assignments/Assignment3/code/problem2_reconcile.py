"""Problem 2 -- Reconcile step, part (g): early exercise premiums and the call's exercise boundary.

Early exercise premium = American value - European value. Both come from a tree with
the same N, so the tree's discretisation error cancels (without dividends the GBSM
value is also shown for reference).

Exercise boundary for the call just before the second dividend (t = 0.40): the holder
compares exercising cum-dividend, S - X, with holding into the ex-dividend price S - D2
with 0.10 years left. No dividend follows, so the American call left after the dividend
is worth the same as the European call (early exercise of a call without dividends is
never optimal). The boundary S* solves
    S* - X = C(S* - D2, X, T - 0.40)
with C from GBSM and, as a check, from the tree with the steps left after the dividend.

Every number is printed in a labelled block and saved to output/problem2_reconcile.txt.
"""

import numpy as np
from scipy.optimize import brentq

from common import P2_DIV_AMOUNTS, P2_DIV_TIMES, P2_RATE, P2_SPOT, P2_STRIKE, P2_T, P2_VOL, Report
from problem2_fit import N_MAIN, div_steps, tree_div, tree_nodiv

import risk545 as rl


def main() -> None:
    rep = Report()

    # ---------------------------------------------------------------- premiums
    rep.section(f"(g) Early exercise premium = American - European (tree, N = {N_MAIN})")
    rep.say(f"{'':<6} {'':<14} {'American':>10} {'European':>10} {'premium':>9}")
    for label, call in (("Call", True), ("Put", False)):
        for case, f in (("no dividends", tree_nodiv), ("with dividends", tree_div)):
            am, eu = f(call, True), f(call, False)
            rep.say(f"{label:<6} {case:<14} {am:10.6f} {eu:10.6f} {am - eu:9.6f}")
    rep.say("\nFor reference, no dividends with the GBSM European instead of the tree European:")
    for label, call in (("Call", True), ("Put", False)):
        am = tree_nodiv(call, True)
        g = rl.gbsm(call, P2_SPOT, P2_STRIKE, P2_T, P2_RATE, P2_RATE, P2_VOL).value
        rep.say(f"  {label:<5} American {am:.6f}  GBSM {g:.6f}  difference {am - g:.6f}")

    # ---------------------------------------------------------------- boundary
    t2, d2 = P2_DIV_TIMES[1], P2_DIV_AMOUNTS[1]
    tau = P2_T - t2
    steps_left = N_MAIN - div_steps(N_MAIN)[1]

    hold_gbsm = lambda s: rl.gbsm(True, s - d2, P2_STRIKE, tau, P2_RATE, P2_RATE, P2_VOL).value
    hold_tree = lambda s: rl.bt_american(True, s - d2, P2_STRIKE, tau, P2_RATE, P2_RATE, P2_VOL, steps_left)
    s_gbsm = brentq(lambda s: (s - P2_STRIKE) - hold_gbsm(s), P2_STRIKE + 1e-6, 4 * P2_STRIKE, xtol=1e-10)
    s_tree = brentq(lambda s: (s - P2_STRIKE) - hold_tree(s), P2_STRIKE + 1e-6, 4 * P2_STRIKE, xtol=1e-10)

    rep.section(f"(g) Call exercise boundary just before the second dividend (t = {t2})")
    rep.say(f"Solve S* - X = C(S* - {d2}, X = {P2_STRIKE}, {tau:.2f} years left).")
    rep.say(f"  hold value from GBSM                     : S* = {s_gbsm:.4f}")
    rep.say(f"  hold value from the tree ({steps_left} steps left) : S* = {s_tree:.4f}")
    rep.say(f"Above S* the holder exercises just before the dividend; below it they hold.\n")

    rep.say("Exercise vs hold around the boundary (GBSM hold value):")
    rep.say(f"  {'S':>7} {'exercise S - X':>15} {'hold C(S - D2)':>15} {'decision':>9}")
    for s in sorted({50.0, 51.0, 52.0, float(np.floor(s_gbsm)), float(np.ceil(s_gbsm)), 54.0, 56.0, 60.0}):
        ex, hd = s - P2_STRIKE, hold_gbsm(s)
        rep.say(f"  {s:7.2f} {ex:15.4f} {hd:15.4f} {'exercise' if ex > hd else 'hold':>9}")
    rep.say(f"\nToday's price S = {P2_SPOT}; S* / S = {s_gbsm / P2_SPOT:.4f}")

    rep.save("problem2_reconcile.txt")


if __name__ == "__main__":
    main()
