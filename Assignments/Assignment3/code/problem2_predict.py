"""Problem 2 -- Predict step: the early exercise condition for the American call.

Part (a): for each dividend D_n paid at t_n, the Week 7 condition
    D_n > X (1 - e^{-r (T - t_n)})
says early exercise of the American call just before that dividend can be optimal:
the dividend lost by holding exceeds the interest earned by paying the strike later.
Also prints the present value of the dividends and the moneyness, for part (b).
No option is priced here.

Every number is printed in a labelled block and saved to output/problem2_predict.txt.
"""

import numpy as np

from common import P2_DIV_AMOUNTS, P2_DIV_TIMES, P2_RATE, P2_SPOT, P2_STRIKE, P2_T, P2_VOL, Report


def main() -> None:
    rep = Report()

    rep.section("Inputs")
    rep.say(f"S = {P2_SPOT}   X = {P2_STRIKE}   T = {P2_T}   r = {P2_RATE}   vol = {P2_VOL}")
    for d, t in zip(P2_DIV_AMOUNTS, P2_DIV_TIMES):
        rep.say(f"dividend {d:.2f} at t = {t:.2f}")
    rep.say(f"Moneyness S / X = {P2_SPOT / P2_STRIKE:.4f}  (call out of the money, put in the money)")

    # ---------------------------------------------------------------- (a)
    rep.section("(a) Early exercise condition D_n > X (1 - e^(-r (T - t_n)))")
    rep.say("Right side = interest saved by paying the strike at T instead of at t_n.\n")
    rep.say(f"{'n':>2} {'t_n':>5} {'T - t_n':>8} {'D_n':>6} {'X(1-e^(-r(T-t_n)))':>20} {'D_n - threshold':>16}  can exercise be optimal?")
    for n, (d, t) in enumerate(zip(P2_DIV_AMOUNTS, P2_DIV_TIMES), start=1):
        thr = P2_STRIKE * (1 - np.exp(-P2_RATE * (P2_T - t)))
        rep.say(f"{n:>2} {t:5.2f} {P2_T - t:8.2f} {d:6.2f} {thr:20.4f} {d - thr:16.4f}  {'yes' if d > thr else 'no'}")

    # ---------------------------------------------------------------- (b) context
    rep.section("(b) Context: size of the dividends")
    pv = [d * np.exp(-P2_RATE * t) for d, t in zip(P2_DIV_AMOUNTS, P2_DIV_TIMES)]
    for n, (d, p) in enumerate(zip(P2_DIV_AMOUNTS, pv), start=1):
        rep.say(f"PV of dividend {n} = {p:.4f}")
    rep.say(f"PV of all dividends = {sum(pv):.4f}  ({sum(pv) / P2_SPOT:.2%} of the stock price)")
    rep.say(f"Stock price less PV of dividends = {P2_SPOT - sum(pv):.4f}")

    rep.save("problem2_predict.txt")


if __name__ == "__main__":
    main()
