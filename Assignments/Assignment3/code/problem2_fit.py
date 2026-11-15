"""Problem 2 -- Fit step: American options with discrete dividends on a binomial tree.

(c) All four options without dividends (GBSM for the Europeans, recombining tree
    for the Americans), then all four with the dividends on the non-recombining
    tree (European = same tree without early exercise). For the Europeans, also
    GBSM with the PV of the dividends taken off the stock price.
(d) The American values (with dividends) as N grows: a chart of every N from 10
    to 400, and a table of neighbouring pairs such as 100 and 105.
(e) Delta and gamma of the two American options (with dividends) by central finite
    differences on the tree, bumps 1, 0.1, 0.01, 0.001, at fixed N.
(f) The sensitivity of each option's value to each dividend amount: central
    difference with a 0.01 bump, per dollar of dividend. The GBSM-on-adjusted-spot
    value of the European derivative, -delta * e^{-r t_n}, is shown as a check.

Every number is printed in a labelled block and saved to output/problem2_fit.txt.
Runs in about four minutes (the non-recombining tree is large at big N).
"""

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, P2_DIV_AMOUNTS, P2_DIV_TIMES, P2_RATE, P2_SPOT, P2_STRIKE, P2_T, P2_VOL, Report

import risk545 as rl

N_MAIN = 500
N_DENSE = list(range(10, 410, 5))             # every multiple of 5, for the chart
N_PAIRS = [25, 50, 100, 200, 400]             # neighbouring pairs (N, N + 5) for the table
N_LARGE = [800, 805]                          # largest trees, used as the reference value
BUMPS = [1.0, 0.1, 0.01, 0.001]
DIV_BUMP = 0.01

OPTIONS = [("European call", True, False), ("American call", True, True),
           ("European put", False, False), ("American put", False, True)]


def div_steps(n: int) -> list:
    """Dividend dates as step indices; exact because N is a multiple of 5."""
    steps = [t / P2_T * n for t in P2_DIV_TIMES]
    assert all(abs(s - round(s)) < 1e-9 for s in steps), "N must put a step on each dividend date"
    return [int(round(s)) for s in steps]


def tree_div(call, american, n=N_MAIN, spot=P2_SPOT, amts=P2_DIV_AMOUNTS):
    return rl.bt_american_div(call, spot, P2_STRIKE, P2_T, P2_RATE, amts, div_steps(n), P2_VOL, n, american)


def tree_nodiv(call, american, n=N_MAIN):
    return rl.bt_american(call, P2_SPOT, P2_STRIKE, P2_T, P2_RATE, P2_RATE, P2_VOL, n, american)


def main() -> None:
    rep = Report()
    pv_divs = sum(d * np.exp(-P2_RATE * t) for d, t in zip(P2_DIV_AMOUNTS, P2_DIV_TIMES))
    adj_spot = P2_SPOT - pv_divs

    rep.section("Inputs")
    rep.say(f"S = {P2_SPOT}   X = {P2_STRIKE}   T = {P2_T}   r = {P2_RATE}   vol = {P2_VOL}")
    rep.say(f"dividends {P2_DIV_AMOUNTS} at t = {P2_DIV_TIMES};  main tree N = {N_MAIN}, "
            f"dividend steps {div_steps(N_MAIN)}")
    rep.say("Tree: CRR, u = e^(vol sqrt(dt)), d = 1/u, pu = (e^(r dt) - d)/(u - d). "
            "With dividends: non-recombining (new sub-tree from S - D at every node on a dividend date).")

    # ---------------------------------------------------------------- (c)
    rep.section(f"(c) Values without and with dividends (tree N = {N_MAIN})")
    rows = {}
    for name, call, american in OPTIONS:
        no_div = tree_nodiv(call, True) if american else rl.gbsm(call, P2_SPOT, P2_STRIKE, P2_T, P2_RATE,
                                                                 P2_RATE, P2_VOL).value
        with_div = tree_div(call, american)
        rows[name] = (no_div, with_div)
    rep.say(f"{'option':<15} {'no dividends':>13} {'method':>10} {'with dividends':>15} {'change':>9}")
    for name, call, american in OPTIONS:
        a, b = rows[name]
        rep.say(f"{name:<15} {a:13.6f} {'tree' if american else 'GBSM':>10} {b:15.6f} {b - a:9.4f}")
    rep.say("(with dividends: all four from the non-recombining tree; Europeans without early exercise)")

    rep.say(f"\nEuropean check, no dividends: tree N = {N_MAIN} vs GBSM")
    for name, call, american in OPTIONS:
        if not american:
            rep.say(f"  {name:<15} tree {tree_nodiv(call, False):.6f}   GBSM {rows[name][0]:.6f}")

    rep.say(f"\nEuropean with dividends: GBSM on S - PV(dividends) vs the tree")
    rep.say(f"  PV(dividends) = {pv_divs:.6f}   S - PV = {adj_spot:.6f}")
    for name, call, american in OPTIONS:
        if not american:
            g = rl.gbsm(call, adj_spot, P2_STRIKE, P2_T, P2_RATE, P2_RATE, P2_VOL).value
            t = rows[name][1]
            rep.say(f"  {name:<15} GBSM(S - PV) {g:.6f}   tree {t:.6f}   tree - GBSM {t - g:+.6f}")

    # ---------------------------------------------------------------- (d)
    rep.section("(d) Convergence of the American values (with dividends) in N")
    print("  computing the N grid ...", flush=True)
    vals = {n: (tree_div(True, True, n), tree_div(False, True, n)) for n in N_DENSE + N_LARGE}
    ref = [(vals[N_LARGE[0]][k] + vals[N_LARGE[1]][k]) / 2 for k in (0, 1)]
    rep.say(f"Reference value = average of N = {N_LARGE[0]} and {N_LARGE[1]}: "
            f"call {ref[0]:.6f}, put {ref[1]:.6f}")
    rep.say("|gap| = distance between the values at N and N + 5;  avg - ref = their average minus the reference value.\n")
    for k, name in ((0, "American call"), (1, "American put")):
        rep.say(f"{name}")
        rep.say(f"  {'N':>4} {'V(N)':>10} {'N+5':>5} {'V(N+5)':>10} {'|gap|':>9} {'average':>10} {'avg - ref':>10}")
        for n in N_PAIRS:
            v1, v2 = vals[n][k], vals[n + 5][k]
            avg = (v1 + v2) / 2
            rep.say(f"  {n:4d} {v1:10.6f} {n + 5:5d} {v2:10.6f} {abs(v1 - v2):9.6f} {avg:10.6f} {avg - ref[k]:+10.6f}")
        rep.say()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k, name, color in ((axes[0], 0, "American call", "#2a78d6"), (axes[1], 1, "American put", "#eb6834")):
        ax.plot(N_DENSE, [vals[n][k] for n in N_DENSE], color=color, linewidth=1.5, marker="o", markersize=3,
                label="tree value at N")
        ax.axhline(ref[k], color="#555555", linewidth=1, linestyle="--",
                   label=f"reference: avg of N = {N_LARGE[0]}, {N_LARGE[1]} ({ref[k]:.4f})")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_title(f"{name}, with dividends")
        ax.set_xlabel("Tree steps N (multiples of 5)")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "problem2_convergence.png"
    fig.savefig(fig_path, dpi=150)
    rep.say(f"Chart of every N from {N_DENSE[0]} to {N_DENSE[-1]} saved to figures/{fig_path.name}")

    # ---------------------------------------------------------------- (e)
    rep.section(f"(e) Delta and gamma of the American options with dividends, central differences (N = {N_MAIN})")
    rep.say("delta = (V(S+h) - V(S-h)) / 2h;   gamma = (V(S+h) - 2 V(S) + V(S-h)) / h^2\n")
    for name, call in (("American call", True), ("American put", False)):
        v0 = tree_div(call, True)
        rep.say(f"{name}: V(S) = {v0:.10f}")
        rep.say(f"  {'h':>6} {'V(S+h)':>14} {'V(S-h)':>14} {'delta':>12} {'gamma':>14}")
        for h in BUMPS:
            up = tree_div(call, True, spot=P2_SPOT + h)
            dn = tree_div(call, True, spot=P2_SPOT - h)
            rep.say(f"  {h:6g} {up:14.10f} {dn:14.10f} {(up - dn) / (2 * h):12.6f} "
                    f"{(up - 2 * v0 + dn) / h**2:14.6f}")
        rep.say()

    # ---------------------------------------------------------------- (f)
    rep.section(f"(f) Sensitivity to each dividend amount, dV/dD_n per dollar (bump {DIV_BUMP}, N = {N_MAIN})")
    rep.say("Check for the Europeans: GBSM on S - PV(div) gives dV/dD_n = -delta * e^(-r t_n).\n")
    rep.say(f"{'option':<15} {'dV/dD1':>10} {'dV/dD2':>10} {'GBSM check D1':>14} {'GBSM check D2':>14}")
    for name, call, american in OPTIONS:
        sens = []
        for k in range(len(P2_DIV_AMOUNTS)):
            up, dn = list(P2_DIV_AMOUNTS), list(P2_DIV_AMOUNTS)
            up[k] += DIV_BUMP
            dn[k] -= DIV_BUMP
            sens.append((tree_div(call, american, amts=up) - tree_div(call, american, amts=dn)) / (2 * DIV_BUMP))
        check = ""
        if not american:
            delta = rl.gbsm(call, adj_spot, P2_STRIKE, P2_T, P2_RATE, P2_RATE, P2_VOL).delta
            check = "".join(f"{-delta * np.exp(-P2_RATE * t):14.6f} " for t in P2_DIV_TIMES)
        rep.say(f"{name:<15} {sens[0]:10.6f} {sens[1]:10.6f} {check}")

    rep.save("problem2_fit.txt")


if __name__ == "__main__":
    main()
