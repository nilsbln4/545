"""Problem 4 -- Reconcile step, part (h): how precisely 240 months pin down the optimizer.

1. Standard error of each asset's mean monthly return, SE = sd / sqrt(T), with the
   mean, a 95% interval (mean +/- 1.96 SE), and the t statistic of the mean excess
   return over rf.
2. Resampling: draw 240 months with replacement (whole months, so the assets keep
   their joint behaviour), re-estimate the mean and covariance, and re-solve the long
   only maximum Sharpe portfolio. The spread of the weights across resamples shows
   which tangency weights the data actually pin down. Also split by whether the
   resample contains June 2008.

Every number is printed in a labelled block and saved to output/problem4_reconcile.txt.
"""

import numpy as np
import pandas as pd

from common import P4_RF_MONTHLY, Report, p4_returns

import risk545 as rl

N_BOOT = 1000
SEED = 545


def main() -> None:
    rep = Report()
    r = p4_returns()
    assets = list(r.columns)
    T = len(r)
    long_only = np.tile([0.0, 1.0], (len(assets), 1))

    # ---------------------------------------------------------------- SEs
    rep.section("(h) Standard error of each asset's mean monthly return, SE = sd / sqrt(T)")
    rep.say(f"T = {T} months; rf = {P4_RF_MONTHLY:.4%} a month\n")
    rep.say(f"{'asset':<5} {'mean':>9} {'sd':>9} {'SE':>9} {'SE/mean':>8} {'95% low':>9} {'95% high':>9} "
            f"{'t(mean - rf)':>13} {'ann mean':>9} {'ann SE':>8}")
    for a in assets:
        x = r[a]
        m, s = x.mean(), x.std(ddof=1)
        se = s / np.sqrt(T)
        rep.say(f"{a:<5} {m:9.6f} {s:9.6f} {se:9.6f} {se / m:8.1%} {m - 1.96 * se:9.6f} {m + 1.96 * se:9.6f} "
                f"{(m - P4_RF_MONTHLY) / se:13.4f} {12 * m:9.4%} {12 * se:8.4%}")

    # ---------------------------------------------------------------- bootstrap
    w_hat = rl.max_sharpe(r.cov().to_numpy(), r.mean().to_numpy(), P4_RF_MONTHLY, long_only)
    rng = np.random.default_rng(SEED)
    june08 = r.index.get_loc(pd.Timestamp("2008-06-30"))
    weights, has_june = [], []
    data = r.to_numpy()
    for _ in range(N_BOOT):
        idx = rng.integers(0, T, size=T)
        sample = data[idx]
        weights.append(rl.max_sharpe(np.cov(sample, rowvar=False), sample.mean(axis=0), P4_RF_MONTHLY, long_only))
        has_june.append(june08 in idx)
    W = pd.DataFrame(weights, columns=assets)
    has_june = np.array(has_june)

    rep.section(f"(h) Long only tangency weights across {N_BOOT} bootstrap resamples of the months (seed {SEED})")
    rep.say(f"{'asset':<5} {'full sample':>11} {'boot mean':>10} {'boot sd':>8} {'5%':>7} {'25%':>7} {'median':>7} "
            f"{'75%':>7} {'95%':>7} {'P(w=0)':>7}")
    for k, a in enumerate(assets):
        w = W[a]
        q = w.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        rep.say(f"{a:<5} {w_hat[k]:11.4f} {w.mean():10.4f} {w.std(ddof=1):8.4f} " +
                " ".join(f"{v:7.4f}" for v in q) + f" {(w < 1e-4).mean():7.1%}")

    rep.section("(h) PUTW weight with and without June 2008 in the resample")
    for label, mask in (("contains 2008-06", has_june), ("misses 2008-06", ~has_june)):
        w = W["PUTW"][mask]
        rep.say(f"{label:<17}: {mask.sum():4d} resamples   mean PUTW weight {w.mean():.4f}   "
                f"median {w.median():.4f}   5%-95% {w.quantile(0.05):.4f} to {w.quantile(0.95):.4f}")

    rep.save("problem4_reconcile.txt")


if __name__ == "__main__":
    main()
