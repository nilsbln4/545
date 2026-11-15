"""Problem 4 -- Predict step: moments and Sharpe ratios of the four assets.

Part (a): mean, standard deviation (ddof = 1), skewness and excess kurtosis
(population, as StatsBase) of each asset's monthly returns, and the annualized
Sharpe ratio (mean - rf) / sd x sqrt(12) with rf = 3% / 12 a month. Then, for each
asset, the single most extreme month and the same statistics with that month
removed, to see how much of any one number comes from one month.
No portfolio is optimized here.

Every number is printed in a labelled block and saved to output/problem4_predict.txt.
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import P4_RF_ANNUAL, P4_RF_MONTHLY, Report, p4_returns


def describe(x: pd.Series) -> dict:
    sd = x.std(ddof=1)
    return {"mean": x.mean(), "sd": sd, "skew": stats.skew(x, bias=True),
            "ex kurt": stats.kurtosis(x, fisher=True, bias=True),
            "min": x.min(), "max": x.max(),
            "ann mean": 12 * x.mean(), "ann vol": sd * np.sqrt(12),
            "Sharpe (ann)": (x.mean() - P4_RF_MONTHLY) / sd * np.sqrt(12)}


def main() -> None:
    rep = Report()
    r = p4_returns()

    rep.section("Inputs")
    rep.say(f"{len(r)} monthly returns, {r.index[0]:%Y-%m} to {r.index[-1]:%Y-%m};  rf = {P4_RF_ANNUAL:.0%} a year "
            f"= {P4_RF_MONTHLY:.4%} a month")

    rep.section("(a) Moments and annualized Sharpe ratio")
    rep.say("Sharpe = (monthly mean - rf) / monthly sd x sqrt(12); ann mean = 12 x mean, ann vol = sd x sqrt(12)\n")
    tbl = pd.DataFrame({c: describe(r[c]) for c in r.columns}).T
    rep.say(tbl.to_string(float_format=lambda v: f"{v:10.5f}"))

    rep.section("(a) The single most extreme month of each asset, and the statistics without it")
    rows = {}
    for c in r.columns:
        x = r[c]
        z = (x - x.mean()) / x.std(ddof=1)
        when = z.abs().idxmax()
        rest = describe(x.drop(when))
        full = describe(x)
        rows[c] = {"month": f"{when:%Y-%m}", "return": x[when], "z-score": z[when],
                   "skew": full["skew"], "skew w/o": rest["skew"],
                   "ex kurt": full["ex kurt"], "kurt w/o": rest["ex kurt"],
                   "Sharpe": full["Sharpe (ann)"], "Sharpe w/o": rest["Sharpe (ann)"]}
    rep.say(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:9.4f}"))

    rep.section("(a) PUTW: its five worst and five best months")
    p = r["PUTW"].sort_values()
    rep.say("worst: " + ", ".join(f"{d:%Y-%m} {v:+.4f}" for d, v in p.head(5).items()))
    rep.say("best:  " + ", ".join(f"{d:%Y-%m} {v:+.4f}" for d, v in p.tail(5).items()))
    rep.say(f"months with PUTW > 0: {(r['PUTW'] > 0).sum()} of {len(r)};  "
            f"most common return {p.round(8).mode().iloc[0]:.6f} "
            f"({(p.round(8) == p.round(8).mode().iloc[0]).sum()} months)")

    rep.section("Correlations of monthly returns (context for (b))")
    rep.say(r.corr().to_string(float_format=lambda v: f"{v:8.4f}"))

    rep.save("problem4_predict.txt")


if __name__ == "__main__":
    main()
