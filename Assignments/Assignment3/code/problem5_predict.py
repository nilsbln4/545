"""Problem 5 -- Predict step: Sharpe ratios, the CAPM, and what its residuals correlate with.

(a) Each fund's mean daily excess return (over the Fama-French RF), volatility and
    Sharpe ratio (annualized with 255 days), and the CAPM regression of the fund's
    excess return on Mkt-RF: alpha, t(alpha), beta, R^2.
(b) Correlation of each fund's CAPM residual with SMB, HML and Mom.
The multi-factor models are not fitted here.

Every number is printed in a labelled block and saved to output/problem5_predict.txt.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import P5_DAYS_PER_YEAR, Report, p5_data

FUNDS = ["FUND1", "FUND2"]


def main() -> None:
    rep = Report()
    d = p5_data()
    A = P5_DAYS_PER_YEAR

    rep.section("Inputs")
    rep.say(f"{len(d)} days, {d.index[0]:%Y-%m-%d} to {d.index[-1]:%Y-%m-%d}; factors divided by 100; "
            f"annualized with {A} days")

    rep.section("(a) Excess return, volatility, Sharpe ratio")
    rep.say(f"{'fund':<6} {'mean ex/day':>12} {'ann mean ex':>12} {'sd/day':>10} {'ann vol':>9} {'Sharpe (ann)':>13}")
    for f in FUNDS:
        x = d[f"{f}_ex"]
        rep.say(f"{f:<6} {x.mean():12.6f} {A * x.mean():12.4%} {x.std(ddof=1):10.6f} "
                f"{x.std(ddof=1) * np.sqrt(A):9.4%} {x.mean() / x.std(ddof=1) * np.sqrt(A):13.4f}")
    m = d["Mkt-RF"]
    rep.say(f"{'Mkt-RF':<6} {m.mean():12.6f} {A * m.mean():12.4%} {m.std(ddof=1):10.6f} "
            f"{m.std(ddof=1) * np.sqrt(A):9.4%} {m.mean() / m.std(ddof=1) * np.sqrt(A):13.4f}")

    rep.section("(a) CAPM: fund excess return on Mkt-RF (OLS, residual variance over T - 2)")
    rep.say(f"{'fund':<6} {'alpha/day':>10} {'alpha ann':>10} {'se(alpha)':>10} {'t(alpha)':>9} {'beta':>8} "
            f"{'se(beta)':>9} {'R^2':>7}")
    resid = {}
    for f in FUNDS:
        fit = sm.OLS(d[f"{f}_ex"].to_numpy(), sm.add_constant(m.to_numpy())).fit()
        resid[f] = pd.Series(fit.resid, index=d.index)
        rep.say(f"{f:<6} {fit.params[0]:10.6f} {A * fit.params[0]:10.4%} {fit.bse[0]:10.6f} {fit.tvalues[0]:9.4f} "
                f"{fit.params[1]:8.4f} {fit.bse[1]:9.4f} {fit.rsquared:7.4f}")

    rep.section("(b) Correlation of each fund's CAPM residual with SMB, HML, Mom")
    corr = pd.DataFrame({f: [resid[f].corr(d[c]) for c in ("SMB", "HML", "Mom")] for f in FUNDS},
                        index=["SMB", "HML", "Mom"]).T
    rep.say(corr.to_string(float_format=lambda v: f"{v:9.4f}"))

    rep.save("problem5_predict.txt")


if __name__ == "__main__":
    main()
