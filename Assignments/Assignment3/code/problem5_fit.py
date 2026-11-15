"""Problem 5 -- Fit step: CAPM, Fama-French 3 and Carhart 4 factor models for two funds.

(c) Each model fitted by OLS to each fund's daily excess return. Coefficients with
    standard errors (residual variance over T - p), alpha's t statistic, R^2 and
    AICc. AICc uses the Gaussian log-likelihood at the maximum likelihood residual
    variance (SSR / T) with k = number of coefficients + 1 (the variance), the same
    parameter counting as library/fitted_model.jl's aicc.
(d) Expected annual return from the four-factor betas and the average daily factor
    returns of the most recent 10 years in the factor files (x 255), with and without
    the fitted alpha; shown as excess return and with the 10 year average RF added.
(e) The worst daily momentum return in the factor file; the one day loss on
    $10,000,000 in each fund from beta_Mom x that return, other factors flat (alpha
    left out), and that loss as a multiple of the fund's daily standard deviation.

Every number is printed in a labelled block and saved to output/problem5_fit.txt.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import P5_DAYS_PER_YEAR, Report, p5_data, p5_factors

FUNDS = ["FUND1", "FUND2"]
MODELS = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"], "Carhart": ["Mkt-RF", "SMB", "HML", "Mom"]}
NOTIONAL = 10_000_000


def fit(y: pd.Series, X: pd.DataFrame):
    return sm.OLS(y.to_numpy(), sm.add_constant(X.to_numpy())).fit()


def gaussian_aicc(res) -> float:
    n = int(res.nobs)
    k = len(res.params) + 1
    sigma2 = res.ssr / n
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def main() -> None:
    rep = Report()
    d = p5_data()
    A = P5_DAYS_PER_YEAR

    rep.section("Inputs")
    rep.say(f"{len(d)} days, {d.index[0]:%Y-%m-%d} to {d.index[-1]:%Y-%m-%d}; excess return = fund - RF")

    # ---------------------------------------------------------------- (c)
    rep.section("(c) Factor regressions (daily; alpha also annualized x 255)")
    fits = {}
    for f in FUNDS:
        for name, cols in MODELS.items():
            res = fit(d[f"{f}_ex"], d[cols])
            fits[(f, name)] = res
            rep.say(f"\n{f} -- {name}")
            rep.say(f"  {'':<8} {'coef':>11} {'se':>10} {'t':>9}")
            for lab, b, se, t in zip(["alpha"] + cols, res.params, res.bse, res.tvalues):
                rep.say(f"  {lab:<8} {b:11.6f} {se:10.6f} {t:9.4f}")
            rep.say(f"  alpha annualized = {A * res.params[0]:.4%}   R^2 = {res.rsquared:.4f}   "
                    f"adj R^2 = {res.rsquared_adj:.4f}   AICc = {gaussian_aicc(res):.4f}")

    rep.section("(c) Summary")
    rep.say(f"{'fund':<6} {'model':<8} {'alpha ann':>10} {'t(alpha)':>9} {'R^2':>7} {'AICc':>12}")
    for (f, name), res in fits.items():
        rep.say(f"{f:<6} {name:<8} {A * res.params[0]:10.4%} {res.tvalues[0]:9.4f} {res.rsquared:7.4f} "
                f"{gaussian_aicc(res):12.4f}")

    # ---------------------------------------------------------------- (d)
    fac = p5_factors()
    end = fac.index[-1]
    recent = fac[fac.index > end - pd.DateOffset(years=10)]
    avg = recent.mean()
    rep.section("(d) Expected annual return from the Carhart betas and the last 10 years of factors")
    rep.say(f"factor window {recent.index[0]:%Y-%m-%d} to {end:%Y-%m-%d} ({len(recent)} days)")
    rep.say("average factor return, annualized x 255: " +
            "  ".join(f"{c} {A * avg[c]:.4%}" for c in MODELS["Carhart"] + ["RF"]) + "\n")
    rep.say(f"{'fund':<6} " + " ".join(f"{'b x ' + c:>12}" for c in MODELS["Carhart"]) +
            f" {'excess, no a':>13} {'alpha':>9} {'excess + a':>11} {'total, no a':>12} {'total + a':>10}")
    for f in FUNDS:
        res = fits[(f, "Carhart")]
        contrib = [A * b * avg[c] for b, c in zip(res.params[1:], MODELS["Carhart"])]
        ex = sum(contrib)
        alpha = A * res.params[0]
        rf = A * avg["RF"]
        rep.say(f"{f:<6} " + " ".join(f"{v:12.4%}" for v in contrib) +
                f" {ex:13.4%} {alpha:9.4%} {ex + alpha:11.4%} {ex + rf:12.4%} {ex + alpha + rf:10.4%}")

    # ---------------------------------------------------------------- (e)
    worst_day = fac["Mom"].idxmin()
    worst = fac.loc[worst_day, "Mom"]
    rep.section("(e) Worst daily momentum return, replayed on $10,000,000 (other factors flat)")
    rep.say(f"worst Mom day in the factor file: {worst_day:%Y-%m-%d}, Mom = {worst:.4%}\n")
    rep.say(f"{'fund':<6} {'beta_Mom':>9} {'return':>9} {'P&L ($)':>14} {'daily sd':>9} {'loss / sd':>10}")
    for f in FUNDS:
        b = fits[(f, "Carhart")].params[4]
        ret = b * worst
        sd = d[f].std(ddof=1)
        rep.say(f"{f:<6} {b:9.4f} {ret:9.4%} {NOTIONAL * ret:14,.0f} {sd:9.4%} {-ret / sd:10.2f}")

    rep.save("problem5_fit.txt")


if __name__ == "__main__":
    main()
