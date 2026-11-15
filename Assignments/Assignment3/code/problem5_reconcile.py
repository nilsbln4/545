"""Problem 5 -- Reconcile step, part (f): accounting for the change in alpha.

With an intercept, OLS makes the fitted line pass through the means, so for each model
    alpha = mean(excess return) - sum_k beta_k x mean(factor_k).
Subtracting the Carhart equation from the CAPM equation gives an exact split of the
change in alpha:
    alpha_CAPM - alpha_Carhart = (beta_Mkt,Carhart - beta_Mkt,CAPM) x mean(Mkt-RF)
                                 + beta_SMB x mean(SMB) + beta_HML x mean(HML) + beta_Mom x mean(Mom)
Each term is a loading times the factor premium earned during the sample.

Every number is printed in a labelled block and saved to output/problem5_reconcile.txt.
"""

from common import P5_DAYS_PER_YEAR, Report, p5_data
from problem5_fit import FUNDS, MODELS, fit

A = P5_DAYS_PER_YEAR


def main() -> None:
    rep = Report()
    d = p5_data()
    prem = d[MODELS["Carhart"]].mean()

    rep.section("(f) Factor premia during the fund sample (annualized x 255)")
    rep.say("  ".join(f"{c} {A * prem[c]:.4%}" for c in MODELS["Carhart"]))

    rep.section("(f) Change in alpha, CAPM -> Carhart, split by factor (annualized)")
    rep.say("term = loading x premium; alpha_CAPM - alpha_Carhart = sum of the terms\n")
    for f in FUNDS:
        capm = fit(d[f"{f}_ex"], d[MODELS["CAPM"]])
        car = fit(d[f"{f}_ex"], d[MODELS["Carhart"]])
        b = dict(zip(MODELS["Carhart"], car.params[1:]))
        terms = {"Mkt-RF (beta change)": (b["Mkt-RF"] - capm.params[1]) * prem["Mkt-RF"],
                 "SMB": b["SMB"] * prem["SMB"], "HML": b["HML"] * prem["HML"], "Mom": b["Mom"] * prem["Mom"]}
        rep.say(f"{f}: CAPM alpha {A * capm.params[0]:.4%} -> Carhart alpha {A * car.params[0]:.4%}  "
                f"(change {A * (car.params[0] - capm.params[0]):+.4%})")
        for k, v in terms.items():
            loading = b[k.split()[0]] - capm.params[1] if k.startswith("Mkt") else b[k]
            rep.say(f"    {k:<22} loading {loading:8.4f}   x premium {A * prem[k.split()[0]]:8.4%}   "
                    f"= {A * v:8.4%}")
        rep.say(f"    {'sum':<22} {A * sum(terms.values()):58.4%}   "
                f"(check: CAPM - Carhart alpha = {A * (capm.params[0] - car.params[0]):.4%})\n")

    rep.save("problem5_reconcile.txt")


if __name__ == "__main__":
    main()
