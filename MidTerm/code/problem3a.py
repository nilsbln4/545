"""Midterm Problem 3(a) -- fit a normal, a Student's t and a NIG to each stock.

    python problem3a.py

Prints every fitted parameter and, for each fit, the log likelihood and AICc.
Reads DEMEAN_Q3 from settings.py. Output is also saved to output/problem3a.txt.
"""

import numpy as np
import pandas as pd

from common import Report, fit_margin, load_q3_returns, rl

rep = Report()
returns = load_q3_returns()

KINDS = ("normal", "t", "nig")
PARAMS = {"normal": ("mu", "sigma"), "t": ("mu", "sigma", "nu"), "nig": ("mu", "alpha", "beta", "delta")}

# ---------------------------------------------------------------------------
# Problem 3(a): fits
# ---------------------------------------------------------------------------
param_rows, fit_rows = [], []
for stock in returns.columns:
    x = returns[stock].to_numpy()
    for kind in KINDS:
        try:
            fit = fit_margin(x, kind)
        except Exception as e:  # a fit can fail; report it rather than stop
            rep.say(f"{stock} {kind}: fit failed ({e})")
            continue
        loglik = float(np.sum(fit.error_model.logpdf(x)))
        param_rows.append({"stock": stock, "distribution": kind,
                           **{p: fit.params[p] for p in PARAMS[kind]}})
        fit_rows.append({"stock": stock, "distribution": kind, "parameters": fit.n_error_params,
                         "log likelihood": loglik, "AICc": rl.aicc(fit, x)})

rep.section(f"Problem 3(a): fitted parameters ({len(returns)} returns per stock)")
params = pd.DataFrame(param_rows)
for kind in KINDS:
    sub = params[params["distribution"] == kind].drop(columns="distribution").set_index("stock")
    rep.say(f"\n{kind}:")
    rep.say(sub.dropna(axis=1, how="all").to_string(float_format=lambda v: f"{v: .6f}"))

rep.section("Problem 3(a): log likelihood and AICc by stock and distribution (lower AICc is better)")
fits = pd.DataFrame(fit_rows)
rep.say(fits.pivot(index="stock", columns="distribution", values="AICc")
        .reindex(returns.columns)[list(KINDS)].to_string(float_format=lambda v: f"{v: .4f}"))
rep.say("\nlog likelihood:")
rep.say(fits.pivot(index="stock", columns="distribution", values="log likelihood")
        .reindex(returns.columns)[list(KINDS)].to_string(float_format=lambda v: f"{v: .4f}"))

rep.save("problem3a.txt")
