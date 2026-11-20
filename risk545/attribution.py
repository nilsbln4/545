"""Ex-post return and volatility attribution (library/expost_factor.jl).

expost_factor(w, stock_returns, factor_returns, betas) follows the Julia function
step for step:
  * weights drift with returns: each period the holdings grow by (1 + r) and are
    renormalized, so the weight used in period t is the start-of-period weight;
  * the factor weight of factor j in period t is sum_i beta_ij w_it;
  * the portfolio return is sum_i w_it r_it, and the residual ("Alpha") is the part
    not explained by factor weight x factor return;
  * multi-period return attribution uses Carino's linking coefficients, so the
    attributions add up exactly to the compounded portfolio return;
  * volatility attribution regresses each factor's (and the residual's) contribution
    on the portfolio return; beta x sd(portfolio) gives component SDs that add up to
    the portfolio's standard deviation (ddof = 1).
For plain attribution by asset (test 11.1), pass the stock returns as the "factors"
with an identity beta matrix and drop the Alpha column.
"""

import numpy as np
import pandas as pd


def expost_factor(w, stock_returns: pd.DataFrame, factor_returns: pd.DataFrame, betas) -> pd.DataFrame:
    """Rows TotalReturn, Return Attribution, Vol Attribution; columns = factors, Alpha,
    Portfolio (the Julia function's Attribution table)."""
    R = stock_returns.to_numpy(dtype=float)
    F = factor_returns.to_numpy(dtype=float)
    B = np.asarray(betas, dtype=float)              # stocks x factors
    factors = list(factor_returns.columns)
    n = R.shape[0]

    last_w = np.asarray(w, dtype=float).copy()
    p_ret = np.empty(n)
    resid = np.empty(n)
    factor_w = np.empty((n, len(factors)))
    for t in range(n):
        factor_w[t] = B.T @ last_w                  # factor weight = sum over stocks of beta x weight
        grown = last_w * (1.0 + R[t])
        pr = grown.sum()
        last_w = grown / pr                         # drift the weights, renormalize to one
        p_ret[t] = pr - 1.0
        resid[t] = p_ret[t] - factor_w[t] @ F[t]

    total = np.expm1(np.log1p(p_ret).sum())
    k = np.log1p(total) / total                     # Carino K
    carino_k = np.log1p(p_ret) / p_ret / k          # per-period linking coefficient

    cols = factors + ["Alpha"]
    attrib = np.column_stack([F * factor_w * carino_k[:, None], resid * carino_k])
    series = np.column_stack([F, resid])
    total_ret = [np.expm1(np.log1p(series[:, j]).sum()) for j in range(len(cols))] + [total]
    return_attr = list(attrib.sum(axis=0)) + [total]

    # Volatility attribution: regress each period's contribution on the portfolio return.
    Y = np.column_stack([F * factor_w, resid])
    X = np.column_stack([np.ones(n), p_ret])
    slope = np.linalg.solve(X.T @ X, X.T @ Y)[1]
    sd = p_ret.std(ddof=1)
    vol_attr = list(slope * sd) + [sd]

    out = pd.DataFrame([total_ret, return_attr, vol_attr], columns=cols + ["Portfolio"])
    out.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    return out
