"""Value at Risk and Expected Shortfall.

Port of library/RiskStats.jl. Both are reported as positive numbers for a loss
(so VaR of a P&L sample is minus its alpha-quantile).
"""

import numpy as np
import pandas as pd
from scipy import integrate


def _is_distribution(a) -> bool:
    return hasattr(a, "ppf") and hasattr(a, "pdf")


def _sample_var_level(x: np.ndarray, alpha: float) -> float:
    """The alpha-quantile of a sorted sample, taken as the midpoint of the two
    order statistics either side of n*alpha (they coincide when n*alpha is an
    integer), matching the reference's convention for historical VaR."""
    n = len(x)
    n_up = int(np.ceil(n * alpha))
    n_dn = int(np.floor(n * alpha))
    if n_dn < 1:
        raise ValueError(f"Sample of {n} is too small for alpha={alpha}: need n*alpha >= 1")
    # The reference indexes from 1; Python from 0.
    return 0.5 * (x[n_up - 1] + x[n_dn - 1])


def value_at_risk(a, alpha: float = 0.05) -> float:
    """VaR from either a P&L sample (historical / simulated) or a frozen
    scipy.stats distribution (parametric)."""
    if _is_distribution(a):
        return -a.ppf(alpha)
    x = np.sort(np.asarray(a, dtype=float))
    return -_sample_var_level(x, alpha)


def expected_shortfall(a, alpha: float = 0.05) -> float:
    """ES: the average loss in the alpha tail, from a P&L sample or a frozen
    scipy.stats distribution."""
    if _is_distribution(a):
        v = value_at_risk(a, alpha)
        start = a.ppf(1e-12)
        integral, _ = integrate.quad(lambda x: x * a.pdf(x), start, -v, limit=200)
        return -integral / alpha

    x = np.sort(np.asarray(a, dtype=float))
    v = _sample_var_level(x, alpha)
    return -x[x <= v].mean()


def agg_risk(df: pd.DataFrame, agg_level: list) -> pd.DataFrame:
    """VaR/ES and summary statistics of simulated P&L, by group and in total.

    `df` has one row per (group, iteration) with columns currentValue,
    simulatedValue and pnl, plus the grouping columns named in `agg_level` and
    an `iteration` column. P&L is summed within each iteration before the risk
    numbers are taken, so a group's risk reflects its positions moving together.
    With agg_level=["Stock"] the result has one row per stock plus a "Total"
    row for the whole portfolio.
    """
    value_cols = ["currentValue", "simulatedValue", "pnl"]

    if agg_level:
        per_iter = df.groupby(list(agg_level) + ["iteration"], as_index=False, sort=False)[value_cols].sum()
        grouped = per_iter.groupby(list(agg_level), sort=False)
    else:
        per_iter = df.groupby("iteration", as_index=False)[value_cols].sum()
        grouped = [((), per_iter)]

    rows = []
    for key, g in grouped:
        pnl = g["pnl"].to_numpy()
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(agg_level, key))
        row.update({
            "currentValue": g["currentValue"].iloc[0],
            "VaR95": value_at_risk(pnl, alpha=0.05),
            "ES95": expected_shortfall(pnl, alpha=0.05),
            "VaR99": value_at_risk(pnl, alpha=0.01),
            "ES99": expected_shortfall(pnl, alpha=0.01),
            "Standard_Dev": pnl.std(ddof=1),
            "min": pnl.min(),
            "max": pnl.max(),
            "mean": pnl.mean(),
        })
        rows.append(row)

    risk = pd.DataFrame(rows)
    for col in ("VaR95", "VaR99", "ES95", "ES99"):
        risk[f"{col}_Pct"] = risk[col] / risk["currentValue"]

    if not agg_level:
        return risk

    # Roll up one level (dropping the last grouping column) and label it "Total".
    uplevel = agg_risk(df, list(agg_level[:-1]))
    uplevel[agg_level[-1]] = "Total"
    return pd.concat([risk, uplevel], ignore_index=True)
