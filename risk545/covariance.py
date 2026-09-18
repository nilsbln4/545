"""Covariance and correlation estimation: missing data and exponential weighting.

Port of library/missing_cov.jl and library/ewCov.jl.
"""

import numpy as np
import pandas as pd


def missing_cov(x, skip_miss: bool = True, method: str = "cov") -> np.ndarray:
    """Covariance or correlation matrix in the presence of missing values.

    skip_miss=True drops any row with a missing value in *any* column before
    computing. skip_miss=False computes each pairwise entry from the rows
    where both of that pair's columns are present -- pandas' cov()/corr()
    already do exactly this pairwise-complete-observations calculation, so
    that branch is just a pass-through.
    """
    df = pd.DataFrame(x).astype(float)
    data = df.dropna() if skip_miss else df

    if method == "cov":
        return data.cov().to_numpy()
    elif method == "cor":
        return data.corr().to_numpy()
    raise ValueError(f"method must be 'cov' or 'cor', got {method!r}")


def ew_cov(x, lam: float) -> np.ndarray:
    """Exponentially-weighted covariance matrix.

    Rows are assumed ordered oldest-to-newest -- the last row gets the
    largest weight, (1-lam), matching library/ewCov.jl's convention.
    """
    x = np.asarray(x, dtype=float)
    m, _ = x.shape
    w = _exp_weights(m, lam)
    weighted_mean = w @ x
    xm = np.sqrt(w)[:, None] * (x - weighted_mean)
    return xm.T @ xm


def ew_corr(x, lam: float) -> np.ndarray:
    """Exponentially-weighted correlation matrix (same lambda for var and corr)."""
    cov = ew_cov(x, lam)
    sd = 1.0 / np.sqrt(np.diag(cov))
    return np.diag(sd) @ cov @ np.diag(sd)


def _exp_weights(m: int, lam: float) -> np.ndarray:
    i = np.arange(m)
    w = (1 - lam) * lam ** (m - 1 - i)
    return w / w.sum()
