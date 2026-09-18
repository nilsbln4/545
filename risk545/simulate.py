"""Monte Carlo simulation from a covariance matrix: direct and PCA-based.

Port of library/simulate.jl's simulateNormal and simulate_pca. Note the RNG
itself is not expected to match the Julia reference -- these tests check
that the simulated sample's covariance recovers the *input* covariance, not
that individual draws match a Julia run bit for bit.
"""

import numpy as np

from .psd import chol_psd, near_psd


def simulate_normal(n_sim: int, cov: np.ndarray, mean=None, seed: int = 1234,
                     fix_method=near_psd) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance matrix is not square ({n},{m})")

    _mean = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)
    if _mean.shape[0] != n:
        raise ValueError(f"Mean ({_mean.shape[0]}) is not the size of cov ({n},{n})")

    try:
        l = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # A genuinely non-PSD matrix can drive chol_psd's diagonal negative;
        # that NaN is expected here and handled by the fallback below, so it
        # doesn't need to surface as a warning.
        with np.errstate(invalid="ignore"):
            l = chol_psd(cov)
        if np.isnan(l).any():
            l = chol_psd(fix_method(cov))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_sim))

    return (l @ z).T + _mean


def simulate_pca(a: np.ndarray, n_sim: int, pct_exp: float = 1.0, mean=None,
                  seed: int = 1234) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    _mean = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)

    vals, vecs = np.linalg.eigh(a)
    # numpy's eigh returns ascending order; PCA wants largest-variance first.
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    total_var = vals.sum()
    pos = np.where(vals >= 1e-8)[0]

    if pct_exp < 1:
        cum, n_val = 0.0, 0
        for idx in pos:
            cum += vals[idx] / total_var
            n_val += 1
            if cum >= pct_exp:
                break
        pos = pos[:n_val]

    vals = vals[pos]
    vecs = vecs[:, pos]

    b = vecs @ np.diag(np.sqrt(vals))

    rng = np.random.default_rng(seed)
    r = rng.standard_normal((len(vals), n_sim))

    return (b @ r).T + _mean
