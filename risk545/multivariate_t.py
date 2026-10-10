"""Multivariate t: fitting and simulation.

Port of library/multivariate_t.jl. The rank-based route from Week 5: mu at the
sample mean, R from Kendall's tau, the scales pinned to nu by the sample
standard deviations, and nu itself by a one-dimensional profile.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .psd import higham_nearest_psd
from .simulate import simulate_pca


def fix_correlation(R: np.ndarray, tol: float = 1e-8, ridge: float = 1e-8) -> np.ndarray:
    """Make a correlation matrix safe to Cholesky-factor.

    Kendall's tau matrix is PSD on complete data, but the entrywise sin()
    transform is not guaranteed to preserve that, so check and repair. Higham
    returns a PSD matrix, which can still be singular, so nudge it to positive
    definite afterwards -- a singular R breaks the Cholesky root the copula
    densities need.
    """
    R = np.asarray(R, dtype=float)
    if np.linalg.eigvalsh((R + R.T) / 2).min() < -tol:
        R = higham_nearest_psd(R)
    # Higham can return a matrix that is asymmetric in the last bits and only
    # just PSD. Both break the Cholesky.
    R = (R + R.T) / 2
    if np.linalg.eigvalsh(R).min() < ridge:
        R = (R + ridge * np.eye(R.shape[0])) / (1 + ridge)
    return R


def kendall_correlation(X) -> np.ndarray:
    """Correlation matrix from Kendall's tau: rho = sin(pi * tau / 2).

    Exact for any elliptical distribution, and nu does not appear in it, so R
    can be estimated once and held fixed while nu is searched.
    """
    tau = pd.DataFrame(np.asarray(X, dtype=float)).corr(method="kendall").to_numpy()
    R = np.sin(np.pi * tau / 2)
    np.fill_diagonal(R, 1.0)
    return fix_correlation(R)


def mvt_scale(sd: np.ndarray, R: np.ndarray, nu: float) -> np.ndarray:
    """A multivariate t carries a scale matrix, not a covariance:
    cov = nu/(nu-2) * S. So at a given nu the scale of variable i has to be
    sd_i * sqrt((nu-2)/nu), and every parameter becomes a function of nu."""
    d = np.asarray(sd) * np.sqrt((nu - 2) / nu)
    return np.outer(d, d) * R


def mvt_loglikelihood(X, mu, sd, R, nu) -> float:
    dist = stats.multivariate_t(loc=mu, shape=mvt_scale(sd, R, nu), df=nu)
    return float(np.sum(dist.logpdf(np.asarray(X, dtype=float))))


def profile_nu(ll, lo: float = 0.01, hi: float = 0.49, n: int = 200):
    """Maximize ll(nu) over nu, searching on theta = 1/nu so the grid
    resolution goes where it matters. The interval stops at 0.49 rather than
    0.5 to keep nu off its lower bound, where the scale collapses to zero and
    S stops being invertible. Returns (nu, ll at nu)."""
    thetas = np.linspace(lo, hi, n)
    lls = np.array([ll(1.0 / th) for th in thetas])
    i = int(np.argmax(lls))

    # refine between the neighbours of the grid maximum
    fine = np.linspace(thetas[max(i - 1, 0)], thetas[min(i + 1, n - 1)], n)
    fine_lls = np.array([ll(1.0 / th) for th in fine])
    j = int(np.argmax(fine_lls))

    return 1.0 / fine[j], float(fine_lls[j])


def fit_multivariate_t(X):
    """Returns the fitted mean vector, scale matrix S, degrees of freedom nu,
    and log likelihood."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    R = kendall_correlation(X)

    nu, ll = profile_nu(lambda nu: mvt_loglikelihood(X, mu, sd, R, nu))

    return mu, mvt_scale(sd, R, nu), nu, ll


def simulate_mvt(mu, S, nu: float, n_sim: int, seed: int = 1234) -> np.ndarray:
    """The mixture representation: one chi-square draw scales the whole normal
    vector, and that common shock is what puts weight in the joint tail."""
    Z = simulate_pca(S, n_sim, seed=seed)
    w = nu / np.random.default_rng(seed + 1).chisquare(nu, n_sim)
    return np.sqrt(w)[:, None] * Z + np.asarray(mu)
