"""Gaussian and t copulas: densities, fitting, simulation, tail dependence.

Port of library/copula.jl. A copula density is the joint density divided by the
product of the marginal densities of the transformed variables. The two models
here differ by exactly one free parameter, nu, which is what makes the
comparison clean.
"""

import numpy as np
from scipy import stats

from .multivariate_t import kendall_correlation, profile_nu
from .simulate import simulate_pca


def gaussian_copula_ll(U, R) -> float:
    Z = stats.norm.ppf(np.asarray(U, dtype=float))
    joint = stats.multivariate_normal(mean=np.zeros(Z.shape[1]), cov=R).logpdf(Z).sum()
    margins = stats.norm.logpdf(Z).sum()
    return float(joint - margins)


def t_copula_ll(U, R, nu: float) -> float:
    T = stats.t.ppf(np.asarray(U, dtype=float), df=nu)
    joint = stats.multivariate_t(loc=np.zeros(T.shape[1]), shape=R, df=nu).logpdf(T).sum()
    margins = stats.t.logpdf(T, df=nu).sum()
    return float(joint - margins)


def fit_gaussian_copula(U):
    """Returns (R, log likelihood). tau is rank based, so it does not matter
    whether it is computed on the returns or on U; the same R serves both
    copulas."""
    R = kendall_correlation(U)
    return R, gaussian_copula_ll(U, R)


def fit_t_copula(U):
    """Returns (R, nu, log likelihood). The bounds put nu in [2, 100]; past 100
    the t copula and the Gaussian are indistinguishable at any sample size we
    will have."""
    R = kendall_correlation(U)
    nu, ll = profile_nu(lambda nu: t_copula_ll(U, R, nu))
    return R, nu, ll


def simulate_gaussian_copula(R, n_sim: int, seed: int = 1234) -> np.ndarray:
    Z = simulate_pca(R, n_sim, seed=seed)
    return stats.norm.cdf(Z)


def simulate_t_copula(R, nu: float, n_sim: int, seed: int = 1234) -> np.ndarray:
    """Same mixture shock as the multivariate t, then back through the t CDF.
    The common chi-square draw is what leaves dependence in the joint tail."""
    Z = simulate_pca(R, n_sim, seed=seed)
    w = nu / np.random.default_rng(seed + 1).chisquare(nu, n_sim)
    return stats.t.cdf(np.sqrt(w)[:, None] * Z, df=nu)


def tail_dependence_t(rho: float, nu: float) -> float:
    """Lower tail dependence coefficient of the t copula. It is exactly 0 under
    the Gaussian copula, whatever the correlation."""
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)


def copula_aicc(ll: float, k: int, m: int) -> float:
    return -2 * ll + 2 * k + (2 * k ** 2 + 2 * k) / (m - k - 1)


def copula_bic(ll: float, k: int, m: int) -> float:
    return k * np.log(m) - 2 * ll
