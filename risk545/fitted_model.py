"""Distribution and regression fitting.

Port of library/fitted_model.jl. Where Julia used Ipopt for the MLE
optimizations (fit_general_t, fit_regression_t), this uses
scipy.optimize.minimize -- a different optimizer converging on the same
likelihood surface, so results match to numerical tolerance rather than
bit for bit.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import optimize, stats


@dataclass
class FittedModel:
    """Holds everything downstream code (VaR/ES, copula simulation, later in
    the semester) needs from a fit: the error distribution, any regression
    beta, the realized errors and their probability-integral-transform (u)
    values, and an `eval` that maps u back to the modeled variable.

    `eval` here is just a field name -- mirroring the Julia struct's
    `eval::Function` field -- holding a distribution's `.ppf` or a small
    closure. It is never Python's builtin `eval()`; nothing in this module
    calls that.

    `params` holds the named error-distribution parameters (e.g. mu/sigma,
    or mu/alpha/beta/delta for the NIG) as plain floats. Prefer reading
    parameters from here over reaching into `error_model`'s scipy internals,
    which differ in shape depending on how each distribution was frozen.
    """
    beta: Optional[np.ndarray]
    error_model: object          # a frozen scipy.stats distribution
    eval: Callable
    errors: np.ndarray
    u: np.ndarray
    n_error_params: int          # parameter count of error_model, for AICc
    params: dict = field(default_factory=dict)


def aicc(model, data, n_error_params: int = None) -> float:
    """Corrected AIC.

    `model` can be a FittedModel (k is read off it automatically, including
    any regression beta), or a bare frozen scipy.stats distribution together
    with n_error_params. Either way, `data` is run through the error model's
    logpdf directly -- pass the raw sample for a location fit (fit_normal,
    fit_general_t, the NIG fits), or the regression residuals for a
    fit_regression_t model, whose error_model has no location of its own.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if isinstance(model, FittedModel):
        d = model.error_model
        k = model.n_error_params + (len(model.beta) if model.beta is not None else 0)
    else:
        if n_error_params is None:
            raise ValueError("n_error_params is required when model is a raw distribution")
        d = model
        k = n_error_params

    ll = np.sum(d.logpdf(data))
    return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

def fit_normal(x: np.ndarray) -> FittedModel:
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=1)

    error_model = stats.norm(loc=m, scale=s)
    errors = x - m
    u = error_model.cdf(x)

    return FittedModel(beta=None, error_model=error_model, eval=error_model.ppf,
                        errors=errors, u=u, n_error_params=2,
                        params={"mu": m, "sigma": s})


# ---------------------------------------------------------------------------
# Generalized (location-scale) Student's t
# ---------------------------------------------------------------------------

def _t_start_params(resid: np.ndarray):
    k = stats.kurtosis(resid, fisher=True, bias=True)
    start_nu = 6.0 / k + 4.0 if k > 0 else 10.0
    start_nu = max(start_nu, 2.1)
    start_s = np.sqrt(resid.var(ddof=1) * (start_nu - 2) / start_nu)
    return start_s, start_nu


def fit_general_t(x: np.ndarray) -> FittedModel:
    x = np.asarray(x, dtype=float)
    start_m = x.mean()
    start_s, start_nu = _t_start_params(x - start_m)

    def neg_loglik(params):
        mu, log_s, log_nu_m2 = params
        s = np.exp(log_s)
        nu = 2.0 + np.exp(log_nu_m2)
        return -np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=s))

    x0 = [start_m, np.log(start_s), np.log(start_nu - 2)]
    res = optimize.minimize(neg_loglik, x0, method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    mu, log_s, log_nu_m2 = res.x
    s = np.exp(log_s)
    nu = 2.0 + np.exp(log_nu_m2)

    error_model = stats.t(df=nu, loc=mu, scale=s)
    errors = x - mu
    u = error_model.cdf(x)

    return FittedModel(beta=None, error_model=error_model, eval=error_model.ppf,
                        errors=errors, u=u, n_error_params=3,
                        params={"mu": mu, "sigma": s, "nu": nu})


# ---------------------------------------------------------------------------
# Regression with t-distributed errors
# ---------------------------------------------------------------------------

def fit_regression_t(y: np.ndarray, x: np.ndarray) -> FittedModel:
    y = np.asarray(y, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] != len(y):
        x = x.T
    n = len(y)
    design = np.column_stack([np.ones(n), x])

    beta_ols, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta_ols
    start_s, start_nu = _t_start_params(resid)

    def neg_loglik(params):
        log_s, log_nu_m2 = params[0], params[1]
        beta = params[2:]
        s = np.exp(log_s)
        nu = 2.0 + np.exp(log_nu_m2)
        e = y - design @ beta
        return -np.sum(stats.t.logpdf(e, df=nu, loc=0.0, scale=s))

    x0 = np.concatenate([[np.log(start_s), np.log(start_nu - 2)], beta_ols])
    res = optimize.minimize(neg_loglik, x0, method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 30000})
    log_s, log_nu_m2 = res.x[0], res.x[1]
    beta = res.x[2:]
    s = np.exp(log_s)
    nu = 2.0 + np.exp(log_nu_m2)

    error_model = stats.t(df=nu, loc=0.0, scale=s)

    def eval_fn(x_new, u):
        x_new = np.atleast_2d(np.asarray(x_new, dtype=float))
        if x_new.shape[1] != x.shape[1]:
            x_new = x_new.T
        design_new = np.column_stack([np.ones(x_new.shape[0]), x_new])
        return design_new @ beta + error_model.ppf(u)

    errors = y - eval_fn(x, np.full(n, 0.5))
    u = error_model.cdf(errors)

    params = {"sigma": s, "nu": nu, "Alpha": beta[0]}
    params.update({f"B{i}": b for i, b in enumerate(beta[1:], start=1)})

    return FittedModel(beta=beta, error_model=error_model, eval=eval_fn,
                        errors=errors, u=u, n_error_params=2, params=params)


# ---------------------------------------------------------------------------
# Normal Inverse Gaussian
# ---------------------------------------------------------------------------

def fit_nig_moments(x: np.ndarray) -> FittedModel:
    """Closed-form method of moments: matches the sample's mean, variance,
    skewness, and excess kurtosis exactly."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    v = x.var(ddof=1)
    skew = stats.skew(x, bias=True)
    kurt = stats.kurtosis(x, fisher=True, bias=True)

    if kurt <= 0:
        raise ValueError(f"NIG method of moments needs positive excess kurtosis, got {kurt}")
    t = skew ** 2 / kurt
    if not (t < 3 / 5):
        raise ValueError("Sample is outside the NIG region: excess kurtosis must exceed (5/3)*skew^2")

    rho2 = t / (3 - 4 * t)
    rho = np.sign(skew) * np.sqrt(rho2)

    d_gamma = 3 * (1 + 4 * rho2) / kurt  # delta * gamma
    alpha = np.sqrt(d_gamma / (v * (1 - rho2) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho2)
    delta = d_gamma / gamma
    mu = m - delta * beta / gamma

    return _nig_fitted_model(mu, alpha, beta, delta, x)


def fit_nig_mle(x: np.ndarray) -> FittedModel:
    """Maximum likelihood via scipy.stats.norminvgauss.fit."""
    x = np.asarray(x, dtype=float)
    a, b, loc, scale = stats.norminvgauss.fit(x)
    mu, delta = loc, scale
    alpha, beta = a / delta, b / delta

    return _nig_fitted_model(mu, alpha, beta, delta, x)


def _nig_fitted_model(mu, alpha, beta, delta, x) -> FittedModel:
    a, b = alpha * delta, beta * delta
    error_model = stats.norminvgauss(a, b, loc=mu, scale=delta)
    errors = x - error_model.mean()
    u = error_model.cdf(x)

    return FittedModel(beta=None, error_model=error_model, eval=error_model.ppf,
                        errors=errors, u=u, n_error_params=4,
                        params={"mu": mu, "alpha": alpha, "beta": beta, "delta": delta})
