"""Portfolio optimizers (library/optimizers.jl).

max_sharpe follows maxSR: maximize (w'mu - rf) / sqrt(w' S w) subject to
sum(w) = 1 and per-asset bounds. The instructor solves it with Ipopt; here SLSQP
from several starting points. With no bounds the tangency portfolio has the closed
form w ~ S^-1 (mu - rf), scaled to sum to one.
"""

import numpy as np
from scipy.optimize import minimize


def portfolio_sharpe(w, mean, cov, rf) -> float:
    w = np.asarray(w, dtype=float)
    return float((w @ mean - rf) / np.sqrt(w @ cov @ w))


def max_sharpe(cov, mean, rf: float, bounds=None) -> np.ndarray:
    """Weights of the maximum Sharpe ratio portfolio. bounds: (n, 2) array of
    lower / upper limits per asset; None means unconstrained (shorting allowed),
    and the default of the instructor's maxSR without bounds is long only, [0, 1],
    which callers pass explicitly."""
    cov = np.asarray(cov, dtype=float)
    mean = np.asarray(mean, dtype=float)
    n = len(mean)
    if bounds is None:
        w = np.linalg.solve(cov, mean - rf)
        return w / w.sum()

    bounds = np.asarray(bounds, dtype=float)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    neg_sr = lambda w: -portfolio_sharpe(w, mean, cov, rf)
    starts = [np.full(n, 1.0 / n)] + [np.eye(n)[i] for i in range(n)]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, bounds[:, 0], bounds[:, 1])
        res = minimize(neg_sr, x0, method="SLSQP", bounds=[tuple(b) for b in bounds], constraints=cons,
                       options={"ftol": 1e-14, "maxiter": 1000})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("max_sharpe: optimizer did not converge from any start")
    return best.x


def min_variance_for_return(cov, mean, target: float, bounds) -> np.ndarray:
    """Minimum variance weights with w'mu = target, sum(w) = 1 and bounds
    (one point of the efficient frontier)."""
    cov = np.asarray(cov, dtype=float)
    mean = np.asarray(mean, dtype=float)
    n = len(mean)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w: w @ mean - target}]
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[tuple(b) for b in np.asarray(bounds, dtype=float)], constraints=cons,
                   options={"ftol": 1e-15, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"min_variance_for_return: {res.message}")
    return res.x
