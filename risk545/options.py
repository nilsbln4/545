"""Option pricing: generalized Black-Scholes-Merton with Greeks, and implied volatility.

Follows library/gbsm.jl. The cost of carry b selects the model:
    b = rf        Black-Scholes 1973 (no dividends)
    b = rf - q    Merton 1973, continuous dividend yield q
    b = 0         Black 1976 futures options
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class GBSM:
    value: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    c_rho: float = 0.0     # sensitivity to the cost of carry b


def gbsm(call: bool, underlying: float, strike: float, ttm: float, rf: float, b: float,
         ivol: float, include_greeks: bool = False) -> GBSM:
    """Price (and optionally Greeks) of a European option. Vega, rho and c_rho are per
    1.00 change (not per point); theta is dV/dt in years (negative for time decay)."""
    s, k, t, v = underlying, strike, ttm, ivol
    sqt = np.sqrt(t)
    d1 = (np.log(s / k) + (b + v**2 / 2) * t) / (v * sqt)
    d2 = d1 - v * sqt
    carry = np.exp((b - rf) * t)
    disc = np.exp(-rf * t)

    if call:
        delta = carry * norm.cdf(d1)
        value = s * delta - k * disc * norm.cdf(d2)
    else:
        delta = carry * (norm.cdf(d1) - 1)
        value = k * disc * norm.cdf(-d2) - s * carry * norm.cdf(-d1)

    if not include_greeks:
        return GBSM(value=value, delta=delta)

    pdf1 = norm.pdf(d1)
    gamma = pdf1 * carry / (s * v * sqt)
    vega = s * carry * pdf1 * sqt
    if call:
        theta = (-s * carry * pdf1 * v / (2 * sqt)
                 - (b - rf) * s * carry * norm.cdf(d1)
                 - rf * k * disc * norm.cdf(d2))
        rho = t * k * disc * norm.cdf(d2)
        c_rho = t * s * carry * norm.cdf(d1)
    else:
        theta = (-s * carry * pdf1 * v / (2 * sqt)
                 + (b - rf) * s * carry * norm.cdf(-d1)
                 + rf * k * disc * norm.cdf(-d2))
        rho = -t * k * disc * norm.cdf(-d2)
        c_rho = -t * s * carry * norm.cdf(-d1)
    return GBSM(value, delta, gamma, vega, theta, rho, c_rho)


def european_bounds(call: bool, underlying: float, strike: float, ttm: float, rf: float,
                    b: float) -> tuple:
    """No-arbitrage price bounds of a European option: (lower, upper). The GBSM price
    tends to the lower bound as vol -> 0 and to the upper bound as vol -> infinity."""
    fwd_leg = underlying * np.exp((b - rf) * ttm)
    pv_k = strike * np.exp(-rf * ttm)
    if call:
        return max(fwd_leg - pv_k, 0.0), fwd_leg
    return max(pv_k - fwd_leg, 0.0), pv_k


def implied_vol(call: bool, underlying: float, strike: float, ttm: float, rf: float, b: float,
                price: float, lo: float = 1e-6, hi: float = 5.0, tol: float = 1e-12) -> float:
    """The vol at which the GBSM price equals `price` (the instructor solves the same
    equation with find_zero). Returns NaN when no vol in [lo, hi] reproduces the price,
    i.e. the price is at or below the no-arbitrage lower bound (intrinsic value on the
    forward) or at or above the upper bound."""
    f = lambda x: gbsm(call, underlying, strike, ttm, rf, b, x).value - price
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    if f_lo == 0:
        return lo
    return brentq(f, lo, hi, xtol=tol)


# ----------------------------------------------------------------------------
# Binomial trees (library/bt_american.jl)
# ----------------------------------------------------------------------------
# Cox-Ross-Rubinstein tree: u = e^{vol sqrt(dt)}, d = 1/u, pu = (e^{b dt} - d) / (u - d).
# Node i at step j (i up moves out of j) has price S u^i d^(j-i).
#
# The Julia code walks the tree node by node. Python loops are too slow for the
# non-recombining dividend tree (a fresh sub-tree starts at every node on each
# dividend date), so here every step is a numpy operation over all nodes at once,
# and a whole batch of sub-trees (one per starting price) is rolled back together.
# The arithmetic is the same as the Julia code, node for node.

_CHUNK = 4000        # sub-trees rolled back together; bounds memory for large N


def _roll_back(call, s0, strike, n_steps, u, d, pu, df, american, terminal):
    """Backward induction over n_steps for a batch of starting prices s0 (shape (m,)).
    terminal: option values at step n_steps, shape (m, n_steps + 1)."""
    z = 1.0 if call else -1.0
    v = terminal
    for j in range(n_steps - 1, -1, -1):
        v = df * (pu * v[:, 1:j + 2] + (1 - pu) * v[:, 0:j + 1])
        if american:
            i = np.arange(j + 1)
            price = s0[:, None] * u**i * d**(j - i)
            v = np.maximum(v, np.maximum(0.0, z * (price - strike)))
    return v[:, 0]


def _tree(call, s0, strike, ttm, rf, b, vol, n, divs, american):
    """Values for a batch of starting prices. divs: list of (step, amount) with steps
    measured from the start of this (sub-)tree."""
    if len(s0) > _CHUNK:
        return np.concatenate([_tree(call, s0[k:k + _CHUNK], strike, ttm, rf, b, vol, n, divs, american)
                               for k in range(0, len(s0), _CHUNK)])
    dt = ttm / n
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    df = np.exp(-rf * dt)
    z = 1.0 if call else -1.0

    if not divs or divs[0][0] > n:
        i = np.arange(n + 1)
        price = s0[:, None] * u**i * d**(n - i)
        return _roll_back(call, s0, strike, n, u, d, pu, df, american, np.maximum(0.0, z * (price - strike)))

    k, amt = divs[0]
    i = np.arange(k + 1)
    price = s0[:, None] * u**i * d**(k - i)              # cum-dividend prices on the dividend date
    rest = [(step - k, a) for step, a in divs[1:]]
    # Value of holding through the dividend: a new tree from the ex-dividend price.
    hold = _tree(call, (price - amt).ravel(), strike, ttm - k * dt, rf, b, vol, n - k, rest, american)
    hold = hold.reshape(price.shape)
    v = np.maximum(hold, np.maximum(0.0, z * (price - strike))) if american else hold
    return _roll_back(call, s0, strike, k, u, d, pu, df, american, v)


def bt_american(call: bool, underlying: float, strike: float, ttm: float, rf: float, b: float,
                ivol: float, n: int, american: bool = True) -> float:
    """American option on a recombining binomial tree with cost of carry b (continuous
    dividend yield q gives b = rf - q). american=False gives the European value."""
    return float(_tree(call, np.array([float(underlying)]), strike, ttm, rf, b, ivol, n, [], american)[0])


def bt_american_div(call: bool, underlying: float, strike: float, ttm: float, rf: float,
                    div_amts, div_steps, ivol: float, n: int, american: bool = True) -> float:
    """American option with discrete cash dividends on a non-recombining tree
    (library/bt_american.jl, second method). div_steps are the dividend dates as
    step indices on the grid 0..n; div_amts the cash amounts. On a dividend date the
    holder compares exercising cum-dividend with holding into a new tree started
    from the ex-dividend price. Between dividends the cost of carry is rf.
    american=False gives the European value on the same tree."""
    divs = [(int(s), float(a)) for s, a in zip(div_steps, div_amts)]
    return float(_tree(call, np.array([float(underlying)]), strike, ttm, rf, rf, ivol, n, divs, american)[0])


def _central_step(x: float) -> float:
    """FiniteDiff.jl's default central-difference step: cbrt(eps) x max(|x|, 1)."""
    return float(np.cbrt(np.finfo(float).eps) * max(abs(x), 1.0))


def bt_american_greeks(call: bool, underlying: float, strike: float, ttm: float, rf: float, b: float,
                       ivol: float, n: int, gamma_bump: float = 1.5) -> dict:
    """Value and finite-difference Greeks of an American option on the recombining tree,
    as the reference for functional test 12.2 computes them (testfiles/test_setup.jl):

    delta, vega, rho, theta -- central differences of the tree value in S, ivol, rf and
        ttm with FiniteDiff.jl's default step. rho moves rf with b held fixed (b is its
        own argument), and theta is dV/d(ttm): positive when more time adds value. That
        is the sign the reference output uses (gbsm's theta is the opposite, dV/dt).
    gamma -- (V(S+d) - 2 V(S) + V(S-d)) / d^2 with d = 1.5. The bump has to be large
        because the tree value is piecewise linear in S between nodes.
    """
    p = {"underlying": float(underlying), "strike": float(strike), "ttm": float(ttm),
         "rf": float(rf), "b": float(b), "ivol": float(ivol)}

    def value(q):
        return bt_american(call, q["underlying"], q["strike"], q["ttm"], q["rf"], q["b"], q["ivol"], n)

    def bumped(name, h):
        q = dict(p)
        q[name] += h
        return value(q)

    def central(name):
        h = _central_step(p[name])
        return (bumped(name, h) - bumped(name, -h)) / (2 * h)

    v0 = value(p)
    d = gamma_bump
    return {"value": v0,
            "delta": central("underlying"),
            "gamma": (bumped("underlying", d) - 2 * v0 + bumped("underlying", -d)) / d**2,
            "vega": central("ivol"),
            "rho": central("rf"),
            "theta": central("ttm")}
