"""Repairing non-PSD matrices, and Cholesky factorization that tolerates PSD input.

Port of library/simulate.jl's near_psd, chol_psd!, and higham_nearestPSD.
"""

import numpy as np


def _is_correlation(a: np.ndarray) -> bool:
    return np.allclose(np.diag(a), 1.0)


def near_psd(a: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """Rebonato-Jackel nearest-PSD repair: clip negative eigenvalues, rescale."""
    a = np.asarray(a, dtype=float)
    out = a.copy()

    inv_sd = None
    if not _is_correlation(out):
        inv_sd = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = inv_sd @ out @ inv_sd

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)

    t = 1.0 / ((vecs * vecs) @ vals)
    t = np.diag(np.sqrt(t))
    root_vals = np.diag(np.sqrt(vals))
    b = t @ vecs @ root_vals
    out = b @ b.T

    if inv_sd is not None:
        inv_sd = np.diag(1.0 / np.diag(inv_sd))
        out = inv_sd @ out @ inv_sd

    return out


def chol_psd(a: np.ndarray, epsilon: float = -1e-8) -> np.ndarray:
    """Cholesky factor of a PSD matrix. A diagonal that lands in [epsilon, 0]
    is clamped to exactly 0 rather than left as tiny floating-point noise;
    once a diagonal is 0, the rest of that row is left at its zero-init."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = root[j, :j] @ root[j, :j] if j > 0 else 0.0
        temp = a[j, j] - s
        if epsilon <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            continue

        ir = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = root[i, :j] @ root[j, :j]
            root[i, j] = (a[i, j] - s) * ir

    return root


def _get_a_plus(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(a)
    vals = np.diag(np.maximum(vals, 0.0))
    return vecs @ vals @ vecs.T


def _get_ps(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    w05 = np.sqrt(w)
    iw = np.linalg.inv(w05)
    return iw @ _get_a_plus(w05 @ a @ w05) @ iw


def _get_pu(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    np.fill_diagonal(out, 1.0)
    return out


def _wgt_norm(a: np.ndarray, w: np.ndarray) -> float:
    w05 = np.sqrt(w)
    m = w05 @ a @ w05
    return float(np.sum(m * m))


def higham_nearest_psd(a: np.ndarray, w: np.ndarray = None,
                        epsilon: float = 1e-9, max_iter: int = 100,
                        tol: float = 1e-9) -> np.ndarray:
    """Higham (2002) alternating-projections nearest correlation matrix."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    if w is None:
        w = np.eye(n)

    delta_s = np.zeros((n, n))
    y_k = a.copy()

    inv_sd = None
    if not _is_correlation(y_k):
        inv_sd = np.diag(1.0 / np.sqrt(np.diag(y_k)))
        y_k = inv_sd @ y_k @ inv_sd

    y_o = y_k.copy()
    norm_last = np.inf

    for _ in range(max_iter):
        r_k = y_k - delta_s
        x_k = _get_ps(r_k, w)
        delta_s = x_k - r_k
        y_k = _get_pu(x_k)
        norm = _wgt_norm(y_k - y_o, w)
        min_eig = np.min(np.linalg.eigvalsh(y_k))

        if abs(norm - norm_last) < tol and min_eig > -epsilon:
            break
        norm_last = norm

    if inv_sd is not None:
        inv_sd = np.diag(1.0 / np.diag(inv_sd))
        y_k = inv_sd @ y_k @ inv_sd

    return y_k
