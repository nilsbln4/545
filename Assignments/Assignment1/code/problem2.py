"""Problem 2 -- A Regression Whose Errors Are Not Normal.

Fits y ~ alpha + beta * x on problem2.csv three ways -- OLS, MLE under a
Normal error, MLE under a Student's t error -- and compares them with AICc.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

from common import DATA_DIR, FIGURES_DIR, aicc, section

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = beta_hat
    resid = y - X @ beta_hat

    # Classical OLS standard errors: sigma^2 uses the n-2 (unbiased) divisor.
    dof = n - 2
    sigma2_unbiased = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2_unbiased * xtx_inv))

    return {
        "alpha": alpha,
        "beta": beta,
        "se_alpha": se[0],
        "se_beta": se[1],
        "sigma": np.sqrt(sigma2_unbiased),
        "resid": resid,
    }


def fit_mle_normal(x: np.ndarray, y: np.ndarray, ols: dict) -> dict:
    n = len(x)
    # Closed form: the mean parameters that maximize a Normal likelihood
    # are the OLS estimates. Only sigma's estimator differs (divide by n,
    # not n-2), since the MLE has no small-sample correction.
    alpha, beta = ols["alpha"], ols["beta"]
    resid = ols["resid"]
    sigma_mle = np.sqrt((resid @ resid) / n)
    loglik = stats.norm.logpdf(resid, loc=0, scale=sigma_mle).sum()
    return {"alpha": alpha, "beta": beta, "sigma": sigma_mle, "loglik": loglik, "k": 3}


def fit_mle_t(x: np.ndarray, y: np.ndarray, ols: dict) -> dict:
    n = len(x)

    def neg_loglik(params):
        alpha, beta, log_scale, log_nu = params
        scale = np.exp(log_scale)
        nu = np.exp(log_nu)
        resid = y - alpha - beta * x
        return -stats.t.logpdf(resid, df=nu, loc=0, scale=scale).sum()

    x0 = [ols["alpha"], ols["beta"], np.log(ols["sigma"]), np.log(10.0)]
    result = optimize.minimize(neg_loglik, x0, method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000})
    alpha, beta, log_scale, log_nu = result.x
    scale, nu = np.exp(log_scale), np.exp(log_nu)
    return {"alpha": alpha, "beta": beta, "scale": scale, "nu": nu,
             "loglik": -result.fun, "k": 4, "converged": result.success}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem2.csv")
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(x)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.6, color="steelblue", edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("problem2.csv: y vs x")
    fig.tight_layout()
    out_path = FIGURES_DIR / "problem2_scatter.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved scatter to {out_path.relative_to(DATA_DIR)}")

    ols = fit_ols(x, y)
    section("Model 1: OLS")
    print(f"alpha       = {ols['alpha']:.6f}  (se = {ols['se_alpha']:.6f})")
    print(f"beta        = {ols['beta']:.6f}  (se = {ols['se_beta']:.6f})")
    print(f"sigma (n-2) = {ols['sigma']:.6f}")

    mle_n = fit_mle_normal(x, y, ols)
    section("Model 2: MLE, Normal error")
    print(f"alpha = {mle_n['alpha']:.6f}")
    print(f"beta  = {mle_n['beta']:.6f}")
    print(f"sigma = {mle_n['sigma']:.6f}")
    print(f"log-likelihood = {mle_n['loglik']:.4f}")

    mle_t = fit_mle_t(x, y, ols)
    section("Model 3: MLE, Student's t error")
    print(f"alpha      = {mle_t['alpha']:.6f}")
    print(f"beta       = {mle_t['beta']:.6f}")
    print(f"scale      = {mle_t['scale']:.6f}")
    print(f"nu (dof)   = {mle_t['nu']:.6f}")
    print(f"log-likelihood = {mle_t['loglik']:.4f}")
    print(f"optimizer converged = {mle_t['converged']}")

    aicc_n = aicc(mle_n["loglik"], mle_n["k"], n)
    aicc_t = aicc(mle_t["loglik"], mle_t["k"], n)

    section("Model comparison (AICc)")
    print(f"{'model':<22}{'k':>4}{'loglik':>14}{'AICc':>14}")
    print(f"{'MLE Normal':<22}{mle_n['k']:>4}{mle_n['loglik']:>14.4f}{aicc_n:>14.4f}")
    print(f"{'MLE Student t':<22}{mle_t['k']:>4}{mle_t['loglik']:>14.4f}{aicc_t:>14.4f}")
    print(f"delta AICc (Normal - t) = {aicc_n - aicc_t:.4f}")
    print("\nNote: OLS's alpha/beta are numerically identical to the Normal MLE's;")
    print("only the sigma divisor differs (n-2 vs n), so OLS is not given a separate")
    print("AICc row -- it is the same model as 'MLE Normal' up to that dof correction.")

    best_name, best_aicc = ("MLE Student t", aicc_t) if aicc_t < aicc_n else ("MLE Normal", aicc_n)
    section("Model selected by AICc")
    print(f"{best_name}  (AICc = {best_aicc:.4f})")

    # Quantiles of the two fitted error distributions, at 95% and 99.5%.
    q_levels = [0.95, 0.995]
    normal_q = [stats.norm.ppf(q, loc=0, scale=mle_n["sigma"]) for q in q_levels]
    t_q = [stats.t.ppf(q, df=mle_t["nu"], loc=0, scale=mle_t["scale"]) for q in q_levels]

    section("Error-distribution quantiles")
    print(f"{'quantile':<12}{'Normal':>12}{'Student t':>12}")
    for q, nq, tq in zip(q_levels, normal_q, t_q):
        print(f"{q:<12}{nq:>12.6f}{tq:>12.6f}")


if __name__ == "__main__":
    main()
