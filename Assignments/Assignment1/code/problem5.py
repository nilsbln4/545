"""Problem 5 -- Identifying an AR or MA Order.

Plots the series, its ACF, and its PACF, then fits AR(1)-AR(3) and
MA(1)-MA(3) and compares them with AICc.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from common import DATA_DIR, FIGURES_DIR, aicc, section

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_arma(x: np.ndarray, order: tuple) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(x, order=order, trend="c")
        res = model.fit()
    n = len(x)
    k = len(res.params)  # includes const, ar/ma coefficients, and sigma2
    return {"order": order, "loglik": res.llf, "k": k, "aicc": aicc(res.llf, k, n),
             "params": res.params}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem5.csv")
    x = df["x"].to_numpy()
    n = len(x)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    axes[0].plot(x, color="steelblue", lw=1)
    axes[0].set_title("problem5.csv: series")
    axes[0].set_xlabel("t")
    plot_acf(x, ax=axes[1], lags=20, title="ACF")
    plot_pacf(x, ax=axes[2], lags=20, title="PACF", method="ywm")
    fig.tight_layout()
    out_path = FIGURES_DIR / "problem5_series_acf_pacf.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved series/ACF/PACF plot to {out_path.relative_to(DATA_DIR)}")

    specs = [("AR", (1, 0, 0)), ("AR", (2, 0, 0)), ("AR", (3, 0, 0)),
             ("MA", (0, 0, 1)), ("MA", (0, 0, 2)), ("MA", (0, 0, 3))]

    section("Fitted models")
    results = []
    for name, order in specs:
        r = fit_arma(x, order)
        results.append((name, order, r))
        p = order[0]
        q = order[2]
        label = f"{name}({p if name == 'AR' else q})"
        print(f"{label:<8} k={r['k']}  loglik={r['loglik']:.4f}  AICc={r['aicc']:.4f}")
        print(f"         params: {np.round(r['params'], 4)}")

    best_name, best_order, best_r = min(results, key=lambda t: t[2]["aicc"])
    p, q = best_order[0], best_order[2]
    best_label = f"{best_name}({p if best_name == 'AR' else q})"

    section("Model selected by AICc")
    print(f"{best_label}  (AICc = {best_r['aicc']:.4f})")

    ar2 = next(r for name, order, r in results if order == (2, 0, 0))
    ar3 = next(r for name, order, r in results if order == (3, 0, 0))
    section("AR(2) vs AR(3), for the reconciliation writeup")
    print(f"AR(2): loglik={ar2['loglik']:.4f}  AICc={ar2['aicc']:.4f}  params={np.round(ar2['params'], 4)}")
    print(f"AR(3): loglik={ar3['loglik']:.4f}  AICc={ar3['aicc']:.4f}  params={np.round(ar3['params'], 4)}")


if __name__ == "__main__":
    main()
